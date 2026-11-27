"""OTDFTiered: official OTDF with a separate near layer of source data (adaptive-source batch, groups rho*T / rho80R).

Only train() of the official class (code/otdf_official/algo/OTDF.py, commit 41e2e62, unchanged) is overridden;
configure_tiers() is a new method that stores the layer settings. Everything else (networks, VAE pre-training,
update_q_functions, update_target, select_action, save) is the official code.

Per step (batch_size = 128 as the official config):
  target batch : 128 rows from the official target ReplayBuffer.sample (unchanged)
  far batch    : 128 rows drawn uniformly from all source rows outside D_near, then the official in-batch
                 normalisation (c - max) / (max - min), topk(int(128 * proportion)) and soft weight exp(c)
  near batch   : 64 rows drawn uniformly from D_near (without replacement when |D_near| >= 64, with replacement
                 when 0 < |D_near| < 64; no near rows when D_near is empty, counted in stats); soft weight
                 = exp((c - max_far) / (max_far - min_far)) with the max/min of the same step's 128-row far batch,
                 clipped to <= 1
  layer weights: target 1, far w_far, near w_near (0.5 / 0.5 in the experiments)
  Q loss       : official update_q_functions with self.weight = official soft weight x layer weight
  V loss       : official asymmetric L2 loss with every row multiplied by its layer weight
  actor        : bc_loss (B,) multiplied by the layer weight first, then the official exp_adv (B, 1) x bc (B,)
                 broadcast is kept unchanged; the CVAE support term is not weighted
With D_near empty and all layer weights 1 the step is numerically identical to OTDF.train (j013 unit test a).
"""
import numpy as np
import torch

from algo.OTDF import OTDF


class OTDFTiered(OTDF):

    def configure_tiers(self, n_source, near_index, w_target=1.0, w_far=0.5, w_near=0.5, near_batch=64, near_seed=0):
        near = np.asarray(near_index, dtype=np.int64).reshape(-1)
        assert len(np.unique(near)) == len(near), 'near_index has duplicates'
        assert len(near) == 0 or (near.min() >= 0 and near.max() < n_source), 'near_index out of range'
        mask = np.ones(int(n_source), bool)
        mask[near] = False
        self.near_index = np.sort(near)
        self.far_index = np.flatnonzero(mask).astype(np.int64)
        assert len(self.far_index) > 0, 'far layer is empty'
        self.w_target, self.w_far, self.w_near = float(w_target), float(w_far), float(w_near)
        self.near_batch = int(near_batch)
        self.near_rng = np.random.RandomState(int(near_seed))
        self.tier_stats = dict(steps=0, near_empty_steps=0, near_with_replacement_steps=0, near_rows=0)
        self._near_weight_sum = None
        self._near_weight_clipped = None
        self.last_losses = None

    def near_weight_stats(self):
        """Mean clipped soft weight of the near rows and fraction clipped at 1 (synchronises the device)."""
        n = self.tier_stats['near_rows']
        if not n or self._near_weight_sum is None:
            return dict(mean=None, clipped_fraction=None)
        return dict(mean=float(self._near_weight_sum)/n, clipped_fraction=float(self._near_weight_clipped)/n)

    def _near_rows(self, n, k):
        """k row positions in range(n): Floyd's algorithm (uniform subset, exactly k draws) if n >= k,
        otherwise k draws with replacement."""
        rng = self.near_rng
        if n >= k:
            chosen = set()
            out = np.empty(k, np.int64)
            for i, j in enumerate(range(n-k, n)):
                t = int(rng.randint(0, j+1))
                if t in chosen:
                    t = j
                chosen.add(t)
                out[i] = t
            return out
        self.tier_stats['near_with_replacement_steps'] += 1
        return rng.randint(0, n, size=k).astype(np.int64)

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1
        self.tier_stats['steps'] += 1

        # far layer: uniform over source rows outside D_near (identical to OTReplayBuffer.sample when D_near is empty)
        far = self.far_index
        ind = far[np.random.randint(0, len(far), size=batch_size)]
        src_state = torch.FloatTensor(src_replay_buffer.state[ind]).to(self.device)
        src_action = torch.FloatTensor(src_replay_buffer.action[ind]).to(self.device)
        src_next_state = torch.FloatTensor(src_replay_buffer.next_state[ind]).to(self.device)
        src_reward = torch.FloatTensor(src_replay_buffer.reward[ind]).to(self.device)
        src_not_done = torch.FloatTensor(src_replay_buffer.not_done[ind]).to(self.device)
        src_cost = torch.FloatTensor(src_replay_buffer.cost[ind]).to(self.device)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        src_cost_raw = src_cost
        # normalization (official)
        src_cost = (src_cost - torch.max(src_cost))/(torch.max(src_cost) - torch.min(src_cost))

        # filter out transitions (official)
        src_filter_num = int(batch_size * self.config['proportion'])

        filter_cost, indices = torch.topk(src_cost, src_filter_num)

        src_state = src_state[indices]
        src_action = src_action[indices]
        src_next_state = src_next_state[indices]
        src_reward = src_reward[indices]
        src_not_done = src_not_done[indices]
        src_cost = src_cost[indices]

        # near layer
        n_near_set = len(self.near_index)
        parts = [(src_state, src_action, src_next_state, src_reward, src_not_done)]
        near_w = None
        if n_near_set > 0:
            nind = self.near_index[self._near_rows(n_near_set, self.near_batch)]
            near_state = torch.FloatTensor(src_replay_buffer.state[nind]).to(self.device)
            near_action = torch.FloatTensor(src_replay_buffer.action[nind]).to(self.device)
            near_next_state = torch.FloatTensor(src_replay_buffer.next_state[nind]).to(self.device)
            near_reward = torch.FloatTensor(src_replay_buffer.reward[nind]).to(self.device)
            near_not_done = torch.FloatTensor(src_replay_buffer.not_done[nind]).to(self.device)
            near_cost = torch.FloatTensor(src_replay_buffer.cost[nind]).to(self.device)
            near_cost = (near_cost - torch.max(src_cost_raw))/(torch.max(src_cost_raw) - torch.min(src_cost_raw))
            near_w = torch.clamp(torch.exp(near_cost), max=1.0)
            parts.append((near_state, near_action, near_next_state, near_reward, near_not_done))
            self.tier_stats['near_rows'] += int(near_state.shape[0])
            s = near_w.sum().detach()
            c = (near_w >= 1.0).sum().detach()
            self._near_weight_sum = s if self._near_weight_sum is None else self._near_weight_sum + s
            self._near_weight_clipped = c if self._near_weight_clipped is None else self._near_weight_clipped + c
        else:
            self.tier_stats['near_empty_steps'] += 1
        parts.append((tar_state, tar_action, tar_next_state, tar_reward, tar_not_done))

        state = torch.cat([p[0] for p in parts], 0)
        action = torch.cat([p[1] for p in parts], 0)
        next_state = torch.cat([p[2] for p in parts], 0)
        reward = torch.cat([p[3] for p in parts], 0)
        not_done = torch.cat([p[4] for p in parts], 0)

        n_src = src_state.shape[0]
        n_near = 0 if near_w is None else near_w.shape[0]
        n_tar = tar_state.shape[0]
        layer = torch.cat([torch.full((n_src,), self.w_far), torch.full((n_near,), self.w_near),
                           torch.full((n_tar,), self.w_target)]).to(self.device)

        self.weight = torch.ones_like(reward.flatten()).to(self.device)

        if self.config['weight']:
            # calculate cost weight (official), near rows get their clipped soft weight
            cost_weight = torch.exp(src_cost)
            self.weight[:src_state.shape[0]] = cost_weight
            if n_near:
                self.weight[n_src:n_src + n_near] = near_w

            self.weight = self.weight.unsqueeze(1)

        # Q-loss weight = official soft weight x layer weight
        self.weight = self.weight * (layer.unsqueeze(1) if self.weight.dim() == 2 else layer)

        # V update: official update_v_function / asymmetric_l2_loss, every row multiplied by its layer weight
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state, action)
            q_t = torch.min(q_t1, q_t2)

        v = self.v_func(state)
        adv = q_t - v
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
        u = adv
        v_loss_step = torch.mean(layer.unsqueeze(1) * (torch.abs(self.lam - (u < 0).float()) * u**2))
        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        q_loss_step = self.update_q_functions(state, action, reward, next_state, not_done, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy: official update_policy with the BC term multiplied by the layer weight before the broadcast
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        exp_adv = torch.exp(self.temp * adv.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state, action) * layer
        policy_loss = torch.mean(exp_adv * bc_loss)

        # whether to add policy regularization (official, not weighted)
        if not self.config['noreg']:
            pi, _, _ = self.policy(state)
            log_beta = self.vae_policy.dataset_prob(state, pi, beta=0.4, num_samples=10)
            policy_loss -= self.config['reg_weight'] * log_beta.mean()
        pi_loss_step = policy_loss
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True
        self.last_losses = (v_loss_step.detach(), q_loss_step.detach(), pi_loss_step.detach())
