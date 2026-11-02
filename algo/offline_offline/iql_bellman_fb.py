import numpy as np
import torch
import torch.nn.functional as F

from offline_offline.iql_dyn import IQL_DYN


class IQL_BELLMAN_FB(IQL_DYN):

    def __init__(self, config, device, target_entropy=None):
        super().__init__(config, device, target_entropy)
        self.bellman_eta = config.get('bellman_eta', 1.0)
        self.normalize_bellman = config.get('normalize_bellman', True)
        self.target_state_bank = None
        self.target_action_bank = None
        self.target_next_state_bank = None
        self.target_reward_bank = None
        self.target_not_done_bank = None

    def build_sa_features(self, state, action):
        return torch.cat([state, action], dim=1)

    def initialize_target_feature_bank(self, tar_replay_buffer):
        size = tar_replay_buffer.size
        indices = np.arange(size)
        if size > self.target_bank_size:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(size, size=self.target_bank_size, replace=False)

        state = torch.FloatTensor(tar_replay_buffer.state[indices]).to(self.device)
        action = torch.FloatTensor(tar_replay_buffer.action[indices]).to(self.device)
        next_state = torch.FloatTensor(tar_replay_buffer.next_state[indices]).to(self.device)
        reward = torch.FloatTensor(tar_replay_buffer.reward[indices]).to(self.device)
        not_done = torch.FloatTensor(tar_replay_buffer.not_done[indices]).to(self.device)

        with torch.no_grad():
            target_features = self.build_sa_features(state, action)
            self.target_mean = target_features.mean(dim=0, keepdim=True)
            self.target_std = target_features.std(dim=0, keepdim=True) + 1e-6
            self.target_features_norm = ((target_features - self.target_mean) / self.target_std).detach()
            self.target_state_bank = state.detach()
            self.target_action_bank = action.detach()
            self.target_next_state_bank = next_state.detach()
            self.target_reward_bank = reward.detach()
            self.target_not_done_bank = not_done.detach()

    def compute_knn_indices(self, source_features_norm):
        k = min(self.knn_k, self.target_features_norm.shape[0])
        topk_values = None
        topk_indices = None
        bank_size = self.target_features_norm.shape[0]
        chunk_size = max(1, self.knn_chunk_size)

        with torch.no_grad():
            for start in range(0, bank_size, chunk_size):
                end = min(start + chunk_size, bank_size)
                dist = torch.cdist(source_features_norm, self.target_features_norm[start:end])
                chunk_k = min(k, dist.shape[1])
                values, indices = torch.topk(dist, k=chunk_k, largest=False, dim=1)
                indices = indices + start
                if topk_values is None:
                    topk_values = values
                    topk_indices = indices
                else:
                    candidate_values = torch.cat([topk_values, values], dim=1)
                    candidate_indices = torch.cat([topk_indices, indices], dim=1)
                    topk_values, keep = torch.topk(candidate_values, k=k, largest=False, dim=1)
                    topk_indices = torch.gather(candidate_indices, 1, keep)

        return topk_values.detach(), topk_indices.detach()

    def compute_source_weight(self, src_state, src_action, src_next_state, src_reward, src_not_done):
        with torch.no_grad():
            source_features = self.build_sa_features(src_state, src_action)
            source_features_norm = (source_features - self.target_mean) / self.target_std
            knn_dist, knn_indices = self.compute_knn_indices(source_features_norm)

            src_v_next = self.v_func(src_next_state)
            source_bellman_target = src_reward + src_not_done * self.discount * src_v_next

            flat_indices = knn_indices.reshape(-1)
            target_next_state = self.target_next_state_bank[flat_indices]
            target_reward = self.target_reward_bank[flat_indices]
            target_not_done = self.target_not_done_bank[flat_indices]
            target_v_next = self.v_func(target_next_state)
            target_bellman = target_reward + target_not_done * self.discount * target_v_next
            target_bellman = target_bellman.view(knn_indices.shape[0], knn_indices.shape[1], 1).mean(dim=1)

            bellman_mismatch = torch.abs(source_bellman_target - target_bellman)
            if self.normalize_bellman:
                bellman_for_weight = bellman_mismatch / (bellman_mismatch.mean().detach() + 1e-6)
            else:
                bellman_for_weight = bellman_mismatch

            bellman_weight_raw = torch.exp(-self.bellman_eta * bellman_for_weight)
            source_weight = self.source_budget * bellman_weight_raw / (bellman_weight_raw.mean().detach() + 1e-6)
            source_weight = torch.clamp(source_weight, min=self.min_weight, max=self.max_weight)

        return {
            'weight': source_weight.detach(),
            'bellman_mismatch': bellman_mismatch.detach(),
            'bellman_weight_raw': bellman_weight_raw.detach(),
            'knn_sa_distance': knn_dist.mean(dim=1, keepdim=True).detach(),
        }

    def log_weight_stats(self, stats, writer):
        if writer is None or self.total_it % 5000 != 0:
            return
        source_weight = stats['weight']
        bellman_mismatch = stats['bellman_mismatch']
        bellman_weight_raw = stats['bellman_weight_raw']
        knn_sa_distance = stats['knn_sa_distance']

        def pearson_corr(x, y):
            x = x.detach().view(-1).float()
            y = y.detach().view(-1).float()
            x = x - x.mean()
            y = y - y.mean()
            denom = x.norm() * y.norm() + 1e-6
            return (x * y).sum() / denom

        writer.add_scalar('train/source_weight_mean', source_weight.mean(), self.total_it)
        writer.add_scalar('train/source_weight_min', source_weight.min(), self.total_it)
        writer.add_scalar('train/source_weight_max', source_weight.max(), self.total_it)
        writer.add_scalar('train/source_weight_at_min_ratio', (source_weight <= self.min_weight + 1e-6).float().mean(), self.total_it)
        ess = source_weight.sum().pow(2) / (source_weight.pow(2).sum() + 1e-6)
        writer.add_scalar('train/source_weight_ess_ratio', ess / source_weight.numel(), self.total_it)
        writer.add_scalar('train/source_budget', self.source_budget, self.total_it)

        writer.add_scalar('train/bellman_mismatch_mean', bellman_mismatch.mean(), self.total_it)
        writer.add_scalar('train/bellman_mismatch_min', bellman_mismatch.min(), self.total_it)
        writer.add_scalar('train/bellman_mismatch_max', bellman_mismatch.max(), self.total_it)
        writer.add_scalar('train/bellman_weight_raw_mean', bellman_weight_raw.mean(), self.total_it)
        writer.add_scalar('train/knn_sa_distance_mean', knn_sa_distance.mean(), self.total_it)
        writer.add_scalar('train/corr_bellman_mismatch_source_weight', pearson_corr(bellman_mismatch, source_weight), self.total_it)
        writer.add_scalar('train/corr_bellman_weight_raw_source_weight', pearson_corr(bellman_weight_raw, source_weight), self.total_it)

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        self.total_it += 1

        if self.target_features_norm is None:
            self.initialize_target_feature_bank(tar_replay_buffer)
            if writer is not None:
                writer.add_scalar('train/target_bank_size', self.target_features_norm.shape[0], self.total_it)

        if self.config.get('target_only', False):
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
            src_state, src_action, src_next_state, src_reward, src_not_done = tar_replay_buffer.sample(batch_size)
        else:
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        weight_stats = self.compute_source_weight(src_state, src_action, src_next_state, src_reward, src_not_done)
        source_weight = weight_stats['weight']
        self.log_weight_stats(weight_stats, writer)

        src_v_loss, src_adv, _ = self.v_loss_per_sample(src_state, src_action)
        tar_v_loss, tar_adv, tar_v = self.v_loss_per_sample(tar_state, tar_action)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', tar_adv.mean(), self.total_it)
            writer.add_scalar('train/value', tar_v.mean(), self.total_it)
        v_loss_step = self.combine_target_source_loss(tar_v_loss, src_v_loss, source_weight)
        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        src_q_loss, _, _ = self.q_loss_per_sample(src_state, src_action, src_reward, src_next_state, src_not_done)
        tar_q_loss, tar_q_1, _ = self.q_loss_per_sample(tar_state, tar_action, tar_reward, tar_next_state, tar_not_done)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', tar_q_1.mean(), self.total_it)
        q_loss_step = self.combine_target_source_loss(tar_q_loss, src_q_loss, source_weight)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        for p in self.q_funcs.parameters():
            p.requires_grad = False
        src_policy_loss = self.policy_loss_per_sample(src_adv, src_state, src_action)
        tar_policy_loss = self.policy_loss_per_sample(tar_adv, tar_state, tar_action)
        pi_loss_step = self.combine_target_source_loss(tar_policy_loss, src_policy_loss, source_weight)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True
