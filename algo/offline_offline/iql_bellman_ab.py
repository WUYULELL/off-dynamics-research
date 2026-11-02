import torch

from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB


class IQL_BELLMAN_AB(IQL_BELLMAN_FB):

    def __init__(self, config, device, target_entropy=None):
        super().__init__(config, device, target_entropy)
        self.bellman_alpha = config.get('bellman_alpha', config.get('bellman_eta', 1.0))
        self.bellman_eta = self.bellman_alpha
        self.normalize_bellman = config.get('normalize_bellman_mismatch', config.get('normalize_bellman', True))
        self.budget_mode = config.get('budget_mode', 'adaptive')
        self.fixed_source_budget = config.get('fixed_source_budget', config.get('source_budget', 0.20))
        self.budget_min = config.get('budget_min', 0.10)
        self.budget_max = config.get('budget_max', 0.33)
        self.budget_temp = config.get('budget_temp', 1.0)
        self.budget_ratio_center = config.get('budget_ratio_center', 1.0)
        self.budget_ema_decay = config.get('budget_ema_decay', 0.95)
        self.use_budget_ema = config.get('use_budget_ema', True)
        self.adaptive_budget_update_interval = config.get('adaptive_budget_update_interval', 10)
        self.exclude_self_in_target_knn = config.get('exclude_self_in_target_knn', True)
        self.diagnostic_log_interval = config.get('diagnostic_log_interval', 5000)
        self.adaptive_budget = None
        self.last_budget_batch = torch.tensor(float(self.fixed_source_budget), device=self.device)
        self.last_target_mismatch = torch.tensor(0.0, device=self.device)
        self.last_source_mismatch = torch.tensor(0.0, device=self.device)
        self.last_mismatch_ratio = torch.tensor(1.0, device=self.device)

    def compute_knn_indices(self, query_features_norm, k_override=None, drop_closest=False):
        base_k = self.knn_k if k_override is None else k_override
        available = self.target_features_norm.shape[0]
        k = min(base_k + int(drop_closest), available)
        topk_values = None
        topk_indices = None
        bank_size = self.target_features_norm.shape[0]
        chunk_size = max(1, self.knn_chunk_size)

        with torch.no_grad():
            for start in range(0, bank_size, chunk_size):
                end = min(start + chunk_size, bank_size)
                dist = torch.cdist(query_features_norm, self.target_features_norm[start:end])
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

            if drop_closest and topk_values.shape[1] > 1:
                topk_values = topk_values[:, 1:]
                topk_indices = topk_indices[:, 1:]

        return topk_values.detach(), topk_indices.detach()

    def compute_bellman_mismatch(self, state, action, next_state, reward, not_done, drop_closest=False):
        with torch.no_grad():
            features = self.build_sa_features(state, action)
            features_norm = (features - self.target_mean) / self.target_std
            knn_dist, knn_indices = self.compute_knn_indices(
                features_norm,
                k_override=self.knn_k,
                drop_closest=drop_closest,
            )

            query_v_next = self.v_func(next_state)
            query_bellman = reward + not_done * self.discount * query_v_next

            flat_indices = knn_indices.reshape(-1)
            target_next_state = self.target_next_state_bank[flat_indices]
            target_reward = self.target_reward_bank[flat_indices]
            target_not_done = self.target_not_done_bank[flat_indices]
            target_v_next = self.v_func(target_next_state)
            target_bellman = target_reward + target_not_done * self.discount * target_v_next
            target_bellman = target_bellman.view(knn_indices.shape[0], knn_indices.shape[1], 1).mean(dim=1)

            mismatch = torch.abs(query_bellman - target_bellman)

        return mismatch.detach(), knn_dist.mean(dim=1, keepdim=True).detach()

    def budget_from_mismatch_ratio(self, ratio):
        ratio_excess = torch.relu(ratio - self.budget_ratio_center)
        return self.budget_min + (self.budget_max - self.budget_min) * torch.exp(
            -self.budget_temp * ratio_excess
        )

    def current_source_budget(self, source_mismatch, target_mismatch):
        with torch.no_grad():
            if self.budget_mode == 'fixed':
                budget = torch.tensor(float(self.fixed_source_budget), device=self.device)
                self.last_budget_batch = budget
                self.adaptive_budget = budget
                self.last_source_mismatch = source_mismatch.mean().detach()
                self.last_target_mismatch = target_mismatch.mean().detach()
                self.last_mismatch_ratio = self.last_source_mismatch / (self.last_target_mismatch + 1e-6)
                return torch.clamp(budget, min=self.budget_min, max=self.budget_max).detach()

            should_update = (
                self.adaptive_budget is None
                or self.total_it % max(1, self.adaptive_budget_update_interval) == 0
            )
            if should_update:
                source_mean = source_mismatch.mean().detach()
                target_mean = target_mismatch.mean().detach()
                ratio = source_mean / (target_mean + 1e-6)
                budget_batch = self.budget_from_mismatch_ratio(ratio)
                budget_batch = torch.clamp(budget_batch, min=self.budget_min, max=self.budget_max)

                if self.adaptive_budget is None or not self.use_budget_ema:
                    self.adaptive_budget = budget_batch.detach()
                else:
                    self.adaptive_budget = (
                        self.budget_ema_decay * self.adaptive_budget
                        + (1.0 - self.budget_ema_decay) * budget_batch
                    ).detach()

                self.last_budget_batch = budget_batch.detach()
                self.last_source_mismatch = source_mean.detach()
                self.last_target_mismatch = target_mean.detach()
                self.last_mismatch_ratio = ratio.detach()

            return torch.clamp(self.adaptive_budget, min=self.budget_min, max=self.budget_max).detach()

    def compute_source_weight(self, src_state, src_action, src_next_state, src_reward, src_not_done,
                              tar_state, tar_action, tar_next_state, tar_reward, tar_not_done):
        with torch.no_grad():
            source_mismatch, knn_dist = self.compute_bellman_mismatch(
                src_state, src_action, src_next_state, src_reward, src_not_done, drop_closest=False
            )
            target_mismatch, _ = self.compute_bellman_mismatch(
                tar_state,
                tar_action,
                tar_next_state,
                tar_reward,
                tar_not_done,
                drop_closest=self.exclude_self_in_target_knn,
            )
            source_budget_current = self.current_source_budget(source_mismatch, target_mismatch)

            if self.normalize_bellman:
                bellman_for_weight = source_mismatch / (source_mismatch.mean().detach() + 1e-6)
            else:
                bellman_for_weight = source_mismatch

            bellman_weight_raw = torch.exp(-self.bellman_alpha * bellman_for_weight)
            source_weight = source_budget_current * bellman_weight_raw / (bellman_weight_raw.mean().detach() + 1e-6)
            source_weight = torch.clamp(source_weight, min=self.min_weight, max=self.max_weight)

        return {
            'weight': source_weight.detach(),
            'bellman_mismatch': source_mismatch.detach(),
            'target_bellman_mismatch': target_mismatch.detach(),
            'bellman_weight_raw': bellman_weight_raw.detach(),
            'knn_sa_distance': knn_dist.detach(),
            'source_budget_current': source_budget_current.detach(),
            'adaptive_budget_batch': self.last_budget_batch.detach(),
            'bellman_mismatch_ratio': self.last_mismatch_ratio.detach(),
        }

    def log_weight_stats(self, stats, writer):
        if writer is None or self.total_it % 5000 != 0:
            return
        source_weight = stats['weight']
        bellman_mismatch = stats['bellman_mismatch']
        target_mismatch = stats['target_bellman_mismatch']
        bellman_weight_raw = stats['bellman_weight_raw']
        knn_sa_distance = stats['knn_sa_distance']
        source_budget_current = stats['source_budget_current']
        budget_batch = stats['adaptive_budget_batch']
        mismatch_ratio = stats['bellman_mismatch_ratio']

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

        writer.add_scalar('train/bellman_mismatch_source_mean', bellman_mismatch.mean(), self.total_it)
        writer.add_scalar('train/bellman_mismatch_source_min', bellman_mismatch.min(), self.total_it)
        writer.add_scalar('train/bellman_mismatch_source_max', bellman_mismatch.max(), self.total_it)
        writer.add_scalar('train/bellman_mismatch_target_mean', target_mismatch.mean(), self.total_it)
        writer.add_scalar('train/bellman_mismatch_ratio', mismatch_ratio, self.total_it)

        writer.add_scalar('train/adaptive_source_budget', source_budget_current, self.total_it)
        writer.add_scalar('train/adaptive_source_budget_batch', budget_batch, self.total_it)
        writer.add_scalar('train/budget_min', self.budget_min, self.total_it)
        writer.add_scalar('train/budget_max', self.budget_max, self.total_it)

        writer.add_scalar('train/bellman_weight_raw_mean', bellman_weight_raw.mean(), self.total_it)
        writer.add_scalar('train/knn_sa_distance_mean', knn_sa_distance.mean(), self.total_it)
        writer.add_scalar('train/corr_bellman_mismatch_source_weight', pearson_corr(bellman_mismatch, source_weight), self.total_it)

    def deterministic_policy_action(self, state):
        mu_logstd = self.policy.network(state)
        mu, _ = mu_logstd.chunk(2, dim=1)
        return torch.tanh(mu) * self.policy.max_action

    def log_training_diagnostics(
        self,
        writer,
        source_weight,
        src_state,
        src_action,
        tar_state,
        tar_action,
        src_adv,
        tar_adv,
        src_v_loss,
        tar_v_loss,
        src_q_loss,
        tar_q_loss,
        src_policy_loss,
        tar_policy_loss,
        v_loss_step,
        q_loss_step,
        pi_loss_step,
    ):
        if writer is None or self.total_it % max(1, self.diagnostic_log_interval) != 0:
            return

        with torch.no_grad():
            src_adv_weight = torch.exp(self.temp * src_adv.detach()).clamp(max=100.0)
            tar_adv_weight = torch.exp(self.temp * tar_adv.detach()).clamp(max=100.0)
            all_adv_weight = torch.cat([src_adv_weight, tar_adv_weight], dim=0)

            src_policy_action = self.deterministic_policy_action(src_state)
            tar_policy_action = self.deterministic_policy_action(tar_state)
            src_action_mse = (src_policy_action - src_action).pow(2).mean()
            tar_action_mse = (tar_policy_action - tar_action).pow(2).mean()

            src_policy_q1, src_policy_q2 = self.q_funcs(src_state, src_policy_action)
            tar_policy_q1, tar_policy_q2 = self.q_funcs(tar_state, tar_policy_action)
            src_policy_q_disagreement = torch.abs(src_policy_q1 - src_policy_q2)
            tar_policy_q_disagreement = torch.abs(tar_policy_q1 - tar_policy_q2)
            all_policy_q_disagreement = torch.cat(
                [src_policy_q_disagreement, tar_policy_q_disagreement], dim=0
            )

            writer.add_scalar('train/value_loss', v_loss_step.detach(), self.total_it)
            writer.add_scalar('train/critic_loss', q_loss_step.detach(), self.total_it)
            writer.add_scalar('train/actor_loss', pi_loss_step.detach(), self.total_it)

            writer.add_scalar('train/value_loss_target', tar_v_loss.mean(), self.total_it)
            writer.add_scalar(
                'train/value_loss_source_weighted',
                (source_weight * src_v_loss).mean(),
                self.total_it,
            )
            writer.add_scalar('train/critic_loss_target', tar_q_loss.mean(), self.total_it)
            writer.add_scalar(
                'train/critic_loss_source_weighted',
                (source_weight * src_q_loss).mean(),
                self.total_it,
            )
            writer.add_scalar('train/actor_loss_target', tar_policy_loss.mean(), self.total_it)
            writer.add_scalar(
                'train/actor_loss_source_weighted',
                (source_weight * src_policy_loss).mean(),
                self.total_it,
            )

            writer.add_scalar('train/advantage_weight_mean', all_adv_weight.mean(), self.total_it)
            writer.add_scalar('train/advantage_weight_max', all_adv_weight.max(), self.total_it)
            writer.add_scalar('train/advantage_weight_source_mean', src_adv_weight.mean(), self.total_it)
            writer.add_scalar('train/advantage_weight_source_max', src_adv_weight.max(), self.total_it)
            writer.add_scalar('train/advantage_weight_target_mean', tar_adv_weight.mean(), self.total_it)
            writer.add_scalar('train/advantage_weight_target_max', tar_adv_weight.max(), self.total_it)

            writer.add_scalar('train/policy_action_mse_source', src_action_mse, self.total_it)
            writer.add_scalar('train/policy_action_mse_target', tar_action_mse, self.total_it)
            writer.add_scalar(
                'train/policy_q_disagreement_mean',
                all_policy_q_disagreement.mean(),
                self.total_it,
            )
            writer.add_scalar(
                'train/policy_q_disagreement_max',
                all_policy_q_disagreement.max(),
                self.total_it,
            )
            writer.add_scalar(
                'train/policy_q_disagreement_source_mean',
                src_policy_q_disagreement.mean(),
                self.total_it,
            )
            writer.add_scalar(
                'train/policy_q_disagreement_target_mean',
                tar_policy_q_disagreement.mean(),
                self.total_it,
            )

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

        weight_stats = self.compute_source_weight(
            src_state, src_action, src_next_state, src_reward, src_not_done,
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done,
        )
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

        self.log_training_diagnostics(
            writer=writer,
            source_weight=source_weight,
            src_state=src_state,
            src_action=src_action,
            tar_state=tar_state,
            tar_action=tar_action,
            src_adv=src_adv,
            tar_adv=tar_adv,
            src_v_loss=src_v_loss,
            tar_v_loss=tar_v_loss,
            src_q_loss=src_q_loss,
            tar_q_loss=tar_q_loss,
            src_policy_loss=src_policy_loss,
            tar_policy_loss=tar_policy_loss,
            v_loss_step=v_loss_step,
            q_loss_step=q_loss_step,
            pi_loss_step=pi_loss_step,
        )
