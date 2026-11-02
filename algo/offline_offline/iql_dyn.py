import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from offline_offline.iql_unc import DoubleQFunc, Policy, ValueFunc, asymmetric_l2_loss_per_sample


class IQL_DYN(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config = config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']

        self.lam = config['lam']
        self.temp = config['temp']

        self.reweight_mode = config.get('reweight_mode', 'dyn')
        self.dyn_alpha = config.get('dyn_alpha', 1.0)
        self.unc_alpha = config.get('unc_alpha', 0.5)
        self.unc_factor_min = config.get('unc_factor_min', 0.0)
        self.unc_start_step = config.get('unc_start_step', 0)
        self.source_budget = config.get('source_budget', 0.33)
        self.min_weight = config.get('min_weight', 0.05)
        self.max_weight = config.get('max_weight', 1.0)
        self.knn_k = config.get('knn_k', 5)
        self.target_bank_size = config.get('target_bank_size', 20000)
        self.knn_chunk_size = config.get('knn_chunk_size', 5000)
        self.source_loss_weight = config.get('source_loss_weight', 1.0)
        self.normalize_distance = config.get('normalize_distance', True)
        self.normalize_uncertainty = config.get('normalize_uncertainty', True)
        self.seed = config.get('seed', 0)

        self.total_it = 0
        self.target_features_norm = None
        self.target_mean = None
        self.target_std = None

        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        self.v_func = ValueFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.v_optimizer = torch.optim.Adam(self.v_func.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, config['max_step'])

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1, -1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        return action.squeeze().cpu().numpy()

    def update_target(self):
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def build_transition_features(self, state, action, next_state):
        return torch.cat([state, action, next_state - state], dim=1)

    def initialize_target_feature_bank(self, tar_replay_buffer):
        size = tar_replay_buffer.size
        indices = np.arange(size)
        if size > self.target_bank_size:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(size, size=self.target_bank_size, replace=False)

        state = torch.FloatTensor(tar_replay_buffer.state[indices]).to(self.device)
        action = torch.FloatTensor(tar_replay_buffer.action[indices]).to(self.device)
        next_state = torch.FloatTensor(tar_replay_buffer.next_state[indices]).to(self.device)

        with torch.no_grad():
            target_features = self.build_transition_features(state, action, next_state)
            self.target_mean = target_features.mean(dim=0, keepdim=True)
            self.target_std = target_features.std(dim=0, keepdim=True) + 1e-6
            self.target_features_norm = ((target_features - self.target_mean) / self.target_std).detach()

    def compute_knn_distance(self, source_features_norm):
        k = min(self.knn_k, self.target_features_norm.shape[0])
        topk_values = None
        bank_size = self.target_features_norm.shape[0]
        chunk_size = max(1, self.knn_chunk_size)

        with torch.no_grad():
            for start in range(0, bank_size, chunk_size):
                end = min(start + chunk_size, bank_size)
                dist = torch.cdist(source_features_norm, self.target_features_norm[start:end])
                values = torch.topk(dist, k=min(k, dist.shape[1]), largest=False, dim=1).values
                if topk_values is None:
                    topk_values = values
                else:
                    topk_values = torch.cat([topk_values, values], dim=1)
                    topk_values = torch.topk(topk_values, k=k, largest=False, dim=1).values
            return topk_values.mean(dim=1, keepdim=True).detach()

    def compute_source_weight(self, src_state, src_action, src_next_state):
        with torch.no_grad():
            source_features = self.build_transition_features(src_state, src_action, src_next_state)
            source_features_norm = (source_features - self.target_mean) / self.target_std
            dyn_distance = self.compute_knn_distance(source_features_norm)

            if self.normalize_distance:
                dyn_for_weight = dyn_distance / (dyn_distance.mean().detach() + 1e-6)
            else:
                dyn_for_weight = dyn_distance
            dyn_weight_raw = torch.exp(-self.dyn_alpha * dyn_for_weight)
            raw_weight = dyn_weight_raw

            q_1, q_2 = self.q_funcs(src_state, src_action)
            uncertainty = torch.abs(q_1 - q_2)
            unc_weight_raw = torch.ones_like(raw_weight)
            uncertainty_active = self.reweight_mode == 'dyn_unc' and self.total_it >= self.unc_start_step
            if uncertainty_active:
                if self.normalize_uncertainty:
                    unc_for_weight = uncertainty / (uncertainty.mean().detach() + 1e-6)
                else:
                    unc_for_weight = uncertainty
                unc_weight_raw = torch.exp(-self.unc_alpha * unc_for_weight)
                if self.unc_factor_min > 0.0:
                    unc_weight_raw = self.unc_factor_min + (1.0 - self.unc_factor_min) * unc_weight_raw
                raw_weight = dyn_weight_raw * unc_weight_raw

            source_weight = self.source_budget * raw_weight / (raw_weight.mean().detach() + 1e-6)
            source_weight = torch.clamp(source_weight, min=self.min_weight, max=self.max_weight)

        return {
            'weight': source_weight.detach(),
            'dyn_distance': dyn_distance.detach(),
            'dyn_weight_raw': dyn_weight_raw.detach(),
            'uncertainty': uncertainty.detach(),
            'unc_weight_raw': unc_weight_raw.detach(),
            'uncertainty_active': torch.tensor(float(uncertainty_active), device=self.device),
        }

    def v_loss_per_sample(self, state_batch, action_batch):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)

        v = self.v_func(state_batch)
        adv = q_t - v
        return asymmetric_l2_loss_per_sample(adv, self.lam), adv, v

    def q_loss_per_sample(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch):
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        q_loss = F.mse_loss(q_1, value_target, reduction='none') + F.mse_loss(q_2, value_target, reduction='none')
        return q_loss, q_1, q_2

    def policy_loss_per_sample(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state_batch, action_batch).mean(dim=1, keepdim=True)
        return exp_adv * bc_loss

    def combine_target_source_loss(self, target_loss, source_loss, source_weight):
        return target_loss.mean() + self.source_loss_weight * (source_weight * source_loss).mean()

    def log_weight_stats(self, stats, writer):
        if writer is None or self.total_it % 5000 != 0:
            return
        dyn_distance = stats['dyn_distance']
        dyn_weight_raw = stats['dyn_weight_raw']
        uncertainty = stats['uncertainty']
        unc_weight_raw = stats['unc_weight_raw']
        source_weight = stats['weight']
        uncertainty_active = stats['uncertainty_active']

        def pearson_corr(x, y):
            x = x.detach().view(-1).float()
            y = y.detach().view(-1).float()
            x = x - x.mean()
            y = y - y.mean()
            denom = x.norm() * y.norm() + 1e-6
            return (x * y).sum() / denom

        writer.add_scalar('train/dyn_distance_mean', dyn_distance.mean(), self.total_it)
        writer.add_scalar('train/dyn_distance_min', dyn_distance.min(), self.total_it)
        writer.add_scalar('train/dyn_distance_max', dyn_distance.max(), self.total_it)
        writer.add_scalar('train/dyn_weight_raw_mean', dyn_weight_raw.mean(), self.total_it)
        writer.add_scalar('train/unc_weight_raw_mean', unc_weight_raw.mean(), self.total_it)
        writer.add_scalar('train/source_uncertainty_mean', uncertainty.mean(), self.total_it)
        writer.add_scalar('train/source_uncertainty_min', uncertainty.min(), self.total_it)
        writer.add_scalar('train/source_uncertainty_max', uncertainty.max(), self.total_it)
        writer.add_scalar('train/source_weight_mean', source_weight.mean(), self.total_it)
        writer.add_scalar('train/source_weight_min', source_weight.min(), self.total_it)
        writer.add_scalar('train/source_weight_max', source_weight.max(), self.total_it)
        writer.add_scalar('train/source_weight_at_min_ratio', (source_weight <= self.min_weight + 1e-6).float().mean(), self.total_it)
        ess = source_weight.sum().pow(2) / (source_weight.pow(2).sum() + 1e-6)
        writer.add_scalar('train/source_weight_ess_ratio', ess / source_weight.numel(), self.total_it)
        writer.add_scalar('train/source_budget', self.source_budget, self.total_it)
        writer.add_scalar('train/uncertainty_active', uncertainty_active, self.total_it)
        writer.add_scalar('train/corr_dyn_distance_source_weight', pearson_corr(dyn_distance, source_weight), self.total_it)
        writer.add_scalar('train/corr_uncertainty_source_weight', pearson_corr(uncertainty, source_weight), self.total_it)
        writer.add_scalar('train/corr_dyn_weight_raw_source_weight', pearson_corr(dyn_weight_raw, source_weight), self.total_it)
        writer.add_scalar('train/corr_unc_weight_raw_source_weight', pearson_corr(unc_weight_raw, source_weight), self.total_it)

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

        weight_stats = self.compute_source_weight(src_state, src_action, src_next_state)
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

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.v_func.state_dict(), filename + "_value")
        torch.save(self.v_optimizer.state_dict(), filename + "_value_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.policy_lr_schedule.state_dict(), filename + "_actor_lr_scheduler")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.v_func.load_state_dict(torch.load(filename + "_value"))
        self.v_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.policy_lr_schedule.load_state_dict(torch.load(filename + "_actor_lr_scheduler"))
