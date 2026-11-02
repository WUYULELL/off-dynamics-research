import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributions.transforms import Transform


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        
        return action * self.max_action, logprob, mean * self.max_action
    
    def bc_loss(self, state, action):
        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        pred_action = torch.tanh(mu)
        return (pred_action - action)**2

class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class ValueFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(asymmetric_l2_loss_per_sample(u, tau))


def asymmetric_l2_loss_per_sample(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.abs(tau - (u < 0).float()) * u**2


class IQL_UNC(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config=  config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']

        # IQL hyperparameter
        self.lam = config['lam']
        self.temp = config['temp']

        # Source-domain uncertainty reweighting hyperparameters.
        self.unc_alpha = config.get('unc_alpha', 1.0)
        self.unc_min_weight = config.get('unc_min_weight', 0.05)
        self.source_loss_weight = config.get('source_loss_weight', 1.0)
        self.unc_normalize = config.get('unc_normalize', True)
        self.const_source_weight = config.get('const_source_weight', None)
        
        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka value
        self.v_func = ValueFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.v_optimizer = torch.optim.Adam(self.v_func.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, config['max_step'])
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def compute_source_uncertainty_weight(self, state_batch, action_batch):
        with torch.no_grad():
            q_1, q_2 = self.q_funcs(state_batch, action_batch)
            uncertainty = torch.abs(q_1 - q_2)
            if self.const_source_weight is not None:
                weight = torch.full_like(uncertainty, float(self.const_source_weight))
                weight = torch.clamp(weight, min=0.0, max=1.0)
                return uncertainty.detach(), weight.detach()
            if self.unc_normalize:
                uncertainty_for_weight = uncertainty / (uncertainty.mean().detach() + 1e-6)
            else:
                uncertainty_for_weight = uncertainty
            weight = torch.exp(-self.unc_alpha * uncertainty_for_weight)
            weight = torch.clamp(weight, min=self.unc_min_weight, max=1.0)
        return uncertainty.detach(), weight.detach()

    def update_v_function(self, state_batch, action_batch, writer=None):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)
            
        v = self.v_func(state_batch)
        adv = q_t - v
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
        v_loss = asymmetric_l2_loss(adv, self.lam)
        return v_loss, adv

    def v_loss_per_sample(self, state_batch, action_batch):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)

        v = self.v_func(state_batch)
        adv = q_t - v
        return asymmetric_l2_loss_per_sample(adv, self.lam), adv, v

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch, writer=None):
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t
            
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss

    def q_loss_per_sample(self, state_batch, action_batch, reward_batch, nextstate_batch, not_done_batch):
        with torch.no_grad():
            v_t = self.v_func(nextstate_batch)
            value_target = reward_batch + not_done_batch * self.discount * v_t

        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        q_loss = F.mse_loss(q_1, value_target, reduction='none') + F.mse_loss(q_2, value_target, reduction='none')
        return q_loss, q_1, q_2

    def update_policy(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state_batch, action_batch)
        policy_loss = torch.mean(exp_adv * bc_loss)
        return policy_loss

    def policy_loss_per_sample(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state_batch, action_batch).mean(dim=1, keepdim=True)
        return exp_adv * bc_loss

    def combine_target_source_loss(self, target_loss, source_loss, source_weight):
        return target_loss.mean() + self.source_loss_weight * (source_weight * source_loss).mean()

    def log_uncertainty_stats(self, uncertainty, source_weight, writer):
        if writer is None or self.total_it % 5000 != 0:
            return
        writer.add_scalar('train/source_uncertainty_mean', uncertainty.mean(), self.total_it)
        writer.add_scalar('train/source_uncertainty_min', uncertainty.min(), self.total_it)
        writer.add_scalar('train/source_uncertainty_max', uncertainty.max(), self.total_it)
        writer.add_scalar('train/source_weight_mean', source_weight.mean(), self.total_it)
        writer.add_scalar('train/source_weight_min', source_weight.min(), self.total_it)
        writer.add_scalar('train/source_weight_max', source_weight.max(), self.total_it)

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1

        if self.config.get('target_only', False):
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
            src_state, src_action, src_next_state, src_reward, src_not_done = tar_replay_buffer.sample(batch_size)
        else:
            src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)

        source_uncertainty, source_weight = self.compute_source_uncertainty_weight(src_state, src_action)
        self.log_uncertainty_stats(source_uncertainty, source_weight, writer)

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

        # update policy and temperature parameter
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
        self.v_func.load_state_dict(torch.load(file_name + "_value"))
        self.v_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.policy_lr_schedule.load_state_dict(torch.load(filename + "_actor_lr_scheduler"))
