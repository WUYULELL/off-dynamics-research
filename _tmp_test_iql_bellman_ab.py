import numpy as np
import torch
from algo.call_algo import call_algo

class Buffer:
    def __init__(self, n=256, s=6, a=2):
        self.size = n
        self.state = np.random.randn(n, s).astype('float32')
        self.action = np.random.randn(n, a).astype('float32')
        self.next_state = self.state + 0.1 * np.random.randn(n, s).astype('float32')
        self.reward = np.random.randn(n, 1).astype('float32')
        self.not_done = np.ones((n, 1), dtype='float32')
        self.device = torch.device('cpu')
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.FloatTensor(x[idx]) for x in [self.state, self.action, self.next_state, self.reward, self.not_done])

config = {
    'gamma': 0.99, 'tau': 0.005, 'action_dim': 2, 'state_dim': 6, 'hidden_sizes': 32,
    'max_action': 1.0, 'critic_lr': 1e-3, 'actor_lr': 1e-3, 'max_step': 10,
    'update_interval': 2, 'lam': 0.7, 'temp': 3.0, 'batch_size': 8,
    'reweight_mode': 'bellman_ab', 'bellman_alpha': 1.0, 'budget_mode': 'adaptive',
    'budget_min': 0.10, 'budget_max': 0.33, 'adaptive_budget_update_interval': 1,
    'target_bank_size': 64, 'knn_k': 5, 'knn_chunk_size': 16, 'seed': 0,
}
policy = call_algo('IQL_BELLMAN_AB', config, 3, torch.device('cpu'))
src = Buffer(); tar = Buffer()
policy.train(src, tar, batch_size=8, writer=None)
print(type(policy).__name__)
print(float(policy.adaptive_budget), float(policy.last_mismatch_ratio))
assert 0.10 <= float(policy.adaptive_budget) <= 0.33
