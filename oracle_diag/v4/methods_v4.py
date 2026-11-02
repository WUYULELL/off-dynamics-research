"""Four attribution arms. Single-domain arms use the frozen v3 TARGET learner."""
from support_v4 import *
from methods_v3 import IndexedReplayBuffer, SourceBackup, TargetIQL, MixedIQL
from offline_offline.iql import IQL
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
import torch


class Target64IQL(TargetIQL):
    def train(self, src, tar, batch_size=128, writer=None):
        assert batch_size == 128 and self.config.get('target_only')
        # The base IQL target-only path samples twice its argument.
        return IQL.train(self, src, tar, 32, writer)


class OracleW1IQL(MixedIQL):
    def __init__(self, config, device, source_backup):
        # Do not call MixedIQL.__init__, whose .20 assertion is intentional in v3.
        IQL_BELLMAN_FB.__init__(self, config, device)
        assert self.source_budget == 1.0 and self.source_loss_weight == 1.0
        assert not config.get('target_only', False)
        self.source_backup = source_backup
        self._q_calls = 0
        self.source_buffer = None

    def compute_source_weight(self, s, a, ns, r, nd):
        self._q_calls = 0
        return dict(weight=torch.ones_like(r))

    def log_weight_stats(self, stats, writer):
        assert torch.all(stats['weight'] == 1.0)
        return super().log_weight_stats(stats, writer)


def build(method, config, device, cache_receipt=None):
    if method not in ALL_METHODS:
        raise ValueError(method)
    if method in ('TARGET', 'ORACLE-ONLY', 'SOURCE-ONLY'):
        assert config.get('target_only')
        return TargetIQL(config, device)
    if method == 'TARGET-64':
        return Target64IQL(config, device)
    if cache_receipt is None:
        raise ValueError('ORACLE-W1 needs the validated v3 source cache')
    return OracleW1IQL(config, device, SourceBackup(device, cache_receipt))
