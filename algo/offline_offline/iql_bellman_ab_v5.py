import torch

from offline_offline.iql_bellman_ab import IQL_BELLMAN_AB


class IQL_BELLMAN_AB_V5(IQL_BELLMAN_AB):

    def __init__(self, config, device, target_entropy=None):
        super().__init__(config, device, target_entropy)
        self.budget_anchor = config.get('budget_anchor', 0.20)
        self.budget_response_scale = config.get('budget_response_scale', 2.0)
        if not self.budget_min <= self.budget_anchor <= self.budget_max:
            raise ValueError('budget_anchor must be inside [budget_min, budget_max]')
        if self.budget_response_scale <= 0:
            raise ValueError('budget_response_scale must be positive')
        self.last_budget_response_signal = torch.tensor(0.0, device=self.device)

    def budget_from_mismatch_ratio(self, ratio):
        response = torch.tanh(
            (self.budget_ratio_center - ratio) / self.budget_response_scale
        )
        upper_budget = self.budget_anchor + (self.budget_max - self.budget_anchor) * response
        lower_budget = self.budget_anchor + (self.budget_anchor - self.budget_min) * response
        budget = torch.where(response >= 0, upper_budget, lower_budget)
        self.last_budget_response_signal = response.detach()
        return torch.clamp(budget, min=self.budget_min, max=self.budget_max)

    def log_weight_stats(self, stats, writer):
        super().log_weight_stats(stats, writer)
        if writer is None or self.total_it % 5000 != 0:
            return
        writer.add_scalar('train/budget_anchor', self.budget_anchor, self.total_it)
        writer.add_scalar(
            'train/budget_response_signal',
            self.last_budget_response_signal,
            self.total_it,
        )
        writer.add_scalar(
            'train/budget_response_scale',
            self.budget_response_scale,
            self.total_it,
        )
