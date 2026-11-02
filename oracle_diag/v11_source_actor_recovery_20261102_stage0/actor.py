"""O1 V/Q update order, with the user-corrected source action weighted MSE."""
from common import *
import random
import numpy as np
import torch
from learner import Learner as O1Learner, Sampler

def source_weights(advantage, variant):
    a = advantage.detach()
    weights = torch.exp(10. * a).clamp(max=100.)
    keep = torch.ones_like(a, dtype=torch.bool)
    if variant == 'SA-top':
        # Top 16/64 of the original batch, then intersect A>0; ties by batch position.
        flat = a.cpu().numpy().reshape(-1)
        rank = np.lexsort((np.arange(len(flat)), -flat))[:len(flat) // 4]
        selected = np.zeros(len(flat), dtype=bool)
        selected[rank] = flat[rank] > 0.
        keep = torch.as_tensor(selected, device=a.device).reshape_as(a)
    else:
        assert variant == 'SA-all'
    return weights * keep, keep

class Learner(O1Learner):
    def __init__(self, config, device, variant, actor_source_w):
        super().__init__(config, device)
        assert config['temp'] == 10 and config['source_loss_weight'] == 1.
        assert variant in ('SA-all', 'SA-top') and actor_source_w in (0., 1.)
        self.variant, self.actor_source_w = variant, actor_source_w
        self.diagnostics = {}

    def step(self, target, source):
        self.total_it += 1
        ts, ta, tn, tr, tm = target
        tv, adv, _ = self.v_loss_per_sample(ts, ta)
        v_loss = tv.mean()
        ss, sa, sn, sr, sm = source
        sv, source_adv, _ = self.v_loss_per_sample(ss, sa)
        v_loss = v_loss + self.source_loss_weight * sv.mean()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        tq, _, _ = self.q_loss_per_sample(ts, ta, tr, tn, tm)
        q_loss = tq.mean()
        sq, _, _ = self.q_loss_per_sample(ss, sa, sr, sn, sm)
        q_loss = q_loss + self.source_loss_weight * sq.mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.update_target()
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        target_term = self.policy_loss_per_sample(adv, ts, ta).mean()
        weights, keep = source_weights(source_adv, self.variant)
        count = int(keep.sum())
        denominator = 64 if self.variant == 'SA-all' else count
        source_term = torch.zeros_like(target_term)
        # w_a=0 retains exactly the original actor computational graph/optimizer path.
        if self.actor_source_w:
            per_action = self.policy.bc_loss(ss, sa).mean(dim=1, keepdim=True)
            if denominator:
                source_term = (weights * per_action).sum() / denominator
            else:
                source_term = (per_action * 0.).sum()
            pi_loss = target_term + self.actor_source_w * source_term
        else:
            pi_loss = target_term
        if self.actor_source_w:
            with torch.no_grad():
                target_bc = self.policy.bc_loss(ts, ta).mean(dim=1, keepdim=True)
                target_weights = torch.exp(10. * adv.detach()).clamp(max=100.)
                self.diagnostic_vectors = torch.cat((adv.detach(), source_adv.detach(), target_bc,
                    per_action.detach(), target_weights, weights), dim=1).cpu().numpy()
        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        for p in self.q_funcs.parameters():
            p.requires_grad = True
        with torch.no_grad():
            tw = torch.exp(10. * adv.detach()).clamp(max=100.)
            effective = weights.sum() / denominator if denominator else weights.sum() * 0.
            self.diagnostics = dict(source_A_positive_fraction=float((source_adv.detach() > 0.).float().mean()),
                target_weight_sum=float(tw.sum()), source_weight_sum=float(weights.sum()),
                target_normalized_weight_sum=float(tw.mean()), source_normalized_weight_sum=float(effective),
                source_selected=count, source_denominator=denominator,
                target_actor_term=float(target_term.detach()), source_actor_term=float(source_term.detach()),
                source_coefficient=self.actor_source_w, actor_total=float(pi_loss.detach()))
        return v_loss.detach(), q_loss.detach(), pi_loss.detach()

def initialize(config, device, variant='SA-all', actor_source_w=1.):
    s = r.seed('initialization', config['seed'], config['task'])
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if device != 'cpu': torch.cuda.manual_seed_all(s)
    return Learner(config, device, variant, actor_source_w)
