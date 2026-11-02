"""Own IQL algebra: source in V/Q only; actor uses pre-V target advantage."""
from runtime import *
import numpy as np
import torch
from offline_offline.iql_dyn import IQL_DYN

class Learner(IQL_DYN):
    def step(self, target, source=None):
        self.total_it += 1
        ts, ta, tn, tr, tm = target
        tv, adv, _ = self.v_loss_per_sample(ts, ta)
        v_loss = tv.mean()
        if source is not None:
            ss, sa, sn, sr, sm = source
            sv, _, _ = self.v_loss_per_sample(ss, sa)
            v_loss = v_loss + sv.mean()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        tq, _, _ = self.q_loss_per_sample(ts, ta, tr, tn, tm)
        q_loss = tq.mean()
        if source is not None:
            sq, _, _ = self.q_loss_per_sample(ss, sa, sr, sn, sm)
            q_loss = q_loss + sq.mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.update_target()
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss = self.policy_loss_per_sample(adv, ts, ta).mean()
        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        for p in self.q_funcs.parameters():
            p.requires_grad = True
        return v_loss.detach(), q_loss.detach(), pi_loss.detach()

    def finite(self):
        return all(bool(torch.isfinite(p).all()) for n in
            (self.policy, self.v_func, self.q_funcs, self.target_q_funcs) for p in n.parameters())

    def digest(self):
        h = hashlib.sha256()
        for n in (self.policy, self.v_func, self.q_funcs, self.target_q_funcs):
            for p in n.state_dict().values():
                h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def checkpoint(self):
        return dict(policy=self.policy.state_dict(), V=self.v_func.state_dict(),
            Q=self.q_funcs.state_dict(), target_Q=self.target_q_funcs.state_dict(),
            V_optimizer=self.v_optimizer.state_dict(), Q_optimizer=self.q_optimizer.state_dict(),
            actor_optimizer=self.policy_optimizer.state_dict(),
            actor_scheduler=self.policy_lr_schedule.state_dict(), step=self.total_it,
            config=self.config)

    def restore(self, path):
        d = torch.load(str(path), map_location=self.device)
        for n, k in ((self.policy, 'policy'), (self.v_func, 'V'),
                     (self.q_funcs, 'Q'), (self.target_q_funcs, 'target_Q')):
            n.load_state_dict(d[k])
        return d

    def deterministic_action(self, state):
        # Same mean as the original policy, without consuming its unused rsample.
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            mu = self.policy.network(x).chunk(2, dim=1)[0]
            return torch.tanh(mu).squeeze(0).cpu().numpy()

class Sampler:
    def __init__(self, arrays, run_seed, purpose, device, ids=None):
        self.arrays, self.device = arrays, device
        self.ids = np.arange(len(arrays[0]), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        assert len(self.ids) and len(np.unique(self.ids)) == len(self.ids)
        assert self.ids.min() >= 0 and self.ids.max() < len(arrays[0])
        self.rng = np.random.RandomState(seed(purpose, run_seed))
        self.draws = 0
        self.hash = hashlib.sha256()
        self.first = None

    def sample(self):
        ix = self.ids[self.rng.randint(len(self.ids), size=64)]
        self.draws += len(ix)
        self.hash.update(ix.astype('<i8').tobytes())
        if self.first is None:
            self.first = ix.tolist()
        return tuple(torch.as_tensor(np.asarray(a[ix]), dtype=torch.float32, device=self.device) for a in self.arrays)

def initialize(config, device):
    s = seed('initialization', config['seed'])
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if device != 'cpu':
        torch.cuda.manual_seed_all(s)
    return Learner(config, device)
