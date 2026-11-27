"""Support statistics and source layers (adaptive-source batch).

All distances are Euclidean in the space standardised by the per-dimension mean / standard deviation of the
target training arrays (official call_tar_dataset rows), computed exactly in float64 (deterministic, CPU).
  tau       = 0.9 quantile of the target (s, a) leave-one-trajectory-out nearest-neighbour distances
              (for a target point, every point of its own trajectory is excluded)
  tau_state = the same with states only
  D_ret(rho)  = the first floor(rho * N) source rows when sorted by the stored OT value, largest first
              (stable sort; the cost file stores negative matching costs, larger = closer, as OTDF's topk)
  D_near(rho) = rows of D_ret(rho) whose (s, a) distance to the target set is <= tau
  random tier = the same number of rows drawn without replacement from D_ret(0.8) with a fixed per-task seed
"""
import math

from asrc_common import array_digest, seed_of

SD_FLOOR = 1e-3
QUANTILE = 0.9


def standardizer(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    raw_sd = x.std(0)
    sd = np.maximum(raw_sd, SD_FLOOR)
    return mu, sd, [int(i) for i in np.flatnonzero(raw_sd < SD_FLOOR)]


def nearest(a, b, chunk=2048):
    """Euclidean distance from each row of a to its nearest row of b (float64, exact, deterministic)."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    b2 = (b*b).sum(1)
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        x = a[i:i+chunk]
        d2 = (x*x).sum(1)[:, None]+b2[None, :]-2.*(x@b.T)
        out[i:i+chunk] = np.sqrt(np.maximum(d2.min(1), 0.))
    return out


def loto(points, seg):
    """Leave-one-trajectory-out nearest-neighbour distance of every point."""
    import numpy as np
    out = np.empty(len(points))
    groups = np.unique(seg)
    assert len(groups) >= 2, 'need at least two target trajectories'
    for g in groups:
        inside = seg == g
        out[inside] = nearest(points[inside], points[~inside])
    return out


def n_ret(rho, n):
    return int(math.floor(rho*n+1e-9))


class Support:
    """Support geometry of one task (target training arrays + source training arrays + OT values)."""

    def __init__(self, target, target_seg, source, cost=None):
        import numpy as np
        t_s = np.asarray(target['observations'], dtype=np.float64)
        t_a = np.asarray(target['actions'], dtype=np.float64)
        assert len(target_seg) == len(t_s)
        self.sd_dim = t_s.shape[1]
        self.t_sa = np.concatenate([t_s, t_a], 1)
        self.mu_sa, self.sd_sa, floor_sa = standardizer(self.t_sa)
        self.mu_s, self.sd_s, floor_s = self.mu_sa[:self.sd_dim], self.sd_sa[:self.sd_dim], [i for i in floor_sa if i < self.sd_dim]
        self.T_sa = (self.t_sa-self.mu_sa)/self.sd_sa
        self.T_s = self.T_sa[:, :self.sd_dim]
        self.seg = np.asarray(target_seg)
        self.loto_sa = loto(self.T_sa, self.seg)
        self.loto_s = loto(self.T_s, self.seg)
        self.tau = float(np.quantile(self.loto_sa, QUANTILE))
        self.tau_state = float(np.quantile(self.loto_s, QUANTILE))
        self.floor_dims = dict(sa=floor_sa, s=floor_s)
        self.n_source = None
        if source is not None:
            s_sa = np.concatenate([np.asarray(source['observations'], dtype=np.float64),
                                   np.asarray(source['actions'], dtype=np.float64)], 1)
            S_sa = (s_sa-self.mu_sa)/self.sd_sa
            self.n_source = len(S_sa)
            self.d_sa = nearest(S_sa, self.T_sa)
            self.d_s = nearest(S_sa[:, :self.sd_dim], self.T_s)
            self.near_all = self.d_sa <= self.tau
            self.near_all_state = self.d_s <= self.tau_state
        self.order = None
        if cost is not None:
            self.set_cost(cost)

    def set_cost(self, cost):
        """Attach the stored OT values (larger = closer) and sort the source rows by them (stable)."""
        import numpy as np
        cost = np.asarray(cost, dtype=np.float64).reshape(-1)
        assert self.n_source is not None and len(cost) == self.n_source, 'cost length != source rows'
        assert np.isfinite(cost).all(), 'non-finite OT values'
        self.order = np.argsort(-cost, kind='stable')

    # ------------------------------------------------------------------ layers
    def ret_index(self, rho):
        import numpy as np
        assert self.order is not None
        return np.sort(self.order[:n_ret(rho, self.n_source)]).astype(np.int64)

    def near_index(self, rho):
        import numpy as np
        ret = self.ret_index(rho)
        return ret[self.near_all[ret]].astype(np.int64)

    def random_index(self, rho, size, task_id):
        """size rows of D_ret(rho) without replacement, fixed per-task seed (the random-tier control)."""
        import numpy as np
        ret = self.ret_index(rho)
        rng = np.random.RandomState(seed_of('random-tier', task_id))
        return np.sort(rng.choice(ret, size=int(size), replace=False)).astype(np.int64)

    def layers(self, group, task_id):
        """near_index for a tiered group: rho20T / rho80T / rho100T (true near layer) or rho80R (random)."""
        rho = {'rho20T': .2, 'rho80T': .8, 'rho100T': 1., 'rho80R': .8}[group]
        if group == 'rho80R':
            return self.random_index(.8, len(self.near_index(.8)), task_id)
        return self.near_index(rho)

    # ------------------------------------------------------------------ report
    def summary(self, rhos=(.2, .8, 1.)):
        import numpy as np
        q = (0.1, 0.5, 0.9, 0.99)
        out = dict(target_rows=int(len(self.T_sa)), target_trajectories=int(len(np.unique(self.seg))),
                   standardization='target training arrays, per-dimension mean / std, std floor %g' % SD_FLOOR,
                   std_floor_dims=self.floor_dims, mu_sa=self.mu_sa, sd_sa=self.sd_sa,
                   tau=self.tau, tau_state=self.tau_state, quantile=QUANTILE,
                   target_loto_quantiles_sa={str(k): float(np.quantile(self.loto_sa, k)) for k in q},
                   target_loto_quantiles_s={str(k): float(np.quantile(self.loto_s, k)) for k in q})
        if self.n_source is not None:
            out.update(source_rows=int(self.n_source),
                       source_distance_quantiles_sa={str(k): float(np.quantile(self.d_sa, k)) for k in (0.001, 0.01, 0.1, 0.5)},
                       source_distance_quantiles_s={str(k): float(np.quantile(self.d_s, k)) for k in (0.001, 0.01, 0.1, 0.5)},
                       near_all_rows=int(self.near_all.sum()), near_all_fraction=float(self.near_all.mean()),
                       near_all_state_rows=int(self.near_all_state.sum()), near_all_state_fraction=float(self.near_all_state.mean()))
        if self.order is not None:
            per = {}
            for rho in rhos:
                ret = self.ret_index(rho)
                near = ret[self.near_all[ret]]
                near_s = ret[self.near_all_state[ret]]
                per['%g' % rho] = dict(ret_rows=int(len(ret)), near_rows=int(len(near)),
                                       near_fraction_of_source=float(len(near)/self.n_source),
                                       near_fraction_of_ret=float(len(near)/max(len(ret), 1)),
                                       near_state_rows=int(len(near_s)),
                                       near_state_fraction_of_source=float(len(near_s)/self.n_source),
                                       near_state_fraction_of_ret=float(len(near_s)/max(len(ret), 1)),
                                       near_index_digest=array_digest(near.astype(np.int64))['sha256'])
            out['by_rho'] = per
        return out


def project_support(points_sa, mu, sd, target_T, chunk=4096):
    """Distances of (s, a) rows to the standardised target set (used by the model rollouts)."""
    import numpy as np
    return nearest((np.asarray(points_sa, dtype=np.float64)-mu)/sd, target_T, chunk)
