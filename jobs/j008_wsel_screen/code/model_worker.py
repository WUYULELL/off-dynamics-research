"""j008 model worker: target dynamics ensemble and model-based choice of the source weight.

Kinds
  source  one member of the source dynamics ensemble for one robot: (s, a) -> (s' - s, r), trained on
          D4RL medium (timeout rows dropped), 2% random rows for validation (V20 settings).
  select  one (task, seed): fine-tune every source member on the target data (FT, V20 settings;
          whole target trajectories, 10% of rows held out by seed for early stopping), then roll out
          the four final policies W0 / W25 / W50 / W100 of this (task, seed) in the learned target model
          from start states drawn from the task's reset distribution. Score = mean model return over the
          rollouts (horizon 1000, known termination rule of the robot). The chosen weight is the one with
          the highest score (ties -> the smaller weight). No step of the true simulator is used here;
          env.reset() is only called to draw start states.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np

KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
ORDER = ('W0', 'W25', 'W50', 'W100')
WEIGHTS = {'W0': 0., 'W25': .25, 'W50': .5, 'W100': 1.}
# observation = qpos[1:] + clip(qvel, -10, 10) for both robots (gym v3); first velocity column:
VEL_START = {'hopper': 5, 'walker2d': 8}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seed_of(*parts):
    text = 'J008-20261120|'+'|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


# ----------------------------------------------------------------------------- data (numpy only)

def load(path, sd, ad, drop_timeouts):
    import h5py
    with h5py.File(path, 'r') as f:
        raw = {k: np.asarray(f[k]) for k in KEYS+('timeouts',) if k in f}
    return arrays(raw, sd, ad, drop_timeouts)


def arrays(raw, sd, ad, drop_timeouts):
    obs, act, nxt = [raw[k].astype(np.float64) for k in KEYS[:3]]
    rew = raw['rewards'].astype(np.float64).reshape(-1)
    term = raw['terminals'].reshape(-1)
    n = len(obs)
    assert obs.shape == (n, sd) and act.shape == (n, ad) and nxt.shape == (n, sd) and len(rew) == n == len(term)
    assert all(np.isfinite(a).all() for a in (obs, act, nxt, rew)) and np.isin(term, [0, 1]).all()
    term = term.astype(bool)
    timeouts = raw['timeouts'].reshape(-1).astype(bool) if 'timeouts' in raw else np.zeros(n, bool)
    cont = np.zeros(n, bool)
    cont[:-1] = np.all(np.abs(nxt[:-1]-obs[1:]) <= 1e-5, axis=1)
    end = term | timeouts | ~cont
    end[-1] = True
    info = dict(rows=int(n), terminal_rows=int(term.sum()), timeout_rows=int(timeouts.sum()), segments=int(end.sum()))
    if drop_timeouts and timeouts.any():
        keep = ~timeouts
        obs, act, nxt, rew, term, end = obs[keep], act[keep], nxt[keep], rew[keep], term[keep], end[keep]
        info['rows_after_dropping_timeouts'] = int(keep.sum())
    return dict(obs=obs, act=act, nxt=nxt, rew=rew, term=term, end=end), info


def segments(end):
    out, start = [], 0
    for i in np.flatnonzero(end):
        out.append((start, int(i)+1))
        start = int(i)+1
    assert start == len(end)
    return out


def split_segments(segs, validation_fraction, seed):
    """Whole segments in a seeded random order go to validation until it holds >= validation_fraction of rows."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(segs))
    lens = np.array([b-a for a, b in segs])
    val, train = [], []
    for i in order:
        if lens[val].sum() < validation_fraction*lens.sum():
            val.append(int(i))
        else:
            train.append(int(i))
    assert len(val) >= 1 and len(train) >= 2, (len(val), len(train))
    return sorted(train), sorted(val)


def rows_of(segs, ids):
    return np.concatenate([np.arange(*segs[i]) for i in ids])


def xy(d, rows):
    x = np.concatenate([d['obs'][rows], d['act'][rows]], 1)
    y = np.concatenate([d['nxt'][rows]-d['obs'][rows], d['rew'][rows, None]], 1)
    return x, y


def stats(x):
    return x.mean(0), np.maximum(x.std(0), 1e-3)


# ----------------------------------------------------------------------------- termination and rollout (array-generic)

def healthy(robot, s, ops):
    """gym v3 rules on observations: hopper z > 0.7, |angle| < 0.2, |obs[1:]| < 100; walker 0.8 < z < 2, |angle| < 1."""
    if robot == 'hopper':
        return (s[:, 0] > .7) & (abs(s[:, 1]) < .2) & ops['all1'](abs(s[:, 1:]) < 100.)
    if robot == 'walker2d':
        return (s[:, 0] > .8) & (s[:, 0] < 2.) & (abs(s[:, 1]) < 1.)
    raise ValueError(robot)


def rollout(robot, start, policy, predict, members, horizon, short, ops):
    """start: (B, sd) states; policy(s) -> actions; predict(s, a) -> (K, B, sd+1) next-state change and reward;
    members: (horizon, B) int array, the ensemble member used for each row at each step.
    The reward of the step that ends an episode is counted (as in gym); a non-finite prediction ends the row
    and its reward is not counted. Returns per-row arrays."""
    B, sd = start.shape
    s = start
    alive = ops['ones_bool'](B)
    ret = ops['zeros'](B)
    ret_short = None
    length = ops['zeros'](B)
    spread_sum, spread_n = 0., 0.
    rows = ops['arange'](B)
    for t in range(horizon):
        a = policy(s)
        p = predict(s, a)                                  # (K, B, sd+1)
        chosen = p[ops['index'](members[t]), rows]          # (B, sd+1)
        spread = ops['mean1'](ops['std0'](p[:, :, :sd]))    # ensemble disagreement per row (raw units)
        spread_ok = alive & ops['isfinite'](spread)
        n_alive = float(ops['sum'](spread_ok))
        if n_alive > 0:
            spread_sum += float(ops['sum'](ops['where'](spread_ok, spread, ops['zeros_like'](spread))))
            spread_n += n_alive
        r = chosen[:, sd]
        r_ok = ops['isfinite'](r)
        ret = ret+ops['where'](alive & r_ok, r, ops['zeros_like'](r))
        length = length+ops['cast'](alive)
        nxt = s+chosen[:, :sd]
        v = VEL_START[robot]
        nxt = ops['cat']([nxt[:, :v], ops['clip'](nxt[:, v:], -10., 10.)])
        ok = ops['all1'](ops['isfinite'](nxt)) & r_ok
        nxt = ops['where2'](ok, nxt, s)                     # keep a finite state; the row is ended below
        alive = alive & ok & healthy(robot, nxt, ops)
        s = nxt
        if t+1 == short:
            ret_short = ret
        if float(ops['sum'](alive)) == 0:
            if ret_short is None:
                ret_short = ret
            break
    if ret_short is None:
        ret_short = ret
    return dict(ret=ops['np'](ret), ret_short=ops['np'](ret_short), length=ops['np'](length),
                alive_end=ops['np'](alive).astype(float), spread=spread_sum/max(spread_n, 1.))


def choose(scores):
    """Highest score; ties go to the smaller weight (ORDER is increasing in w)."""
    best = None
    for g in ORDER:
        if best is None or scores[g] > scores[best]:
            best = g
    return best


NUMPY_OPS = dict(ones_bool=lambda n: np.ones(n, bool), zeros=lambda n: np.zeros(n), arange=np.arange,
                 index=lambda m: np.asarray(m), mean1=lambda x: x.mean(1), std0=lambda x: x.std(0), sum=np.sum,
                 where=np.where, where2=lambda c, a, b: np.where(c[:, None], a, b), cast=lambda b: b.astype(float),
                 cat=lambda xs: np.concatenate(xs, 1), clip=np.clip, all1=lambda x: x.all(1), isfinite=np.isfinite,
                 zeros_like=np.zeros_like, np=lambda x: np.asarray(x))


def torch_ops(torch, device):
    return dict(ones_bool=lambda n: torch.ones(n, dtype=torch.bool, device=device),
                zeros=lambda n: torch.zeros(n, dtype=torch.float64, device=device),
                arange=lambda n: torch.arange(n, device=device),
                index=lambda m: torch.as_tensor(np.asarray(m), dtype=torch.long, device=device),
                mean1=lambda x: x.mean(1), std0=lambda x: x.std(0, unbiased=False), sum=lambda x: x.sum().item(),
                where=torch.where, where2=lambda c, a, b: torch.where(c[:, None], a, b), cast=lambda b: b.double(),
                cat=lambda xs: torch.cat(xs, 1), clip=torch.clamp, all1=lambda x: x.all(1), isfinite=torch.isfinite,
                zeros_like=torch.zeros_like, np=lambda x: x.detach().cpu().numpy())


# ----------------------------------------------------------------------------- torch networks

class Nets:
    def __init__(self, hidden):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected exactly one visible GPU'
        self.device = 'cuda:0'
        self.hidden = hidden

    def new(self, n_in, n_out, seed):
        torch = self.torch
        torch.manual_seed(seed)
        layers, d = [], n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.SiLU()]
            d = h
        return torch.nn.Sequential(*layers, torch.nn.Linear(d, n_out)).to(self.device)

    def tensor(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def forward(self, net, x):
        out = []
        with self.torch.no_grad():
            net.eval()
            for i in range(0, len(x), 65536):
                out.append(net(self.tensor(x[i:i+65536])).double().cpu().numpy())
        return np.concatenate(out, 0)

    def fit(self, net, x, y, steps, batch, lr, eval_every, patience, val_fn, seed, progress=None):
        """MSE between net(x) and y (standardized); best validation network kept; patience=None trains all steps."""
        torch = self.torch
        rng = np.random.RandomState(seed)
        xt, yt = self.tensor(x), self.tensor(y)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        best, best_state, best_step, since, history = math.inf, None, 0, 0, []
        step = 0
        for step in range(1, steps+1):
            net.train()
            idx = torch.as_tensor(rng.randint(0, len(x), size=batch), device=self.device)
            loss = ((net(xt[idx])-yt[idx])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % eval_every == 0 or step == steps:
                if not math.isfinite(float(loss)):
                    raise FloatingPointError('non-finite training loss at step %d' % step)
                v = float(val_fn(net))
                if not math.isfinite(v):
                    raise FloatingPointError('non-finite validation error at step %d' % step)
                history.append([step, float(loss), v])
                if v < best:
                    best, best_step, since = v, step, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    since += 1
                if progress:
                    progress(step)
                if patience is not None and since >= patience:
                    break
        net.load_state_dict(best_state)
        return dict(best_validation=best, best_step=best_step, last_step=step, history=history)

    def finite(self, net):
        return all(bool(self.torch.isfinite(p).all()) for p in net.parameters())


# ----------------------------------------------------------------------------- jobs

def run_source(job, out, progress):
    nets = Nets(job['model']['hidden'])
    sd, ad = job['state_dim'], job['action_dim']
    assert sha(job['source_dataset']) == job['source_dataset_sha256']
    d, info = load(job['source_dataset'], sd, ad, drop_timeouts=True)
    rng = np.random.RandomState(seed_of('source-validation', job['robot'], job['member']))
    perm = rng.permutation(len(d['obs']))
    t = job['model']['source_training']
    n_val = int(round(t['validation_fraction']*len(perm)))
    val, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    x_tr, y_tr = xy(d, tr)
    x_va, y_va = xy(d, val)
    x_mu, x_sd = stats(x_tr)
    y_mu, y_sd = stats(y_tr)
    net = nets.new(sd+ad, sd+1, seed_of('source-init', job['robot'], job['member']))
    xs_va, ys_va = (x_va-x_mu)/x_sd, (y_va-y_mu)/y_sd
    fit = nets.fit(net, (x_tr-x_mu)/x_sd, (y_tr-y_mu)/y_sd, job['source_steps'], t['batch'], t['lr'], job['source_eval_every'],
                   None, lambda n: ((nets.forward(n, xs_va)-ys_va)**2).mean(),
                   seed_of('source-batches', job['robot'], job['member']), lambda s: progress(s, job['source_steps']))
    assert nets.finite(net)
    nets.torch.save(net.state_dict(), str(out/'source_model.pt'))
    np.savez(str(out/'source_norm.npz'), x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd)
    return dict(kind='source', robot=job['robot'], member=job['member'], data=info, train_rows=int(len(tr)),
                validation_rows=int(len(val)), fit={k: v for k, v in fit.items() if k != 'history'}, history=fit['history'],
                files={n: sha(out/n) for n in ('source_model.pt', 'source_norm.npz')})


def run_select(job, out, progress):
    nets = Nets(job['model']['hidden'])
    torch = nets.torch
    sys.path.insert(0, job['vendor'])
    sys.path.insert(0, job['algo'])
    from offline_offline import iql_unc
    sd, ad, robot = job['state_dim'], job['action_dim'], job['robot']
    assert sha(job['target_dataset']) == job['target_dataset_sha256'] and sha(job['xml']) == job['xml_sha256']
    d, info = load(job['target_dataset'], sd, ad, drop_timeouts=False)
    segs = segments(d['end'])
    assert len(segs) <= len(d['obs'])/20, 'target rows are not consecutive trajectories'
    train_ids, val_ids = split_segments(segs, job['model']['validation_fraction'], seed_of('split', job['task'], job['seed']))
    tr, va = rows_of(segs, train_ids), rows_of(segs, val_ids)
    x_tr, y_tr = xy(d, tr)
    x_va, y_va = xy(d, va)
    unit = stats(y_tr)[1]                                   # error unit: target training sd of (s'-s, r)
    t = job['model']['target_training']
    members, fits = [], []
    for k in range(job['model']['members']):
        src = Path(job['source_dirs'][k])
        done = read(src/'COMPLETED.json')
        for name, digest in done['result']['files'].items():
            assert sha(src/name) == digest, (str(src), name)
        norm = np.load(str(src/'source_norm.npz'))
        x_mu, x_sd, y_mu, y_sd = (norm[n] for n in ('x_mu', 'x_sd', 'y_mu', 'y_sd'))
        net = nets.new(sd+ad, sd+1, 0)
        net.load_state_dict(torch.load(str(src/'source_model.pt'), map_location=nets.device))
        xs_va = (x_va-x_mu)/x_sd

        def val_err(n):
            return ((((nets.forward(n, xs_va)*y_sd+y_mu)-y_va)/unit)**2).mean()
        source_only = float(val_err(net))
        fit = nets.fit(net, (x_tr-x_mu)/x_sd, (y_tr-y_mu)/y_sd, job['target_max_steps'], t['batch'], t['lr'],
                       job['target_eval_every'], t['patience_evals'], val_err,
                       seed_of('ft-batches', job['task'], job['seed'], k), lambda s: progress(k*job['target_max_steps']+s,
                                                                                         job['model']['members']*job['target_max_steps']))
        assert nets.finite(net)
        fits.append(dict(member=k, source_only_validation=source_only, **{kk: vv for kk, vv in fit.items() if kk != 'history'}))
        members.append((net, [torch.as_tensor(v, dtype=torch.float32, device=nets.device) for v in (x_mu, x_sd, y_mu, y_sd)]))

    def predict(s, a):
        outs = []
        x = torch.cat([s, a], 1)
        for net, (x_mu, x_sd, y_mu, y_sd) in members:
            outs.append(net((x-x_mu)/x_sd)*y_sd+y_mu)
        return torch.stack(outs, 0)

    # start states from the reset distribution of the target task (no simulator steps)
    import gym  # noqa: F401  (registers mujoco envs; same import order as the policy worker)
    constructor = getattr(__import__(job['env_module'], fromlist=[job['env_class']]), job['env_class'])
    env = constructor(xml_file=job['xml'])
    starts = []
    for i in range(job['rollout']['start_states']):
        env.seed(job['rollout']['reset_seed_origin']+i)
        starts.append(np.asarray(env.reset(), dtype=np.float32))
    env.close()
    starts = np.repeat(np.stack(starts, 0), job['rollout']['repeats'], axis=0)
    B, K, H = len(starts), job['model']['members'], job['horizon']
    pick = np.random.RandomState(seed_of('members', job['task'], job['seed'])).randint(K, size=(H, B))
    ops = torch_ops(torch, nets.device)
    results = {}
    for g in ORDER:
        run_dir = Path(job['policy_dirs'][g])
        done = read(run_dir/'COMPLETED.json')
        assert sha(run_dir/'final_checkpoint.pt') == done['checkpoint_sha256'] and done['steps'] == job['policy_steps'], g
        policy = iql_unc.Policy(sd, ad, 1., 256).to(nets.device)
        policy.load_state_dict(torch.load(str(run_dir/'final_checkpoint.pt'), map_location=nets.device)['policy'])
        policy.eval()

        def act(s):
            return torch.tanh(policy.network(s).chunk(2, dim=1)[0])
        with torch.no_grad():
            r = rollout(robot, torch.as_tensor(starts, device=nets.device), act, predict, pick, H, job['short_horizon'], ops)
        lo, hi = job['score_min'], job['score_max']
        results[g] = dict(score=float(100*(r['ret'].mean()-lo)/(hi-lo)), score_short=float(100*(r['ret_short'].mean()-lo)/(hi-lo)),
                          raw_return_mean=float(r['ret'].mean()), raw_return_sd=float(r['ret'].std()),
                          length_mean=float(r['length'].mean()), alive_at_end=float(r['alive_end'].mean()),
                          ensemble_spread=float(r['spread']))
        assert all(math.isfinite(v) for v in results[g].values()), g
    chosen = choose({g: results[g]['score'] for g in ORDER})
    chosen_short = choose({g: results[g]['score_short'] for g in ORDER})
    return dict(kind='select', task=job['task'], seed=job['seed'], data=info, segments=len(segs), train_segments=len(train_ids),
                validation_segments=len(val_ids), train_rows=int(len(tr)), validation_rows=int(len(va)), fits=fits,
                validation_error_FT_mean=float(np.mean([f['best_validation'] for f in fits])),
                validation_error_source_only_mean=float(np.mean([f['source_only_validation'] for f in fits])),
                rollouts=B, horizon=H, short_horizon=job['short_horizon'], scores=results, chosen=chosen, chosen_short=chosen_short)


def main():
    job = read(sys.argv[1])
    out = Path(job['output'])
    started = time.time()

    def progress(step, total):
        write(out/'progress.json', dict(step=int(step), total=int(total), time=time.time()))
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(job['gpu'])
        result = (run_source if job['kind'] == 'source' else run_select)(job, out, progress)
        write(out/'COMPLETED.json', dict(completed=True, kind=job['kind'], wall_seconds=time.time()-started, result=result))
    except BaseException:
        write(out/'FAILED.json', dict(time=time.time(), traceback=traceback.format_exc(), job=job['id']))
        print(traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
