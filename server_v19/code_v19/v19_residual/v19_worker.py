"""V19 worker: residual dynamics-model pilot. No policy training, no simulator.

Job kinds
  source  train the source dynamics model f_S on D4RL medium (one robot, one seed)
  task    for one task and one repetition: split the target data into blocks, then for each
          train fraction fit T, FT and RES and evaluate S, T, FT, RES on the held-out test blocks

A model maps (s, a) to (s' - s, r). Every error is reported in the units of the full target
training set of that repetition, so all models are measured with the same ruler.
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

TEST = os.environ.get('V19_TEST_BACKEND')          # local test only: linear models, npz data


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
    text = 'V19-20261117|'+'|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


# ----------------------------------------------------------------------------- data

def load(path, sd, ad, drop_timeouts):
    keys = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
    if TEST:
        with np.load(path) as f:
            raw = {k: np.asarray(f[k]) for k in f.files}
    else:
        import h5py
        with h5py.File(path, 'r') as f:
            raw = {k: np.asarray(f[k]) for k in keys+('timeouts',) if k in f}
    obs, act, nxt = [raw[k].astype(np.float64) for k in keys[:3]]
    rew = raw['rewards'].astype(np.float64).reshape(-1)
    term = raw['terminals'].reshape(-1)
    n = len(obs)
    assert obs.shape == (n, sd) and act.shape == (n, ad) and nxt.shape == (n, sd) and len(rew) == n == len(term)
    assert all(np.isfinite(a).all() for a in (obs, act, nxt, rew)) and np.isin(term, [0, 1]).all()
    term = term.astype(bool)
    timeouts = raw['timeouts'].reshape(-1).astype(bool) if 'timeouts' in raw else np.zeros(n, bool)
    cont = np.zeros(n, bool)
    cont[:-1] = np.all(np.abs(nxt[:-1]-obs[1:]) <= 1e-5, axis=1)
    end = term | timeouts | ~cont                    # last row of a trajectory segment
    info = dict(rows=int(n), terminal_rows=int(term.sum()), timeout_rows=int(timeouts.sum()),
                has_timeouts='timeouts' in raw, segments=int(end.sum()))
    if drop_timeouts and timeouts.any():
        keep = ~timeouts
        obs, act, nxt, rew, term, end = obs[keep], act[keep], nxt[keep], rew[keep], term[keep], end[keep]
        info['rows_after_dropping_timeouts'] = int(keep.sum())
    return dict(obs=obs, act=act, nxt=nxt, rew=rew, term=term, end=end), info


def make_blocks(end, length, min_block):
    """Consecutive rows of one segment, cut every `length` rows; a tail shorter than min_block joins the previous block."""
    blocks, start = [], 0
    for i in np.flatnonzero(end):
        seg = list(range(start, i+1, length))
        cuts = [(a, min(a+length, i+1)) for a in seg]
        if len(cuts) > 1 and cuts[-1][1]-cuts[-1][0] < min_block:
            cuts[-2] = (cuts[-2][0], cuts[-1][1])
            cuts.pop()
        blocks += cuts
        start = i+1
    assert start == len(end)
    return blocks


def split(blocks, split_cfg, fractions, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(blocks))
    n_test = int(round(split_cfg['test_fraction']*len(blocks)))
    n_val = int(round(split_cfg['validation_fraction']*len(blocks)))
    test = sorted(order[:n_test].tolist())
    val = sorted(order[n_test:n_test+n_val].tolist())
    train = order[n_test+n_val:].tolist()           # kept in random order for the nested fractions
    assert n_test > 0 and n_val > 0 and len(train) >= 4
    subsets = {f: sorted(train[:max(1, int(math.ceil(f*len(train))))]) for f in fractions}
    return test, val, subsets


def rows_of(blocks, ids):
    return np.concatenate([np.arange(*blocks[i]) for i in ids])


def stats(x):
    mu = x.mean(0)
    sd = np.maximum(x.std(0), 1e-3)
    return mu, sd


def xy(d, rows):
    x = np.concatenate([d['obs'][rows], d['act'][rows]], 1)
    y = np.concatenate([d['nxt'][rows]-d['obs'][rows], d['rew'][rows, None]], 1)
    return x, y


# ----------------------------------------------------------------------------- backend

class Torch:
    def __init__(self, cfg_net):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        self.device = 'cuda:0'
        self.hidden = cfg_net['hidden']

    def new(self, n_in, n_out, seed, zero_last=False):
        torch = self.torch
        torch.manual_seed(seed)
        layers, d = [], n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.SiLU()]
            d = h
        last = torch.nn.Linear(d, n_out)
        if zero_last:
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
        net = torch.nn.Sequential(*layers, last).to(self.device)
        return net

    def copy(self, net):
        import copy
        return copy.deepcopy(net)

    def tensor(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def forward(self, net, x):
        """x: standardized inputs (numpy); returns standardized outputs (numpy, float64)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            net.eval()
            for i in range(0, len(x), 65536):
                out.append(net(self.tensor(x[i:i+65536])).double().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, net[-1].out_features))

    def fit(self, net, x, y, steps, batch, lr, eval_every, patience, val_fn, seed, progress=None):
        """Minimise the mean squared error between net(x) and y (both standardized). Every eval_every steps
        val_fn(net) gives a validation error; the best network is returned. patience=None trains all steps."""
        torch = self.torch
        rng = np.random.RandomState(seed)
        xt, yt = self.tensor(x), self.tensor(y)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        best, best_state, best_step, since, history = math.inf, None, 0, 0, []
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

    def save(self, net, path):
        self.torch.save(net.state_dict(), str(path))

    def load(self, n_in, n_out, path):
        net = self.new(n_in, n_out, 0)
        net.load_state_dict(self.torch.load(str(path), map_location=self.device))
        return net


class Linear:
    """Local test backend: ridge regression on [x, 1]; mimics the interface, ignores steps."""
    class Net:
        def __init__(self, n_in, n_out, w=None):
            self.w = np.zeros((n_in+1, n_out)) if w is None else w

    def new(self, n_in, n_out, seed, zero_last=False):
        rng = np.random.RandomState(seed)
        return self.Net(n_in, n_out, None if zero_last else rng.randn(n_in+1, n_out)*0.01)

    def copy(self, net):
        return self.Net(0, 0, net.w.copy())

    def forward(self, net, x):
        return np.concatenate([x, np.ones((len(x), 1))], 1) @ net.w

    def fit(self, net, x, y, steps, batch, lr, eval_every, patience, val_fn, seed, progress=None):
        a = np.concatenate([x, np.ones((len(x), 1))], 1)
        prior = net.w
        net.w = prior+np.linalg.solve(a.T@a+1e-2*np.eye(a.shape[1]), a.T@(y-a@prior))
        v = float(val_fn(net))
        if progress:
            progress(steps)
        return dict(best_validation=v, best_step=steps, last_step=steps, history=[[steps, 0., v]])

    def finite(self, net):
        return bool(np.isfinite(net.w).all())

    def save(self, net, path):
        np.save(str(path), net.w)

    def load(self, n_in, n_out, path):
        return self.Net(0, 0, np.load(str(path)+('' if str(path).endswith('.npy') else '.npy')))


# ----------------------------------------------------------------------------- predictors

class Model:
    """Raw-space predictor: (s, a) -> (s' - s, r)."""
    def __init__(self, be, net, x_mu, x_sd, y_mu, y_sd, base=None):
        self.be, self.net, self.base = be, net, base
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd

    def __call__(self, x):
        out = self.be.forward(self.net, (x-self.x_mu)/self.x_sd)*self.y_sd+self.y_mu
        return out if self.base is None else out+self.base(x)


def errors(model, d, test_rows, blocks, test_ids, units, horizons, near_mask):
    """All errors in units of the full target training set: dy for (s'-s, r), s for states."""
    x, y = xy(d, test_rows)
    pred = model(x)
    sd = d['obs'].shape[1]
    e = ((pred-y)/units['y_sd'])**2
    state_err = e[:, :sd].mean(1)
    res = dict(one_step_state=float(state_err.mean()), one_step_reward=float(e[:, sd].mean()),
               one_step_state_near=float(state_err[near_mask].mean()), one_step_state_far=float(state_err[~near_mask].mean()))
    for H in horizons:
        starts = np.concatenate([np.arange(a, b-H+1) for a, b in (blocks[i] for i in test_ids) if b-a >= H] or [np.zeros(0, int)])
        s = d['obs'][starts].copy()
        for k in range(H):
            step = model(np.concatenate([s, d['act'][starts+k]], 1))[:, :sd]
            s = np.clip(s+step, -1e6, 1e6)
        err = np.minimum((((s-d['nxt'][starts+H-1])/units['s_sd'])**2).mean(1), 1e12)
        res['rollout%d_state' % H] = float(err.mean())
        res['rollout%d_state_median' % H] = float(np.median(err))
        res['rollout%d_starts' % H] = int(len(starts))
    return res


def nn_distance(be, a, b):
    """Euclidean distance from each row of a to its nearest row of b (both already standardized)."""
    if isinstance(be, Linear):
        out = np.empty(len(a))
        for i in range(0, len(a), 256):
            d2 = (a[i:i+256, None, :]-b[None, :, :])**2
            out[i:i+256] = np.sqrt(d2.sum(-1).min(1))
        return out
    torch = be.torch
    bt = be.tensor(b)
    b2 = (bt**2).sum(1)
    out = []
    with torch.no_grad():
        for i in range(0, len(a), 1024):
            at = be.tensor(a[i:i+1024])
            d2 = (at**2).sum(1, keepdim=True)+b2[None, :]-2*at@bt.T
            out.append(torch.sqrt(torch.clamp(d2.min(1).values, min=0)).double().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- jobs

def run_source(job, out, be, progress):
    d, info = load(job['source_dataset'], job['state_dim'], job['action_dim'], drop_timeouts=True)
    rows = np.arange(len(d['obs']))
    rng = np.random.RandomState(seed_of('source-validation', job['robot'], job['seed']))
    perm = rng.permutation(len(rows))
    n_val = int(round(job['train']['validation_fraction']*len(rows)))
    val, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    x_tr, y_tr = xy(d, tr)
    x_va, y_va = xy(d, val)
    x_mu, x_sd = stats(x_tr)
    y_mu, y_sd = stats(y_tr)
    n_in, n_out = x_tr.shape[1], y_tr.shape[1]
    net = be.new(n_in, n_out, seed_of('source-init', job['robot'], job['seed']))
    xs_va, ys_va = (x_va-x_mu)/x_sd, (y_va-y_mu)/y_sd

    def val_fn(n):
        return ((be.forward(n, xs_va)-ys_va)**2).mean()
    t = job['train']
    fit = be.fit(net, (x_tr-x_mu)/x_sd, (y_tr-y_mu)/y_sd, t['steps'], t['batch'], t['lr'], t['eval_every'], None,
                 val_fn, seed_of('source-batches', job['robot'], job['seed']), progress)
    assert be.finite(net)
    be.save(net, out/'source_model.pt')
    np.savez(str(out/'source_norm.npz'), x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd)
    return dict(kind='source', data=info, train_rows=int(len(tr)), validation_rows=int(len(val)), fit=fit,
                files={n: sha(out/n) for n in os.listdir(str(out)) if n.startswith('source_')})


def run_task(job, out, be, progress):
    sd, ad = job['state_dim'], job['action_dim']
    d, info = load(job['target_dataset'], sd, ad, drop_timeouts=False)
    src = Path(job['source_model_dir'])
    completed = read(src/'COMPLETED.json')
    for name, digest in completed['result']['files'].items():
        assert sha(src/name) == digest, name
    norm = np.load(str(src/'source_norm.npz'))
    n_in, n_out = sd+ad, sd+1
    f_s = Model(be, be.load(n_in, n_out, src/'source_model.pt'), norm['x_mu'], norm['x_sd'], norm['y_mu'], norm['y_sd'])

    blocks = make_blocks(d['end'], job['split']['block_length'], job['split']['min_block'])
    assert len(blocks) <= len(d['obs'])/20, 'target rows are not consecutive trajectories (next_observations != observations[i+1])'
    test_ids, val_ids, subsets = split(blocks, job['split'], job['fractions'], seed_of('split', job['task'], job['seed']))
    test_rows, val_rows = rows_of(blocks, test_ids), rows_of(blocks, val_ids)
    full_rows = rows_of(blocks, subsets[max(job['fractions'])])
    x_full, y_full = xy(d, full_rows)
    ux_mu, ux_sd = stats(x_full)
    uy_mu, uy_sd = stats(y_full)
    units = dict(y_sd=uy_sd, s_sd=stats(d['obs'][full_rows])[1])
    x_va, y_va = xy(d, val_rows)
    x_te, _ = xy(d, test_rows)

    # Coverage (descriptive): distance from test rows to the full training blocks and to source rows.
    src_data, _ = load(job['source_dataset'], sd, ad, drop_timeouts=True)
    rng = np.random.RandomState(seed_of('coverage', job['task'], job['seed']))
    pick = np.sort(rng.choice(len(src_data['obs']), size=min(job['coverage_sample'], len(src_data['obs'])), replace=False))
    x_src = np.concatenate([src_data['obs'][pick], src_data['act'][pick]], 1)
    del src_data
    coverage = dict(test_to_train=float(np.median(nn_distance(be, (x_te-ux_mu)/ux_sd, (x_full-ux_mu)/ux_sd))),
                    test_to_source=float(np.median(nn_distance(be, (x_te-ux_mu)/ux_sd, (x_src-ux_mu)/ux_sd))))

    t = job['train']
    fits, metrics = {}, {}
    done = [0]
    total = len(job['fractions'])*3

    def bump(_step):
        progress(done[0]*t['max_steps']+_step, total*t['max_steps'])
    for f in job['fractions']:
        rows = rows_of(blocks, subsets[f])
        x_tr, y_tr = xy(d, rows)
        near = nn_distance(be, (x_te-ux_mu)/ux_sd, (x_tr-ux_mu)/ux_sd)
        near_mask = near <= np.median(near)
        res = {}
        res['S'] = errors(f_s, d, test_rows, blocks, test_ids, units, job['horizons'], near_mask)
        tag = (job['task'], job['seed'], f)

        # T: fresh network, normalization from this fraction's training rows
        tx_mu, tx_sd = stats(x_tr)
        ty_mu, ty_sd = stats(y_tr)
        net = be.new(n_in, n_out, seed_of('init-T', *tag))

        def val_T(n):
            return ((((Model(be, n, tx_mu, tx_sd, ty_mu, ty_sd)(x_va))-y_va)/uy_sd)**2).mean()
        fits['T_%s' % f] = be.fit(net, (x_tr-tx_mu)/tx_sd, (y_tr-ty_mu)/ty_sd, t['max_steps'], t['batch'], t['lr'],
                                  t['eval_every'], t['patience_evals'], val_T, seed_of('batches-T', *tag), bump)
        assert be.finite(net)
        res['T'] = errors(Model(be, net, tx_mu, tx_sd, ty_mu, ty_sd), d, test_rows, blocks, test_ids, units, job['horizons'], near_mask)
        done[0] += 1

        # FT: start from the source network, keep the source normalization
        net = be.copy(f_s.net)

        def val_FT(n):
            return ((((Model(be, n, f_s.x_mu, f_s.x_sd, f_s.y_mu, f_s.y_sd)(x_va))-y_va)/uy_sd)**2).mean()
        fits['FT_%s' % f] = be.fit(net, (x_tr-f_s.x_mu)/f_s.x_sd, (y_tr-f_s.y_mu)/f_s.y_sd, t['max_steps'], t['batch'], t['lr'],
                                   t['eval_every'], t['patience_evals'], val_FT, seed_of('batches-FT', *tag), bump)
        assert be.finite(net)
        res['FT'] = errors(Model(be, net, f_s.x_mu, f_s.x_sd, f_s.y_mu, f_s.y_sd), d, test_rows, blocks, test_ids, units, job['horizons'], near_mask)
        done[0] += 1

        # RES: frozen source model plus a residual; residual inputs and outputs use this fraction's statistics
        resid = y_tr-f_s(x_tr)
        rx_mu, rx_sd = tx_mu, tx_sd
        r_sd = ty_sd                                     # residual measured in the target's own scale
        net = be.new(n_in, n_out, seed_of('init-RES', *tag), zero_last=True)

        def val_RES(n):
            return ((((Model(be, n, rx_mu, rx_sd, 0., r_sd, base=f_s)(x_va))-y_va)/uy_sd)**2).mean()
        fits['RES_%s' % f] = be.fit(net, (x_tr-rx_mu)/rx_sd, resid/r_sd, t['max_steps'], t['batch'], t['lr'],
                                    t['eval_every'], t['patience_evals'], val_RES, seed_of('batches-RES', *tag), bump)
        assert be.finite(net)
        res['RES'] = errors(Model(be, net, rx_mu, rx_sd, 0., r_sd, base=f_s), d, test_rows, blocks, test_ids, units, job['horizons'], near_mask)
        done[0] += 1
        for name, m in res.items():
            assert all(math.isfinite(v) for v in m.values()), (name, f)
        metrics[str(f)] = res
    return dict(kind='task', data=info, blocks=len(blocks), test_blocks=len(test_ids), validation_blocks=len(val_ids),
                train_blocks={str(f): len(v) for f, v in subsets.items()}, test_rows=int(len(test_rows)),
                validation_rows=int(len(val_rows)), coverage=coverage, metrics=metrics,
                fits={k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in fits.items()},
                fit_history={k: v['history'] for k, v in fits.items()})


def main():
    job = read(sys.argv[1])
    out = Path(job['output'])
    started = time.time()

    def progress(step, total=None):
        write(out/'progress.json', dict(step=int(step), total=int(total or job.get('total_steps', step)), time=time.time()))
    try:
        if not TEST:
            assert os.environ['CUDA_VISIBLE_DEVICES'] == str(job['gpu'])
        assert sha(job['source_dataset']) == job['source_dataset_sha256']
        if job['kind'] == 'task':
            assert sha(job['target_dataset']) == job['target_dataset_sha256']
        be = Linear() if TEST else Torch(job['network'])
        result = (run_source if job['kind'] == 'source' else run_task)(job, out, be, progress)
        write(out/'COMPLETED.json', dict(completed=True, kind=job['kind'], wall_seconds=time.time()-started, result=result))
    except BaseException:
        write(out/'WORKER_ERROR.json', dict(traceback=traceback.format_exc()))
        print(traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
