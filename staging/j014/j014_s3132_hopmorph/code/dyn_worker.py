"""j015a worker: probabilistic dynamics ensembles and the reliable horizon H* (adaptive-source batch).

kind 'source' (robot, seed): K = 5 members (s, a) -> Gaussian over (s' - s, r), 4 x 200 SiLU, Gaussian negative
  log-likelihood; 200000 steps, batch 512, Adam 1e-3 on D4RL medium (recorded next_observations, timeout rows
  dropped, as the j008 model worker); 2% random rows per member held out, evaluated every 5000 steps, best kept
  (j008 / V20 protocol). Inputs / outputs standardised with the member's training-row statistics; these
  statistics are kept for fine-tuning. The source stage depends only on (robot, seed, member), so it is
  trained once per robot and seed and shared by the tasks of that robot (identical to training it per task).
kind 'target' (task, seed): fine-tuning of each source member on target data (at most 30000 steps, batch 256,
  Adam 3e-4, validation every 500 steps, stop after 20 validations without improvement, best kept; validation =
  squared error of the mean in units of the fitting rows' std of (s' - s, r)), in two uses:
  * cross calibration: the trajectories are split into F = min(#trajectories, 5) folds (whole trajectories,
    seeded); for fold f the members are fine-tuned on the other trajectories, early stopping uses a random 10% of
    those rows (excluded from fitting); on the held-out trajectories every row is a start point and the recorded
    actions are replayed open-loop with the member's mean prediction (velocity columns clipped to [-10, 10]);
    e(h) = ||(s_hat - s) / sigma_T|| / sqrt(n) after h steps, sigma_T = std of the target states (floor 1e-3);
    errors of the 5 members, all folds and start points are pooled per h; q90(h) = 0.9 quantile;
    H*(eps0) = the largest h before q90 first exceeds eps0 (0 if q90(1) > eps0), eps0 in {0.05, 0.1, 0.2, 0.5};
  * final models: fine-tuned on all target rows except a random 10% used for early stopping; saved with SHA-256.
"""
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

MAX_LOGVAR, MIN_LOGVAR = 0.5, -10.


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(job['output'])
    code = Path(job['code'])
    sys.path.insert(0, str(code))
    from asrc_common import scrub, write_json
    try:
        result = (run_source if job['kind'] == 'source' else run_target)(job, out, code)
        write_json(out/'COMPLETED.json', result)
    except BaseException:
        write_json(out/'FAILED.json', dict(time=time.time(), job=job['id'], traceback=scrub(traceback.format_exc())))
        print(scrub(traceback.format_exc()), flush=True)
        sys.exit(1)


def torch_setup(job):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if job.get('device', 'cuda') == 'cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(job['gpu'])
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected exactly one visible GPU'
        return torch, 'cuda:0'
    return torch, 'cpu'


def make_net(torch, n_in, n_out, hidden, seed, device):
    nn = torch.nn
    F = torch.nn.functional

    class ProbNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.SiLU()]
                d = h
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d, 2*n_out)
            self.n_out = n_out

        def forward(self, x):
            h = self.head(self.body(x))
            mu, raw = h[:, :self.n_out], h[:, self.n_out:]
            logvar = MAX_LOGVAR-F.softplus(MAX_LOGVAR-raw)
            logvar = MIN_LOGVAR+F.softplus(logvar-MIN_LOGVAR)
            return mu, logvar

    torch.manual_seed(seed)
    return ProbNet().to(device)


def nll(torch, mu, logvar, y):
    return (0.5*((mu-y)**2*torch.exp(-logvar)+logvar)).mean()


def fit(torch, net, x, y, steps, batch, lr, eval_every, patience, val_fn, seed, device, progress=None):
    """Gaussian NLL on standardised (x, y); the network with the best validation value is kept."""
    import numpy as np
    rng = np.random.RandomState(seed)
    xt = torch.as_tensor(np.asarray(x, np.float32), device=device)
    yt = torch.as_tensor(np.asarray(y, np.float32), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, best_step, since, history, step = math.inf, None, 0, 0, [], 0
    for step in range(1, steps+1):
        net.train()
        idx = torch.as_tensor(rng.randint(0, len(x), size=batch), device=device)
        mu, logvar = net(xt[idx])
        loss = nll(torch, mu, logvar, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            lv = float(loss)
            if not math.isfinite(lv):
                raise FloatingPointError('non-finite training loss at step %d' % step)
            v = float(val_fn(net))
            if not math.isfinite(v):
                raise FloatingPointError('non-finite validation value at step %d' % step)
            history.append([step, lv, v])
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


def mean_predict(torch, net, x, device, chunk=65536):
    import numpy as np
    out = []
    with torch.no_grad():
        net.eval()
        for i in range(0, len(x), chunk):
            mu, _ = net(torch.as_tensor(np.asarray(x[i:i+chunk], np.float32), device=device))
            out.append(mu.double().cpu().numpy())
    return np.concatenate(out, 0)


def xy(d, rows):
    import numpy as np
    x = np.concatenate([d['obs'][rows], d['act'][rows]], 1)
    y = np.concatenate([d['nxt'][rows]-d['obs'][rows], d['rew'][rows, None]], 1)
    return x, y


def std_stats(x):
    import numpy as np
    return x.mean(0), np.maximum(x.std(0), 1e-3)


# ----------------------------------------------------------------------------- source stage

def run_source(job, out, code):
    import numpy as np
    from asrc_common import seed_of, sha256_file, write_json
    import asrc_data
    torch, device = torch_setup(job)
    sd, ad = job['state_dim'], job['action_dim']
    m = job['model']
    if job.get('synthetic'):
        d, info = synthetic_transitions(np, sd, ad, job['synthetic']['source_rows'], 3)
    else:
        assert sha256_file(job['source_dataset']) == job['source_sha256'], 'source hash mismatch'
        d, info = asrc_data.raw_transitions(job['source_dataset'], sd, ad, drop_timeouts=True)
    members = []
    for k in range(m['members']):
        rng = np.random.RandomState(seed_of('dyn-source-val', job['robot'], job['seed'], k))
        perm = rng.permutation(len(d['obs']))
        n_val = int(round(m['source']['validation_fraction']*len(perm)))
        val, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        x_tr, y_tr = xy(d, tr)
        x_va, y_va = xy(d, val)
        x_mu, x_sd = std_stats(x_tr)
        y_mu, y_sd = std_stats(y_tr)
        net = make_net(torch, sd+ad, sd+1, m['hidden'], seed_of('dyn-source-init', job['robot'], job['seed'], k), device)
        xs_va, ys_va = (x_va-x_mu)/x_sd, (y_va-y_mu)/y_sd

        def val_fn(n):
            return ((mean_predict(torch, n, xs_va, device)-ys_va)**2).mean()

        def progress(step, k=k):
            write_json(out/'progress.json', dict(step=k*m['source']['steps']+step, total=m['members']*m['source']['steps'], time=time.time()))
        r = fit(torch, net, (x_tr-x_mu)/x_sd, (y_tr-y_mu)/y_sd, job['source_steps'], m['source']['batch'], m['source']['lr'],
                job['source_eval_every'], None, val_fn, seed_of('dyn-source-batches', job['robot'], job['seed'], k), device, progress)
        assert all(bool(torch.isfinite(p).all()) for p in net.parameters())
        path = out/('source_member%d.pt' % k)
        torch.save(dict(state=net.state_dict(), x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd, hidden=list(m['hidden']),
                        n_in=sd+ad, n_out=sd+1), str(path))
        members.append(dict(member=k, file=path.name, sha256=sha256_file(path), train_rows=int(len(tr)),
                            validation_rows=int(len(val)), best_validation=r['best_validation'], best_step=r['best_step'],
                            history=r['history']))
    return dict(kind='source', robot=job['robot'], seed=job['seed'], data=info, members=members)


def synthetic_transitions(np, sd, ad, n, seed):
    rng = np.random.RandomState(seed)
    W = rng.randn(sd+ad, sd)*.05
    obs = np.zeros((n, sd))
    act = np.tanh(rng.randn(n, ad))
    nxt = np.zeros((n, sd))
    end = np.zeros(n, bool)
    s = rng.randn(sd)*.1
    for i in range(n):
        obs[i] = s
        nxt[i] = s+np.concatenate([s, act[i]])@W+.01*rng.randn(sd)
        s = nxt[i]
        if (i+1) % 200 == 0:
            end[i] = True
            s = rng.randn(sd)*.1
    end[-1] = True
    obs[:, 0] = 1.25+.05*np.tanh(obs[:, 0])      # healthy heights for both robots
    nxt[:, 0] = 1.25+.05*np.tanh(nxt[:, 0])
    rew = 1.+.1*np.tanh(obs[:, 1])
    return dict(obs=obs, act=act, nxt=nxt, rew=rew, term=np.zeros(n, bool), end=end), dict(rows=int(n), synthetic=True)


# ----------------------------------------------------------------------------- target stage

def load_member(torch, path, device):
    blob = torch.load(str(path), map_location=device)
    net = make_net(torch, blob['n_in'], blob['n_out'], blob['hidden'], 0, device)
    net.load_state_dict(blob['state'])
    return net, blob


def finetune(torch, job, d, fit_rows, es_rows, members, seed_parts, device, progress=None):
    """Fine-tune every member (fresh copy from its source weights) on fit_rows; returns nets and fit records."""
    import numpy as np
    from asrc_common import seed_of
    m = job['model']
    x_fit, y_fit = xy(d, fit_rows)
    x_es, y_es = xy(d, es_rows)
    unit = std_stats(y_fit)[1]
    nets, recs = [], []
    for k, (path, sha) in enumerate(members):
        net, blob = load_member(torch, path, device)
        x_mu, x_sd, y_mu, y_sd = (np.asarray(blob[n]) for n in ('x_mu', 'x_sd', 'y_mu', 'y_sd'))
        xs_es = (x_es-x_mu)/x_sd

        def val_fn(n):
            pred = mean_predict(torch, n, xs_es, device)*y_sd+y_mu
            return (((pred-y_es)/unit)**2).mean()
        before = float(val_fn(net))
        r = fit(torch, net, (x_fit-x_mu)/x_sd, (y_fit-y_mu)/y_sd, job['target_max_steps'], m['target']['batch'],
                m['target']['lr'], job['target_eval_every'], m['target']['patience_evals'], val_fn,
                seed_of('dyn-ft-batches', job['task'], job['seed'], seed_parts, k), device,
                (lambda s, k=k: progress(k, s)) if progress else None)
        assert all(bool(torch.isfinite(p).all()) for p in net.parameters())
        nets.append((net, [torch.as_tensor(np.asarray(v, np.float32), device=device) for v in (x_mu, x_sd, y_mu, y_sd)]))
        recs.append(dict(member=k, source_only_validation=before, best_validation=r['best_validation'],
                         best_step=r['best_step'], last_step=r['last_step']))
    return nets, recs


def openloop_errors(torch, nets, d, segs, ids, sigma, sd, vel_start, device):
    """Pooled errors per horizon h (list index h-1): open-loop replay of recorded actions from every row of the
    held-out trajectories, mean prediction of each member, velocity columns clipped to [-10, 10]."""
    import numpy as np
    per_h = {}
    sig = torch.as_tensor(np.asarray(sigma, np.float64), device=device)
    for g in ids:
        a, b = segs[g]
        L = b-a
        states = np.concatenate([d['obs'][a:b], d['nxt'][b-1:b]], 0)          # L + 1 states
        acts = d['act'][a:b]
        S = torch.as_tensor(states, dtype=torch.float64, device=device)
        A = torch.as_tensor(np.asarray(acts, np.float32), device=device)
        for net, (x_mu, x_sd, y_mu, y_sd) in nets:
            s_hat = S[:L].clone()                                             # start t0 = 0 .. L-1
            t0 = torch.arange(L, device=device)
            with torch.no_grad():
                net.eval()
                for j in range(L):
                    active = L-j                                              # rows t0 <= L-1-j
                    s_cur = s_hat[:active]
                    x = torch.cat([s_cur.float(), A[t0[:active]+j]], 1)
                    mu, _ = net((x-x_mu)/x_sd)
                    delta = (mu*y_sd+y_mu)[:, :sd].double()
                    nxt = s_cur+delta
                    nxt = torch.cat([nxt[:, :vel_start], torch.clamp(nxt[:, vel_start:], -10., 10.)], 1)
                    err = torch.sqrt((((nxt-S[t0[:active]+j+1])/sig)**2).mean(1))
                    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float('inf')))
                    per_h.setdefault(j+1, []).append(err.cpu().numpy().astype(np.float32))
                    nxt = torch.where(torch.isfinite(nxt), nxt, s_cur)
                    s_hat = nxt
    return {h: np.concatenate(v) for h, v in per_h.items()}


def h_star(q90, eps):
    """Largest h before q90 first exceeds eps (q90[h-1] is the quantile at horizon h); 0 if q90(1) > eps."""
    for i, q in enumerate(q90):
        if not (q <= eps):
            return i
    return len(q90)


def run_target(job, out, code):
    import numpy as np
    from asrc_common import read_json, rel, seed_of, sha256_file, write_json
    import asrc_data
    torch, device = torch_setup(job)
    sd, ad = job['state_dim'], job['action_dim']
    m = job['model']
    if job.get('synthetic'):
        d, info = synthetic_transitions(np, sd, ad, job['synthetic']['target_rows'], 5)
    else:
        assert sha256_file(job['target_dataset']) == job['target_sha256'], 'target hash mismatch'
        d, info = asrc_data.raw_transitions(job['target_dataset'], sd, ad, drop_timeouts=False)
    segs = asrc_data.segment_bounds(d['end'])
    n_traj = len(segs)
    assert n_traj >= 2, 'need at least two target trajectories'
    sigma = np.maximum(d['obs'].std(0), 1e-3)
    members = []
    for k in range(m['members']):
        src_dir = Path(job['source_dir'])
        done = read_json(src_dir/'COMPLETED.json')
        rec = next(r for r in done['members'] if r['member'] == k)
        path = src_dir/rec['file']
        assert sha256_file(path) == rec['sha256'], 'source member changed: %s' % rec['file']
        members.append((path, rec['sha256']))
    total = (min(n_traj, m['folds_max'])+1)*m['members']*job['target_max_steps']
    done_units = [0]

    def progress(k, s):
        write_json(out/'progress.json', dict(step=done_units[0]+k*job['target_max_steps']+s, total=total, time=time.time()))

    # ---------------- cross calibration
    F_ = min(n_traj, m['folds_max'])
    order = np.random.RandomState(seed_of('dyn-folds', job['task'], job['seed'])).permutation(n_traj)
    fold_of = np.empty(n_traj, np.int64)
    fold_of[order] = np.arange(n_traj) % F_
    pooled, folds = {}, []
    for f in range(F_):
        held = [int(g) for g in np.flatnonzero(fold_of == f)]
        rest = [int(g) for g in np.flatnonzero(fold_of != f)]
        rows = np.concatenate([np.arange(*segs[g]) for g in rest])
        rng = np.random.RandomState(seed_of('dyn-es', job['task'], job['seed'], 'fold%d' % f))
        es = np.sort(rng.choice(rows, size=max(1, int(round(m['es_fraction']*len(rows)))), replace=False))
        fit_rows = np.setdiff1d(rows, es)
        nets, recs = finetune(torch, job, d, fit_rows, es, members, 'fold%d' % f, device, progress)
        done_units[0] += m['members']*job['target_max_steps']
        errs = openloop_errors(torch, nets, d, segs, held, sigma, sd, job['vel_start'], device)
        for h, e in errs.items():
            pooled.setdefault(h, []).append(e)
        folds.append(dict(fold=f, held_out_trajectories=held, fit_rows=int(len(fit_rows)), es_rows=int(len(es)), fits=recs))
        del nets
    hs = sorted(pooled)
    assert hs == list(range(1, len(hs)+1))
    q90 = [float(np.quantile(np.concatenate(pooled[h]), m['quantile'])) for h in hs]
    q50 = [float(np.quantile(np.concatenate(pooled[h]), .5)) for h in hs]
    counts = [int(sum(len(x) for x in pooled[h])) for h in hs]
    hstar = {'%g' % e: h_star(q90, e) for e in m['eps0']}

    # ---------------- final models (all target data, 10% for early stopping)
    rows = np.arange(len(d['obs']))
    rng = np.random.RandomState(seed_of('dyn-es', job['task'], job['seed'], 'final'))
    es = np.sort(rng.choice(rows, size=max(1, int(round(m['es_fraction']*len(rows)))), replace=False))
    fit_rows = np.setdiff1d(rows, es)
    nets, recs = finetune(torch, job, d, fit_rows, es, members, 'final', device, progress)
    model_dir = Path(job['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for k, (net, norms) in enumerate(nets):
        path = model_dir/('member%d.pt' % k)
        torch.save(dict(state=net.state_dict(), x_mu=norms[0].cpu().numpy(), x_sd=norms[1].cpu().numpy(),
                        y_mu=norms[2].cpu().numpy(), y_sd=norms[3].cpu().numpy(), hidden=list(m['hidden']),
                        n_in=sd+ad, n_out=sd+1), str(path))
        saved.append(dict(member=k, path=rel(path), sha256=sha256_file(path), bytes=path.stat().st_size))
    return dict(kind='target', task=job['task'], seed=job['seed'], data=info, trajectories=n_traj, folds=folds,
                sigma_T=sigma, h_max=len(hs), q90=q90, q50=q50, counts=counts, h_star=hstar,
                final=dict(fit_rows=int(len(fit_rows)), es_rows=int(len(es)), fits=recs), saved_models=saved)


if __name__ == '__main__':
    main()
