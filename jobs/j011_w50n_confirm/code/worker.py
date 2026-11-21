"""j009 worker: one training run (2x2 source-composition screen: far-source weight x near-source term).

Groups (T, BC, BCMIX are the V21/j004-j008 definitions, unchanged):
  T       target-only IQL (frozen iql_dyn.IQL_DYN, target batch of 64 only)
  BC      target-only behaviour cloning (frozen iql_unc.Policy.bc_loss)
  BCMIX   BC on 64 target rows + 1.0 * BC on 64 uniform source rows          (far weight a=1,   near term b=0)
  W50     BC on 64 target rows + 0.5 * BC on 64 uniform source rows          (a=0.5, b=0; = j008 W50)
  W50N    W50 + 0.5 * BC on 32 rows drawn uniformly from the near set         (a=0.5, b=0.5; ~ j007 PB50)
  BCMIXN  BCMIX + 0.5 * BC on 32 rows drawn uniformly from the near set       (a=1,   b=0.5; candidate)
Each term is a batch mean. All source groups of one (task, seed) share the initialization, the target index stream,
the 64-row uniform source index stream and (W50N, BCMIXN) the 32-row near index stream.
Near set: exactly as j005 CG90 / j007 - source rows whose state is within tau of some target state, distances
Euclidean in the state space standardised by the target-state mean / standard deviation, tau = 90% quantile of the
target states' own leave-one-trajectory-out nearest-neighbour distances. Only offline data, fixed before training.

Source rows: D4RL medium, timeout rows dropped, recorded actions (no relabelling). Raw observations.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
IQL_GROUPS = ('T',)
SOURCE_GROUPS = ('BCMIX', 'W50', 'W50N', 'BCMIXN')
EXTRA_INPUTS = {'T': 0, 'BC': 0, 'BCMIX': 0, 'W50': 0, 'W50N': 0, 'BCMIXN': 0}
NEAR_ROWS = {'W50N': 32, 'BCMIXN': 32}       # rows of the separate near batch
FAR_WEIGHT = {'BCMIX': 1., 'W50': .5, 'W50N': .5, 'BCMIXN': 1.}   # weight of the 64-row uniform source term
NEAR_WEIGHT = .5                               # weight of the near term
NEAR_QUANTILE = 0.9                            # j005 CG90 threshold


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


def seed_of(purpose, task, seed):
    text = 'J009-20261121|%s|%d|%s' % (task, seed, purpose)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def load(path, sd, ad, np, h5py, drop_timeouts):
    with h5py.File(path, 'r') as f:
        arrays = [np.asarray(f[k], dtype=np.float32) for k in KEYS]
        timeouts = np.asarray(f['timeouts']).reshape(-1).astype(bool) if 'timeouts' in f else None
    n = len(arrays[0])
    assert all(len(a) == n and np.isfinite(a).all() for a in arrays)
    assert arrays[0].shape == (n, sd) and arrays[1].shape == (n, ad) and arrays[2].shape == (n, sd)
    assert np.isin(arrays[4], [0, 1]).all()
    dropped = 0
    keep = np.ones(n, bool)
    if drop_timeouts and timeouts is not None:
        keep = ~timeouts
        dropped = int(timeouts.sum())
        arrays = [a[keep] for a in arrays]
    arrays[3] = arrays[3].reshape(-1, 1)
    arrays[4] = 1.-arrays[4].reshape(-1, 1)          # not-done mask
    return arrays, keep, dict(rows=int(len(arrays[0])), dropped_timeout_rows=dropped, has_timeouts=timeouts is not None)


def segment_returns(path, np, h5py, lo, hi):
    """Per-row segment id and per-segment normalised return (V21 definition)."""
    with h5py.File(path, 'r') as f:
        obs = np.asarray(f['observations'], dtype=np.float64)
        nxt = np.asarray(f['next_observations'], dtype=np.float64)
        rew = np.asarray(f['rewards'], dtype=np.float64).reshape(-1)
        term = np.asarray(f['terminals']).reshape(-1).astype(bool)
        timeouts = np.asarray(f['timeouts']).reshape(-1).astype(bool) if 'timeouts' in f else np.zeros(len(rew), bool)
    cont = np.zeros(len(rew), bool)
    cont[:-1] = np.all(np.abs(nxt[:-1]-obs[1:]) <= 1e-5, axis=1)
    end = term | timeouts | ~cont
    end[-1] = True
    seg = np.concatenate([[0], np.cumsum(end[:-1])]).astype(np.int64)
    ret = np.bincount(seg, weights=rew)
    return seg, 100*(ret-lo)/(hi-lo)


def nearest(a, b, np, chunk=4096):
    """Euclidean distance from each row of a to its nearest row of b (float64, exact, deterministic)."""
    b2 = (b*b).sum(1)
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        x = a[i:i+chunk]
        d2 = (x*x).sum(1)[:, None]+b2[None, :]-2.*(x@b.T)
        out[i:i+chunk] = np.sqrt(np.maximum(d2.min(1), 0.))
    return out


def proximity_split(target_obs, target_seg, source_obs, quantile, np):
    """Mask over source rows: True = state farther from all target states than the target's own LOTO quantile.
    Identical to j005's coverage_gate; here the complement (~mask) is the near set that is boosted."""
    mu = target_obs.mean(0)
    sd = np.maximum(target_obs.std(0), 1e-3)
    t = ((target_obs-mu)/sd).astype(np.float64)
    s = ((source_obs-mu)/sd).astype(np.float64)
    loto = np.empty(len(t))
    for g in np.unique(target_seg):
        inside = target_seg == g
        assert (~inside).any(), 'need at least two target trajectory segments'
        loto[inside] = nearest(t[inside], t[~inside], np)
    tau = float(np.quantile(loto, quantile))
    ds = nearest(s, t, np, chunk=2048)
    keep = ds > tau
    assert keep.any() and (~keep).sum() >= 64, 'near set must be non-empty (>= 64 rows) and not all rows'
    info = dict(quantile=quantile, tau=tau, target_segments=int(len(np.unique(target_seg))),
                target_loto_quantiles={str(q): float(np.quantile(loto, q)) for q in (0.1, 0.5, 0.9, 0.99)},
                source_distance_quantiles={str(q): float(np.quantile(ds, q)) for q in (0.01, 0.1, 0.5, 0.9)},
                source_rows=int(len(ds)), near_rows=int((~keep).sum()), near_fraction=float((~keep).mean()),
                near_distance_mean=float(ds[~keep].mean()))
    return keep, info


def conditioning(group, path, keep, n_rows, domain_flag, np, h5py, lo, hi):
    """Extra input columns for DCBC / QDCBC rows, and segment statistics."""
    k = EXTRA_INPUTS[group]
    if k == 0:
        return None, None
    cols = [np.full((n_rows, 1), float(domain_flag), np.float32)]
    info = None
    if k == 2:
        seg, norm = segment_returns(path, np, h5py, lo, hi)
        q = (norm[seg]/100.).astype(np.float32)[keep].reshape(-1, 1)
        assert len(q) == n_rows and np.isfinite(q).all()
        cols.append(q)
        info = dict(segments=int(len(norm)), norm_return_mean=float(norm.mean()), norm_return_max=float(norm.max()),
                    norm_return_min=float(norm.min()))
    return np.concatenate(cols, axis=1), info


def run(job, out):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(job['gpu'])
    sys.path.insert(0, job['vendor'])
    sys.path.insert(0, job['algo'])
    import numpy as np
    import torch
    import h5py
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if job.get('device', 'cuda') == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected exactly one visible GPU'
        device = 'cuda:0'
    else:
        device = 'cpu'                                  # local synthetic test only
    from offline_offline import iql_dyn, iql_unc
    assert Path(iql_dyn.__file__).resolve().parent == (Path(job['algo'])/'offline_offline').resolve()
    task, group, steps = job['task'], job['group'], job['steps']
    sd, ad = job['state_dim'], job['action_dim']
    lo, hi = job['score_min'], job['score_max']
    extra = EXTRA_INPUTS[group]

    assert sha(job['target_dataset']) == job['target_dataset_sha256'] and sha(job['xml']) == job['xml_sha256']
    target, t_keep, target_info = load(job['target_dataset'], sd, ad, np, h5py, drop_timeouts=False)
    assert target_info['rows'] == job['target_rows']
    t_cond, t_seg_info = conditioning(group, job['target_dataset'], t_keep, len(target[0]), 1, np, h5py, lo, hi)
    source, source_info, s_cond, s_seg_info, near = None, None, None, None, None
    if group in SOURCE_GROUPS:
        assert sha(job['source_dataset']) == job['source_dataset_sha256']
        source, s_keep, source_info = load(job['source_dataset'], sd, ad, np, h5py, drop_timeouts=True)
        s_cond, s_seg_info = conditioning(group, job['source_dataset'], s_keep, len(source[0]), 0, np, h5py, lo, hi)
        if group in NEAR_ROWS:
            t_seg, _ = segment_returns(job['target_dataset'], np, h5py, lo, hi)
            far, near_info = proximity_split(target[0], t_seg[t_keep], source[0], NEAR_QUANTILE, np)
            near = [a[~far] for a in source]          # source itself stays complete (uniform part samples all rows)
            source_info = dict(source_info, proximity=dict(near_info, near_rows_per_batch=NEAR_ROWS[group], near_weight=NEAR_WEIGHT))
    eval_cond = None
    if extra:
        eval_cond = [1.0] if extra == 1 else [1.0, t_seg_info['norm_return_max']/100.]
    t_in = target[0] if t_cond is None else np.concatenate([target[0], t_cond], axis=1)
    s_in = None if source is None else (source[0] if s_cond is None else np.concatenate([source[0], s_cond], axis=1))

    s = seed_of('initialization', task, job['seed'])
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if device != 'cpu':
        torch.cuda.manual_seed_all(s)
    if group in IQL_GROUPS:
        config = dict(state_dim=sd, action_dim=ad, hidden_sizes=256, max_action=1., gamma=.99, tau=.005,
                      update_interval=2, lam=.7, temp=10., critic_lr=.0003, actor_lr=.0003, max_step=steps,
                      seed=job['seed'], source_budget=1., source_loss_weight=1.)
        model = iql_dyn.IQL_DYN(config, device)
        policy = model.policy
        nets = dict(policy=model.policy, V=model.v_func, Q=model.q_funcs, target_Q=model.target_q_funcs)
    else:
        config = dict(lr=.0003, hidden_sizes=256, cosine_T_max=steps, policy_input_dim=sd+extra,
                      eval_conditioning=eval_cond)
        model = None
        policy = iql_unc.Policy(sd+extra, ad, 1., 256).to(device)
        bc_optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        bc_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(bc_optimizer, steps)
        nets = dict(policy=policy)

    def digest():
        h = hashlib.sha256()
        for net in nets.values():
            for value in net.state_dict().values():
                h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def finite():
        return all(bool(torch.isfinite(p).all()) for net in nets.values() for p in net.parameters())

    write(out/'config.json', dict(job=job, learner_config=config, batch='64 target rows (+ a * mean over 64 uniform source rows, a = FAR_WEIGHT; W50N/BCMIXN + 0.5 * mean over 32 near rows)', far_weight=FAR_WEIGHT.get(group), near_weight=NEAR_WEIGHT if group in NEAR_ROWS else None,
        observation_scaling='raw', reward_scaling=1, threads=1, deterministic_algorithms=True, TF32=True,
        target=target_info, source=source_info, target_segments=t_seg_info, source_segments=s_seg_info,
        iql_dyn_sha256=sha(iql_dyn.__file__), iql_unc_sha256=sha(iql_unc.__file__),
        initialization_sha256=digest(), torch=torch.__version__))

    class Sampler:
        def __init__(self, arrays, purpose, size=64):
            self.arrays, self.n, self.size = arrays, len(arrays[0]), size
            self.rng = np.random.RandomState(seed_of(purpose, task, job['seed']))
            self.hash = hashlib.sha256()

        def sample(self):
            ix = self.rng.randint(self.n, size=self.size)
            self.hash.update(ix.astype('<i8').tobytes())
            return tuple(torch.as_tensor(np.asarray(a[ix]), dtype=torch.float32, device=device) for a in self.arrays)

    near_sampler = None
    if group in IQL_GROUPS:
        target_sampler = Sampler(target, 'target_sample')
        source_sampler = None
    else:
        target_sampler = Sampler([t_in, target[1]], 'target_sample')
        k = NEAR_ROWS.get(group, 0)
        source_sampler = Sampler([s_in, source[1]], 'source_sample', 64) if source is not None else None
        if k:
            assert s_cond is None
            near_sampler = Sampler([near[0], near[1]], 'near_sample', k)

    def iql_step():
        model.total_it += 1
        ts, ta, tn, tr, tm = target_sampler.sample()
        tv, adv, _ = model.v_loss_per_sample(ts, ta)
        v_loss = tv.mean()
        model.v_optimizer.zero_grad()
        v_loss.backward()
        model.v_optimizer.step()
        tq, _, _ = model.q_loss_per_sample(ts, ta, tr, tn, tm)
        q_loss = tq.mean()
        model.q_optimizer.zero_grad()
        q_loss.backward()
        model.q_optimizer.step()
        model.update_target()
        for p in model.q_funcs.parameters():
            p.requires_grad = False
        pi_loss = model.policy_loss_per_sample(adv, ts, ta).mean()
        model.policy_optimizer.zero_grad()
        pi_loss.backward()
        model.policy_optimizer.step()
        model.policy_lr_schedule.step()
        for p in model.q_funcs.parameters():
            p.requires_grad = True
        return dict(v=float(v_loss.detach()), q=float(q_loss.detach()), target_policy=float(pi_loss.detach()))

    def bc_step():
        batch = target_sampler.sample()
        target_loss = policy.bc_loss(batch[0], batch[1]).mean(dim=1, keepdim=True).mean()
        loss = target_loss
        source_loss = None
        near_loss = None
        if source_sampler is not None:
            sbatch = source_sampler.sample()
            assert sbatch[0].shape[0] == 64
            source_loss = policy.bc_loss(sbatch[0], sbatch[1]).mean(dim=1, keepdim=True).mean()
            loss = loss+FAR_WEIGHT[group]*source_loss
            if near_sampler is not None:              # separate near batch, its own batch mean
                nbatch = near_sampler.sample()
                assert nbatch[0].shape[0] == NEAR_ROWS[group]
                near_loss = policy.bc_loss(nbatch[0], nbatch[1]).mean(dim=1, keepdim=True).mean()
                loss = loss+NEAR_WEIGHT*near_loss
        bc_optimizer.zero_grad()
        loss.backward()
        bc_optimizer.step()
        bc_schedule.step()
        return dict(target_bc=float(target_loss.detach()), source_bc=None if source_loss is None else float(source_loss.detach()),
                    near_bc=None if near_loss is None else float(near_loss.detach()))

    cond_tensor = None if eval_cond is None else torch.as_tensor([eval_cond], dtype=torch.float32, device=device)

    def act(state):
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
            if cond_tensor is not None:
                x = torch.cat([x, cond_tensor], dim=1)
            return torch.tanh(policy.network(x).chunk(2, dim=1)[0]).squeeze(0).cpu().numpy()

    if job.get('env_factory'):                          # local synthetic test only
        env = __import__(job['env_factory']).make(sd, ad)
    else:
        import gym
        constructor = getattr(__import__(job['env_module'], fromlist=[job['env_class']]), job['env_class'])
        env = gym.wrappers.TimeLimit(constructor(xml_file=job['xml']), max_episode_steps=1000)
        assert env.unwrapped.frame_skip == 4 and abs(float(env.unwrapped.model.opt.timestep)-.002) < 1e-12
        assert env.observation_space.shape == (sd,) and env.action_space.shape == (ad,)

    def evaluate(step):
        saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if device != 'cpu' else None)
        returns, lengths, seeds = [], [], []
        try:
            for episode in range(10):
                value = 500000+10*(job['seed']-job['seed_origin'])+episode
                seeds.append(value)
                env.seed(value)
                state, done, ret, length = env.reset(), False, 0., 0
                while not done:
                    state, reward, done, _ = env.step(act(state))
                    ret += float(reward)
                    length += 1
                returns.append(ret)
                lengths.append(length)
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            torch.set_rng_state(saved[2])
            if saved[3] is not None:
                torch.cuda.set_rng_state_all(saved[3])
        scores = (np.asarray(returns)-lo)/(hi-lo)*100.
        assert np.isfinite(scores).all()
        return dict(step=step, raw_returns=returns, scores=scores.tolist(), mean=float(scores.mean()),
                    episode_lengths=lengths, env_seeds=seeds, steps=int(sum(lengths)))

    started = time.monotonic()
    update = iql_step if group in IQL_GROUPS else bc_step
    evaluations = 0
    losses = {}
    try:
        for step in range(1, steps+1):
            losses = update()
            if step % 1000 == 0 or step == steps:
                assert all(v is None or math.isfinite(v) for v in losses.values()) and finite(), 'nonfinite at step %d' % step
                write(out/'progress.json', dict(step=step, total=steps, elapsed=time.monotonic()-started, losses=losses,
                                                time=time.time(), pid=os.getpid()))
            if step % job['eval_every'] == 0:
                write(out/('eval_%07d.json' % step), evaluate(step))
                evaluations += 1
    finally:
        env.close()
    checkpoint = out/'final_checkpoint.pt'
    torch.save({name: net.state_dict() for name, net in nets.items()}, str(checkpoint))
    write(out/'COMPLETED.json', dict(job=job, steps=steps, finite=finite(), evaluations=evaluations,
        target_stream_sha256=target_sampler.hash.hexdigest(),
        source_stream_sha256=None if source_sampler is None else source_sampler.hash.hexdigest(),
        near_stream_sha256=None if near_sampler is None else near_sampler.hash.hexdigest(),
        target=target_info, source=source_info, eval_conditioning=eval_cond, final_losses=losses,
        model_digest=digest(), checkpoint_sha256=sha(checkpoint), wall_seconds=time.monotonic()-started,
        GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device != 'cpu' else 0, ended=time.time()))


def main():
    job = read(sys.argv[1])
    out = Path(job['output'])
    try:
        run(job, out)
    except BaseException:
        write(out/'FAILED.json', dict(time=time.time(), traceback=traceback.format_exc(), job=job['id']))
        raise


if __name__ == '__main__':
    main()
