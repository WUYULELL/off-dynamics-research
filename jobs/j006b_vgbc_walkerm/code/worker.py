"""j006 worker: one training run (value-guided BC-mixing screen).

Groups (T, BC, BCMIX are the V21/j004/j005 definitions, unchanged):
  T      target-only IQL (frozen iql_dyn.IQL_DYN, target batch of 64 only)
  BC     target-only behaviour cloning (frozen iql_unc.Policy.bc_loss)
  BCMIX  BC on 64 target rows plus BC on 64 source rows, the two batch means added
  AWM, AFM  (candidates) value-guided BCMIX. A critic (V, double Q, target Q) is trained on TARGET rows only with
         exactly the IQL updates of group T (expectile 0.7, gamma 0.99, Polyak 0.005, lr 3e-4, batch 64): first
         critic_warmup critic-only steps, then each of the `steps` actor steps also does one critic step. The actor is
         a fresh iql_unc.Policy trained like BC/BCMIX (Adam 3e-4, cosine over `steps`) on 64 target + 64 source rows,
         with the advantage A(s,a) = min target-Q(s,a) - V(s) of the TARGET-trained critic (no gradient):
    AWM  advantage-weighted: u = min(exp(beta*A), 100) with beta = 3; weights normalised jointly over the 128 rows,
         w = u / mean(u); loss = 2 * mean(w * mse). beta -> 0 gives BCMIX exactly. The critic decides, per row,
         whether the target or the source action at that state is imitated more.
    AFM  advantage-filtered: target rows weight 1; a source row weight 1 if A > 0 else 0;
         loss = mean_target(mse) + sum_source(keep * mse) / 64.

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
VG_GROUPS = ('AWM', 'AFM')
SOURCE_GROUPS = ('BCMIX', 'AWM', 'AFM')
AWM_BETA = 3.0
WEIGHT_CLIP = 100.0


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
    text = 'J006-20261119|%s|%d|%s' % (task, seed, purpose)
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
    if drop_timeouts and timeouts is not None:
        keep = ~timeouts
        dropped = int(timeouts.sum())
        arrays = [a[keep] for a in arrays]
    arrays[3] = arrays[3].reshape(-1, 1)
    arrays[4] = 1.-arrays[4].reshape(-1, 1)          # not-done mask
    return arrays, dict(rows=int(len(arrays[0])), dropped_timeout_rows=dropped, has_timeouts=timeouts is not None)


class Running:
    """Running means of guidance statistics over all actor steps and over the last 20% of them."""

    def __init__(self, steps):
        self.tail_start = int(0.8*steps)+1
        self.all, self.tail, self.n_all, self.n_tail = {}, {}, 0, 0

    def add(self, step, values):
        self.n_all += 1
        for k, v in values.items():
            self.all[k] = self.all.get(k, 0.)+v
        if step >= self.tail_start:
            self.n_tail += 1
            for k, v in values.items():
                self.tail[k] = self.tail.get(k, 0.)+v

    def result(self):
        return dict(all_steps={k: v/self.n_all for k, v in self.all.items()} if self.n_all else None,
                    last_20pct={k: v/self.n_tail for k, v in self.tail.items()} if self.n_tail else None,
                    actor_steps=self.n_all)


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
    warmup = int(job.get('critic_warmup', 0)) if group in VG_GROUPS else 0
    sd, ad = job['state_dim'], job['action_dim']
    lo, hi = job['score_min'], job['score_max']

    assert sha(job['target_dataset']) == job['target_dataset_sha256'] and sha(job['xml']) == job['xml_sha256']
    target, target_info = load(job['target_dataset'], sd, ad, np, h5py, drop_timeouts=False)
    assert target_info['rows'] == job['target_rows']
    source, source_info = None, None
    if group in SOURCE_GROUPS:
        assert sha(job['source_dataset']) == job['source_dataset_sha256']
        source, source_info = load(job['source_dataset'], sd, ad, np, h5py, drop_timeouts=True)

    s = seed_of('initialization', task, job['seed'])
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if device != 'cpu':
        torch.cuda.manual_seed_all(s)
    iql_config = dict(state_dim=sd, action_dim=ad, hidden_sizes=256, max_action=1., gamma=.99, tau=.005,
                      update_interval=2, lam=.7, temp=10., critic_lr=.0003, actor_lr=.0003, max_step=steps,
                      seed=job['seed'], source_budget=1., source_loss_weight=1.)
    model = None
    if group in IQL_GROUPS:
        config = iql_config
        model = iql_dyn.IQL_DYN(config, device)
        policy = model.policy
        nets = dict(policy=model.policy, V=model.v_func, Q=model.q_funcs, target_Q=model.target_q_funcs)
    else:
        config = dict(lr=.0003, hidden_sizes=256, cosine_T_max=steps, policy_input_dim=sd)
        if group in VG_GROUPS:
            model = iql_dyn.IQL_DYN(dict(iql_config, max_step=warmup+steps), device)   # critic only; its policy unused
            config = dict(config, critic=dict(iql_config, max_step=warmup+steps), critic_warmup=warmup,
                          guidance=group, awm_beta=AWM_BETA, weight_clip=WEIGHT_CLIP)
        policy = iql_unc.Policy(sd, ad, 1., 256).to(device)
        bc_optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        bc_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(bc_optimizer, steps)
        nets = dict(policy=policy)
        if model is not None:
            nets.update(V=model.v_func, Q=model.q_funcs, target_Q=model.target_q_funcs)

    def digest():
        h = hashlib.sha256()
        for net in nets.values():
            for value in net.state_dict().values():
                h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def finite():
        return all(bool(torch.isfinite(p).all()) for net in nets.values() for p in net.parameters())

    write(out/'config.json', dict(job=job, learner_config=config, batch='64 target rows (+64 source rows)',
        observation_scaling='raw', reward_scaling=1, threads=1, deterministic_algorithms=True, TF32=True,
        target=target_info, source=source_info,
        iql_dyn_sha256=sha(iql_dyn.__file__), iql_unc_sha256=sha(iql_unc.__file__),
        initialization_sha256=digest(), torch=torch.__version__))

    class Sampler:
        def __init__(self, arrays, purpose):
            self.arrays, self.n = arrays, len(arrays[0])
            self.rng = np.random.RandomState(seed_of(purpose, task, job['seed']))
            self.hash = hashlib.sha256()

        def sample(self):
            ix = self.rng.randint(self.n, size=64)
            self.hash.update(ix.astype('<i8').tobytes())
            return tuple(torch.as_tensor(np.asarray(a[ix]), dtype=torch.float32, device=device) for a in self.arrays)

    critic_sampler = None
    if group in IQL_GROUPS:
        target_sampler = Sampler(target, 'target_sample')
        source_sampler = None
    else:
        target_sampler = Sampler([target[0], target[1]], 'target_sample')
        source_sampler = Sampler([source[0], source[1]], 'source_sample') if source is not None else None
        if group in VG_GROUPS:
            critic_sampler = Sampler(target, 'critic_sample')

    def critic_step(ts, ta, tn, tr, tm):
        """The V / Q / target-Q part of iql_step, unchanged. Returns (v_loss, q_loss, adv on this batch)."""
        model.total_it += 1
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
        return v_loss, q_loss, adv

    def iql_step():
        ts, ta, tn, tr, tm = target_sampler.sample()
        v_loss, q_loss, adv = critic_step(ts, ta, tn, tr, tm)
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

    def warmup_step():
        v_loss, q_loss, _ = critic_step(*critic_sampler.sample())
        return dict(v=float(v_loss.detach()), q=float(q_loss.detach()))

    def advantage(states, actions):
        with torch.no_grad():
            q1, q2 = model.target_q_funcs(states, actions)
            return torch.min(q1, q2)-model.v_func(states)

    def bc_step():
        batch = target_sampler.sample()
        target_loss = policy.bc_loss(batch[0], batch[1]).mean(dim=1, keepdim=True).mean()
        loss = target_loss
        source_loss = None
        if source_sampler is not None:
            sbatch = source_sampler.sample()
            source_loss = policy.bc_loss(sbatch[0], sbatch[1]).mean(dim=1, keepdim=True).mean()
            loss = loss+source_loss
        bc_optimizer.zero_grad()
        loss.backward()
        bc_optimizer.step()
        bc_schedule.step()
        return dict(target_bc=float(target_loss.detach()), source_bc=None if source_loss is None else float(source_loss.detach()))

    def vg_step(step):
        v_loss, q_loss, _ = critic_step(*critic_sampler.sample())
        batch = target_sampler.sample()
        sbatch = source_sampler.sample()
        a_t = advantage(batch[0], batch[1])
        a_s = advantage(sbatch[0], sbatch[1])
        mse_t = policy.bc_loss(batch[0], batch[1]).mean(dim=1, keepdim=True)
        mse_s = policy.bc_loss(sbatch[0], sbatch[1]).mean(dim=1, keepdim=True)
        record = step % 10 == 0                         # guidance statistics every 10th step (fewer GPU syncs)
        stats = None
        if record:
            stats = dict(adv_target_mean=float(a_t.mean()), adv_source_mean=float(a_s.mean()),
                         source_adv_positive=float((a_s > 0).float().mean()), target_adv_positive=float((a_t > 0).float().mean()))
        if group == 'AWM':
            u = torch.exp(AWM_BETA*torch.cat([a_t, a_s], dim=0)).clamp(max=WEIGHT_CLIP)
            w = u/u.mean().clamp(min=1e-12)
            loss = 2.*(w*torch.cat([mse_t, mse_s], dim=0)).mean()
            if record:
                stats.update(source_weight_share=float(w[64:].sum()/w.sum()), weight_max=float(w.max()),
                             clipped_fraction=float((u >= WEIGHT_CLIP).float().mean()))
        else:
            keep = (a_s > 0).float()
            loss = mse_t.mean()+(keep*mse_s).sum()/64.
            if record:
                stats.update(source_kept_fraction=float(keep.mean()))
        bc_optimizer.zero_grad()
        loss.backward()
        bc_optimizer.step()
        bc_schedule.step()
        return dict(v=float(v_loss.detach()), q=float(q_loss.detach()), actor=float(loss.detach()),
                    target_bc=float(mse_t.mean().detach()), source_bc=float(mse_s.mean().detach())), stats

    def act(state):
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
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
    evaluations = 0
    losses = {}
    guidance = Running(steps) if group in VG_GROUPS else None
    try:
        for step in range(1, warmup+1):                 # critic-only warm-up (candidates only), no evaluation
            losses = warmup_step()
            if step % 1000 == 0 or step == warmup:
                assert all(math.isfinite(v) for v in losses.values()) and finite(), 'nonfinite at warm-up step %d' % step
                write(out/'progress.json', dict(step=step-warmup, total=steps, phase='critic_warmup', warmup=warmup,
                                                elapsed=time.monotonic()-started, losses=losses, time=time.time(), pid=os.getpid()))
        for step in range(1, steps+1):
            if group in IQL_GROUPS:
                losses = iql_step()
            elif group in VG_GROUPS:
                losses, stats = vg_step(step)
                if stats is not None:
                    guidance.add(step, stats)
            else:
                losses = bc_step()
            if step % 1000 == 0 or step == steps:
                assert all(v is None or math.isfinite(v) for v in losses.values()) and finite(), 'nonfinite at step %d' % step
                write(out/'progress.json', dict(step=step, total=steps, elapsed=time.monotonic()-started, losses=losses,
                                                guidance=None if guidance is None else guidance.result(),
                                                time=time.time(), pid=os.getpid()))
            if step % job['eval_every'] == 0:
                write(out/('eval_%07d.json' % step), evaluate(step))
                evaluations += 1
    finally:
        env.close()
    checkpoint = out/'final_checkpoint.pt'
    torch.save({name: net.state_dict() for name, net in nets.items()}, str(checkpoint))
    write(out/'COMPLETED.json', dict(job=job, steps=steps, critic_warmup=warmup, finite=finite(), evaluations=evaluations,
        target_stream_sha256=target_sampler.hash.hexdigest(),
        source_stream_sha256=None if source_sampler is None else source_sampler.hash.hexdigest(),
        critic_stream_sha256=None if critic_sampler is None else critic_sampler.hash.hexdigest(),
        target=target_info, source=source_info, guidance=None if guidance is None else guidance.result(),
        final_losses=losses, model_digest=digest(), checkpoint_sha256=sha(checkpoint),
        wall_seconds=time.monotonic()-started,
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
