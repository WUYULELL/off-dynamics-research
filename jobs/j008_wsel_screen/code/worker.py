"""j008 policy worker: one BC-mix training run with a fixed source weight w.

Groups (loss on one step = mean BC loss over 64 target rows + w * mean BC loss over 64 source rows):
  W0    w = 0     identical to the BC control of V21/j004-j007 (no source data loaded)
  W25   w = 0.25
  W50   w = 0.5
  W100  w = 1     identical to the BCMIX control of V21/j004-j007
All groups of one (task, seed) share the initialization and the target / source batch index streams
(seeds depend on task and seed only), so they differ only in w.

Source rows: D4RL medium, timeout rows dropped, recorded actions. Raw observations.
Evaluation: every 5000 steps, 10 episodes, Last5 (unchanged). The final policy is saved for the
model-based weight selection (model_worker.py, kind 'select').
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
WEIGHTS = {'W0': 0., 'W25': .25, 'W50': .5, 'W100': 1.}


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
    text = 'J008-20261120|%s|%d|%s' % (task, seed, purpose)
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
    return arrays, dict(rows=int(len(arrays[0])), dropped_timeout_rows=dropped, has_timeouts=timeouts is not None)


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
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected exactly one visible GPU'
    device = 'cuda:0'
    from offline_offline import iql_unc
    assert Path(iql_unc.__file__).resolve().parent == (Path(job['algo'])/'offline_offline').resolve()
    task, group, steps = job['task'], job['group'], job['steps']
    w = WEIGHTS[group]
    sd, ad = job['state_dim'], job['action_dim']
    lo, hi = job['score_min'], job['score_max']

    assert sha(job['target_dataset']) == job['target_dataset_sha256'] and sha(job['xml']) == job['xml_sha256']
    target, target_info = load(job['target_dataset'], sd, ad, np, h5py, drop_timeouts=False)
    assert target_info['rows'] == job['target_rows']
    source, source_info = None, None
    if w > 0:
        assert sha(job['source_dataset']) == job['source_dataset_sha256']
        source, source_info = load(job['source_dataset'], sd, ad, np, h5py, drop_timeouts=True)

    s = seed_of('initialization', task, job['seed'])
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    config = dict(lr=.0003, hidden_sizes=256, cosine_T_max=steps, source_weight=w)
    policy = iql_unc.Policy(sd, ad, 1., 256).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    def digest():
        h = hashlib.sha256()
        for value in policy.state_dict().values():
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def finite():
        return all(bool(torch.isfinite(p).all()) for p in policy.parameters())

    write(out/'config.json', dict(job=job, learner_config=config, batch='64 target rows (+64 source rows when w > 0)',
        observation_scaling='raw', threads=1, deterministic_algorithms=True, TF32=True, target=target_info, source=source_info,
        iql_unc_sha256=sha(iql_unc.__file__), initialization_sha256=digest(), torch=torch.__version__))

    class Sampler:
        def __init__(self, arrays, purpose, size=64):
            self.arrays, self.n, self.size = arrays, len(arrays[0]), size
            self.rng = np.random.RandomState(seed_of(purpose, task, job['seed']))
            self.hash = hashlib.sha256()

        def sample(self):
            ix = self.rng.randint(self.n, size=self.size)
            self.hash.update(ix.astype('<i8').tobytes())
            return tuple(torch.as_tensor(np.asarray(a[ix]), dtype=torch.float32, device=device) for a in self.arrays)

    target_sampler = Sampler([target[0], target[1]], 'target_sample')
    source_sampler = Sampler([source[0], source[1]], 'source_sample') if source is not None else None

    def bc_step():
        batch = target_sampler.sample()
        target_loss = policy.bc_loss(batch[0], batch[1]).mean(dim=1, keepdim=True).mean()
        loss = target_loss
        source_loss = None
        if source_sampler is not None:
            sbatch = source_sampler.sample()
            source_loss = policy.bc_loss(sbatch[0], sbatch[1]).mean(dim=1, keepdim=True).mean()
            loss = loss+w*source_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        return dict(target_bc=float(target_loss.detach()), source_bc=None if source_loss is None else float(source_loss.detach()))

    def act(state):
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
            return torch.tanh(policy.network(x).chunk(2, dim=1)[0]).squeeze(0).cpu().numpy()

    import gym
    constructor = getattr(__import__(job['env_module'], fromlist=[job['env_class']]), job['env_class'])
    env = gym.wrappers.TimeLimit(constructor(xml_file=job['xml']), max_episode_steps=1000)
    assert env.unwrapped.frame_skip == 4 and abs(float(env.unwrapped.model.opt.timestep)-.002) < 1e-12
    assert env.observation_space.shape == (sd,) and env.action_space.shape == (ad,)

    def evaluate(step):
        saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
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
            torch.cuda.set_rng_state_all(saved[3])
        scores = (np.asarray(returns)-lo)/(hi-lo)*100.
        assert np.isfinite(scores).all()
        return dict(step=step, raw_returns=returns, scores=scores.tolist(), mean=float(scores.mean()),
                    episode_lengths=lengths, env_seeds=seeds, steps=int(sum(lengths)))

    started = time.monotonic()
    evaluations = 0
    losses = {}
    try:
        for step in range(1, steps+1):
            losses = bc_step()
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
    torch.save({'policy': policy.state_dict()}, str(checkpoint))
    write(out/'COMPLETED.json', dict(job=job, steps=steps, finite=finite(), evaluations=evaluations, source_weight=w,
        target_stream_sha256=target_sampler.hash.hexdigest(),
        source_stream_sha256=None if source_sampler is None else source_sampler.hash.hexdigest(),
        target=target_info, source=source_info, final_losses=losses, model_digest=digest(),
        checkpoint_sha256=sha(checkpoint), wall_seconds=time.monotonic()-started,
        GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), ended=time.time()))


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
