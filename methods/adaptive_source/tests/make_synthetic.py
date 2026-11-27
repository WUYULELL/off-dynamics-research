"""Synthetic D4RL-like source files and ODRL-like target files for the local tests (written to a scratch folder).

usage: python make_synthetic.py <out_dir> <source_rows> <target_rows>
Writes hopper / walker2d source files (rewards (N,), terminals (N,), timeouts (N,)) and one target file per task id
(rewards (N, 1), terminals (N, 1), no timeouts, 400-step trajectories), plus files.json with their SHA-256."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_env import FakeEnv  # noqa: E402

TASKS = {'hopper-kinematic-footjnt-medium': 'hopper', 'walker2d-gravity-2.0': 'walker2d', 'walker2d-morph-leg-hard': 'walker2d',
         'hopper-kinematic-legjnt-medium': 'hopper', 'walker2d-kinematic-thighjnt-hard': 'walker2d',
         'walker2d-gravity-0.5': 'walker2d', 'hopper-morph': 'hopper'}
DIMS = {'hopper': (11, 3), 'walker2d': (17, 6)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rollout(sd, ad, n, ep_len, seed, noise):
    env = FakeEnv(sd, ad)
    env.seed(seed)
    rng = np.random.RandomState(seed+1)
    obs, act, nxt, rew, term, tout = [], [], [], [], [], []
    s = env.reset()
    t = 0
    while len(obs) < n:
        a = np.clip(np.tanh(.3*s[:ad])+noise*rng.randn(ad), -1, 1)
        s2, r, done, info = env.step(a)
        t += 1
        timeout = t >= ep_len and not done
        obs.append(s)
        act.append(a)
        nxt.append(s2)
        rew.append(r)
        term.append(bool(done and not info.get('TimeLimit.truncated')))
        tout.append(bool(timeout))
        s = s2
        if done or timeout:
            s = env.reset()
            t = 0
    f = lambda x, d: np.asarray(x, dtype=d)
    return f(obs, np.float32), f(act, np.float32), f(nxt, np.float32), f(rew, np.float32), f(term, bool), f(tout, bool)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    n_src, n_tar = int(sys.argv[2]), int(sys.argv[3])
    files = dict(source_files={}, target_files={})
    for robot, (sd, ad) in DIMS.items():
        o, a, n, r, te, to = rollout(sd, ad, n_src, 500, 1 if robot == 'hopper' else 2, .5)
        p = out/('%s_medium-v2.hdf5' % robot)
        with h5py.File(str(p), 'w') as h:
            h['observations'], h['actions'], h['next_observations'] = o, a, n
            h['rewards'], h['terminals'], h['timeouts'] = r, te, to
            h['infos/qpos'] = np.zeros((len(o), 2), np.float32)
        files['source_files'][robot] = dict(path=str(p), sha256=sha(p))
    for i, (tid, robot) in enumerate(TASKS.items()):
        sd, ad = DIMS[robot]
        o, a, n, r, te, to = rollout(sd, ad, n_tar, 400, 10+i, .2)
        p = out/('%s_target.hdf5' % tid)
        with h5py.File(str(p), 'w') as h:
            h['observations'], h['actions'], h['next_observations'] = o, a, n
            h['rewards'] = r.reshape(-1, 1)
            h['terminals'] = te.reshape(-1, 1).astype(np.float64)
        files['target_files'][tid] = dict(path=str(p), sha256=sha(p))
    (out/'files.json').write_text(json.dumps(files, indent=2))
    print('written', len(files['target_files']), 'targets')


if __name__ == '__main__':
    main()
