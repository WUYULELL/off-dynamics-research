"""Shared helpers for the oracle diagnosis of IQL_BELLMAN_FB.

Everything here reuses the project's own loaders (ReplayBuffer.convert_D4RL /
ReplayBuffer.subsample / call_tar_dataset / call_mujoco_env / d4rl) so that the
data seen here is byte-identical to what train.py feeds the algorithm in
paper-style mode (ODRL_PAPER_STYLE_SOURCE_TARGET=1).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

# same prelude as train.py so that mujoco_py can find the dll
_mujoco = os.path.expanduser('~/.mujoco210/mujoco210')
if os.path.isdir(_mujoco):
    os.environ.setdefault('MUJOCO_PY_MUJOCO_PATH', _mujoco)
    os.environ['PATH'] = os.path.join(_mujoco, 'bin') + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')
for p in [str(ROOT), str(ROOT / 'algo')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import gym  # noqa: E402
import d4rl  # noqa: E402,F401
import torch  # noqa: E402

import algo.utils as utils  # noqa: E402
from dataset.call_dataset import call_tar_dataset  # noqa: E402
from envs.mujoco.call_mujoco_env import call_mujoco_env  # noqa: E402

ENVS = [
    'walker2d-kinematic-footjnt',
    'hopper-kinematic-legjnt',
    'halfcheetah-kinematic-footjnt',
]
SHIFT_LEVEL = 'medium'
SRCTYPE = 'medium'
TARTYPE = 'medium'
TARGET_FRACTION = 0.1
GAMMA = 0.99
# defaults of IQL_DYN / IQL_BELLMAN_FB (config/mujoco/iql_bellman_fb/*.yaml)
KNN_K = 5
TARGET_BANK_SIZE = 20000
KNN_CHUNK_SIZE = 5000
MIN_WEIGHT = 0.05
MAX_WEIGHT = 1.0
BELLMAN_ETA = 1.0
SOURCE_BUDGET = 0.20
QVEL_CLIP = 10.0


def d4rl_name(env_name):
    return env_name.split('-')[0] + '-' + SRCTYPE + '-v2'


def clips_qvel(env_name):
    # gym Hopper/Walker2d (v2 and v3) clip qvel to [-10, 10] in _get_obs; HalfCheetah does not
    return 'halfcheetah' not in env_name


def make_target_env(env_name):
    """Normal-dynamics env = paper-style target eval env (gym.make of the D4RL id)."""
    return gym.make(d4rl_name(env_name))


def make_shifted_env(env_name):
    """Shifted env = paper-style source env (call_mujoco_env with the modified XML)."""
    return call_mujoco_env({'env_name': env_name, 'shift_level': SHIFT_LEVEL})


def load_target_buffer(env_name, seed, fraction=TARGET_FRACTION, device='cpu'):
    """Exactly train.py L436-446: D4RL qlearning_dataset -> ReplayBuffer -> subsample(fraction, seed)."""
    env = make_target_env(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buf = utils.ReplayBuffer(obs_dim, act_dim, device)
    buf.convert_D4RL(d4rl.qlearning_dataset(env))
    full_size = buf.size
    buf.subsample(fraction, seed=seed)
    return buf, full_size


def target_subsample_indices(full_size, fraction, seed, obs_dim, act_dim):
    """Indices (into the qlearning_dataset arrays) chosen by ReplayBuffer.subsample.

    Computed by running the project's own subsample() on a buffer whose 'state'
    column holds the row index, so no re-implementation of the sampling logic.
    """
    buf = utils.ReplayBuffer(1, 1, 'cpu', max_size=1)
    idx = np.arange(full_size, dtype=np.float64).reshape(-1, 1)
    buf.convert_D4RL({
        'observations': idx, 'actions': idx, 'next_observations': idx,
        'rewards': idx.ravel(), 'terminals': np.zeros(full_size),
    })
    buf.subsample(fraction, seed=seed)
    return buf.state.ravel().astype(np.int64)


def qlearning_raw_indices(env_name):
    """Raw hdf5 row index for each row of d4rl.qlearning_dataset (terminate_on_end=False).

    Used only to fetch infos/qpos, infos/qvel for the exact-state replay; the
    caller asserts the mapping by comparing observations.
    """
    env = make_target_env(env_name)
    ds = env.get_dataset()
    n = ds['rewards'].shape[0]
    timeouts = ds['timeouts']
    keep = [i for i in range(n - 1) if not timeouts[i]]
    return np.asarray(keep, dtype=np.int64), ds


def load_source_buffer(env_name, device='cpu'):
    """Exactly train.py L450-452 (tar_* before the paper-style swap)."""
    env = make_target_env(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buf = utils.ReplayBuffer(obs_dim, act_dim, device)
    buf.convert_D4RL(call_tar_dataset(env_name, SHIFT_LEVEL, TARTYPE))
    return buf


def source_hdf5_path(env_name):
    return ROOT / 'dataset' / 'mujoco' / '{}_{}_{}.hdf5'.format(env_name.replace('-', '_'), SHIFT_LEVEL, TARTYPE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def obs_to_qpos_qvel(env, obs):
    """Invert gym's _get_obs: obs = [qpos[1:], qvel(clipped)], qpos[0] (x) := 0."""
    u = env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    obs = np.asarray(obs, dtype=np.float64)
    assert obs.shape[0] == nq - 1 + nv, (obs.shape, nq, nv)
    qpos = np.concatenate([[0.0], obs[:nq - 1]])
    qvel = obs[nq - 1:].copy()
    return qpos, qvel


def is_qvel_clipped(env, env_name, obs, tol=1e-4):
    """True if any velocity component sits on the +-10 clip boundary (true qvel unknown)."""
    if not clips_qvel(env_name):
        return np.zeros(len(obs), dtype=bool)
    nq = env.unwrapped.model.nq
    vel = np.asarray(obs)[:, nq - 1:]
    return (np.abs(vel) >= QVEL_CLIP - tol).any(axis=1)


def oracle_step(env, s, a, qpos=None, qvel=None):
    """Set env to state s (or exact qpos/qvel if given), execute a once.

    Uses env.unwrapped.step so the TimeLimit counter is not involved; this is
    the same step() that generated the datasets. Returns (r, s', done).
    """
    u = env.unwrapped
    if qpos is None:
        qpos, qvel = obs_to_qpos_qvel(env, s)
    u.set_state(qpos, qvel)
    next_obs, reward, done, _ = u.step(np.asarray(a, dtype=np.float64))
    return float(reward), np.asarray(next_obs, dtype=np.float64), bool(done)


def oracle_batch(env, S, A):
    n = len(S)
    R = np.zeros(n)
    S2 = np.zeros((n, S.shape[1]))
    D = np.zeros(n, dtype=bool)
    for i in range(n):
        R[i], S2[i], D[i] = oracle_step(env, S[i], A[i])
    return R, S2, D


def oracle_batch_exact(env, QP, QV, A):
    n = len(A)
    R = np.zeros(n)
    S2 = None
    D = np.zeros(n, dtype=bool)
    for i in range(n):
        r, s2, d = oracle_step(env, None, A[i], qpos=QP[i], qvel=QV[i])
        if S2 is None:
            S2 = np.zeros((n, s2.shape[0]))
        R[i], S2[i], D[i] = r, s2, d
    return R, S2, D


def recover_states(gen_env, obs, actions, prev_index, clipped, max_depth=200, tol=1e-4):
    """Recover the full internal (qpos, qvel) behind each observation.

    Unclipped rows are exact from the observation. For a row whose qvel was
    clipped, walk back along its own trajectory (prev_index[i] = row of the
    previous step, -1 at an episode start) to the nearest unclipped ancestor and
    re-simulate forward in the env that generated the data, checking at every
    step that the simulated (clipped) observation equals the recorded one.

    Returns qpos, qvel, status (0 exact-from-obs, 1 recovered, 2 unrecoverable:
    clipped with no unclipped ancestor / chain too deep / verification failed),
    max_verify_err.
    """
    u = gen_env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    n = len(obs)
    qpos = np.zeros((n, nq))
    qvel = np.zeros((n, nv))
    status = np.full(n, 2, dtype=np.int8)
    verr = np.zeros(n)
    for i in range(n):
        qpos[i], qvel[i] = obs_to_qpos_qvel(gen_env, obs[i])
        if not clipped[i]:
            status[i] = 0
    for i in np.where(clipped)[0]:
        chain = [i]
        j = prev_index[i]
        while j >= 0 and clipped[j] and len(chain) < max_depth:
            chain.append(j)
            j = prev_index[j]
        if j < 0 or clipped[j]:
            continue
        # j is an unclipped ancestor; if an intermediate row was already recovered, start there
        chain = chain[::-1]
        start_qp, start_qv = qpos[j].copy(), qvel[j].copy()
        k0 = 0
        for idx, c in enumerate(chain[:-1]):
            if status[c] == 1:
                start_qp, start_qv, k0 = qpos[c].copy(), qvel[c].copy(), idx + 1
                j = c
        u.set_state(start_qp, start_qv)
        prev = j
        ok = True
        worst = 0.0
        for c in chain[k0:]:
            o, _, _, _ = u.step(np.asarray(actions[prev], dtype=np.float64))
            e = float(np.abs(o - obs[c]).max())
            worst = max(worst, e)
            if e > tol:
                ok = False
                break
            qp = u.sim.data.qpos.copy()
            qp[0] = 0.0
            if status[c] != 1:
                qpos[c], qvel[c], status[c], verr[c] = qp, u.sim.data.qvel.copy(), 1, worst
            prev = c
        if not ok:
            status[i] = 2
    return qpos, qvel, status, verr


def prev_index_from_boundaries(n, episode_start):
    prev = np.arange(n) - 1
    prev[episode_start] = -1
    return prev


def load_policy_config(env_name, policy='iql_bellman_fb', overrides=None):
    """Same yaml as train.py, plus the fields train.py fills in before call_algo."""
    import yaml
    base = env_name.split('-')[0]
    with open(ROOT / 'config' / 'mujoco' / policy / '{}.yaml'.format(base), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    env = make_target_env(env_name)
    config.update({
        'env_name': env_name,
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'max_action': float(env.action_space.high[0]),
        'tar_env_interact_interval': 10,
        'shift_level': SHIFT_LEVEL,
    })
    if overrides:
        config.update(overrides)
    return config


def build_bellman_agent(env_name, seed, tar_buffer, device='cpu', overrides=None):
    """IQL_BELLMAN_FB instance whose target feature bank is built by its own
    initialize_target_feature_bank (bank indices, mean/std, knn_k all original)."""
    from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
    ov = {'seed': int(seed), 'source_budget': SOURCE_BUDGET, 'bellman_eta': BELLMAN_ETA, 'max_step': 500000}
    if overrides:
        ov.update(overrides)
    config = load_policy_config(env_name, 'iql_bellman_fb', ov)
    agent = IQL_BELLMAN_FB(config, torch.device(device))
    agent.initialize_target_feature_bank(tar_buffer)
    return agent


def target_obs_std(agent, obs_dim):
    """State block of the bank's std (target standardisation used by the kNN)."""
    return agent.target_std[0, :obs_dim].cpu().numpy().astype(np.float64)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'n': 0}
    return {
        'n': int(x.size),
        'mean': float(x.mean()),
        'median': float(np.median(x)),
        'p95': float(np.percentile(x, 95)),
        'max': float(x.max()),
    }


def dump_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
