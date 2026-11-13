"""V16 worker side: per source trajectory, open-loop replay (B1), one-step closed-loop
tracking (B2) and an instrument self-check, in one shifted target simulator. CPU only.

Physics convention (cold sim.reset, set_state, zero qacc_warmstart, one env step),
and the bounded solver are the frozen v9/v13 definitions, loaded read-only from
$HOME/v13_action_fix/stage0. The B1/B2 loop is the V15 loop (which reproduced V14b
on 1000/1000 records), limited to the first H steps.
"""
import atexit
import os
from pathlib import Path
import sys

SHIM = os.environ.get('V16_TEST_SHIM')          # synthetic local test only; never set on the server
if SHIM and SHIM not in sys.path:
    sys.path.insert(0, SHIM)

import numpy as np

from core import module, read    # $HOME/v14_diag/core.py, read-only

V13_CODE = Path.home()/'v13_action_fix/stage0'
ENV_SEED = 2026111404
RUNTIME = {}
ROBOTS = {'hopper': ('gym.envs.mujoco.hopper_v3', 'HopperEnv', 6),
          'walker2d': ('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv', 9)}


def initialize(task, robot, xml, vendor, scale, solver_cfg, boundary_tolerance):
    if task in RUNTIME:
        return RUNTIME[task]
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import mujoco_py
    assert Path(vendor).resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
    tag = ''.join(c if c.isalnum() else '_' for c in task)
    legacy = module(V13_CODE/'legacy_replay.py', 'v16_restoration_'+tag)
    ns = legacy.load_restoration(V13_CODE/'references', read(V13_CODE/'RESTORATION_AST.json'))
    module_name, class_name, nq = ROBOTS[robot]
    env = getattr(__import__(module_name, fromlist=[class_name]), class_name)(xml_file=str(xml))
    atexit.register(env.close)
    env.seed(ENV_SEED)
    assert env.frame_skip == 4 and abs(float(env.model.opt.timestep)-.002) < 1e-12 and env.model.nq == nq
    r = dict(env=env, scale=np.asarray(scale, dtype=np.float64), solver_cfg=solver_cfg,
             solve=module(V13_CODE/'solver.py', 'v16_solver_'+tag).solve, step=ns['step'],
             boundary_tolerance=boundary_tolerance)
    RUNTIME[task] = r
    return r


def finite(*values):
    if not all(np.isfinite(np.asarray(x, dtype=np.float64)).all() for x in values):
        raise FloatingPointError('Nonfinite simulator output')


def follow(r, qpos, qvel, actions, wanted_next, mode, H):
    """mode 'open': execute the given actions. mode 'track': solve for the action closest to wanted_next."""
    p, v = np.asarray(qpos, dtype=np.float64).copy(), np.asarray(qvel, dtype=np.float64).copy()
    e1s, acts, bound, rewards, terminal = [], [], [], 0.0, False
    for given, wanted in zip(actions[:H], wanted_next[:H]):
        def probe(action):
            return r['step'](r['env'], p, v, action)[:3]
        if mode == 'open':
            action = np.asarray(given, dtype=np.float64)
        else:
            start = np.clip(np.asarray(given, dtype=np.float64), -1, 1)
            original = probe(start)
            action, _, _, e1, _, _, _ = r['solve'](probe, start, wanted, r['scale'], r['solver_cfg'], original)
        ns, reward, terminal, p_next, v_next = r['step'](r['env'], p, v, action)
        finite(ns, reward, p_next, v_next)
        actual = float(np.linalg.norm((ns-wanted)/r['scale']))
        if mode == 'track':
            assert np.isclose(actual, e1, atol=1e-12, rtol=1e-12), 'Chosen-action replay differs from solver output'
        e1s.append(actual)
        acts.append(np.asarray(action, dtype=np.float64))
        bound.append(bool(np.any(1-np.abs(action) <= r['boundary_tolerance'])))
        rewards += float(reward)
        p, v = p_next, v_next
        if terminal:
            break
    return dict(e1=np.asarray(e1s), actions=np.asarray(acts), boundary=np.asarray(bound), reward=rewards, terminal=bool(terminal))


def continuous_reference(r, qpos, qvel, actions):
    """A real target-domain trajectory produced the way datasets are: one state set, then plain env.step
    calls without resets, observations stored as float32. Used only to test the tracking instrument."""
    u = r['env'].unwrapped
    u.sim.reset()
    u.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    nexts, terminals = [], []
    for action in actions:
        obs, reward, done, _ = u.step(np.asarray(action, dtype=np.float32))
        finite(obs, reward)
        nexts.append(np.asarray(obs, dtype=np.float32))
        terminals.append(bool(done))
        if done:
            break
    return np.asarray(nexts), np.asarray(terminals)


def run(job):
    r = initialize(*job['init'])
    H, L = job['H'], job['length']
    out = dict(trajectory_id=job['trajectory_id'], L=L, L_H=min(L, H), worker_pid=os.getpid())
    for name, mode in (('B1', 'open'), ('B2', 'track')):
        f = follow(r, job['qpos'], job['qvel'], job['actions'], job['next_observations'], mode, H)
        n = len(f['e1'])
        out[name] = dict(T_H=n, fell_within_H=f['terminal'], reward_H=f['reward'],
            boundary_fraction=float(f['boundary'].mean()), e1_median=float(np.median(f['e1'])),
            e1_p90=float(np.percentile(f['e1'], 90)), e1_first50_median=float(np.median(f['e1'][:50])),
            e1=f['e1'].astype(np.float32))
        if name == 'B2':
            b2_actions = f['actions']
    if job['selfcheck']:
        # Instrument test: a genuine target trajectory (continuous simulation of the B2 actions) must be
        # re-trackable with the cold-reset one-step procedure, as the real target data were in V14b/V15.
        nexts, terminals = continuous_reference(r, job['qpos'], job['qvel'], b2_actions.astype(np.float32))
        f = follow(r, job['qpos'], job['qvel'], b2_actions.astype(np.float32)[:len(nexts)], nexts, 'track', H)
        n = len(f['e1'])
        over = np.flatnonzero(f['e1'] > job['gate'])
        reached = n == len(nexts) and f['terminal'] == bool(terminals[n-1])
        out['selfcheck'] = dict(reference_length=int(len(nexts)), tracked=n, reached_end=bool(reached),
            e1_max=float(f['e1'].max()), first_exceed_step=None if not len(over) else int(over[0])+1,
            passed=bool(reached and not len(over)))
    return out


def task_entry(job):
    try:
        return run(job)
    except BaseException:
        import traceback
        from core import utc, write
        write(Path(job['output'])/('trajectory%05d.FAILED.json' % job['trajectory_id']),
              dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False))
        raise
