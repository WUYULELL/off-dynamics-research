"""V15 worker side: generic one-step tracking in a target simulator. CPU only.

The physics convention (cold sim.reset, set_state, zero qacc_warmstart, one env
step), the observation inversion and the bounded solver are the frozen v9/v13
definitions, loaded read-only from $HOME/v13_action_fix/stage0. The loop below
follows v14_diag/diagnostic_b.track step for step; it only removes the
Hopper-only assumptions so that the same code runs Walker2d.
"""
import atexit
import os
from pathlib import Path
import resource
import sys

SHIM = os.environ.get('V15_TEST_SHIM')          # synthetic local test only; never set on the server
if SHIM and SHIM not in sys.path:
    sys.path.insert(0, SHIM)

import numpy as np

from core import module, read, sha, stats, utc, write    # $HOME/v14_diag/core.py, read-only

V13_CODE = Path.home()/'v13_action_fix/stage0'
ENV_SEED = 2026111404                                    # same constant as v14 B
RUNTIME = {}
ENVIRONMENTS = {'hopper-morph': ('gym.envs.mujoco.hopper_v3', 'HopperEnv', 6),
                'walker2d-morph': ('gym.envs.mujoco.walker2d_v3', 'Walker2dEnv', 9)}


def initialize(task, xml, vendor, scale, solver_cfg, boundary_tolerance):
    if task in RUNTIME:
        return RUNTIME[task]
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import mujoco_py
    assert Path(vendor).resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
    legacy = module(V13_CODE/'legacy_replay.py', 'v15_restoration_'+task.replace('-', '_'))
    ns = legacy.load_restoration(V13_CODE/'references', read(V13_CODE/'RESTORATION_AST.json'))
    module_name, class_name, nq = ENVIRONMENTS[task]
    env = getattr(__import__(module_name, fromlist=[class_name]), class_name)(xml_file=str(xml))
    atexit.register(env.close)
    env.seed(ENV_SEED)
    assert env.frame_skip == 4 and abs(float(env.model.opt.timestep)-.002) < 1e-12 and env.model.nq == nq
    r = dict(env=env, scale=np.asarray(scale, dtype=np.float64), solver_cfg=solver_cfg,
             solve=module(V13_CODE/'solver.py', 'v15_solver_'+task.replace('-', '_')).solve,
             ns=ns, steps=1, boundary_tolerance=boundary_tolerance)
    original = ns['step']

    def counted(*args):
        r['steps'] += 1
        value = original(*args)
        if not all(np.isfinite(x).all() for x in (value[0], value[1], value[3], value[4])):
            raise FloatingPointError('Nonfinite simulator output')
        return value
    ns['step'] = counted
    RUNTIME[task] = r
    return r


def track(job):
    r = initialize(*job['init'])
    first = r['steps']
    record, mode, folder = job['record'], job['mode'], Path(job['output'])
    p, v = np.asarray(job['qpos'], dtype=np.float64).copy(), np.asarray(job['qvel'], dtype=np.float64).copy()
    rows = {k: [] for k in ('action', 'e0', 'e1', 'raw_linf', 'action_delta', 'boundary', 'reward', 'terminal', 'iterations')}
    for source_action, wanted in zip(job['actions'], job['next_observations']):
        # Every candidate starts from the same current actual p/v. Never from a later reference row.
        def probe(action):
            return r['ns']['step'](r['env'], p, v, action)[:3]
        if mode == 'B1':
            action = np.asarray(source_action, dtype=np.float64)
            original = probe(action)
            e0 = e1 = float(np.linalg.norm((original[0]-wanted)/r['scale']))
            nit = 0
        else:
            # The frozen solver requires a start inside the box; identity for in-range data.
            start = np.clip(np.asarray(source_action, dtype=np.float64), -1, 1)
            original = probe(start)
            action, _, e0, e1, nit, _, _ = r['solve'](probe, start, wanted, r['scale'], r['solver_cfg'], original)
        # Explicitly execute the accepted action, because the solver's last probe can be rejected.
        ns, reward, terminal, p_next, v_next = r['ns']['step'](r['env'], p, v, action)
        actual_e1 = float(np.linalg.norm((ns-wanted)/r['scale']))
        assert np.isclose(actual_e1, e1, atol=1e-12, rtol=1e-12), 'Chosen-action replay differs from solver output'
        values = (action, e0, actual_e1, float(np.max(np.abs(ns-wanted))),
                  float(np.linalg.norm(action-np.asarray(source_action, dtype=np.float64))),
                  bool(np.any(1-np.abs(action) <= r['boundary_tolerance'])), reward, terminal, nit)
        for key, value in zip(rows, values):
            rows[key].append(value)
        p, v = p_next, v_next
        if terminal:
            break
    arrays = {k: np.asarray(x) for k, x in rows.items()}
    assert all(np.isfinite(x).all() for x in arrays.values())
    n = len(arrays['e1'])
    name = '%s_trajectory%05d' % (mode, record['trajectory_id'])
    reference_terminal = np.asarray(job['terminals'][:n]).astype(bool)
    np.savez_compressed(folder/(name+'.npz'), **arrays, step=np.arange(n), reference_terminal=reference_terminal)
    total = float(arrays['reward'].sum())
    return dict(mode=mode, trajectory_id=record['trajectory_id'], first_row=record['first_row'],
        last_row=record['last_row'], L=record['length'], T=n, T_over_L=n/record['length'],
        physical_terminal=bool(arrays['terminal'][-1]), reached_record_end=n == record['length'],
        terminal_mismatches=int(np.count_nonzero(arrays['terminal'].astype(bool) != reference_terminal)),
        boundary_fraction=float(arrays['boundary'].mean()), raw_return=total,
        normalized_score=(total-job['score_min'])/(job['score_max']-job['score_min'])*100,
        e1=stats(arrays['e1']), raw_linf=stats(arrays['raw_linf']), action_delta=stats(arrays['action_delta']),
        simulator_steps=r['steps']-first, raw_sha256=sha(folder/(name+'.npz')),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, worker_pid=os.getpid())


def task_entry(job):
    try:
        return track(job)
    except BaseException:
        import traceback
        name = '%s_trajectory%05d.FAILED.json' % (job['mode'], job['record']['trajectory_id'])
        write(Path(job['output'])/name, dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False))
        raise
