"""Full CPU action correction and independent replay, using the frozen stage-0 solver."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from common import sha, utc, write
import resources
from science import Project
from stage1_core import CONFIG, PROJECT, STAGE0, SUCCESS, TASK, old_solver_config, verify_worker


def stats(a):
    assert np.isfinite(a).all()
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(np.mean(a, dtype=np.float64)), min=float(np.min(a)),
                median=float(np.median(a)), p90=float(np.percentile(a,90)),
                p99=float(np.percentile(a,99)), max=float(np.max(a)))


def describe(arrays, filter_limit, boundary_tolerance):
    src_bound = (1-np.abs(arrays['original_action']) <= boundary_tolerance).any(axis=1)
    fix_bound = (1-np.abs(arrays['action_fixed']) <= boundary_tolerance).any(axis=1)
    assert np.array_equal(fix_bound, arrays['boundary'])
    converged = np.isin(arrays['solver_status'], sorted(SUCCESS))
    kept = arrays['e1'] <= filter_limit
    binned = {}
    for bound in (False,True):
        for conv in (False,True):
            key = ('boundary' if bound else 'interior') + ('_converged' if conv else '_not_converged')
            binned[key] = stats(arrays['e1'][(fix_bound == bound) & (converged == conv)])
    names, counts = np.unique(arrays['solver_status'], return_counts=True)
    new_bound = int((~src_bound & fix_bound).sum())
    interior_count = int((~src_bound).sum())
    source_terminal = arrays['original_terminal']
    fixed_terminal = arrays['terminal_actual']
    return dict(rows=len(kept), selected_rows=int(kept.sum()), selected_fraction=float(kept.mean()),
                source_boundary_fraction=float(src_bound.mean()), corrected_boundary_fraction=float(fix_bound.mean()),
                new_boundary_count=new_bound, new_boundary_fraction_all=new_bound/len(kept),
                new_boundary_fraction_among_source_interior=new_bound/interior_count if interior_count else None,
                convergence_fraction=float(converged.mean()), e1_by_boundary_and_convergence=binned,
                solver_status_counts={str(k):int(v) for k,v in zip(names,counts)},
                source_terminal_count=int(source_terminal.sum()), corrected_terminal_count=int(fixed_terminal.sum()),
                source_terminal_fraction=float(source_terminal.mean()), corrected_terminal_fraction=float(fixed_terminal.mean()),
                terminal_changed_count=int((source_terminal!=fixed_terminal).sum()),
                action_delta=stats(arrays['action_delta']), e0=stats(arrays['e0']), e1=stats(arrays['e1']),
                solver_iterations=stats(arrays['iterations']), simulator_steps=stats(arrays['simulator_steps']),
                row_seconds=stats(arrays['row_seconds'])), np.flatnonzero(kept).astype('<i8')


def independently_replay(legacy, out, xml, sample, data, arrays):
    from gym.envs.mujoco.hopper_v3 import HopperEnv

    spec = CONFIG['cache']
    restored = legacy.load_restoration(STAGE0/'references', legacy.read(STAGE0/'RESTORATION_AST.json'))
    env = HopperEnv(xml_file=str(xml))
    env.seed(spec['audit_seed'])
    tracker = legacy.CountedEnvironment(env)
    tracker.phase = 'independent_replay'
    assert env.model.nq == env.model.nv == 6 and env.frame_skip == 4
    assert abs(float(env.model.opt.timestep)-.002) < 1e-12
    next_obs = np.empty((len(sample),11))
    reward = np.empty(len(sample))
    terminal = np.empty(len(sample),dtype=bool)
    try:
        for j,i in enumerate(sample):
            next_obs[j],reward[j],terminal[j],_,_ = restored['step'](
                env,data['qpos'][i],data['qvel'][i],arrays['action_fixed'][i])
    finally:
        env.close()
    state_error = np.max(np.abs(next_obs-arrays['next_actual'][sample]),axis=1)
    reward_error = np.abs(reward-arrays['reward_actual'][sample])
    terminal_same = terminal == arrays['terminal_actual'][sample]
    np.savez_compressed(out/'CACHE_REPLAY_RAW.npz', canonical_position=sample, raw_row=data['row_id'][sample],
        next_observation=next_obs,reward=reward,terminal=terminal,state_error=state_error,
        reward_error=reward_error,terminal_same=terminal_same)
    passed = bool(np.all(state_error <= spec['audit_state_linf']) and np.all(reward_error <= spec['audit_reward_abs'])
                  and np.all(terminal_same) and np.isfinite(state_error).all() and np.isfinite(reward_error).all())
    evidence = dict(passed=passed, replay_count=len(sample), state_linf=stats(state_error), reward_abs=stats(reward_error),
                    terminal_disagreement=int((~terminal_same).sum()), total_steps=1+tracker.steps,
                    state_tolerance=spec['audit_state_linf'], reward_tolerance=spec['audit_reward_abs'],
                    sample_seed=spec['audit_seed'], sample_sha256=sha(out/'audit_positions.npy'),
                    replay_sha256=sha(out/'CACHE_REPLAY_RAW.npz'))
    write(out/'CACHE_REPLAY_CHECK.json', evidence)
    assert passed, 'Independent cache replay failed; see CACHE_REPLAY_RAW.npz'
    return evidence


def merge(stage0, out, folders, order, data, scale):
    n = len(data['row_id'])
    arrays = {}
    offset = 0
    steps = 0
    for name in order:
        folder = folders[name]
        receipt = stage0.legacy.read(folder/'COMPLETED.json')
        assert sha(folder/'rows.npz') == receipt['rows_sha256']
        part = stage0.load_rows(folder)
        end = offset+len(part['row_id'])
        assert np.array_equal(part['row_id'],data['row_id'][offset:end])
        assert np.array_equal(part['original_action'],data['actions'][offset:end])
        assert np.array_equal(part['original_terminal'],data['terminals'][offset:end].reshape(-1))
        assert (part['e1'] <= part['e0']).all() and (part['e1'] >= 0).all()
        assert (part['iterations'] <= 10).all() and np.all(np.abs(part['action_fixed']) <= 1)
        residual = (part['next_actual'] - data['next_observations'][offset:end]) / scale
        assert np.allclose(np.linalg.norm(residual,axis=1),part['e1'],atol=1e-12,rtol=1e-12)
        for key,value in part.items():
            assert len(value) == end-offset
            if value.dtype.kind in 'if':
                assert np.isfinite(value).all(), key
            if key not in arrays:
                # String width is fixed across chunks even if the pilot lacks a solver status.
                dtype = '<U40' if key == 'solver_status' else value.dtype
                arrays[key] = np.lib.format.open_memmap(out/(key+'.npy'), mode='w+', dtype=dtype,
                                                        shape=(n,)+value.shape[1:])
            arrays[key][offset:end] = value
        steps += sum(receipt['environment_step_counts'].values())
        offset = end
    assert offset == n
    for a in arrays.values():
        a.flush()
    np.save(out/'source_observations.npy', data['observations'].astype('<f4',copy=False))
    return arrays,steps


def run(job):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert os.getppid() == job['controller_pid']
    assert job['job'] == 'full_cache'
    out = Path(job['output'])
    started, host_before = time.monotonic(), resources.cpu_ticks()
    old_cfg = old_solver_config()
    sys.path.insert(0,str(STAGE0))
    import run_stage0 as stage0
    assert Path(stage0.__file__).resolve() == STAGE0/'run_stage0.py'
    legacy = stage0.legacy
    limits = resources.limits_snapshot()
    assert limits['cgroup_mapping_found'] and not limits['unreadable_fields']
    project = Project(PROJECT)
    task = project.spec(TASK)
    xml = Path(task['xml'])
    source = project.path(task['source_dataset'])
    target = project.path(task['target_dataset'])
    oracle = {k:project.path('data/oracle/hopper/'+k+'.npy') for k in ('raw_indices','next_state','reward','terminal')}
    ids = np.load(oracle['raw_indices'],allow_pickle=False)
    n = CONFIG['canonical_source_rows']
    assert len(ids) == n and np.all(ids[1:]>ids[:-1])
    audit = CONFIG['cache']
    sample = np.random.RandomState(audit['audit_seed']).choice(n,audit['audit_rows'],replace=False)
    np.save(out/'audit_positions.npy',sample)
    write(out/'PRE_CACHE_COMMITMENT.json',dict(job=job, audit_seed=audit['audit_seed'], sample_sha256=sha(out/'audit_positions.npy'),
        correction_solver=old_cfg['solver'], filter=CONFIG['residual_filter'], time=utc()))
    target_data = stage0.target_input(target, ('observations',))
    scale = target_data['observations'].astype(np.float64).std(axis=0,ddof=1)+old_cfg['std_epsilon']
    assert np.isfinite(scale).all() and np.all(scale>0)
    np.save(out/'target_scale.npy',scale)
    data = stage0.source_input(source,ids,('observations','actions','next_observations','rewards','terminals'))
    rec = Path(__file__).resolve().parent/'restoration'
    qpos = np.load(rec/'qpos.npy',mmap_mode='r',allow_pickle=False)
    qvel = np.load(rec/'qvel.npy',mmap_mode='r',allow_pickle=False)
    status = np.load(rec/'recovery_status.npy',mmap_mode='r',allow_pickle=False)
    assert qpos.shape == qvel.shape == (1000000,6) and status.shape == (1000000,)
    assert np.isin(status[ids],[0,1,3]).all()
    data.update(qpos=qpos[ids],qvel=qvel[ids],row_id=ids,
                oracle_next=np.load(oracle['next_state'],mmap_mode='r',allow_pickle=False),
                oracle_reward=np.load(oracle['reward'],mmap_mode='r',allow_pickle=False).reshape(-1),
                oracle_terminal=np.load(oracle['terminal'],mmap_mode='r',allow_pickle=False).reshape(-1))
    assert all(len(a)==n and np.isfinite(a).all() for a in data.values())
    assert (data['qpos'][:,0] == 0).all()
    binding, before, private = legacy.prepare_private_binding(out)
    import mujoco_py
    assert private in Path(mujoco_py.cymj.__file__).resolve().parents
    (out/'runs').mkdir()
    common = (xml,out/'private_runtime',old_cfg,scale)

    def subset(i,j):
        return {k:v[i:j] for k,v in data.items()}

    pilot = audit['pilot_rows']
    print('CACHE: single-process RSS measurement, first',pilot,'rows count toward the full cache',flush=True)
    pilot_folder = stage0.run_jobs(out,common,[('source_pilot','source',subset(0,pilot))],1)['source_pilot']
    rss = legacy.read(pilot_folder/'COMPLETED.json')['peak_RSS_bytes']
    current = resources.limits_snapshot()
    parent = resources.owned_process(os.getpid())['RSS_bytes']
    max_mem = current['effective_available_memory_bytes']*CONFIG['resource_policy']['memory_max_fraction']
    workers = min(audit['cpu_workers_max'],int(current['effective_visible_cpu_cores']),int((max_mem-parent)/rss))
    assert 1 <= workers <= 32
    # The multi-row chunks need margin over the 256-row pilot.
    estimate = int(rss*1.25)
    workers = min(workers,int((max_mem-parent)/estimate))
    assert workers >= 1
    budget = dict(pilot_RSS_bytes=rss,parent_RSS_bytes=parent,workers=workers,predicted_RSS_bytes=parent+workers*estimate,
        worker_RSS_estimate_bytes=estimate,effective_available_memory_bytes=current['effective_available_memory_bytes'],
        nproc=current['nproc'],effective_cpu=current['effective_visible_cpu_cores'])
    write(out/'CACHE_RSS_BEFORE_PARALLEL.json',budget)
    print('CACHE RSS / CPU BUDGET',budget,flush=True)
    chunk = audit['chunk_rows']
    jobs = [(f'source_{i:06d}','source',subset(i,i+chunk)) for i in range(pilot,n,chunk)]
    folders = stage0.run_jobs(out,common,jobs,workers,estimate)
    folders['source_pilot'] = pilot_folder
    arrays, solve_steps = merge(stage0,out,folders,['source_pilot']+[row[0] for row in jobs],data,scale)
    summary, selected = describe(arrays,CONFIG['residual_filter']['threshold'],audit['boundary_abs_tolerance'])
    assert 0 < len(selected) < n
    np.save(out/'selected_positions.npy',selected)
    write(out/'CACHE_STATISTICS.json',summary)
    print('CACHE: selected',len(selected),'/',n,'rows; starting independent replay',flush=True)
    evidence = independently_replay(legacy,out,xml,sample,data,arrays)
    assert legacy.hash_tree(binding) == before and 'torch' not in sys.modules
    files = {path.name:sha(path) for path in sorted(out.glob('*.npy'))}
    receipt = dict(rows=n,filter=CONFIG['residual_filter'],selected_rows=len(selected),
        cache_sha256=files,statistics_sha256=sha(out/'CACHE_STATISTICS.json'),solver_config_sha256=sha(STAGE0/'config.json'),
        stage2_authorized=False,training_updates=0,policy_evaluations=0,GPU_jobs=0,
        full_correction_successful_environment_steps=solve_steps,wall_seconds=time.monotonic()-started,
        host_CPU_busy_percent=resources.busy_percent(host_before,resources.cpu_ticks()),completed_utc=utc(),**evidence)
    write(out/'CACHE_ACCEPTANCE.json',receipt)
    print('CACHE_ACCEPTANCE passed',out/'CACHE_ACCEPTANCE.json',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',type=Path)
    spec=verify_worker(p.parse_args().job,1)
    try:
        run(spec)
    except Exception as exc:
        write(Path(spec['output'])/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
