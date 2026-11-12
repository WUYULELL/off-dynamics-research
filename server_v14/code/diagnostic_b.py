"""B0 target self-check, B1 open-loop, B2 actual-state closed-loop; CPU only."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from pathlib import Path
import resource
import sys

import numpy as np

from core import A_LABELS,bins,csv_write,histogram_svg,interpret_b,module,read,sha,stats,utc,write
from context import CODE,V13

RUNTIME = None


def initialize(xml, vendor, scale, cfg):
    global RUNTIME
    sys.path.insert(0,str(vendor))
    import mujoco_py
    assert Path(vendor).resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
    legacy = module(CODE/'legacy_replay.py','v14_B_restoration')
    ns = legacy.load_restoration(CODE/'references',read(CODE/'RESTORATION_AST.json'))
    from gym.envs.mujoco.hopper_v3 import HopperEnv
    env = HopperEnv(xml_file=str(xml))
    import atexit
    atexit.register(env.close)
    env.seed(2026111404)
    assert env.frame_skip==4 and abs(float(env.model.opt.timestep)-.002)<1e-12
    RUNTIME = dict(env=env,scale=np.asarray(scale),config=cfg,solver=module(CODE/'solver.py','v14_B_solver').solve,
                   restoration=ns,steps=1)
    original = ns['step']
    def counted(*args):
        RUNTIME['steps'] += 1
        value = original(*args)
        if not all(np.isfinite(x).all() for x in (value[0],value[1],value[3],value[4])):
            raise FloatingPointError('Nonfinite B simulator output')
        return value
    ns['step'] = counted


def track(job):
    r = RUNTIME
    first = r['steps']
    record,mode,folder = job['record'],job['mode'],Path(job['output'])
    p,v = job['qpos'].copy(),job['qvel'].copy()
    rows = {k:[] for k in ('observation','action','wanted_next','actual_next','e0','e1','raw_linf','action_delta','boundary','reward','terminal','iterations','probe_steps')}
    for t,(source_action,wanted) in enumerate(zip(job['actions'],job['next_observations'])):
        current_obs = np.r_[p[1:],np.clip(v,-10,10)]
        # Every candidate starts from the same current actual p/v. Never from a later source row.
        def probe(action):
            return r['restoration']['step'](r['env'],p,v,action)[:3]
        original = probe(source_action)
        if mode=='B1':
            action = np.asarray(source_action,dtype='float64')
            e0 = e1 = float(np.linalg.norm((original[0]-wanted)/r['scale']))
            nit,calls = 0,1
        else:
            action,_,e0,e1,nit,calls,_ = r['solver'](probe,source_action,wanted,r['scale'],r['config']['solver'],original)
        # Explicitly execute accepted action, because the solver's last probe can be rejected.
        ns,reward,terminal,p_next,v_next = r['restoration']['step'](r['env'],p,v,action)
        actual_e1 = float(np.linalg.norm((ns-wanted)/r['scale']))
        assert np.isclose(actual_e1,e1,atol=1e-12,rtol=1e-12), 'Chosen-action replay differs from solver output'
        raw_linf = float(np.max(np.abs(ns-wanted)))
        values = (current_obs,action,wanted,ns,e0,actual_e1,raw_linf,float(np.linalg.norm(action-source_action)),
                  bool(np.any(1-np.abs(action)<=r['config']['action_boundary_abs_tolerance'])),reward,terminal,nit,calls)
        for key,value in zip(rows,values):
            rows[key].append(value)
        p,v = p_next,v_next
        if terminal:
            break
    arrays = {k:np.asarray(v) for k,v in rows.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    n = len(arrays['e1'])
    name = f'{mode}_trajectory{record["trajectory_id"]:05d}'
    np.savez_compressed(folder/(name+'.npz'),**arrays,step=np.arange(n),reference_terminal=job['terminals'][:n])
    score = (float(arrays['reward'].sum())-job['score_min'])/(job['score_max']-job['score_min'])*100
    result = dict(mode=mode,trajectory_id=record['trajectory_id'],first_row=record['first_row'],
        last_row=record['last_row'],L=record['length'],T=n,T_over_L=n/record['length'],
        physical_terminal=bool(arrays['terminal'][-1]),reached_record_end=n==record['length'],
        boundary_fraction=float(arrays['boundary'].mean()),raw_return=float(arrays['reward'].sum()),normalized_score=score,
        e1=stats(arrays['e1']),raw_linf=stats(arrays['raw_linf']),action_delta=stats(arrays['action_delta']),
        simulator_steps=r['steps']-first,raw_sha256=sha(folder/(name+'.npz')),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,worker_pid=os.getpid())
    if mode=='B0':
        from core import b0_gate
        result['self_check'] = b0_gate(arrays['raw_linf'],arrays['terminal'],job['terminals'],
            record['length'],r['config']['target_control_acceptance']['corrected_observation_linf'])
    write(folder/(name+'.json'),result)
    return result


def source_task(item):
    initial,work = item
    try:
        if RUNTIME is None:
            initialize(*initial)
        return track(work)
    except BaseException:
        import traceback
        name = f'{work["mode"]}_trajectory{work["record"]["trajectory_id"]:05d}.FAILED.json'
        write(Path(work['output'])/name,dict(time=utc(),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def job(inputs,out,record,mode,qpos,qvel,domain):
    start,end = record['first_row'],record['last_row']+1
    data = inputs.data[domain]
    assert record['length']<=1000
    return dict(record=record,mode=mode,output=str(out),qpos=np.asarray(qpos[start]),qvel=np.asarray(qvel[start]),
        actions=data['actions'][start:end],next_observations=data['next_observations'][start:end],
        terminals=data['terminals'][start:end].reshape(-1).astype(bool),score_min=inputs.spec['score_min'],score_max=inputs.spec['score_max'])


def b0(inputs,out,vendor,heartbeat):
    initialize(inputs.xml,vendor,inputs.scale,inputs.old)
    r = RUNTIME
    data = inputs.data['target']
    starts = np.zeros(5000,dtype=bool)
    starts[[s['first_row'] for s in inputs.target_segments]] = True
    qpos,qvel,status,error,_ = r['restoration']['recover'](r['env'],data['observations'],data['actions'],starts,'hopper-morph')
    np.savez_compressed(out/'B0_restoration.npz',qpos=qpos,qvel=qvel,status=status,error=error)
    records = []
    try:
        for record in inputs.target_segments:
            if status[record['first_row']]==2:
                r0 = dict(mode='B0',trajectory_id=record['trajectory_id'],first_row=record['first_row'],
                    last_row=record['last_row'],L=record['length'],T=0,T_over_L=0.,physical_terminal=False,
                    reached_record_end=False,boundary_fraction=0.,raw_return=0.,
                    normalized_score=(0-inputs.spec['score_min'])/(inputs.spec['score_max']-inputs.spec['score_min'])*100,
                    e1=stats([]),raw_linf=stats([]),action_delta=stats([]),simulator_steps=0,
                    self_check=dict(passed=False,reached_record_end=False,first_exceed_step=None,
                        terminal_mismatch_steps=[],raw_linf=stats([]),reason='Initial physical-state recovery unavailable'))
                np.savez_compressed(out/f'B0_trajectory{record["trajectory_id"]:05d}.npz',e1=np.empty(0),step=np.empty(0,dtype=int))
                records.append(r0)
                write(out/f'B0_trajectory{record["trajectory_id"]:05d}.json',r0)
            else:
                records.append(track(job(inputs,out,record,'B0',qpos,qvel,'target')))
            heartbeat('B0',completed=len(records),total=len(inputs.target_segments))
    finally:
        r['env'].close()
    result = dict(passed=all(r['self_check']['passed'] for r in records),trajectories=records,
        raw_linf_limit=inputs.old['target_control_acceptance']['corrected_observation_linf'],
        definition='Raw observation max absolute error after committed tracking action versus recorded target next observation, at every executed step. All recorded segments including tail; no early termination; matching last terminal.',
        simulator_steps=r['steps'],source_tracking_started=False)
    write(out/'B0_RESULT.json',result)
    return result


def source_tracks(inputs,out,vendor,heartbeat,workers):
    qpos = np.load(V13/'stage1/restoration/qpos.npy',mmap_mode='r',allow_pickle=False)
    qvel = np.load(V13/'stage1/restoration/qvel.npy',mmap_mode='r',allow_pickle=False)
    status = np.load(V13/'stage1/restoration/recovery_status.npy',mmap_mode='r',allow_pickle=False)
    assert qpos.shape==qvel.shape==(1000000,6)
    assert np.isin(status[[r['first_row'] for r in inputs.selected_source]],[0,1,3]).all()
    jobs = [job(inputs,out,r,mode,qpos,qvel,'source') for r in inputs.selected_source for mode in ('B1','B2')]
    # One real selected trajectory, two modes, is the RSS pilot and is counted in the 500.
    initialize(inputs.xml,vendor,inputs.scale,inputs.old)
    records = []
    try:
        records.extend(track(j) for j in jobs[:2])
    finally:
        RUNTIME['env'].close()
    import limits
    available = limits.limits_snapshot()['effective_available_memory_bytes']
    pilot = max(r['peak_RSS_bytes'] for r in records)
    count = min(workers,max(1,int(available*.60/max(pilot,1))))
    assert available>pilot*2, 'Insufficient memory for B CPU tracking'
    write(out/'CPU_ADMISSION.json',dict(requested_workers=workers,actual_workers=count,pilot_peak_RSS_bytes=pilot,
        projected_worker_RSS_bytes=count*pilot,effective_available_memory_bytes=available,
        process_ceiling_including_controller_and_helpers=32,no_GPU=True))
    ctx = mp.get_context('spawn')
    initial = (str(inputs.xml),str(vendor),inputs.scale,inputs.old)
    # Initialization occurs in the task, so an import/build exception reaches the
    # controller instead of Pool endlessly respawning a failing initializer.
    executor = ProcessPoolExecutor(max_workers=count,mp_context=ctx)
    futures = []
    try:
        futures = [executor.submit(source_task,(initial,j)) for j in jobs[2:]]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            heartbeat('B1_B2',completed=len(records),total=len(jobs))
    except BaseException:
        for future in futures:
            future.cancel()
        # Only processes created and owned by this executor are stopped.
        for process in list((executor._processes or {}).values()):
            if process.is_alive():
                process.terminate()
        raise
    finally:
        executor.shutdown(wait=True)
    assert len(records)==2*len(inputs.selected_source)
    write(out/'B_SOURCE_TRACKS.json',records)
    return records


def summarize(out,b0_result,records):
    all_records = b0_result['trajectories']+records
    simple,bybin,hist_rows,groups = [],[],[],{}
    counts_by_mode = {}
    for mode in ('B0','B1','B2'):
        rows = sorted([r for r in all_records if r['mode']==mode],key=lambda r:r['trajectory_id'])
        if not rows:
            continue
        lengths = np.asarray([r['T'] for r in rows])
        ratios = np.asarray([r['T_over_L'] for r in rows])
        groups[mode] = dict(episodes=len(rows),survival_steps=stats(lengths),T_over_L=stats(ratios),
            fraction_T_under_200=float((lengths<200).mean()),fraction_T_over_L_at_least_08=float((ratios>=.8).mean()),
            normalized_score=stats([r['normalized_score'] for r in rows]),physical_terminal_count=sum(r['physical_terminal'] for r in rows))
        if mode=='B2':
            groups[mode]['interpretation'] = interpret_b(lengths,ratios)
        errors,steps = [],[]
        hist = np.zeros(50,dtype=np.int64)
        for r in rows:
            simple.append({k:r[k] for k in ('mode','trajectory_id','first_row','last_row','L','T','T_over_L',
                'physical_terminal','reached_record_end','boundary_fraction','raw_return','normalized_score')})
            with np.load(out/f'{mode}_trajectory{r["trajectory_id"]:05d}.npz',allow_pickle=False) as f:
                errors.append(f['e1']);steps.append(f['step'])
            if r['physical_terminal']:
                hist[(r['T']-1)//20] += 1
        e,t = np.concatenate(errors),np.concatenate(steps)
        for j,label in enumerate(A_LABELS):
            bybin.append(dict(mode=mode,step_bin=label,**stats(e[bins(t)==j])))
        counts_by_mode[mode] = hist.tolist()
        for i,value in enumerate(hist):
            hist_rows.append(dict(mode=mode,executed_step_first=i*20+1,executed_step_last=(i+1)*20,count=int(value)))
    csv_write(out/'B_trajectories.csv',simple)
    csv_write(out/'B_residual_step_bins.csv',bybin)
    csv_write(out/'B_physical_termination_histogram.csv',hist_rows)
    histogram_svg(out/'B_physical_termination_histogram.svg',counts_by_mode)
    result = dict(B0=b0_result,B1_B2_started=bool(records),groups=groups,residual_bins=bybin,
        histogram=hist_rows,source_episode_count=len(records)//2,
        limitation='B2 is one-step greedy tracking with v13 cold-reset/zero-warmstart candidate convention; the actual committed qpos/qvel are carried forward, not replaced by source states. Failure does not rule out lookahead controllers.',
        histogram_note='Counts include physical environment termination, including natural termination on the recorded final step; horizon/source-record endings without physical termination are not falls.',
        score_note='Normalized cumulative return over each tracked record; variable record length and early termination, not a formal policy-performance comparison.',
        simulator_steps=sum(r['simulator_steps'] for r in records)+b0_result['simulator_steps']+len({r['worker_pid'] for r in records}),
        new_training_updates=0,GPU_jobs=0)
    write(out/'RESULT.json',result)
    write(out/'COMPLETED.json',dict(completed=bool(b0_result['passed'] and records),
        status='B_COMPLETED' if records else 'B0_FAILED_B1_B2_NOT_STARTED',result_sha256=sha(out/'RESULT.json'),ended_utc=utc()))
    return result
