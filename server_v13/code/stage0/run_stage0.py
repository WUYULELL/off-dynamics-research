"""V13 stage 0 only: no learner, training, policy evaluation, or CUDA."""
import argparse
import fcntl
import importlib.metadata as metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
from scipy.spatial import cKDTree

import legacy_replay as legacy
import legacy_resources as resources
from solver import solve


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = Path.home() / 'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
GIB = 1024 ** 3
TASK = 'hopper-morph'
SUCCESS = {'observation_noise_floor', 'projected_gradient', 'step_tolerance', 'cost_tolerance'}


def publish(path, value):
    scratch = path.with_suffix('.tmp')
    scratch.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    scratch.replace(path)


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {'n': 0}
    assert np.isfinite(x).all()
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def source_input(path, rows, fields):
    with h5py.File(path, 'r') as f:
        assert f['observations'].shape == (1000000, 11)
        return {k: np.asarray(f[k])[rows] for k in fields}


def target_input(path, fields):
    with h5py.File(path, 'r') as f:
        result = {k: np.asarray(f[k]) for k in fields}
    assert result['observations'].shape == (5000, 11)
    return result


def worker(folder, xml, vendor, cfg, scale, data, kind):
    started = time.monotonic()
    env = None
    try:
        sys.path.insert(0, str(vendor))
        import mujoco_py
        assert vendor in Path(mujoco_py.cymj.__file__).resolve().parents
        from gym.envs.mujoco.hopper_v3 import HopperEnv

        ns = legacy.load_restoration(HERE/'references', legacy.read(HERE/'RESTORATION_AST.json'))
        env = HopperEnv(xml_file=str(xml))
        env.seed(cfg['source_sample_seed'])
        assert env.model.nq == env.model.nv == 6 and env.frame_skip == 4
        assert abs(float(env.model.opt.timestep) - .002) < 1e-12
        assert abs(float(env.model.geom_size[env.model.geom_name2id('torso_geom'), 0]) - .125) < 1e-12
        tracker = legacy.CountedEnvironment(env)
        obs, actions, nexts = [data[k] for k in ('observations', 'actions', 'next_observations')]
        assert actions.shape == (len(obs), 3) and obs.shape == nexts.shape == (len(obs), 11)
        assert all(np.isfinite(a).all() for a in (obs, actions, nexts))
        assert set(np.unique(data['terminals'])).issubset({0, 1})
        tracker.phase = 'restore'
        if kind == 'target':
            boundaries = np.max(np.abs(nexts[:-1]-obs[1:]), axis=1) > 1e-6
            starts = np.r_[True, boundaries | data['terminals'].reshape(-1)[:-1].astype(bool)]
            assert np.flatnonzero(starts).tolist() == [0, 652, 1320, 1963, 2618, 3265, 3911, 4564]
            qpos, qvel, status, recovery_error, _ = ns['recover'](env, obs, actions, starts, TASK)
            if np.any(status == 2):
                legacy.write(folder/'RESTORE_FAILURE.json', {'rows': np.flatnonzero(status == 2).tolist()})
                raise RuntimeError('Target state restoration failed')
        else:
            qpos, qvel = data['qpos'], data['qvel']
            recovery_error = np.zeros(len(obs))
        tracker.phase = 'solve'
        rows = []
        for i, (p, v, a, target) in enumerate(zip(qpos, qvel, actions, nexts)):
            t0 = time.monotonic()

            def replay(act):
                next_obs, reward, terminal, _, _ = ns['step'](env, p, v, act)
                if not np.isfinite(next_obs).all() or not np.isfinite(reward):
                    raise FloatingPointError('Non-finite simulator result at row %d' % data['row_id'][i])
                return next_obs, reward, terminal

            restored = np.r_[p[1:], np.clip(v, -10, 10)]
            recovery_error[i] = float(np.max(np.abs(restored-obs[i])))
            assert recovery_error[i] <= cfg['source_observation_restoration_linf'], 'Restored observation mismatch'
            original = replay(a)
            if kind == 'source':
                gate = cfg['original_oracle_comparison']
                assert (np.max(np.abs(original[0] - data['oracle_next'][i])) <= gate['observation_linf']
                        and abs(original[1] - data['oracle_reward'][i]) <= gate['reward_abs']
                        and original[2] == bool(data['oracle_terminal'][i])), ('V9 oracle replay mismatch', int(data['row_id'][i]))
            fixed, actual, e0, e1, nit, calls, status = solve(replay, a, target, scale, cfg['solver'], original)
            assert np.isfinite([e0, e1]).all() and e1 <= e0 and nit <= 10
            rows.append((fixed, actual[0], actual[1], actual[2], e0, e1, float(np.linalg.norm(fixed-a)), nit,
                         calls, status, np.any(1-np.abs(fixed) <= cfg['action_boundary_abs_tolerance']),
                         float(np.max(np.abs(original[0]-target))), float(np.max(np.abs(actual[0]-target))),
                         float(abs(original[1]-data['rewards'].reshape(-1)[i])),
                         original[2] == bool(data['terminals'].reshape(-1)[i]), time.monotonic()-t0))
            if (i+1) % 1000 == 0 or i+1 == len(obs):
                publish(folder/'heartbeat.json', dict(time=legacy.utc(), pid=os.getpid(), done=i+1, total=len(obs),
                        environment_steps=1+tracker.steps))
        names = ('action_fixed', 'next_actual', 'reward_actual', 'terminal_actual', 'e0', 'e1', 'action_delta',
                 'iterations', 'simulator_steps', 'solver_status', 'boundary', 'original_raw_linf', 'corrected_raw_linf',
                 'original_reward_error', 'original_terminal_matches', 'row_seconds')
        result = {k: np.asarray(x) for k, x in zip(names, zip(*rows))}
        result.update(row_id=data['row_id'], original_terminal=data['terminals'].reshape(-1).astype(bool),
                      original_action=actions, recovery_error=recovery_error)
        assert int(result['simulator_steps'].sum()) == tracker.by_phase['solve']
        assert 'torch' not in sys.modules
        np.savez_compressed(folder/'rows.npz', **result)
        legacy.write(folder/'COMPLETED.json', dict(rows=len(obs), rows_sha256=legacy.sha(folder/'rows.npz'),
            wall_seconds=time.monotonic()-started, peak_RSS_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment_step_counts=dict(constructor=1, **tracker.by_phase), completed_utc=legacy.utc()))
    except BaseException:
        legacy.write(folder/'FAILED.json', dict(traceback=traceback.format_exc(), time=legacy.utc()))
        raise
    finally:
        if env is not None:
            env.close()


def run_jobs(out, common, jobs, limit, worker_rss=0):
    pending = [(name, kind, rows, 0) for name, kind, rows in jobs]
    active = []
    completed = {}
    exhausted = []
    context = mp.get_context('spawn')
    cfg = common[2]
    limits = resources.limits_snapshot()
    parent_rss = resources.owned_process(os.getpid())['RSS_bytes']
    with (out/'queue_events.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        def event(**data):
            log.write(json.dumps(dict(time=legacy.utc(), **data), ensure_ascii=False)+'\n')
        try:
            while pending or active:
                for item in active[:]:
                    process, folder, name, kind, rows, retry = item
                    code = process.exitcode
                    if code is None:
                        continue
                    process.join()
                    active.remove(item)
                    event(job=name, attempt=retry, pid=process.pid, exitcode=code, physical_gpu=None)
                    if code == 0:
                        receipt = legacy.read(folder/'COMPLETED.json')
                        assert receipt['rows_sha256'] == legacy.sha(folder/'rows.npz')
                        completed[name] = folder
                    elif code in (-9, -15) and not (folder/'FAILED.json').exists():
                        try:
                            evidence = subprocess.run(['dmesg', '--level=err,warn'], text=True,
                                                      capture_output=True, timeout=5)
                            fragment = (evidence.stdout or evidence.stderr).splitlines()[-20:]
                        except (OSError, subprocess.TimeoutExpired) as error:
                            fragment = [str(error)]
                        legacy.write(folder/'INTERRUPTED.json', dict(exitcode=code, physical_gpu=None, dmesg_tail=fragment))
                        if retry < cfg['external_retry_limit']:
                            pending.append((name, kind, rows, retry+1))
                        else:
                            exhausted.append(name)
                    else:
                        raise RuntimeError('Stage 0 worker failed: %s, exit %s; see %s' % (name, code, folder))
                mem = resources.key_values('/proc/meminfo')
                effective = resources.dynamic_snapshot(limits)['effective_available_memory_bytes']
                host_ok = mem['MemAvailable'] >= cfg['host_memory_stop_fraction']*mem['MemTotal']
                budget_ok = parent_rss + (len(active)+1)*worker_rss <= cfg['memory_fraction']*effective
                while pending and len(active) < limit and host_ok and budget_ok:
                    name, kind, rows, retry = pending.pop(0)
                    folder = out/'runs'/(name if retry == 0 else '%s_retry%d' % (name, retry))
                    folder.mkdir()
                    process = context.Process(target=worker, args=(folder, *common, rows, kind))
                    process.start()
                    event(job=name, attempt=retry, pid=process.pid, state='started', physical_gpu=None)
                    active.append((process, folder, name, kind, rows, retry))
                    budget_ok = parent_rss + (len(active)+1)*worker_rss <= cfg['memory_fraction']*effective
                    mem = resources.key_values('/proc/meminfo')
                    host_ok = mem['MemAvailable'] >= cfg['host_memory_stop_fraction']*mem['MemTotal']
                publish(out/'progress.json', dict(time=legacy.utc(), active={x[2]: x[0].pid for x in active},
                    phase=jobs[0][1], current_batch_completed=len(completed), pending=len(pending), exhausted=exhausted,
                    admission_waiting=bool(pending and not active), effective_available_memory_bytes=effective))
                if pending or active:
                    time.sleep(1)
        finally:
            for process, *_ in active:
                if process.is_alive():
                    process.terminate()
                process.join()
    if exhausted:
        raise RuntimeError('External retry limit reached: '+', '.join(exhausted))
    return completed


def load_rows(folder):
    with np.load(folder/'rows.npz', allow_pickle=False) as raw:
        return dict(raw)


def describe(data):
    result = {k: summarize(data[k]) for k in ('e0', 'e1', 'action_delta', 'iterations', 'simulator_steps',
              'original_raw_linf', 'corrected_raw_linf', 'row_seconds')}
    nonzero = data['e0'] > 0
    statuses, counts = np.unique(data['solver_status'], return_counts=True)
    result.update(e1_over_e0=summarize(data['e1'][nonzero]/data['e0'][nonzero]), zero_e0=int((~nonzero).sum()),
        boundary_fraction=float(data['boundary'].mean()), original_terminal_fraction=float(data['original_terminal'].mean()),
        corrected_terminal_fraction=float(data['terminal_actual'].mean()),
        solver_status_counts={str(k): int(v) for k,v in zip(statuses, counts)},
        not_converged_fraction=float(np.mean([s not in SUCCESS for s in data['solver_status']])))
    return result


def report(out, target, failures, source=None, bins=None, budget=None, error=None):
    lines = ['# V13 阶段 0 源动作修正可行性报告', '',
        'Hopper-morph。优化器逐条在目标仿真器中试算动作。训练更新、策略评估和 GPU 作业均为 0。',
        '距离为按目标当前观测逐维样本标准差加 1e-6 缩放后的 L2。V9 物理回放验收的逐维最大绝对误差另列。',
        '目标域 5000 行各一次。源域在 V9 验收的 999998 个 canonical 行中固定种子不放回取 20000 行。',
        '源状态沿用 V9 已独立验收的物理状态，原动作的 Linux 回放与 V9 oracle 核对。', '']
    for label, result in [('目标数据自检', target), ('源数据修正', source)]:
        if result is None:
            continue
        lines += ['## '+label, '', '|指标|n|均值|中位数|P90|P99|最大值|', '|---|---:|---:|---:|---:|---:|---:|']
        for k,v in result.items():
            if isinstance(v, dict) and 'n' in v:
                lines.append('|'+k+'|'+str(v['n'])+'|'+'|'.join(format(v[t], '.8g') if t in v else '—'
                             for t in ('mean', 'median', 'p90', 'p99', 'max'))+'|')
        lines += ['', '原终止比例 %.4f%%，修正终止比例 %.4f%%，动作触边 %.4f%%，未收敛 %.4f%%。e0 为 0 的 %d 行另计。' %
                  tuple([100*result[k] for k in ('original_terminal_fraction','corrected_terminal_fraction',
                         'boundary_fraction','not_converged_fraction')]+[result['zero_e0']]),
                  '停止状态计数为 `'+json.dumps(result['solver_status_counts'], ensure_ascii=False)+'`。', '']
    lines += ['目标自检超限行数为 %d，预先冻结的技术门槛见 config.json。' % len(failures), '']
    if bins is not None:
        lines += ['## 到目标状态的最近邻距离与 e1（仅描述）', '',
                  '|档|行数|距离范围|距离中位数|e1 均值|e1 中位数|e1 P90|e1 P99|', '|---|---:|---|---:|---:|---:|---:|---:|']
        for i, b in enumerate(bins, 1):
            s = b['e1']
            lines.append(f'|{i}|{s["n"]}|{b["min"]:.7g}–{b["max"]:.7g}|{b["median"]:.7g}|'
                         f'{s["mean"]:.7g}|{s["median"]:.7g}|{s["p90"]:.7g}|{s["p99"]:.7g}|')
    if budget:
        lines += ['', '## 资源与全量估算', '',
                  '单进程试算峰值 RSS %.3f GiB，控制器 RSS %.3f GiB，最多放行 %d 个 CPU 进程，估计总 RSS %.3f GiB。' %
                  (budget['pilot_RSS_bytes']/GIB, budget['parent_RSS_bytes']/GIB, budget['workers'], budget['predicted_RSS_bytes']/GIB),
                  'nproc %d，cgroup/亲和性可用 %.2f 核。源样本实测阶段墙钟 %.2f 秒，整机 CPU 占用 %.2f%%。' %
                  (budget['nproc'], budget['effective_cpu'], budget['source_wall_seconds'], budget['host_busy_percent']),
                  '按实测墙钟外推全量约 %.3f 小时，按单行计算时间和选定并发的理想下限约 %.3f 小时。' %
                  (budget['full_wall_hours_estimate'], budget['full_ideal_hours_estimate']),
                  '墙钟包含试算进程和调度开销。全量另需写缓存、独立回放验收和 I/O，外推以本次服务器负载为基准。']
    lines += ['', '## 停点', '', error or 'AWAITING_STAGE1_CONFIRMATION。请依据阶段 0 残差报告决定过滤阈值及是否启动阶段 1。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def validate_environment(cfg, out):
    assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    assert sys.version_info[:3] == (3, 8, 13)
    for name, version in cfg['expected_packages'].items():
        assert metadata.version(name) == version, ('Unexpected runtime package version', name)
    gym = Path(metadata.distribution('gym').locate_file('gym')).resolve()
    files = {}
    for name in ('envs/mujoco/hopper_v3.py', 'envs/mujoco/mujoco_env.py', 'utils/seeding.py', 'utils/ezpickle.py'):
        path = gym/name
        files[str(path)] = legacy.sha(path)
        assert files[str(path)] == legacy.sha(PROJECT/'environment/gym_source_reference'/name)
    library = Path(os.environ['MUJOCO_PY_MUJOCO_PATH'])/'bin/libmujoco210.so'
    files[str(library)] = legacy.sha(library)
    conda = Path(sys.prefix)/'conda-meta'
    meta_before = legacy.hash_tree(conda)
    for name in ('tmp', 'cache'):
        (out/name).mkdir()
    os.environ.update(TMPDIR=str(out/'tmp'), XDG_CACHE_HOME=str(out/'cache'),
                      CYTHON_CACHE_DIR=str(out/'cache/cython'))
    binding, binding_hashes, private = legacy.prepare_private_binding(out)
    import mujoco_py
    extension = Path(mujoco_py.cymj.__file__).resolve()
    assert private in extension.parents
    files[str(extension)] = legacy.sha(extension)
    legacy.write(out/'ENVIRONMENT.json', dict(python=sys.version, executable=sys.executable,
                 platform=platform.platform(), packages={d.metadata['Name']:d.version for d in metadata.distributions()},
                 verified_runtime_files=files, conda_metadata_hashes=meta_before,
                 CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                 GPU_inventory=legacy.resource_snapshot()['nvidia_smi_snapshot']))
    return files, conda, meta_before, binding, binding_hashes


def confirm_inputs(out, before, manifest, environment):
    files, conda, meta, binding, binding_hashes = environment
    assert legacy.verify_project(PROJECT, manifest) == before
    assert legacy.hash_tree(conda) == meta and legacy.hash_tree(binding) == binding_hashes
    assert all(legacy.sha(path) == digest for path,digest in files.items())
    legacy.write(out/'INPUT_PROTECTION_AFTER.json', dict(passed=True, transfer_files=len(before),
                 conda_metadata_files=len(meta), installed_mujoco_binding_files=len(binding_hashes)))


def stage0(out, cfg, freeze):
    manifest = freeze['transfer_manifest_sha256']
    before = legacy.verify_project(PROJECT, manifest)
    environment = validate_environment(cfg, out)
    limits = resources.limits_snapshot()
    assert limits['cgroup_mapping_found'] and not limits['unreadable_fields'], 'Resolve CPU/memory limits before launch'
    legacy.write(out/'RESOURCES_BEFORE.json', limits)
    mapping = legacy.read(PROJECT/'cluster/paths.example.json')['tasks'][TASK]
    xml = PROJECT/mapping['xml']
    target_path = PROJECT/mapping['target_dataset']
    source_path = PROJECT/mapping['source_dataset']
    v9 = legacy.read(HERE/'references/V9_PROVENANCE.json')
    assert before[mapping['source_dataset']] == v9['source_sha256']
    assert before[mapping['xml']] == v9['xml_sha256']
    assert before['data/oracle/hopper/raw_indices.npy'] == v9['raw_indices_sha256']
    for name,digest in v9['oracle_arrays_sha256'].items():
        assert before['data/oracle/hopper/'+name] == digest
    source_count = mapping['source_rows']
    oracle = PROJECT/'data/oracle/hopper'
    ids = np.load(oracle/'raw_indices.npy', allow_pickle=False)
    assert ids.shape == (source_count,) and source_count == 999998 and np.all(ids[:-1] < ids[1:])
    positions = np.random.RandomState(cfg['source_sample_seed']).choice(source_count, cfg['source_sample_count'], replace=False)
    source_rows = ids[positions]
    np.savez_compressed(out/'sample_plan.npz', source_row_id=source_rows, canonical_position=positions,
                        target_row_id=np.arange(cfg['target_sample_count']))
    legacy.write(out/'PRE_SIMULATOR_COMMITMENT.json', dict(time=legacy.utc(), sample_sha256=legacy.sha(out/'sample_plan.npz'),
                 config_sha256=legacy.sha(out/'config.json'), transfer_manifest_sha256=manifest,
                 freeze_sha256=legacy.sha(HERE/'FROZEN.json'), v9_provenance=v9))
    fields = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')
    target = target_input(target_path, fields)
    target['row_id'] = np.arange(len(target['observations']))
    assert len(target['row_id']) == cfg['target_sample_count']
    scale = np.std(target['observations'].astype(np.float64), axis=0, ddof=1)+cfg['std_epsilon']
    assert np.isfinite(scale).all() and np.all(scale > 0)
    np.save(out/'target_observation_scale.npy', scale)
    common = (xml, out/'private_runtime', cfg, scale)
    (out/'runs').mkdir()
    print('阶段 0 | 目标域 5000 行回放和求解自检', flush=True)
    target_dir = run_jobs(out, common, [('target', 'target', target)], 1)['target']
    target_records = load_rows(target_dir)
    assert np.array_equal(target_records['row_id'], target['row_id'])
    target_summary = describe(target_records)
    gate = cfg['target_control_acceptance']
    fail = np.flatnonzero((target_records['original_raw_linf'] > gate['original_observation_linf']) |
                         (target_records['corrected_raw_linf'] > gate['corrected_observation_linf']) |
                         (target_records['action_delta'] > gate['action_change_l2']) |
                         (target_records['original_reward_error'] > gate['original_reward_abs']) |
                         ~target_records['original_terminal_matches']).tolist()
    legacy.write(out/'TARGET_CONTROL.json', dict(passed=not fail, failing_rows=fail, **target_summary))
    if fail:
        confirm_inputs(out, before, manifest, environment)
        report(out, target_summary, fail, error='TARGET_CONTROL_FAILED。源数据试算和后续阶段均未放行。')
        raise RuntimeError('Target positive control failed, see TARGET_CONTROL.json')

    source = source_input(source_path, source_rows, fields)
    with np.load(HERE/'references/source_restoration_sample.npz', allow_pickle=False) as recovery:
        assert np.array_equal(recovery['row_id'], source_rows)
        assert np.isin(recovery['status'], [0, 1, 3]).all()
        source.update(qpos=recovery['qpos'], qvel=recovery['qvel'], row_id=source_rows)
    for name, filename in [('oracle_next', 'next_state'), ('oracle_reward', 'reward'), ('oracle_terminal', 'terminal')]:
        values = np.load(oracle/(filename+'.npy'), allow_pickle=False, mmap_mode='r')[positions]
        source[name] = values if name == 'oracle_next' else values.reshape(-1)

    def subset(start, end):
        return {k: values[start:end] for k,values in source.items()}

    print('阶段 0 | 源域单进程 RSS 试算，计入原定样本', flush=True)
    source_started = time.monotonic()
    cpu_before = resources.cpu_ticks()
    pilot_count = cfg['pilot_count']
    pilot_dir = run_jobs(out, common, [('source_pilot', 'source_pilot', subset(0, pilot_count))], 1)['source_pilot']
    pilot_rss = legacy.read(pilot_dir/'COMPLETED.json')['peak_RSS_bytes']
    assert pilot_rss > 0
    limits_now = resources.limits_snapshot()
    parent_rss = resources.owned_process(os.getpid())['RSS_bytes']
    available = limits_now['effective_available_memory_bytes']
    memory_cap = max(0, int((available*cfg['memory_fraction']-parent_rss)/pilot_rss))
    workers = min(cfg['max_cpu_workers'], cfg['cpu_core_budget'], int(limits_now['effective_visible_cpu_cores']), memory_cap)
    assert workers >= 1, 'RSS estimate leaves no admissible worker'
    budget = dict(pilot_RSS_bytes=pilot_rss, parent_RSS_bytes=parent_rss, workers=workers,
                  predicted_RSS_bytes=parent_rss + workers*pilot_rss,
                  effective_available_memory_bytes=available, nproc=limits_now['nproc'],
                  effective_cpu=limits_now['effective_visible_cpu_cores'])
    legacy.write(out/'MEASURED_RSS_BEFORE_PARALLEL.json', budget)
    print('并行放行前 RSS 核验', json.dumps(budget, ensure_ascii=False), flush=True)
    size = cfg['source_chunk_size']
    jobs = [('source_%05d' % start, 'source', subset(start,start+size))
            for start in range(pilot_count, len(source_rows), size)]
    folders = run_jobs(out, common, jobs, workers, pilot_rss)
    source_wall = time.monotonic()-source_started
    host_busy = resources.busy_percent(cpu_before, resources.cpu_ticks())
    ordered = [pilot_dir] + [folders[name] for name, *_ in jobs]
    pieces = [load_rows(folder) for folder in ordered]
    solved = {key:np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    assert np.array_equal(solved['row_id'], source_rows) and len(solved['row_id']) == cfg['source_sample_count']
    distance, nearest = cKDTree(target['observations']/scale).query(source['observations']/scale, k=1)
    bins = []
    for idx in np.array_split(np.argsort(distance, kind='stable'), 5):
        stats = summarize(distance[idx])
        stats['e1'] = summarize(solved['e1'][idx])
        bins.append(stats)
    np.savez_compressed(out/'source_results.npz', **solved, nearest_target_row=nearest,
                        nearest_distance=distance, canonical_position=positions)
    source_summary = describe(solved)
    budget.update(source_wall_seconds=source_wall, host_busy_percent=host_busy,
                  full_wall_hours_estimate=source_wall*source_count/len(source_rows)/3600,
                  full_ideal_hours_estimate=source_summary['row_seconds']['mean']*source_count/workers/3600,
                  full_serial_hours_estimate=source_summary['row_seconds']['mean']*source_count/3600)
    environment_steps = {}
    for folder in [target_dir] + ordered:
        for k,n in legacy.read(folder/'COMPLETED.json')['environment_step_counts'].items():
            environment_steps[k] = environment_steps.get(k, 0)+n
    assert 'torch' not in sys.modules
    confirm_inputs(out, before, manifest, environment)
    report(out, target_summary, fail, source_summary, bins, budget)
    with (out/'REPORT_ZH.md').open('a', encoding='utf-8') as f:
        f.write('\n本次成功任务的仿真步分类为 `'+json.dumps(environment_steps)+'`。'
                '如有外部中断，失败尝试的步数下界见心跳，退出原因见 queue_events.jsonl。\n')
    legacy.write(out/'RESULT.json', dict(target=target_summary, source=source_summary, nearest_neighbor_bins=bins,
                 resource_budget=budget, successful_environment_step_counts=environment_steps,
                 accepted_source_rows=len(solved['row_id']), no_learner_constructed=True,
                 training_updates=0, policy_evaluations=0, GPU_jobs=0))
    evidence = ['config.json','PRE_SIMULATOR_COMMITMENT.json','MEASURED_RSS_BEFORE_PARALLEL.json','INPUT_PROTECTION_AFTER.json',
                'TARGET_CONTROL.json','RESULT.json','REPORT_ZH.md','source_results.npz','sample_plan.npz','queue_events.jsonl']
    legacy.write(out/'STAGE0_COMPLETED.json', dict(completed=True, target_control_passed=True, time=legacy.utc(),
                 acceptance_status='AWAITING_STAGE1_CONFIRMATION', later_stage_started=False,
                 source_count=20000, target_count=5000, report_sha256=legacy.sha(out/'REPORT_ZH.md'),
                 evidence_sha256={name:legacy.sha(out/name) for name in evidence}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-reboot', action='store_true')
    args = parser.parse_args()
    assert sys.platform == 'linux' and os.environ.get('TMUX')
    cfg = legacy.read(HERE/'config.json')
    freeze = legacy.read(HERE/'FROZEN.json')
    assert cfg['stage'] == 0 and not cfg['stage1_authorized'] and not cfg['stage2_authorized']
    for name,digest in freeze['files'].items():
        assert legacy.sha(HERE/name) == digest, ('Changed package input', name)
    output = BASE/'outputs'
    output.mkdir(exist_ok=True)
    with (output/'.stage0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = sorted(output.glob('stage0_*'), key=lambda p:p.stat().st_mtime)
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        started = dict(pid=os.getpid(), boot_id=boot_id, start_time=legacy.utc(), code_freeze_sha256=legacy.sha(HERE/'FROZEN.json'))
        if args.after_reboot:
            assert attempts and len(attempts) <= cfg['external_retry_limit']
            assert all(not (p/'STAGE0_COMPLETED.json').exists() and not (p/'FAILED.json').exists() for p in attempts)
            old = legacy.read(attempts[-1]/'STARTED.json')
            assert old['boot_id'] != boot_id and old['code_freeze_sha256'] == started['code_freeze_sha256']
            started.update(external_interruption='boot_id changed', previous=str(attempts[-1]), previous_boot_id=old['boot_id'])
            name = attempts[0].name+'_retry'+str(len(attempts))
        else:
            assert not attempts, 'An earlier stage 0 attempt exists; inspect it before invoking any restart'
            name = 'stage0_'+time.strftime('%Y%m%d')
        out = output/name
        out.mkdir()
        legacy.write(out/'STARTED.json', started)
        legacy.write(out/'config.json', cfg)
        print('输出目录', out, flush=True)
        try:
            stage0(out, cfg, freeze)
        except BaseException:
            failure = traceback.format_exc()
            legacy.write(out/'FAILED.json', dict(time=legacy.utc(), traceback=failure, later_stage_started=False))
            if not (out/'REPORT_ZH.md').exists():
                report(out, None, [], error='FAILED。详见 FAILED.json 与队列、任务日志。')
            raise
        print('AWAITING_STAGE1_CONFIRMATION', out/'REPORT_ZH.md', flush=True)


if __name__ == '__main__':
    main()
