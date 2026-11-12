"""Rebuild the V13 stage-0 report from the audited-on-write CPU results."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree


GIB = 1024 ** 3
REQUIRED = ('STARTED.json', 'FAILED.json', 'config.json', 'PRE_SIMULATOR_COMMITMENT.json',
            'INPUT_PROTECTION_AFTER.json', 'sample_plan.npz', 'target_observation_scale.npy',
            'MEASURED_RSS_BEFORE_PARALLEL.json', 'TARGET_CONTROL.json', 'queue_events.jsonl', 'source_results.npz')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def npz(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def report_functions(path):
    # Reuse only the original, hash-verified reporting definitions. No runtime imports.
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    wanted = {'summarize', 'describe', 'report'}
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in funcs} == wanted and len(funcs) == len(wanted)
    success = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'SUCCESS' for t in node.targets))
    scope = dict(np=np, json=json, GIB=GIB, SUCCESS=ast.literal_eval(success))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), scope)
    return scope['summarize'], scope['describe'], scope['report']


def accepted_attempts(old, events, jobs):
    live, done, interrupted, snapshots = {}, {}, [], {}
    peak_source_jobs = 0
    for event in events:
        name, attempt = event['job'], event['attempt']
        key = (name, attempt)
        assert name in jobs and type(attempt) is int and 0 <= attempt <= 3
        assert event['physical_gpu'] is None
        if event.get('state') == 'started':
            assert key not in live and name not in done and not any(k[0] == name for k in live)
            live[key] = event['pid']
            peak_source_jobs = max(peak_source_jobs, sum(k[0] != 'target' for k in live))
            continue
        code = event['exitcode']
        assert live.pop(key) == event['pid']
        folder = old/'runs'/(name if attempt == 0 else f'{name}_retry{attempt}')
        assert old in folder.resolve().parents
        if code != 0:
            assert code in (-9, -15) and not (folder/'FAILED.json').exists()
            path = folder/'INTERRUPTED.json'
            assert read(path)['exitcode'] == code
            snapshots[path] = sha(path)
            interrupted.append(dict(job=name, attempt=attempt, exitcode=code, time=event['time']))
            continue
        assert name not in done
        receipt_file, rows_file = folder/'COMPLETED.json', folder/'rows.npz'
        receipt = read(receipt_file)
        digest = sha(rows_file)
        rows = npz(rows_file)
        assert receipt['rows_sha256'] == digest and receipt['rows'] == len(rows['row_id']) == jobs[name]
        assert all(len(a) == jobs[name] for a in rows.values())
        assert receipt['environment_step_counts']['solve'] == int(rows['simulator_steps'].sum())
        assert receipt['environment_step_counts']['constructor'] == 1
        snapshots[receipt_file], snapshots[rows_file] = sha(receipt_file), digest
        done[name] = receipt, rows
    assert not live and done.keys() == jobs.keys(), 'Incomplete CPU work. Report cannot be recovered from this run.'
    return done, interrupted, snapshots, peak_source_jobs


def neighbor_bins(project, freeze, old, plan, source, cfg, summarize):
    manifest_file = project/'MANIFEST.json'
    assert sha(manifest_file) == freeze['transfer_manifest_sha256']
    manifest = {item['path']:item['sha256'] for item in read(manifest_file)['files']}
    base, target = 'data/source/hopper_medium-v2.hdf5', 'data/target/hopper_morph_medium.hdf5'
    verified = {manifest_file:sha(manifest_file)}
    for name in (base, target, 'data/oracle/hopper/raw_indices.npy'):
        path = project/name
        assert project in path.resolve().parents and sha(path) == manifest[name], ('Changed input', name)
        verified[path] = manifest[name]
    ids = np.load(project/'data/oracle/hopper/raw_indices.npy', allow_pickle=False)
    assert len(ids) == 999998
    pick = np.random.RandomState(cfg['source_sample_seed']).choice(len(ids), cfg['source_sample_count'], replace=False)
    assert np.array_equal(pick, plan['canonical_position']) and np.array_equal(ids[pick], plan['source_row_id'])
    with h5py.File(project/base, 'r') as f:
        source_obs = np.asarray(f['observations'])[plan['source_row_id']]
    with h5py.File(project/target, 'r') as f:
        target_obs = np.asarray(f['observations'])
    scale = np.std(target_obs.astype(np.float64), axis=0, ddof=1) + cfg['std_epsilon']
    assert np.array_equal(np.load(old/'target_observation_scale.npy', allow_pickle=False), scale)
    distance, closest = cKDTree(target_obs/scale).query(source_obs/scale, k=1)
    assert np.array_equal(source['nearest_target_row'], closest)
    assert np.array_equal(source['nearest_distance'], distance)
    bins = []
    for idx in np.array_split(np.argsort(distance, kind='stable'), 5):
        # The original reporter used `min`, which the original summary did not produce.
        stats = dict(summarize(distance[idx]), min=float(distance[idx].min()))
        stats['e1'] = summarize(source['e1'][idx])
        bins.append(stats)
    return bins, verified


def verify(old, project, package):
    before = {old/name:sha(old/name) for name in REQUIRED}
    assert not (old/'STAGE0_COMPLETED.json').exists()
    failure = read(old/'FAILED.json')
    trace = failure['traceback']
    assert trace.rstrip().endswith("KeyError: 'min'") and 'in report' in trace
    assert 'report(out, target_summary, fail, source_summary, bins, budget)' in trace
    assert failure['later_stage_started'] is False
    assert read(old/'INPUT_PROTECTION_AFTER.json')['passed'] is True
    frozen_file = package/'FROZEN.json'
    freeze = read(frozen_file)
    start, commit = read(old/'STARTED.json'), read(old/'PRE_SIMULATOR_COMMITMENT.json')
    assert sha(frozen_file) == start['code_freeze_sha256'] == commit['freeze_sha256']
    before[frozen_file] = commit['freeze_sha256']
    for name,digest in freeze['files'].items():
        path = package/name
        assert package in path.resolve().parents and sha(path) == digest
        before[path] = digest
    assert commit['transfer_manifest_sha256'] == freeze['transfer_manifest_sha256']
    assert before[old/'config.json'] == commit['config_sha256']
    assert before[old/'sample_plan.npz'] == commit['sample_sha256']
    cfg = read(old/'config.json')
    assert cfg == read(package/'config.json'), 'Run config differs from the frozen config'
    assert cfg['stage'] == 0 and not cfg['stage1_authorized'] and not cfg['stage2_authorized']
    n, n_target, pilot, size = (cfg[key] for key in ('source_sample_count', 'target_sample_count', 'pilot_count', 'source_chunk_size'))
    assert (n,n_target) == (20000,5000)
    names = ['source_pilot'] + [f'source_{i:05d}' for i in range(pilot,n,size)]
    jobs = dict(target=n_target, source_pilot=pilot)
    jobs.update({f'source_{i:05d}': min(size,n-i) for i in range(pilot,n,size)})
    events = [json.loads(line) for line in (old/'queue_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    done, interrupted, snapshots, peak = accepted_attempts(old, events, jobs)
    before.update(snapshots)
    plan = npz(old/'sample_plan.npz')
    source, target = npz(old/'source_results.npz'), done['target'][1]
    assert np.array_equal(target['row_id'], np.arange(n_target)) and np.array_equal(target['row_id'], plan['target_row_id'])
    assert np.array_equal(source['row_id'], plan['source_row_id']) and np.array_equal(source['canonical_position'],plan['canonical_position'])
    parts = [done[name][1] for name in names]
    assert set(source) == set(parts[0]) | {'nearest_target_row','nearest_distance','canonical_position'}
    for key in parts[0]:
        assert np.array_equal(source[key], np.concatenate([p[key] for p in parts])), ('Merged rows differ', key)
    for rows in (source, target):
        assert np.all(np.isfinite(rows['action_fixed'])) and np.all(np.abs(rows['action_fixed']) <= 1)
        assert np.isfinite(rows['next_actual']).all() and np.isfinite(rows['reward_actual']).all()
        assert np.all((rows['e1'] >= 0) & (rows['e1'] <= rows['e0']))
        assert np.all((rows['iterations'] >= 0) & (rows['iterations'] <= cfg['solver']['max_outer_iterations']))
    gate = cfg['target_control_acceptance']
    assert np.all(target['original_raw_linf'] <= gate['original_observation_linf'])
    assert np.all(target['corrected_raw_linf'] <= gate['corrected_observation_linf'])
    assert np.all(target['action_delta'] <= gate['action_change_l2'])
    assert np.all(target['original_reward_error'] <= gate['original_reward_abs'])
    assert np.all(target['original_terminal_matches'])
    summarize, describe, report = report_functions(package/'run_stage0.py')
    target_summary, source_summary = describe(target), describe(source)
    assert read(old/'TARGET_CONTROL.json') == dict(passed=True, failing_rows=[], **target_summary)
    bins, verified = neighbor_bins(project, freeze, old, plan, source, cfg, summarize)
    before.update(verified)
    budget = read(old/'MEASURED_RSS_BEFORE_PARALLEL.json')
    assert budget['pilot_RSS_bytes'] == done['source_pilot'][0]['peak_RSS_bytes']
    assert 1 <= peak <= budget['workers'] <= 32
    time0 = min(datetime.fromisoformat(e['time']) for e in events if e['job'] != 'target' and e.get('state') == 'started')
    time1 = max(datetime.fromisoformat(e['time']) for e in events if e['job'] != 'target' and 'exitcode' in e)
    elapsed = (time1-time0).total_seconds()
    assert elapsed > 0
    budget.update(source_queue_wall_seconds_reconstructed=elapsed, measured_peak_source_workers=peak,
                  host_busy_percent=None, host_busy_percent_status='not persisted before report exception',
                  full_wall_hours_estimate=elapsed*999998/n/3600,
                  full_ideal_hours_estimate=source_summary['row_seconds']['mean']*999998/budget['workers']/3600)
    steps = {}
    for receipt, _ in done.values():
        for kind,count in receipt['environment_step_counts'].items():
            steps[kind] = steps.get(kind,0)+count
    results = dict(target=target_summary, source=source_summary, nearest_neighbor_bins=bins, resource_budget=budget,
                   successful_environment_step_counts=steps, interrupted_attempts=interrupted,
                   accepted_source_rows=n, accepted_target_rows=n_target, training_updates=0, policy_evaluations=0,
                   GPU_jobs=0, new_simulator_steps=0, original_run=str(old), source_workers_verified=len(names))
    return results, before, report


def render(out, result, report):
    report(out, result['target'], [], result['source'], result['nearest_neighbor_bins'])
    path = out/'REPORT_ZH.md'
    text, marker, stopping = path.read_text(encoding='utf-8').partition('\n## 停点\n')
    assert marker
    b = result['resource_budget']
    details = f'''\n## 资源与全量估算

- `nproc` 为 {b['nproc']}，cgroup 和亲和性可用 {b['effective_cpu']:.2f} 核。源单进程实测峰值 RSS {b['pilot_RSS_bytes']/GIB:.3f} GiB，控制器 RSS {b['parent_RSS_bytes']/GIB:.3f} GiB。
- 准入上限 {b['workers']}，按该上限估计合计 RSS {b['predicted_RSS_bytes']/GIB:.3f} GiB。队列实际同时在途的源任务最多 {b['measured_peak_source_workers']} 个。
- 源样本队列由首次放行与最后一次退出的日志时间重建，约 {b['source_queue_wall_seconds_reconstructed']:.2f} 秒。整机 CPU 占用在报告异常前未落盘。该时间不包括目标自检、全量缓存写入和独立验收。
- 按同等吞吐外推 999,998 行约 {b['full_wall_hours_estimate']:.3f} 小时。按单行耗时和准入并发的理想下限约 {b['full_ideal_hours_estimate']:.3f} 小时。
- 已完成任务的仿真步分类为 `{json.dumps(result['successful_environment_step_counts'])}`。中断尝试 {len(result['interrupted_attempts'])} 次。其退出记录见结果 JSON 和原队列日志。

## 报告修复记录

原运行在最近邻距离表取 `min` 时抛出 `KeyError`。本次从原始距离数组补算每档最小值，并对照原任务完成凭据、合并样本、目标自检、输入哈希以及最近邻结果。修复新增仿真步、训练更新和 GPU 作业均为 0。原输出与故障凭据保留。
'''
    path.write_text(text+details+marker+stopping, encoding='utf-8')


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    base = (Path.home()/'v13_action_fix').resolve()
    old = args.run.resolve()
    assert old.parent == base/'outputs' and old.name.startswith('stage0_')
    project = (Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104').resolve()
    out = old.with_name(old.name+'_report_repaired')
    with (old.parent/'.stage0.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not out.exists(), 'Repair output already exists: '+str(out)
        print('Verifying saved stage-0 artifacts. No simulator or training is run.', flush=True)
        results, original, report = verify(old, project, base/'stage0')
        assert all(sha(path) == digest for path,digest in original.items()), 'Original input changed during repair'
        assert 'mujoco_py' not in sys.modules and 'torch' not in sys.modules
        out.mkdir()
        render(out, results, report)
        write(out/'RESULT.json', results)
        write(out/'SOURCE_HASHES.json', {str(p):d for p,d in sorted(original.items())})
        write(out/'STAGE0_COMPLETED.json', dict(completed=True, target_control_passed=True,
              acceptance_status='AWAITING_STAGE1_CONFIRMATION', later_stage_started=False, report_recovery_only=True,
              source_count=results['accepted_source_rows'], target_count=results['accepted_target_rows'],
              new_simulator_steps=0, ended_utc=datetime.now(timezone.utc).isoformat(), original_run=str(old),
              evidence_sha256={name:sha(out/name) for name in ('REPORT_ZH.md', 'RESULT.json', 'SOURCE_HASHES.json')}))
    print('AWAITING_STAGE1_CONFIRMATION', out/'REPORT_ZH.md', flush=True)


if __name__ == '__main__':
    main()
