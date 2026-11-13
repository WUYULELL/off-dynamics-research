"""V16: tracking-feasibility screening of D4RL medium source trajectories across ODRL shift types.

For Hopper and Walker2d and each ODRL shifted simulator (friction, gravity, kinematic,
morphology; plus the unshifted simulator as a control), on the same 500 source
trajectories used in V14b/V15: B1 open-loop replay, B2 one-step closed-loop tracking,
and an instrument self-check. Everything is judged within the first H steps.

No target dataset is needed or read. No training, no optimizer, no GPU, no retry.
Reads $HOME/v14_diag/core.py, $HOME/v13_action_fix/stage0 and the two source datasets
read-only; writes only under $HOME/v16_shift/outputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
V14 = Path.home()/'v14_diag'
V13 = Path.home()/'v13_action_fix'
PROJECT = Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(V14))

import numpy as np

import core as v14_core
from core import csv_write, module, read, segments, sha, stats, utc, write
import v16_tracker

TEST = bool(v16_tracker.SHIM)
SOURCE_TASK = {'hopper': 'hopper-morph', 'walker2d': 'walker2d-morph'}   # keys of the transfer project's path table


def own_freeze():
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V16 package hash mismatch', name)


def protected(cfg):
    hashes = {}
    manifest = {str((PROJECT/e['path']).resolve()): e for e in read(PROJECT/'MANIFEST.json')['files']}
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    for robot in cfg['robots']:
        path = PROJECT/spec[SOURCE_TASK[robot]]['source_dataset']
        digest = sha(path)
        entry = manifest[str(path.resolve())]
        assert entry['sha256'] == digest and path.stat().st_size == entry['bytes'], str(path)
        hashes[str(path)] = digest
    code = V13/'stage0'
    for name, digest in read(code/'FROZEN.json')['files'].items():
        assert sha(code/name) == digest, ('Changed v13 solver package', name)
    for path in sorted(code.rglob('*')):
        if path.is_file():
            hashes[str(path)] = sha(path)
    for name, digest in read(V14/'FROZEN.json')['files'].items():
        assert sha(V14/name) == digest, ('Changed v14 package', name)
        hashes[str(V14/name)] = digest
    return hashes


def load_source(robot):
    import h5py
    spec = read(PROJECT/'cluster/paths.example.json')['tasks'][SOURCE_TASK[robot]]
    with h5py.File(PROJECT/spec['source_dataset'], 'r') as f:
        data = {k: np.asarray(f[k]) for k in ('observations', 'actions', 'next_observations', 'rewards', 'terminals')}
        if 'timeouts' in f:
            data['timeouts'] = np.asarray(f['timeouts'])
    assert all(np.isfinite(a).all() for a in data.values()) and np.isin(data['terminals'], [0, 1]).all()
    return data


def start_state(obs, nq):
    obs = np.asarray(obs, dtype=np.float64)
    return np.concatenate([[0.0], obs[:nq-1]]), obs[nq-1:].copy()


def select(robot, data, cfg):
    nq = v16_tracker.ROBOTS[robot][2]
    source_segments, _ = segments(data, source=True)
    complete = [r for r in source_segments if r['complete']]
    eligible = [r for r in complete if not (np.abs(data['observations'][r['first_row']][nq-1:]) >= 10-1e-6).any()]
    order = np.random.RandomState(cfg['source_seed'][robot]).choice(len(eligible), min(cfg['source_trajectories'], len(eligible)), replace=False)
    return [eligible[int(i)] for i in order], len(complete), len(eligible)


def run_pool(jobs, workers, heartbeat, phase):
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
    records, futures = [], []
    try:
        futures = [executor.submit(v16_tracker.task_entry, j) for j in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 100 == 0 or len(records) == len(jobs):
                heartbeat(phase, completed=len(records), total=len(jobs))
    except BaseException:
        for future in futures:
            future.cancel()
        for process in list((executor._processes or {}).values()):
            if process.is_alive():
                process.terminate()
        raise
    finally:
        executor.shutdown(wait=True)
    return sorted(records, key=lambda r: r['trajectory_id'])


def classify(mode_rows, cfg):
    TH = np.asarray([r['T_H'] for r in mode_rows])
    ratio = np.asarray([r['ratio_H'] for r in mode_rows])
    fraction = float((ratio >= .8).mean())
    residual = float(np.median([r['e1_median'] for r in mode_rows]))
    if np.median(TH) < 200 and fraction < .1:
        return '不可跟踪'
    if fraction >= .5:
        return '候选' if residual <= cfg['candidate_residual_limit'] else '存活但未跟上'
    return '不确定'


def summarize(task, records, cfg):
    out = dict(task=task['id'], robot=task['robot'], shift=task['shift'], trajectories=len(records))
    for mode in ('B1', 'B2'):
        rows = [dict(r[mode], ratio_H=r[mode]['T_H']/r['L_H']) for r in records]
        e = np.concatenate([r[mode]['e1'] for r in records]).astype(np.float64)
        out[mode] = dict(T_H=stats([r['T_H'] for r in rows]), ratio_H=stats([r['ratio_H'] for r in rows]),
            fraction_ratio_H_at_least_08=float(np.mean([r['ratio_H'] >= .8 for r in rows])),
            fraction_fell_within_H=float(np.mean([r['fell_within_H'] for r in rows])),
            boundary_fraction_mean=float(np.mean([r['boundary_fraction'] for r in rows])),
            residual_per_trajectory_median=stats([r['e1_median'] for r in rows]),
            residual_first50_per_trajectory_median=stats([r['e1_first50_median'] for r in rows]),
            residual_all_steps=stats(e), fraction_steps_residual_within_limit=float((e <= cfg['candidate_residual_limit']).mean()))
        if mode == 'B2':
            out['B2']['class_before_instrument_check'] = classify(rows, cfg)
    checks = [r['selfcheck'] for r in records if 'selfcheck' in r]
    out['selfcheck'] = dict(n=len(checks), passed=bool(checks) and all(c['passed'] for c in checks),
        failed=int(sum(not c['passed'] for c in checks)), e1_max=max(c['e1_max'] for c in checks),
        reference_length=stats([c['reference_length'] for c in checks]),
        earliest_exceed_step=min([c['first_exceed_step'] for c in checks if c['first_exceed_step']], default=None))
    out['verdict'] = out['B2']['class_before_instrument_check'] if out['selfcheck']['passed'] else '工具不可靠，不判定'
    return out


def fmt(x, digits=3):
    return '无' if x is None else ('%.*g' % (digits, x) if isinstance(x, float) else str(x))


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(fmt(c) for c in r)+'|' for r in rows])


def report(out, cfg, results, elapsed):
    H = cfg['horizon_steps']
    text = ['# V16 不同偏移类型下源状态轨迹可跟踪性筛查', '',
        '仅 CPU：新训练更新 0、优化器实例 0、GPU 作业 0。不读取任何目标数据集。源数据为 D4RL medium，每个机器人使用与 V14b/V15 相同的 500 条完整源轨迹。',
        '本次输出：`%s`。' % out, '',
        '## 预设规则（结果产生前写定，全部只看前 H = %d 步）' % H, '',
        '- T_H = min(T, H)，L_H = min(L, H)。归一化尺度为该机器人源数据当前观测逐维样本标准差（ddof=1）+ 1e-6（没有目标数据可用，因此残差数值不能与 V14b/V15 直接比较）。',
        '- 工具自检：对前 %d 条轨迹，把 B2 得到的动作在目标仿真器里连续执行（不逐步重置、观测存为 float32），得到一条真实的目标轨迹，再用同样的一步跟踪去跟它。'
        '全部到达末尾且每步残差 ≤ %s 为通过；不通过则该任务标为“工具不可靠，不判定”。' % (cfg['selfcheck_trajectories'], fmt(cfg['selfcheck_gate'])),
        '- 不可跟踪：B2 的 T_H 中位数 < 200，且 T_H/L_H ≥ 0.8 的轨迹不到 10%。',
        '- 候选：T_H/L_H ≥ 0.8 的轨迹达到 50%%，且各轨迹残差中位数的中位数 ≤ %s。' % fmt(cfg['candidate_residual_limit']),
        '- 存活但未跟上：T_H/L_H ≥ 0.8 达到 50%，但残差超过上述限值。',
        '- 其他：不确定。',
        '- 对照（无偏移）：同一程序在未修改的仿真器上运行。若对照本身不是“候选”，说明源数据在本机仿真器里就无法复现，所有偏移任务的结论都受此影响，需单独说明。',
        '- 局限：一步贪心跟踪；“候选”只表示源状态轨迹可以被复现，不表示修正后的数据对策略学习有用。', '']
    for robot in cfg['robots']:
        rows = [r for r in results if r['robot'] == robot]
        if not rows:
            continue
        text += ['## %s' % robot, '',
            table(['偏移', '判定', 'B2 T_H中位数', 'B2 T_H/L_H≥0.8', 'B2 H内摔倒', 'B2 残差(轨迹中位数的中位数)', 'B2 前50步残差', 'B2 触边', 'B1 T_H中位数', 'B1 T_H/L_H≥0.8', 'B1 残差', '自检'],
                [[r['shift'], r['verdict'], r['B2']['T_H']['median'], r['B2']['fraction_ratio_H_at_least_08'],
                  r['B2']['fraction_fell_within_H'], r['B2']['residual_per_trajectory_median']['median'],
                  r['B2']['residual_first50_per_trajectory_median']['median'], r['B2']['boundary_fraction_mean'],
                  r['B1']['T_H']['median'], r['B1']['fraction_ratio_H_at_least_08'],
                  r['B1']['residual_per_trajectory_median']['median'],
                  '通过' if r['selfcheck']['passed'] else '不通过(%d/%d，最早第%s步)' % (r['selfcheck']['failed'], r['selfcheck']['n'], fmt(r['selfcheck']['earliest_exceed_step']))]
                 for r in rows]), '']
    control = {r['robot']: r['verdict'] for r in results if r['shift'] == 'none'}
    text += ['## 对照结论', '',
        '；'.join('%s 无偏移对照：%s' % (k, v) for k, v in control.items())+'。', '',
        '## 停点', '',
        '完成 %d 个任务，总耗时 %.1f 秒。各任务的逐轨迹结果见 tasks/<任务>/TRAJECTORIES.csv。本次不提出也不启动下一项实验。' % (len(results), elapsed), '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    assert 1 <= args.workers <= 32
    if not TEST:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert Path(v14_core.__file__).resolve().parent == V14.resolve()
    own_freeze()
    cfg = read(HERE/'config.json')
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v16.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('shift_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (out/'tasks').mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V16 output:', out, flush=True)
    started = time.monotonic()
    before = None

    def heartbeat(phase, **kw):
        write(out/'STATUS.json', dict(time=utc(), phase=phase, **kw), replace=True)
        print(phase, kw, flush=True)

    try:
        heartbeat('VERIFYING_PROTECTED_INPUTS')
        before = protected(cfg)
        write(out/'PROTECTED_BEFORE.json', before)
        write(out/'config.json', dict(cfg, test_shim=TEST))
        if TEST:
            vendor, workers = Path(v16_tracker.SHIM), min(args.workers, 4)
        else:
            import limits
            runtime = limits.limits_snapshot()
            workers = min(args.workers, max(1, int(runtime['effective_visible_cpu_cores'])-4))
            legacy = module(V13/'stage0/legacy_replay.py', 'v16_binding_loader')
            binding, binding_before, private = legacy.prepare_private_binding(out)
            vendor = private.parent
            import mujoco_py
            assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        old = read(V13/'stage0/config.json')
        results = []
        for robot in cfg['robots']:
            data = load_source(robot)
            nq = v16_tracker.ROBOTS[robot][2]
            scale = data['observations'].astype(np.float64).std(axis=0, ddof=1)+old['std_epsilon']
            selected, complete, eligible = select(robot, data, cfg)
            outside = int(np.count_nonzero(np.abs(data['actions']) > 1))
            write(out/('PLAN_%s.json' % robot), dict(robot=robot, source_rows=int(len(data['observations'])), source_complete=complete,
                source_eligible=eligible, source_seed=cfg['source_seed'][robot], selected=selected, scale=scale.tolist(),
                source_actions_outside_box=outside, solver=old['solver']))
            for task in [t for t in cfg['tasks'] if t['robot'] == robot]:
                tdir = out/'tasks'/task['id']
                tdir.mkdir()
                init = (task['id'], robot, str(HERE/'assets'/task['xml']), str(vendor), scale, old['solver'], old['action_boundary_abs_tolerance'])
                jobs = []
                for i, r in enumerate(selected):
                    a, b = r['first_row'], r['last_row']+1
                    p, v = start_state(data['observations'][a], nq)
                    jobs.append(dict(init=init, output=str(tdir), trajectory_id=r['trajectory_id'], length=r['length'],
                        qpos=p, qvel=v, actions=data['actions'][a:b][:cfg['horizon_steps']],
                        next_observations=data['next_observations'][a:b][:cfg['horizon_steps']],
                        H=cfg['horizon_steps'], selfcheck=i < cfg['selfcheck_trajectories'], gate=cfg['selfcheck_gate']))
                records = run_pool(jobs, workers, heartbeat, task['id'])
                assert len(records) == len(selected)
                res = summarize(task, records, cfg)
                csv_write(tdir/'TRAJECTORIES.csv', [dict(trajectory_id=r['trajectory_id'], L=r['L'], L_H=r['L_H'],
                    **{'%s_%s' % (m, k): r[m][k] for m in ('B1', 'B2') for k in ('T_H', 'fell_within_H', 'reward_H', 'boundary_fraction', 'e1_median', 'e1_p90', 'e1_first50_median')},
                    selfcheck_passed=r.get('selfcheck', {}).get('passed', ''), selfcheck_e1_max=r.get('selfcheck', {}).get('e1_max', ''),
                    selfcheck_first_exceed_step=r.get('selfcheck', {}).get('first_exceed_step', '')) for r in records])
                np.savez_compressed(tdir/'B2_residuals.npz', trajectory_id=np.asarray([r['trajectory_id'] for r in records]),
                    length=np.asarray([len(r['B2']['e1']) for r in records]), e1=np.concatenate([r['B2']['e1'] for r in records]))
                write(tdir/'RESULT.json', res)
                results.append(res)
                print(task['id'], res['verdict'], flush=True)
        heartbeat('VERIFYING_PROTECTED_INPUTS_AFTER')
        after = protected(cfg)
        assert before == after, 'Protected original inputs changed'
        if not TEST:
            assert legacy.hash_tree(binding) == binding_before, 'Installed mujoco-py binding changed'
        write(out/'PROTECTED_AFTER.json', dict(passed=True, files=len(after)))
        elapsed = time.monotonic()-started
        report(out, cfg, results, elapsed)
        write(out/'RESULT.json', dict(tasks=results, new_training_updates=0, GPU_jobs=0))
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW', tasks=len(results),
            verdicts={r['task']: r['verdict'] for r in results}, new_training_updates=0, optimizer_instances=0, GPU_jobs=0,
            elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V16 DONE:', out/'REPORT_ZH.md', flush=True)
    except BaseException:
        failed = dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False, new_training_updates=0, GPU_jobs=0)
        if before:
            failed['protected_input_differences'] = [p for p, h in before.items() if not Path(p).is_file() or sha(p) != h]
        write(out/'FAILED.json', failed)
        print(failed['traceback'], flush=True)
        raise


if __name__ == '__main__':
    main()
