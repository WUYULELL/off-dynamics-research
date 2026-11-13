"""V15: tracking-feasibility screening of source state trajectories. CPU only.

For each task: B0 self-check on the target data's own trajectories, then B1
(open-loop replay of source actions) and B2 (one-step closed-loop tracking of
source next observations) on a fixed sample of complete source trajectories.
Hopper runs first as a regression test against the finished v14b run; the new
task (Walker2d) is started only if the regression reproduces v14b.

No training, no optimizer, no GPU, no retry. Reads $HOME/v14_diag (core.py,
limits.py), $HOME/v13_action_fix/stage0 (solver, restoration) and the transfer
project read-only; writes only under $HOME/v15_screen/outputs.
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
V14B = Path.home()/'v14b_track'
V13 = Path.home()/'v13_action_fix'
PROJECT = Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(V14))            # frozen core.py / limits.py; this package has no module of the same name

import numpy as np

import core as v14_core
from core import (A_LABELS, bins, csv_write, histogram_svg, module, read, segments,
                  sha, stats, utc, write)
import v15_tracker

TEST = bool(v15_tracker.SHIM)


def own_freeze():
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V15 package hash mismatch', name)


def protected(cfg):
    """Hashes of every original file this run reads; compared before and after."""
    hashes = {}
    manifest = {str((PROJECT/e['path']).resolve()): e for e in read(PROJECT/'MANIFEST.json')['files']}
    spec = read(PROJECT/'cluster/paths.example.json')['tasks']
    for task in cfg['tasks']:
        for key in ('xml', 'source_dataset', 'target_dataset'):
            relative = spec[task][key]
            path = PROJECT/relative
            digest = sha(path)
            entry = manifest[str(path.resolve())]
            assert entry['sha256'] == digest and path.stat().st_size == entry['bytes'], relative
            hashes[str(path)] = digest
    for path in [PROJECT/'MANIFEST.json', PROJECT/'cluster/paths.example.json']:
        hashes[str(path)] = sha(path)
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


def load(task):
    import h5py
    spec = read(PROJECT/'cluster/paths.example.json')['tasks'][task]
    data = {}
    for domain in ('source', 'target'):
        with h5py.File(PROJECT/spec[domain+'_dataset'], 'r') as f:
            data[domain] = {k: np.asarray(f[k]) for k in ('observations', 'actions', 'next_observations', 'rewards', 'terminals')}
            if 'timeouts' in f:
                data[domain]['timeouts'] = np.asarray(f['timeouts'])
        assert all(np.isfinite(a).all() for a in data[domain].values())
        assert np.isin(data[domain]['terminals'], [0, 1]).all()
    return spec, data


def start_state(obs, nq):
    """Frozen v9 inversion of gym _get_obs: qpos=[0, obs[:nq-1]], qvel=obs[nq-1:]."""
    obs = np.asarray(obs, dtype=np.float64)
    return np.concatenate([[0.0], obs[:nq-1]]), obs[nq-1:].copy()


def clipped(obs, nq):
    return bool((np.abs(np.asarray(obs)[nq-1:]) >= 10-1e-6).any())


def job(init, out, record, mode, qpos, qvel, data, spec):
    start, end = record['first_row'], record['last_row']+1
    assert record['length'] <= 1000
    return dict(init=init, record=record, mode=mode, output=str(out), qpos=qpos, qvel=qvel,
        actions=data['actions'][start:end], next_observations=data['next_observations'][start:end],
        terminals=data['terminals'][start:end].reshape(-1).astype(bool),
        score_min=spec['score_min'], score_max=spec['score_max'])


def run_pool(jobs, workers, heartbeat, phase):
    if not jobs:
        return []
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
    records, futures = [], []
    try:
        futures = [executor.submit(v15_tracker.task_entry, j) for j in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 50 == 0 or len(records) == len(jobs):
                heartbeat(phase, completed=len(records), total=len(jobs))
    except BaseException:
        for future in futures:
            future.cancel()
        for process in list((executor._processes or {}).values()):   # only this executor's own children
            if process.is_alive():
                process.terminate()
        raise
    finally:
        executor.shutdown(wait=True)
    return sorted(records, key=lambda r: (r['mode'], r['trajectory_id']))


def interpret(lengths, ratios):
    """Same thresholds as v14 interpret_b; wording without the Hopper-specific remark."""
    fraction = float((np.asarray(ratios) >= .8).mean())
    if np.median(lengths) < 200 and fraction < .1:
        return '不可跟踪（仅限一步贪心跟踪）'
    if fraction >= .5:
        return '可跟踪（按存活步数；是否跟得上还需结合跟踪残差）'
    return '不确定'


def summarize(out, records, reference):
    groups, bybin, hist_rows, counts, simple = {}, [], [], {}, []
    for mode in ('B0', 'B1', 'B2'):
        rows = [r for r in records if r['mode'] == mode]
        if not rows:
            continue
        lengths = np.asarray([r['T'] for r in rows])
        ratios = np.asarray([r['T_over_L'] for r in rows])
        groups[mode] = dict(episodes=len(rows), survival_steps=stats(lengths), T_over_L=stats(ratios),
            fraction_T_under_200=float((lengths < 200).mean()),
            fraction_T_over_L_at_least_08=float((ratios >= .8).mean()),
            normalized_score=stats([r['normalized_score'] for r in rows]),
            boundary_fraction=stats([r['boundary_fraction'] for r in rows]),
            physical_terminal_count=int(sum(r['physical_terminal'] for r in rows)),
            reference_length=stats([r['L'] for r in rows]))
        if mode == 'B2':
            groups[mode]['interpretation'] = interpret(lengths, ratios)
        errors, steps = [], []
        hist = np.zeros(50, dtype=np.int64)
        for r in rows:
            simple.append({k: r[k] for k in ('mode', 'trajectory_id', 'first_row', 'last_row', 'L', 'T', 'T_over_L',
                'physical_terminal', 'reached_record_end', 'boundary_fraction', 'raw_return', 'normalized_score')})
            with np.load(out/('%s_trajectory%05d.npz' % (mode, r['trajectory_id'])), allow_pickle=False) as f:
                errors.append(f['e1'])
                steps.append(f['step'])
            if r['physical_terminal']:
                hist[(r['T']-1)//20] += 1
        e, t = np.concatenate(errors), np.concatenate(steps)
        groups[mode]['residual_all_steps'] = stats(e)
        groups[mode]['fraction_steps_residual_within_reference'] = float((e <= reference).mean())
        for j, label in enumerate(A_LABELS):
            bybin.append(dict(mode=mode, step_bin=label, **stats(e[bins(t) == j])))
        counts[mode] = hist.tolist()
        hist_rows += [dict(mode=mode, executed_step_first=i*20+1, executed_step_last=(i+1)*20, count=int(c)) for i, c in enumerate(hist)]
    csv_write(out/'B_trajectories.csv', simple)
    csv_write(out/'B_residual_step_bins.csv', bybin)
    csv_write(out/'B_physical_termination_histogram.csv', hist_rows)
    histogram_svg(out/'B_physical_termination_histogram.svg', counts)
    return dict(groups=groups, residual_bins=bybin,
        simulator_steps=int(sum(r['simulator_steps'] for r in records)+len({r['worker_pid'] for r in records})))


def gate(records, limit):
    rows = []
    for r in records:
        ok = bool(r['reached_record_end'] and r['terminal_mismatches'] == 0 and r['e1']['max'] <= limit)
        rows.append(dict(trajectory_id=r['trajectory_id'], L=r['L'], T=r['T'], reached_record_end=r['reached_record_end'],
            terminal_mismatches=r['terminal_mismatches'], e1_max=r['e1']['max'], e1_median=r['e1']['median'],
            raw_linf_max=r['raw_linf']['max'], action_delta_max=r['action_delta']['max'], passed=ok))
    return dict(passed=bool(rows) and all(x['passed'] for x in rows), normalized_residual_limit=limit, trajectories=rows)


def regression(records, cfg):
    """Hopper B1/B2 from this generic tracker must reproduce the finished v14b run."""
    reference = read(V14B/'outputs'/cfg['regression_reference']/'B/B_SOURCE_TRACKS.json')
    old = {(r['mode'], r['trajectory_id']): r for r in reference}
    new = {(r['mode'], r['trajectory_id']): r for r in records if r['mode'] in ('B1', 'B2')}
    assert set(old) == set(new) and len(old) > 0, 'Regression sets differ'
    bad = []
    for key in sorted(old):
        a, b = old[key], new[key]
        same = a['T'] == b['T'] and abs(a['raw_return']-b['raw_return']) <= cfg['regression_return_tolerance']*max(1., abs(a['raw_return']))
        if not same:
            bad.append(dict(mode=key[0], trajectory_id=key[1], T_v14b=a['T'], T_v15=b['T'],
                            return_v14b=a['raw_return'], return_v15=b['raw_return']))
    return dict(passed=not bad, compared=len(old), mismatches=len(bad), first_mismatches=bad[:20],
        reference=str(V14B/'outputs'/cfg['regression_reference']),
        rule='identical T and relative return difference <= %g for every B1/B2 record' % cfg['regression_return_tolerance'])


def run_task(task, cfg, out, vendor, workers, heartbeat):
    tdir = out/task
    tdir.mkdir()
    spec, data = load(task)
    nq = v15_tracker.ENVIRONMENTS[task][2]
    old = read(V13/'stage0/config.json')
    scale = data['target']['observations'].astype(np.float64).std(axis=0, ddof=1)+old['std_epsilon']
    init = (task, str(PROJECT/spec['xml']), str(vendor), scale, old['solver'], old['action_boundary_abs_tolerance'])
    target_segments, _ = segments(data['target'])
    source_segments, _ = segments(data['source'], source=True)
    complete = [r for r in source_segments if r['complete']]
    eligible = [r for r in complete if not clipped(data['source']['observations'][r['first_row']], nq)]
    seed = cfg['source_seed'][task]
    order = np.random.RandomState(seed).choice(len(eligible), min(cfg['source_trajectories'], len(eligible)), replace=False)
    selected = [eligible[int(i)] for i in order]
    out_of_box = int(np.count_nonzero(np.abs(data['source']['actions']) > 1))
    plan = dict(task=task, xml=spec['xml'], source_dataset=spec['source_dataset'], target_dataset=spec['target_dataset'],
        target_scale=scale.tolist(), solver=old['solver'], source_seed=seed,
        source_rows=int(len(data['source']['observations'])), target_rows=int(len(data['target']['observations'])),
        target_segments=target_segments, source_complete=len(complete), source_eligible_unclipped_start=len(eligible),
        selected_source_trajectories=selected, source_actions_outside_box=out_of_box)
    hopper_states = None
    if task == 'hopper-morph':
        # Identical inputs to v14b: same selection, same recorded initial physical states.
        earlier = read(V14/'outputs'/cfg['earlier_v14_run']/'config.json')
        assert selected == earlier['selected_B_source_trajectories'], 'Hopper selection differs from the frozen v14 plan'
        assert scale.tolist() == earlier['target_scale']
        qpos = np.load(V13/'stage1/restoration/qpos.npy', mmap_mode='r', allow_pickle=False)
        qvel = np.load(V13/'stage1/restoration/qvel.npy', mmap_mode='r', allow_pickle=False)
        hopper_states = (qpos, qvel)
        difference = 0.
        for r in selected:
            p, v = start_state(data['source']['observations'][r['first_row']], nq)
            difference = max(difference, float(np.max(np.abs(np.r_[p-qpos[r['first_row']], v-qvel[r['first_row']]]))))
        plan['hopper_recorded_vs_observation_start_state_max_abs'] = difference
    write(tdir/'PLAN.json', plan)

    b0_jobs, unavailable = [], []
    for r in target_segments:
        first = data['target']['observations'][r['first_row']]
        if clipped(first, nq):
            unavailable.append(r['trajectory_id'])
            continue
        p, v = start_state(first, nq)
        b0_jobs.append(job(init, tdir, r, 'B0', p, v, data['target'], spec))
    heartbeat(task+'_B0', total=len(b0_jobs))
    b0_records = run_pool(b0_jobs, min(workers, max(1, len(b0_jobs))), heartbeat, task+'_B0')
    b0 = gate(b0_records, cfg['B0_normalized_residual_limit'])
    b0['segments_without_recoverable_start'] = unavailable
    b0['passed'] = bool(b0['passed'] and not unavailable)
    write(tdir/'B0_RESULT.json', b0)
    result = dict(task=task, plan=plan, B0=b0, B1_B2_started=False)
    records = list(b0_records)
    if b0['passed']:
        jobs = []
        for r in selected:
            if hopper_states is not None:
                p, v = np.asarray(hopper_states[0][r['first_row']]), np.asarray(hopper_states[1][r['first_row']])
            else:
                p, v = start_state(data['source']['observations'][r['first_row']], nq)
            jobs += [job(init, tdir, r, mode, p, v, data['source'], spec) for mode in ('B1', 'B2')]
        heartbeat(task+'_B1_B2', total=len(jobs))
        source_records = run_pool(jobs, workers, heartbeat, task+'_B1_B2')
        assert len(source_records) == 2*len(selected)
        write(tdir/'B_SOURCE_TRACKS.json', source_records)
        records += source_records
        result['B1_B2_started'] = True
        if task == 'hopper-morph':
            result['regression'] = regression(source_records, cfg)
    result.update(summarize(tdir, records, cfg['residual_reference']))
    write(tdir/'RESULT.json', result)
    return result


def fmt(x, digits=4):
    return '无' if x is None else ('%.*g' % (digits, x))


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(str(c) for c in r)+'|' for r in rows])


def report(out, cfg, results, stopped, elapsed):
    text = ['# V15 源状态轨迹可跟踪性筛查报告', '',
        '仅 CPU：新训练更新 0、优化器实例 0、GPU 作业 0。物理约定、观测反演和求解器为 v9/v13 冻结定义（只读加载），跟踪循环按 v14 的 B 诊断逐步对应，仅去掉了只适用于 Hopper 的假设。',
        '入口：`bash ~/v15_screen/run_in_tmux.sh`；本次输出：`%s`。' % out, '',
        '## 预设规则（结果产生前写定）', '',
        '- B0 自检：目标数据的每段轨迹都跟踪到记录末尾、终止标签一致，且每步归一化残差 ≤ %s。不通过则该任务不运行 B1、B2。' % fmt(cfg['B0_normalized_residual_limit']),
        '- 回归检验：Hopper 的 B1、B2 须与 v14b 逐轨迹一致（存活步数相同，累计回报相对差 ≤ %g）。不一致则不启动新任务。' % cfg['regression_return_tolerance'],
        '- B2 解读：存活步数中位数 < 200 且 T/L ≥ 0.8 的轨迹不到 10% 为“不可跟踪”；T/L ≥ 0.8 达到 50% 以上为“可跟踪”；其他为“不确定”。',
        '- 局限：B2 是一步贪心跟踪，结论仅限这种跟踪方式。', '']
    for res in results:
        task, plan, b0 = res['task'], res['plan'], res['B0']
        text += ['## %s' % task, '',
            '源数据 %d 行，目标数据 %d 行；完整源轨迹 %d 条，其中起点可直接恢复的 %d 条，固定种子 %d 选取 %d 条。源动作超出 [-1,1] 的元素 %d 个。'
            % (plan['source_rows'], plan['target_rows'], plan['source_complete'], plan['source_eligible_unclipped_start'],
               plan['source_seed'], len(plan['selected_source_trajectories']), plan['source_actions_outside_box']), '',
            '### B0 自检', '',
            table(['轨迹', 'L', 'T', '到达末尾', '终止不一致', '归一化残差最大值', '归一化残差中位数', '原始误差最大值', '动作差最大值', '结果'],
                [[r['trajectory_id'], r['L'], r['T'], r['reached_record_end'], r['terminal_mismatches'], fmt(r['e1_max']),
                  fmt(r['e1_median']), fmt(r['raw_linf_max']), fmt(r['action_delta_max']), '通过' if r['passed'] else '不通过']
                 for r in b0['trajectories']]), '',
            'B0 结论：%s。%s' % ('通过' if b0['passed'] else '不通过，B1/B2 未启动',
                '' if not b0['segments_without_recoverable_start'] else '起点无法直接恢复的目标段：%s。' % b0['segments_without_recoverable_start']), '']
        if 'regression' in res:
            g = res['regression']
            text += ['### 与 v14b 的回归检验', '',
                '比较 %d 条记录，不一致 %d 条。结论：%s。' % (g['compared'], g['mismatches'], '通过' if g['passed'] else '不通过'), '']
            if 'hopper_recorded_vs_observation_start_state_max_abs' in plan:
                text += ['所选轨迹起点处，v13 记录的物理状态与由观测直接反演的状态最大绝对差：%s。' % fmt(plan['hopper_recorded_vs_observation_start_state_max_abs']), '']
        if res['B1_B2_started']:
            g = res['groups']
            text += ['### B1 开环重放与 B2 闭环跟踪', '',
                table(['模式', '轨迹数', '源轨迹长度中位数', 'T均值', 'T中位数', 'T P10', 'T P90', 'T/L中位数', 'T<200比例', 'T/L≥0.8比例', '物理终止条数', '触边比例均值'],
                    [[m, g[m]['episodes'], fmt(g[m]['reference_length']['median']), fmt(g[m]['survival_steps']['mean']),
                      fmt(g[m]['survival_steps']['median']), fmt(g[m]['survival_steps']['p10']), fmt(g[m]['survival_steps']['p90']),
                      fmt(g[m]['T_over_L']['median']), fmt(g[m]['fraction_T_under_200']), fmt(g[m]['fraction_T_over_L_at_least_08']),
                      g[m]['physical_terminal_count'], fmt(g[m]['boundary_fraction']['mean'])] for m in ('B1', 'B2')]), '',
                '归一化分数（各轨迹按自身记录长度结束，不是正式策略回报比较）：', '',
                table(['模式', '均值', '中位数', 'P10', 'P90', '最小', '最大'],
                    [[m]+[fmt(g[m]['normalized_score'][k]) for k in ('mean', 'median', 'p10', 'p90', 'min', 'max')] for m in ('B0', 'B1', 'B2')]), '',
                '跟踪残差按步数分档（归一化）：', '',
                table(['模式', '步数档', 'n', '均值', '中位数', 'P90', '最大'],
                    [[r['mode'], r['step_bin'], r['n'], fmt(r['mean']), fmt(r['median']), fmt(r['p90']), fmt(r['max'])]
                     for r in res['residual_bins'] if r['mode'] in ('B1', 'B2')]), '',
                '全部已执行步的归一化跟踪残差（描述性；%s 是 v13 在 Hopper 上用过的筛选阈值，这里只作参照，不参与判定）：' % fmt(cfg['residual_reference']), '',
                table(['模式', '步数', '均值', '中位数', 'P90', '残差≤参照值的步数比例'],
                    [[m, g[m]['residual_all_steps']['n'], fmt(g[m]['residual_all_steps']['mean']), fmt(g[m]['residual_all_steps']['median']),
                      fmt(g[m]['residual_all_steps']['p90']), fmt(g[m]['fraction_steps_residual_within_reference'])] for m in ('B1', 'B2')]), '',
                '**B2 判定：%s**' % g['B2']['interpretation'], '',
                '注意：判定只看存活步数。机器人没摔倒但残差很大，说明它并没有跟上源轨迹，这种情况不能当作“源状态可以复现”。', '']
    text += ['## 停点', '', stopped, '',
        '总耗时 %.1f 秒。读取前后的原始输入哈希核对见 PROTECTED_BEFORE.json 与 PROTECTED_AFTER.json。' % elapsed,
        '逐轨迹结果见各任务目录下的 B_trajectories.csv，摔倒步数直方图见 B_physical_termination_histogram.svg。',
        '本次不提出也不启动下一项实验。任何训练、V13 阶段 2 和其他任务均未启动。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    assert 1 <= args.workers <= 24
    if not TEST:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert Path(v14_core.__file__).resolve().parent == V14.resolve(), 'core must come from the frozen v14_diag'
    own_freeze()
    cfg = read(HERE/'config.json')
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v15.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('screen_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V15 output:', out, flush=True)
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
            vendor = Path(v15_tracker.SHIM)
            workers = min(args.workers, 4)
        else:
            import limits
            runtime = limits.limits_snapshot()
            workers = min(args.workers, max(1, int(runtime['effective_visible_cpu_cores'])-2))
            legacy = module(V13/'stage0/legacy_replay.py', 'v15_binding_loader')
            binding, binding_before, private = legacy.prepare_private_binding(out)
            vendor = private.parent
            import mujoco_py                     # build the private copy once, before workers start
            assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        results, stopped = [], '全部任务按计划完成。'
        for task in cfg['tasks']:
            res = run_task(task, cfg, out, vendor, workers, heartbeat)
            results.append(res)
            if not res['B0']['passed']:
                stopped = '%s 的 B0 自检不通过，该任务的 B1/B2 未运行。' % task
                if task == 'hopper-morph':
                    stopped += ' Hopper 回归未完成，后续任务未启动。'
                    break
                continue
            if 'regression' in res and not res['regression']['passed']:
                stopped = 'Hopper 回归检验与 v14b 不一致，后续任务未启动；这是技术核验失败，需要先检查代码。'
                break
        heartbeat('VERIFYING_PROTECTED_INPUTS_AFTER')
        after = protected(cfg)
        assert before == after, 'Protected original inputs changed'
        if not TEST:
            assert legacy.hash_tree(binding) == binding_before, 'Installed mujoco-py binding changed'
        write(out/'PROTECTED_AFTER.json', dict(passed=True, files=len(after)))
        elapsed = time.monotonic()-started
        report(out, cfg, results, stopped, elapsed)
        compact = [{k: v for k, v in r.items() if k != 'plan'} for r in results]
        write(out/'RESULT.json', dict(tasks=compact, stopped=stopped, new_training_updates=0, GPU_jobs=0))
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW', stopped=stopped,
            tasks_run=[r['task'] for r in results],
            B0_passed={r['task']: r['B0']['passed'] for r in results},
            regression_passed=next((r['regression']['passed'] for r in results if 'regression' in r), None),
            B2_interpretation={r['task']: r['groups']['B2']['interpretation'] for r in results if r['B1_B2_started']},
            new_training_updates=0, optimizer_instances=0, GPU_jobs=0,
            environment_steps=int(sum(r['simulator_steps'] for r in results)), elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V15 DONE:', out/'REPORT_ZH.md', flush=True)
    except BaseException:
        failed = dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False, new_training_updates=0, GPU_jobs=0)
        if before:
            failed['protected_input_differences'] = [p for p, h in before.items() if not Path(p).is_file() or sha(p) != h]
        write(out/'FAILED.json', failed)
        print(failed['traceback'], flush=True)
        raise


if __name__ == '__main__':
    main()
