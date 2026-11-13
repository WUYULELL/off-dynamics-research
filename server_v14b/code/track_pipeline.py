"""V14b: re-judge the existing B0 records with the revised gate, then run source B1/B2.

CPU only. No training, no optimizer, no GPU, no retry.
All numerical work (state restoration, solver, tracking, summary) is done by the
frozen v14_diag modules, imported read-only from $HOME/v14_diag. This file adds
only: the revised B0 gate applied to the already saved B0 arrays, the controller,
and the report. Nothing under v14_diag, v13_action_fix or the transfer project is
written to.
"""
import argparse
import fcntl
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
V14 = Path.home()/'v14_diag'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
# Frozen v14 modules first on the path; this package has no module with the same name.
sys.path.insert(0, str(V14))

import numpy as np

from core import read, sha, stats, utc, write, A_LABELS   # v14_diag/core.py
import core as v14_core


def own_freeze():
    for name, digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name) == digest, ('V14b package hash mismatch', name)


def rejudge(prior, limit):
    """Apply the revised gate to the saved per-step arrays of the earlier B0 run.

    Pass: every target segment reached its recorded end, every terminal label
    equals the recorded one, and every step has normalized residual e1 <= limit.
    No simulation is executed here.
    """
    old = read(prior/'B/B0_RESULT.json')
    done = read(prior/'COMPLETED.json')
    assert old['passed'] is False and old['source_tracking_started'] is False
    assert done['B0_passed'] is False and done['B1_B2_started'] is False
    rows = []
    for t in old['trajectories']:
        assert t['mode'] == 'B0'
        name = 'B0_trajectory%05d' % t['trajectory_id']
        path = prior/'B'/(name+'.npz')
        assert sha(path) == t['raw_sha256'], ('Earlier B0 array changed', name)
        with np.load(path, allow_pickle=False) as f:
            e1 = np.asarray(f['e1'], dtype=np.float64)
            raw = np.asarray(f['raw_linf'], dtype=np.float64)
            terminal = np.asarray(f['terminal']).astype(bool)
            reference = np.asarray(f['reference_terminal']).astype(bool)
        assert len(e1) == len(raw) == len(terminal) == len(reference) == t['T'] > 0
        assert np.isfinite(e1).all() and np.isfinite(raw).all()
        assert abs(float(e1.max())-t['e1']['max']) <= 1e-15, 'Saved array differs from earlier summary'
        over = np.flatnonzero(e1 > limit)
        mismatch = np.flatnonzero(terminal != reference)
        reached = len(e1) == t['L']
        rows.append(dict(trajectory_id=t['trajectory_id'], first_row=t['first_row'], last_row=t['last_row'],
            L=t['L'], T=int(len(e1)), reached_record_end=bool(reached),
            terminal_mismatch_steps=(mismatch+1).tolist(),
            first_exceed_step=None if not len(over) else int(over[0])+1,
            steps_over_limit=int(len(over)), e1=stats(e1), raw_linf=stats(raw),
            passed=bool(reached and not len(over) and not len(mismatch)),
            earlier_gate_passed=bool(t['self_check']['passed']), array_sha256=t['raw_sha256']))
    return dict(passed=all(r['passed'] for r in rows), trajectories=rows,
        normalized_residual_limit=limit, earlier_raw_linf_limit=old['raw_linf_limit'],
        earlier_result_passed=old['passed'], earlier_result_sha256=sha(prior/'B/B0_RESULT.json'),
        earlier_run=str(prior), new_simulator_steps=0,
        note='Gate revised by the user on 2026-11-12 after the earlier B0 result was known. '
             'The earlier result (not passed under raw L-infinity <= 2e-6) is kept unchanged.')


def fmt(x, digits=4):
    return '无' if x is None else ('%.*g' % (digits, x))


def table(header, rows):
    lines = ['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
    lines += ['|'+'|'.join(str(c) for c in r)+'|' for r in rows]
    return '\n'.join(lines)


def report(out, cfg, gate, b, resources, count, workers):
    text = ['# V14b Hopper 源轨迹闭环跟踪诊断报告', '',
        '仅 CPU：新训练更新 0、优化器实例 0、GPU 作业 0。数值计算全部调用只读的 v14_diag 冻结模块，本包只新增 B0 重判、控制器和报告。',
        '入口：`bash ~/v14b_track/run_in_tmux.sh`；本次输出：`%s`。' % out, '',
        '## B0 重判（不重新仿真）', '',
        '门槛修订由用户于 2026-11-12 在看到原 B0 结果之后作出。原门槛为原始观测最大绝对误差 ≤ 2e-6（v13 单步回放验收标准），'
        '原结果“不通过”原样保留在 `%s`。' % gate['earlier_run'],
        '新条件：每段目标轨迹跟踪到记录末尾、终止标签逐步一致，且每一步归一化跟踪残差 ≤ %s（v13 筛选阈值 0.0840 的 1%%）。' % fmt(gate['normalized_residual_limit']),
        '判定直接读取原 B0 逐步数组（已核对 SHA-256），新增仿真步 0。', '',
        table(['轨迹', 'L', 'T', '到达末尾', '终止不一致步数', '超限步数', '归一化残差最大值', '归一化残差中位数', '原始误差最大值', '新条件', '原条件'],
            [[r['trajectory_id'], r['L'], r['T'], r['reached_record_end'], len(r['terminal_mismatch_steps']), r['steps_over_limit'],
              fmt(r['e1']['max']), fmt(r['e1']['median']), fmt(r['raw_linf']['max']),
              '通过' if r['passed'] else '不通过', '通过' if r['earlier_gate_passed'] else '不通过'] for r in gate['trajectories']]), '',
        'B0 重判结论：%s。' % ('通过' if gate['passed'] else '不通过，B1/B2 未启动'), '']
    if b is not None:
        g = b['groups']
        text += ['## B1 开环重放与 B2 闭环跟踪', '',
            '源轨迹：可识别的完整轨迹 %d 条，按固定种子无放回选取 %d 条，选取结果与 v14 首次运行冻结的清单逐项一致。实际并行工作进程 %d。' % (count, b['source_episode_count'], workers),
            'B1 从源轨迹初始状态出发顺次执行源动作。B2 每步从仿真器当前实际状态出发，用 v13 求解器求最接近源轨迹下一观测的有界动作并执行；仅起点使用源状态。', '',
            table(['模式', '轨迹数', 'T均值', 'T中位数', 'T P10', 'T P90', 'T/L均值', 'T/L中位数', 'T<200比例', 'T/L≥0.8比例', '物理终止条数'],
                [[m, g[m]['episodes'], fmt(g[m]['survival_steps']['mean']), fmt(g[m]['survival_steps']['median']),
                  fmt(g[m]['survival_steps']['p10']), fmt(g[m]['survival_steps']['p90']),
                  fmt(g[m]['T_over_L']['mean']), fmt(g[m]['T_over_L']['median']),
                  fmt(g[m]['fraction_T_under_200']), fmt(g[m]['fraction_T_over_L_at_least_08']),
                  g[m]['physical_terminal_count']] for m in ('B1', 'B2')]), '',
            '### 归一化分数（各轨迹按自身记录长度结束，不是正式策略回报比较）', '',
            table(['模式', '均值', '中位数', 'P10', 'P90', '最小', '最大'],
                [[m]+[fmt(g[m]['normalized_score'][k]) for k in ('mean', 'median', 'p10', 'p90', 'min', 'max')] for m in ('B1', 'B2')]), '',
            '### 跟踪残差按步数分档（归一化）', '',
            table(['模式', '步数档', 'n', '均值', '中位数', 'P90', '最大'],
                [[r['mode'], r['step_bin'], r['n'], fmt(r['mean']), fmt(r['median']), fmt(r['p90']), fmt(r['max'])]
                 for r in b['residual_bins'] if r['mode'] in ('B1', 'B2')]), '',
            '### 预设解读（针对 B2，结果产生前写定，未修改）', '',
            '- 不可跟踪：存活步数中位数 < 200，且 T/L ≥ 0.8 的轨迹不到 10%。',
            '- 可跟踪：T/L ≥ 0.8 的轨迹达到 50% 以上。',
            '- 其他：不确定。', '',
            '**本次判定：%s**' % g['B2']['interpretation'], '',
            '局限：B2 是一步贪心跟踪，不能排除带前瞻的控制器跟踪得更好；“不可跟踪”的结论仅限这种跟踪方式。',
            '摔倒步数直方图（每 20 步一档）：`B/B_physical_termination_histogram.svg` 及同名 CSV；逐轨迹结果：`B/B_trajectories.csv`；逐步原始值：`B/*.npz`。', '']
    text += ['## 资源与保护核对', '',
        '总耗时 %.1f 秒；B 仿真步（含试算）%s。' % (resources['elapsed_seconds'], 'n/a' if b is None else b['simulator_steps']),
        '读取前后对 v13 模型/缓存/代码、迁移项目与环境元数据做了哈希核对，见 INPUT_HASHES_BEFORE.json 与 INPUT_PROTECTION_AFTER.json。',
        '本次不提出也不启动下一项实验。V13 阶段 2、其他任务和任何训练均未启动。', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    assert 1 <= args.workers <= 24
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
    assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert Path(v14_core.__file__).resolve().parent == V14.resolve(), 'core must come from the frozen v14_diag'
    own_freeze()
    from context import Inputs, inventory, verify_code   # v14_diag/context.py
    import limits                                         # v14_diag/limits.py
    verify_code()                                         # v14_diag files equal their own FROZEN.json
    cfg = read(HERE/'config.json')
    prior = V14/'outputs'/cfg['earlier_v14_run']
    assert prior.is_dir(), 'Earlier v14 run not found: '+str(prior)
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v14b.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('track_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (out/'B').mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V14b output:', out, flush=True)
    started = time.monotonic()
    before = None

    def heartbeat(phase, **kw):
        write(out/'STATUS.json', dict(time=utc(), phase=phase, **kw), replace=True)
        print(phase, kw, flush=True)

    try:
        heartbeat('REJUDGING_B0_FROM_SAVED_ARRAYS')
        gate = rejudge(prior, cfg['B0_normalized_residual_limit'])
        write(out/'B0_REJUDGED.json', gate)
        if not gate['passed']:
            resources = dict(elapsed_seconds=time.monotonic()-started)
            report(out, cfg, gate, None, resources, 0, 0)
            write(out/'COMPLETED.json', dict(completed=True, B0_rejudged_passed=False, B1_B2_started=False,
                status='B0_REJUDGED_FAILED_AWAITING_USER_REVIEW', new_training_updates=0, GPU_jobs=0, ended_utc=utc()))
            heartbeat('STOPPED_AWAITING_USER_REVIEW')
            print('B0 did not pass the revised gate. B1/B2 not started.', flush=True)
            return
        heartbeat('VERIFYING_PROTECTED_INPUTS')
        before = inventory()
        write(out/'INPUT_HASHES_BEFORE.json', before)
        inputs = Inputs()
        earlier_cfg = read(prior/'config.json')
        assert inputs.selected_source == earlier_cfg['selected_B_source_trajectories'], 'Source selection differs from the frozen v14 plan'
        assert inputs.old['solver'] == earlier_cfg['solver']
        assert inputs.scale.tolist() == earlier_cfg['target_scale']
        random.seed(inputs.config['model_loading_seed'])
        np.random.seed(inputs.config['model_loading_seed'])
        write(out/'config.json', dict(cfg, v14_config_sha256=sha(V14/'config.json'),
            v14_frozen_sha256=sha(V14/'FROZEN.json'), solver=inputs.old['solver'],
            target_scale=inputs.scale.tolist(), selected_B_source_trajectories=inputs.selected_source,
            source_complete_available=inputs.complete_source_count, B_environment_seed=inputs.config['B_environment_seed'],
            B_interpretation=inputs.config['B_interpretation']))
        runtime = limits.limits_snapshot()
        write(out/'RESOURCES_BEFORE.json', runtime)
        _, _, private = inputs.legacy.prepare_private_binding(out)
        vendor = private.parent
        import mujoco_py
        assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        import diagnostic_b                                # v14_diag/diagnostic_b.py
        cores = max(1, int(runtime['effective_visible_cpu_cores'])-2)
        heartbeat('B1_B2_STARTING', requested_workers=min(args.workers, cores))
        records = diagnostic_b.source_tracks(inputs, out/'B', vendor, heartbeat, min(args.workers, cores))
        stub = dict(passed=True, trajectories=[], simulator_steps=0, source_tracking_started=True,
            note='B0 was not re-simulated; see ../B0_REJUDGED.json for the revised gate applied to the earlier arrays.')
        b = diagnostic_b.summarize(out/'B', stub, records)
        heartbeat('VERIFYING_PROTECTED_INPUTS_AFTER')
        after = inventory()
        assert before == after, 'Protected original inputs changed'
        write(out/'INPUT_PROTECTION_AFTER.json', dict(passed=True, files=len(after)))
        resources = dict(limits=runtime, elapsed_seconds=time.monotonic()-started)
        write(out/'RESOURCES_AFTER.json', resources)
        admission = read(out/'B/CPU_ADMISSION.json')
        report(out, cfg, gate, b, resources, inputs.complete_source_count, admission['actual_workers'])
        write(out/'RESULT.json', dict(B0_rejudged=gate, B=b, new_training_updates=0, optimizer_instances=0,
            GPU_jobs=0, stage2_started=False))
        write(out/'COMPLETED.json', dict(completed=True, B0_rejudged_passed=True, B1_B2_started=True,
            B1_B2_records=len(records), status='AWAITING_USER_REVIEW', new_training_updates=0,
            optimizer_instances=0, GPU_jobs=0, environment_steps=b['simulator_steps'], stage2_started=False,
            ended_utc=utc(), evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json', 'B0_REJUDGED.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V14b DONE:', out/'REPORT_ZH.md', flush=True)
    except BaseException:
        failed = dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False,
            new_training_updates=0, GPU_jobs=0)
        if before:
            failed['protected_input_differences'] = [p for p, h in before.items() if not Path(p).is_file() or sha(p) != h]
        write(out/'FAILED.json', failed)
        print(failed['traceback'], flush=True)
        raise


if __name__ == '__main__':
    main()
