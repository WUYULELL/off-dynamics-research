"""V15c: walker2d-morph source tracking (B1/B2) judged within the instrument horizon H.

H comes from the V15 B0 self-check on target data only (V15b diagnosis): every
target segment stays within the 8.4e-4 residual gate for at least 403 steps, so
the one-step tracking instrument is treated as reliable for the first H = 400
steps. The rule below was fixed after that diagnosis and before any walker2d
source-trajectory result existed.

All tracking code is the frozen V15 package in $HOME/v15_screen (read-only),
with the same source selection, solver and physics. CPU only, no training,
no GPU, no retry. Writes only under $HOME/v15c_walker/outputs.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
V15 = Path.home()/'v15_screen'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(V15))

import numpy as np

import v15_pipeline as vp            # $HOME/v15_screen, frozen; it puts $HOME/v14_diag on the path
from core import module, read, sha, stats, utc, write

TASK = 'walker2d-morph'


def verify_packages():
    for root in (HERE, V15):
        for name, digest in read(root/'FROZEN.json')['files'].items():
            assert sha(root/name) == digest, ('Package hash mismatch', str(root/name))
    assert Path(vp.__file__).resolve().parent == V15.resolve()


def horizon_check(cfg):
    """Recompute, from the saved V15 B0 arrays, the steps each target segment stays within the gate."""
    tdir = V15/'outputs'/cfg['v15_run']/TASK
    b0 = read(tdir/'B0_RESULT.json')
    rows = []
    for seg in b0['trajectories']:
        with np.load(tdir/('B0_trajectory%05d.npz' % seg['trajectory_id']), allow_pickle=False) as f:
            e1 = np.asarray(f['e1'], dtype=np.float64)
        over = np.flatnonzero(e1 > cfg['B0_normalized_residual_limit'])
        within = int(over[0]) if len(over) else len(e1)
        rows.append(dict(trajectory_id=seg['trajectory_id'], L=seg['L'], T=seg['T'], steps_within_gate=within,
                         exceeded=bool(len(over))))
    exceeded = [r['steps_within_gate'] for r in rows if r['exceeded']]
    minimum = min(exceeded) if exceeded else min(r['steps_within_gate'] for r in rows)
    assert minimum >= cfg['horizon_steps'], 'H is not supported by the saved B0 arrays'
    return dict(segments=rows, min_steps_within_gate_over_exceeding_segments=minimum, H=cfg['horizon_steps'])


def truncated(records, H, out):
    """Per-record statistics restricted to the first H executed steps."""
    rows = []
    for r in records:
        TH, LH = min(r['T'], H), min(r['L'], H)
        with np.load(out/('%s_trajectory%05d.npz' % (r['mode'], r['trajectory_id'])), allow_pickle=False) as f:
            e1 = np.asarray(f['e1'], dtype=np.float64)[:H]
        rows.append(dict(mode=r['mode'], trajectory_id=r['trajectory_id'], L=r['L'], T=r['T'], L_H=LH, T_H=TH,
                         ratio_H=TH/LH, fell_within_H=bool(r['physical_terminal'] and r['T'] <= H), e1_H=e1))
    return rows


def interpret(TH, ratio):
    fraction = float((np.asarray(ratio) >= .8).mean())
    if np.median(TH) < 200 and fraction < .1:
        return '不可跟踪（前 H 步内；仅限一步贪心跟踪）'
    if fraction >= .5:
        return '可跟踪（前 H 步内按存活步数；是否跟得上还需结合跟踪残差）'
    return '不确定'


def group(rows, reference):
    out = {}
    for mode in ('B1', 'B2'):
        g = [r for r in rows if r['mode'] == mode]
        TH = np.asarray([r['T_H'] for r in g])
        ratio = np.asarray([r['ratio_H'] for r in g])
        e = np.concatenate([r['e1_H'] for r in g])
        out[mode] = dict(episodes=len(g), T_H=stats(TH), ratio_H=stats(ratio),
            fraction_T_H_under_200=float((TH < 200).mean()), fraction_ratio_H_at_least_08=float((ratio >= .8).mean()),
            fraction_fell_within_H=float(np.mean([r['fell_within_H'] for r in g])),
            fraction_L_longer_than_H=float(np.mean([r['L'] > r['L_H'] for r in g])),
            residual_H=stats(e), fraction_steps_residual_within_reference=float((e <= reference).mean()),
            interpretation=interpret(TH, ratio))
    return out


def hopper_consistency(cfg):
    """Same truncated rule applied to the finished V15 hopper records (no simulation)."""
    records = read(V15/'outputs'/cfg['v15_run']/'hopper-morph/B_SOURCE_TRACKS.json')
    H = cfg['horizon_steps']
    res = {}
    for mode in ('B1', 'B2'):
        g = [r for r in records if r['mode'] == mode]
        TH = np.asarray([min(r['T'], H) for r in g])
        ratio = np.asarray([min(r['T'], H)/min(r['L'], H) for r in g])
        res[mode] = dict(T_H_median=float(np.median(TH)), fraction_ratio_H_at_least_08=float((ratio >= .8).mean()),
                         interpretation=interpret(TH, ratio))
    return res


def fmt(x, digits=4):
    return '无' if x is None else ('%.*g' % (digits, x) if isinstance(x, float) else str(x))


def table(header, rows):
    return '\n'.join(['|'+'|'.join(header)+'|', '|'+'|'.join(['---']*len(header))+'|']
                     + ['|'+'|'.join(fmt(c) for c in r)+'|' for r in rows])


def report(out, cfg, hz, plan, groups, full, hopper, elapsed):
    H = cfg['horizon_steps']
    text = ['# V15c walker2d 形态偏移源轨迹跟踪（前 H 步判定）', '',
        '仅 CPU：新训练更新 0、优化器实例 0、GPU 作业 0。跟踪代码为只读的 V15 冻结包，源轨迹选取、求解器和物理约定与 V15 相同。',
        '本次输出：`%s`。' % out, '',
        '## 规则来历（如实说明）', '',
        'V15 中 walker2d 的 B0 自检不通过：3 段走满 1000 步的目标轨迹残差随步数累积超限，按 V15 预设规则未运行源轨迹跟踪。',
        'V15b 只读诊断显示，所有目标段从第 1 步起至少 %d 步都在门槛 %s 以内。用户于 2026-11-12 确认：这一步只调整工具的可用时长，'
        '在看到任何 walker2d 源轨迹结果之前，取 H = %d 步，只在前 H 步内判定。这是第二次在看到 B0 结果后调整规则。'
        % (hz['min_steps_within_gate_over_exceeding_segments'], fmt(cfg['B0_normalized_residual_limit']), H), '',
        table(['目标段', 'L', 'T', '从第 1 步起保持在门槛内的步数', '是否超限'],
            [[s['trajectory_id'], s['L'], s['T'], s['steps_within_gate'], s['exceeded']] for s in hz['segments']]), '',
        '## 预设规则（前 H 步内；阈值与 V14/V15 相同）', '',
        '- 记 T_H = min(T, H)、L_H = min(L, H)。',
        '- 不可跟踪：B2 的 T_H 中位数 < 200，且 T_H/L_H ≥ 0.8 的轨迹不到 10%。',
        '- 可跟踪：B2 中 T_H/L_H ≥ 0.8 的轨迹达到 50% 以上。',
        '- 其他：不确定。',
        '- 判定只看存活步数；同时报告前 H 步内的跟踪残差，没摔但残差很大的不能算“源状态可以复现”。',
        '- 局限：一步贪心跟踪；H 之后的结果不可靠，下文全长统计仅作参考。', '',
        '## 结果（前 H = %d 步）' % H, '',
        '源数据 %d 行；完整源轨迹 %d 条，其中起点可直接恢复的 %d 条；固定种子 %d 选取 %d 条，与 V15 冻结的清单逐项一致。'
        % (plan['source_rows'], plan['source_complete'], plan['source_eligible_unclipped_start'], plan['source_seed'],
           len(plan['selected_source_trajectories'])), '',
        table(['模式', '轨迹数', 'T_H中位数', 'T_H P10', 'T_H P90', 'T_H/L_H中位数', 'T_H<200比例', 'T_H/L_H≥0.8比例', 'H内摔倒比例'],
            [[m, groups[m]['episodes'], groups[m]['T_H']['median'], groups[m]['T_H']['p10'], groups[m]['T_H']['p90'],
              groups[m]['ratio_H']['median'], groups[m]['fraction_T_H_under_200'], groups[m]['fraction_ratio_H_at_least_08'],
              groups[m]['fraction_fell_within_H']] for m in ('B1', 'B2')]), '',
        '前 H 步内的归一化跟踪残差（描述性；%s 是 v13 在 Hopper 上用过的筛选阈值，只作参照）：' % fmt(cfg['residual_reference']), '',
        table(['模式', '步数', '中位数', 'P90', '残差≤参照值的步数比例'],
            [[m, groups[m]['residual_H']['n'], groups[m]['residual_H']['median'], groups[m]['residual_H']['p90'],
              groups[m]['fraction_steps_residual_within_reference']] for m in ('B1', 'B2')]), '',
        '**B2 判定：%s**' % groups['B2']['interpretation'], '',
        '## 参考', '',
        '同一前 H 步规则用于 V15 已完成的 hopper 记录（不仿真）：B2 的 T_H 中位数 %s，T_H/L_H ≥ 0.8 比例 %s，判定“%s”。'
        % (fmt(hopper['B2']['T_H_median']), fmt(hopper['B2']['fraction_ratio_H_at_least_08']), hopper['B2']['interpretation']), '',
        '全长统计（H 之后工具不可靠，仅作参考）：', '',
        table(['模式', '源轨迹长度中位数', 'T中位数', 'T/L≥0.8比例', '物理终止条数'],
            [[m, full[m]['reference_length']['median'], full[m]['survival_steps']['median'],
              full[m]['fraction_T_over_L_at_least_08'], full[m]['physical_terminal_count']] for m in ('B1', 'B2')]), '',
        '## 停点', '',
        '总耗时 %.1f 秒。逐轨迹结果见 %s/B_trajectories.csv 与 TRUNCATED.csv。本次不提出也不启动下一项实验。' % (elapsed, TASK), '']
    (out/'REPORT_ZH.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    assert 1 <= args.workers <= 24
    if not vp.TEST:
        assert sys.platform == 'linux' and sys.version_info[:3] == (3, 8, 13)
        assert Path(sys.prefix).resolve() == (Path.home()/'conda-envs/offdynamics38').resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    verify_packages()
    cfg = read(HERE/'config.json')
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v15c.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = HERE/'outputs'/('walker_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n', encoding='utf-8')
    print('V15c output:', out, flush=True)
    started = time.monotonic()
    before = None

    def heartbeat(phase, **kw):
        write(out/'STATUS.json', dict(time=utc(), phase=phase, **kw), replace=True)
        print(phase, kw, flush=True)

    try:
        heartbeat('VERIFYING_PROTECTED_INPUTS')
        scope = dict(tasks=[TASK])
        before = vp.protected(scope)
        write(out/'PROTECTED_BEFORE.json', before)
        hz = horizon_check(cfg)
        v15_plan = read(V15/'outputs'/cfg['v15_run']/TASK/'PLAN.json')
        v15_cfg = read(V15/'config.json')
        spec, data = vp.load(TASK)
        nq = vp.v15_tracker.ENVIRONMENTS[TASK][2]
        old = read(vp.V13/'stage0/config.json')
        scale = data['target']['observations'].astype(np.float64).std(axis=0, ddof=1)+old['std_epsilon']
        assert scale.tolist() == v15_plan['target_scale']
        source_segments, _ = vp.segments(data['source'], source=True)
        eligible = [r for r in source_segments if r['complete'] and not vp.clipped(data['source']['observations'][r['first_row']], nq)]
        seed = v15_cfg['source_seed'][TASK]
        order = np.random.RandomState(seed).choice(len(eligible), min(v15_cfg['source_trajectories'], len(eligible)), replace=False)
        selected = [eligible[int(i)] for i in order]
        assert selected == v15_plan['selected_source_trajectories'], 'Selection differs from the frozen V15 plan'
        write(out/'config.json', dict(cfg, horizon_check=hz, v15_frozen_sha256=sha(V15/'FROZEN.json'), test_shim=vp.TEST))
        if vp.TEST:
            vendor = Path(vp.v15_tracker.SHIM)
            workers = min(args.workers, 4)
        else:
            import limits
            runtime = limits.limits_snapshot()
            workers = min(args.workers, max(1, int(runtime['effective_visible_cpu_cores'])-2))
            legacy = module(vp.V13/'stage0/legacy_replay.py', 'v15c_binding_loader')
            binding, binding_before, private = legacy.prepare_private_binding(out)
            vendor = private.parent
            import mujoco_py
            assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        tdir = out/TASK
        tdir.mkdir()
        init = (TASK, str(vp.PROJECT/spec['xml']), str(vendor), scale, old['solver'], old['action_boundary_abs_tolerance'])
        jobs = []
        for r in selected:
            p, v = vp.start_state(data['source']['observations'][r['first_row']], nq)
            jobs += [vp.job(init, tdir, r, mode, p, v, data['source'], spec) for mode in ('B1', 'B2')]
        heartbeat('B1_B2', total=len(jobs))
        records = vp.run_pool(jobs, workers, heartbeat, 'B1_B2')
        assert len(records) == 2*len(selected)
        write(tdir/'B_SOURCE_TRACKS.json', records)
        full = vp.summarize(tdir, records, cfg['residual_reference'])['groups']
        rows = truncated(records, cfg['horizon_steps'], tdir)
        vp.csv_write(tdir/'TRUNCATED.csv', [{k: v for k, v in r.items() if k != 'e1_H'} for r in rows])
        groups = group(rows, cfg['residual_reference'])
        hopper = hopper_consistency(cfg)
        heartbeat('VERIFYING_PROTECTED_INPUTS_AFTER')
        after = vp.protected(scope)
        assert before == after, 'Protected original inputs changed'
        if not vp.TEST:
            assert legacy.hash_tree(binding) == binding_before, 'Installed mujoco-py binding changed'
        write(out/'PROTECTED_AFTER.json', dict(passed=True, files=len(after)))
        elapsed = time.monotonic()-started
        plan = dict(source_rows=int(len(data['source']['observations'])),
                    source_complete=sum(r['complete'] for r in source_segments),
                    source_eligible_unclipped_start=len(eligible), source_seed=seed, selected_source_trajectories=selected)
        report(out, cfg, hz, plan, groups, full, hopper, elapsed)
        write(out/'RESULT.json', dict(task=TASK, horizon=hz, truncated=groups, full_length=full, hopper_consistency=hopper,
                                      new_training_updates=0, GPU_jobs=0))
        write(out/'COMPLETED.json', dict(completed=True, status='AWAITING_USER_REVIEW',
            B2_interpretation_within_H=groups['B2']['interpretation'], H=cfg['horizon_steps'],
            records=len(records), new_training_updates=0, optimizer_instances=0, GPU_jobs=0,
            elapsed_seconds=elapsed, ended_utc=utc(),
            evidence_sha256={n: sha(out/n) for n in ('REPORT_ZH.md', 'RESULT.json', 'config.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V15c DONE:', out/'REPORT_ZH.md', flush=True)
    except BaseException:
        failed = dict(time=utc(), traceback=traceback.format_exc(), automatic_retry=False, new_training_updates=0, GPU_jobs=0)
        if before:
            failed['protected_input_differences'] = [p for p, h in before.items() if not Path(p).is_file() or sha(p) != h]
        write(out/'FAILED.json', failed)
        print(failed['traceback'], flush=True)
        raise


if __name__ == '__main__':
    main()
