"""Prelaunch budget and the frozen V12 H_root analysis."""
import math
from pathlib import Path
from statistics import mean, stdev

from stage3_setup import CAP, GPU_UUIDS, ROOT_COMMIT, SEEDS, STEPS, points, read, sha
from stage3_worker import read_eval_records


def decision(delta, se, positive_count=None):
    if delta >= 5 and delta > 2 * se and (positive_count is None or positive_count >= 4):
        return '成立'
    if delta < 2:
        return '不成立'
    return '不确定'


def prelaunch_budget(out, common, initial, remaining_groups, solve_weights):
    # ROOT phase 2 includes VAE and 100 estimated evaluations. T and BC use observed planning anchors.
    hours = {'ROOT': 5.44, 'T': 5069.1 / 3600, 'BC-all': .25}
    waves = [remaining_groups[i:i + CAP] for i in range(0, len(remaining_groups), CAP)]
    weights = 76 / 3600 if solve_weights else 0.
    total = sum(max(hours[g] for g in wave) for wave in waves) + weights
    serial = sum(hours[g] for g in remaining_groups) + weights
    result = dict(max_GPU_processes=3, per_job_hours=hours, nominal_waves=waves,
                  weights_once_hours_anchor=weights, nominal_wall_hours_before_wait_and_overhead=total,
                  single_worker_wall_hours_before_wait_and_overhead=serial,
                  selected_GPU=initial['gpu'], initial_admissible_own_workers=initial['slots'],
                  budget_hours_with_margin=[17., 19.] if len(remaining_groups) == 15 and solve_weights else None,
                  assumed_parallel_speedup='three concurrent jobs, same per-job time',
                  stage0_T_anchor='IQL 500k extrapolation from one-process Hopper short run',
                  stage1B_BC_anchor='Walker-containing BC waves, about 15 minutes per wave',
                  stage2_ROOT_anchor='Walker2d-morph 20k trial extrapolation, no 3-way ROOT measurement',
                  gpu_occupancy_wait_seconds='unknown until launch', resource_limited_concurrency_can_extend_duration=True)
    common.write(out / 'BUDGET_ESTIMATE.json', result)
    text = [
        '# V12 阶段 3 启动前预算', '',
        f'剩余训练固定使用一张 A6000，整卡并发上限 {CAP}。',
        f"选定物理 {initial['index']} 号卡，首次准入可开 {initial['slots']} 个本实验进程。共享时最多开 1 个。",
        f'正式范围共 15 项。本次余 {len(remaining_groups)} 项训练，最多三并发时分 {len(waves)} 批。',
        '本次计算一份 OT 权重。' if solve_weights else '本次复用已完整写出且已核对哈希的 OT 权重。', '',
        '|组别|单次步数|单次耗时参考|依据|', '|---|---:|---:|---|',
        '|ROOT|1,000,000|5.44 小时|阶段 2 实测外推，含 VAE 和估计的评估|',
        '|T|500,000|约 1.41 小时|阶段 0.3 的单进程 IQL 外推，Hopper 作为 Walker 的时间参照|',
        '|BC-all|200,000|约 15 分钟|阶段 1B 含 Walker 的批次墙钟参照|', '',
        f'按每批最慢一项计，三并发基准为 {total:.2f} 小时。若并发受资源限制，将减少每批进程数。',
        f'如果全程只能开一个进程，串行基准约 {serial:.2f} 小时，GPU 等待和共享造成的减速另计。',
        '剩余批次顺序为 ' + (' → '.join(' + '.join(w) for w in waves) if waves else '只汇总和补审计') + '。', '',
    ]
    with (out / 'PRELAUNCH_REPORT_ZH.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(text))
    return result


def q_lines(probe):
    q = probe['losses']
    ratios = probe['ratios']
    return [
        'CPU 在同一个真实批次上复算三种损失。源域和目标域各 128 条，批次抽取种子 501。',
        '只读加载阶段 2 种子 901 的 20,000 步 Q/V 参数和已验收的阶段 2 权重，核验阶段 3 选定数据口径的批次。',
        '该检查执行 0 次优化更新、0 环境步，CUDA 未初始化。', '',
        '|Q 损失|数值|相对不加权|', '|---|---:|---:|',
        f"|上游广播|{q['upstream']:.9g}|{ratios['upstream_over_unweighted']:.9g}|",
        f"|逐样本加权|{q['per_sample_weighted']:.9g}|{ratios['per_sample_over_unweighted']:.9g}|",
        f"|不加权|{q['unweighted']:.9g}|1|", '',
        f"上游广播/逐样本加权为 {ratios['upstream_over_per_sample']:.9g}，批内平均权重为 {probe['mean_weight']:.9g}。",
        '原权重形状 [256,1,1] 与残差 [256,1] 广播至 [256,256,1]。因此上游 Q 损失在浮点误差内等于批内平均权重乘不加权损失。',
        f"上游 ROOT.py 第 {probe['actor_call_line']} 行为 `{probe['actor_call']}`，没有传入样本权重。",
        'proportion=0.25 保留在配置中，源码的过滤分支处于注释状态。训练直接调用原 ROOT.train 和 train_vae。', '',
    ]


def weight_lines(comparison, proof):
    lines = ['## OT 权重对照', '',
             '冻结求解器只读取 epsilon、lambda_src、lambda_tar，按固定顺序构造特征和分块，无训练种子输入或随机数调用。',
             f"求解器 SHA-256 `{proof['solver_sha256']}`。正式权重计算一次，种子 501–505 共用。",
             f"阶段 3 权重 SHA-256 `{comparison['stage3_file_sha256']}`。",
             f"阶段 2 权重 SHA-256 `{comparison['stage2_file_sha256']}`。",
             f"按同顺序逐行对齐 {comparison['aligned_source_rows']:,} 条源数据。以下统计用于描述。", '',
             '|阶段|权重|均值|最小|P01|P25|P50|P75|P95|P99|最大|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for stage, title in (('stage3', '3'), ('stage2', '2')):
        dist = comparison['distributions'][stage]
        for key, name in (('raw', '原始'), ('normalized_float32', '归一化'), ('clipped', '截断后')):
            record = dist[key]
            vals = [record['mean'], *(record['quantiles'][k] for k in ('min', 'p01', 'p25', 'p50', 'p75', 'p95', 'p99', 'max'))]
            lines.append(f'|{title}|{name}|' + '|'.join(f'{v:.8g}' for v in vals) + '|')
    lines.append('')
    for stage, title in (('stage3', '3'), ('stage2', '2')):
        ratio = comparison['distributions'][stage]['fraction_clipped_at_5']
        lines.append(f'阶段 {title} 归一化权重严格大于 5 并实际截断的比例为 {ratio:.6%}。')
    pairs = []
    for key, name in (('raw', '原始'), ('normalized_float32', '归一化'), ('clipped', '截断后')):
        value = comparison['pearson'][key]
        pairs.append(f'{name} {value:.8f}' if value is not None else f'{name}不可计算（常量列）')
    lines += ['逐行配对 Pearson 相关系数为' + '、'.join(pairs) + '。', '']
    return lines


def summary(out, jobs, legacy, resources_summary, waves, budget, waits, audit_paths):
    import numpy as np
    _, common, _, _, _, science = legacy
    project = science.Project(jobs[0]['project'])
    groups = {g: [] for g in STEPS}
    indices = {uuid: index for index, uuid in GPU_UUIDS.items()}
    for job in jobs:
        base = Path(job['output'])
        audit = read(audit_paths[job['id']])
        assert audit['passed'] and audit['completed_sha256'] == sha(base / 'COMPLETED.json')
        assert audit['checkpoint_sha256'] == sha(base / 'final_checkpoint.pt')
        assert len(audit['evaluation_sha256']) == len(points(job['group']))
        for step, digest in audit['evaluation_sha256'].items():
            assert sha(base / f'eval_{int(step):07d}.json') == digest
        raw = read_eval_records(job, project, common)
        record = dict(seed=job['seed'], last5=mean(e['mean'] for e in raw[-5:]), final=raw[-1]['mean'],
                      gpu_index=indices[job['gpu_uuid']], gpu_uuid=job['gpu_uuid'],
                      last5_lengths=[x for e in raw[-5:] for x in e['episode_lengths']], final_lengths=raw[-1]['episode_lengths'],
                      audited_output=str(base), audit_path=str(audit_paths[job['id']]),
                      completed=read(base / 'COMPLETED.json'))
        if job['group'] == 'ROOT':
            mid = [e['mean'] for e in raw if 460000 <= e['step'] <= 500000]
            assert len(mid) == 5
            record.update(last5_46_50w=mean(mid), maximum=max(e['mean'] for e in raw), last10=mean(e['mean'] for e in raw[-10:]))
        groups[job['group']].append(record)
    stats = {}
    for group, seeds in groups.items():
        seeds.sort(key=lambda x: x['seed'])
        assert tuple(x['seed'] for x in seeds) == SEEDS
        last5 = [x['last5'] for x in seeds]
        final = [x['final'] for x in seeds]
        worst = min(seeds, key=lambda x: (x['last5'], x['seed']))
        stats[group] = dict(last5_mean=mean(last5), last5_sd=stdev(last5), worst_seed=worst['seed'],
                            worst_last5=worst['last5'], last5_q1=float(np.quantile(last5, .25)),
                            final_mean=mean(final), final_sd=stdev(final), seeds=seeds,
                            last5_lengths=common.length_stats([v for x in seeds for v in x['last5_lengths']]),
                            final_lengths=common.length_stats([v for x in seeds for v in x['final_lengths']]))
    diffs = [r['last5'] - t['last5'] for r, t in zip(groups['ROOT'], groups['T'])]
    t = dict(delta=mean(diffs), SE=stdev(diffs) / math.sqrt(5), positive_count=sum(x > 0 for x in diffs), paired_differences=diffs)
    bc = dict(delta=stats['ROOT']['last5_mean'] - stats['BC-all']['last5_mean'],
              SE=math.sqrt((stats['ROOT']['last5_sd'] ** 2 + stats['BC-all']['last5_sd'] ** 2) / 5))
    t['rule_if_baseline'] = decision(t['delta'], t['SE'], t['positive_count'])
    bc['rule_if_baseline'] = decision(bc['delta'], bc['SE'])
    if stats['T']['last5_mean'] == stats['BC-all']['last5_mean']:
        baseline = 'T 与 BC-all 精确并列'
        outcome = t['rule_if_baseline'] if t['rule_if_baseline'] == bc['rule_if_baseline'] else '并列基线下规则结果不同，待解释'
    else:
        baseline = max(('T', 'BC-all'), key=lambda g: stats[g]['last5_mean'])
        outcome = (t if baseline == 'T' else bc)['rule_if_baseline']
    probe = read(out / 'ROOT_Q_READONLY_PROBE.json')
    inputs = read(out / 'ROOT_INPUTS.json')
    weights = read(out / 'ROOT_TRAINING_WEIGHTS.json')
    proof = read(out / 'ROOT_OT_SEED_INDEPENDENCE.json')
    comparison = read(out / 'ROOT_WEIGHT_COMPARISON.json')
    handoff = read(out / 'ORIGINAL_QUEUE_HANDOFF.json')
    restart_plan = read(out / 'RESTART_PLAN.json')
    reused_ids = sorted(restart_plan['reused_jobs'])
    reaudit_ids = [name for name in reused_ids if restart_plan['reused_jobs'][name]['audit'] is None]
    gpu_by_job = {job['id']: dict(index=indices[job['gpu_uuid']], uuid=job['gpu_uuid']) for job in jobs}
    assert sha(weights['path']) == weights['sha256'] == comparison['stage3_file_sha256']
    assert weights['seed_independence_sha256'] == sha(out / 'ROOT_OT_SEED_INDEPENDENCE.json')
    assert weights['comparison_sha256'] == sha(out / 'ROOT_WEIGHT_COMPARISON.json')
    result = dict(stage='3', groups=stats, ROOT_minus_T=t, ROOT_minus_BC_all=bc, selected_baseline=baseline, H_root=outcome,
                  q_probe=probe, source_inputs=inputs, weights=weights, OT_solver_seed_independence=proof,
                  OT_weight_comparison=comparison, budget=budget, waves=waves, resources=resources_summary,
                  original_queue_handoff=handoff, original_CPU_probe_reused=(out / 'CPU_PROBE_REUSED.json').is_file(),
                  reused_logical_jobs=reused_ids, reaudited_logical_jobs=reaudit_ids,
                  physical_GPU_by_job=gpu_by_job,
                  new_logical_jobs=restart_plan['newly_scheduled_jobs'], interrupted_attempts=restart_plan['previous_interrupted_runs'],
                  gpu_waits=waits, total_gpu_wait_seconds=sum(x['blocked_wait_seconds'] for x in waits), later_stage_started=False)
    common.write(out / 'STAGE3_RESULT.json', result)

    def length(value):
        return f"{value['mean']:.2f} / {value['median']:.1f} / {value['minimum']} / {value['fraction_under_200']:.1%}"

    limits = resources_summary['limits']
    lines = ['# V12 阶段 3 ROOT 正式复现报告', '',
             f"tmux `{resources_summary['tmux_session']}`，入口 `{resources_summary['launch']}`。配置 `{out / 'PROTOCOL.json'}`。",
             f'上游 commit `{ROOT_COMMIT}`，ROOT、T、BC-all 各五个种子，15 项审计通过。',
             f"本次复用 {len(reused_ids)} 项完整训练（其中 {len(reaudit_ids)} 项在本次补做 CPU 审计），执行 {len(restart_plan['newly_scheduled_jobs'])} 项训练。",
             '中断种子从原种子的第 0 步重跑，旧目录保留。各项出处见 `RESTART_PLAN.json` 和汇总 JSON。',
             f"原 0 号卡的已审计结果保留。换卡后剩余工作使用物理 {budget['selected_GPU']['index']} 号 RTX A6000。",
             '比较时合并两张物理卡上的结果，训练和判定协议一致。逐种子的设备记录见下表。',
             'ROOT 100 万步并保留 1 万步 VAE，T 50 万步，BC-all 20 万步。统一 Walker2d-morph、种子 501–505、'
             '目标 XML、归一化和逐回合评估种子。ROOT/T 每 10k，BC 每 5k，每点 10 回合。',
             'Last5 固定取 ROOT 96–100 万步、T 46–50 万步、BC-all 18–20 万步的各自末五个评估点。',
             'ROOT 从源环境读取动作维度，metric 为 euclidean，原 Q 广播、actor 调用和 CVAE 正则保留。'
             '三组目标数据均显式读取 next_observations，保留 5,000 行。相对上游相邻行 loader，'
             '共同覆盖的 4,999 行中 7 条下一观测不同，其中 3 条非终止，另保留末行。',
             'ROOT 源域按上游 D4RL 读取相邻观测。与阶段 2 的显式 next 相比，513 条终止转移不同，非终止转移差异 0 条。', '',
             '## 启动前 CPU 核验', '', *q_lines(probe),
             *weight_lines(comparison, proof),
             '## 组别和预设判定', '',
             '|组|Last5 均值|样本 SD|最差值与种子|Q1|Final 均值|', '|---|---:|---:|---|---:|---:|']
    for group, s in stats.items():
        lines.append(f"|{group}|{s['last5_mean']:.4f}|{s['last5_sd']:.4f}|{s['worst_last5']:.4f}（{s['worst_seed']}）|{s['last5_q1']:.4f}|{s['final_mean']:.4f}|")
    lines += ['',
              f"ROOT−T 的同种子配对差均值为 {t['delta']:+.4f}，配对 SE {t['SE']:.4f}，"
              f"正号 {t['positive_count']}/5。逐差 {', '.join(f'{x:+.4f}' for x in diffs)}。",
              f"ROOT−BC-all 的均值差为 {bc['delta']:+.4f}，Welch SE {bc['SE']:.4f}。",
              f'基线 B 为 {baseline}，H_root 按预设规则判为「{outcome}」。',
              '成立须差值至少 5 且严格大于 2×SE，B 为 T 时另须至少 4 个配对差为正。差值低于 2 判不成立。'
              '其他情况不确定。取较高基线带有同批选择效应，因此同时列出两项比较。', '',
              '## 各种子和回合长度', '',
              '|组|种子|物理 GPU|Last5|Final|Last5 长度均值／中位数／最短／<200|Final 长度均值／中位数／最短／<200|',
              '|---|---:|---:|---:|---:|---|---|']
    for group, s in stats.items():
        for row in s['seeds']:
            a, b = common.length_stats(row['last5_lengths']), common.length_stats(row['final_lengths'])
            lines.append(f"|{group}|{row['seed']}|{row['gpu_index']}|{row['last5']:.4f}|{row['final']:.4f}|{length(a)}|{length(b)}|")
        lines.append(f"|{group}|合计|—|{s['last5_mean']:.4f}|{s['final_mean']:.4f}|{length(s['last5_lengths'])}|{length(s['final_lengths'])}|")
    lines += ['', '## ROOT 描述性指标', '', '|种子|46–50 万步 Last5|所有评估点最高值|最后十个评估点平均|', '|---:|---:|---:|---:|']
    for row in groups['ROOT']:
        lines.append(f"|{row['seed']}|{row['last5_46_50w']:.4f}|{row['maximum']:.4f}|{row['last10']:.4f}|")
    lines.append('|均值|' + '|'.join(f"{mean(x[k] for x in groups['ROOT']):.4f}" for k in ('last5_46_50w', 'maximum', 'last10')) + '|')
    lines += ['', '描述性参照为 V12 所列 Windows R13 T 38.21、V9 BC-all 38.85、OTDF 本地复现 43.73 和 ROOT 论文值 57.3。',
              '', '## 资源与耗时', '',
              f"`nproc --all` 为 {limits['nproc_all']}，`nproc` 为 {limits['nproc']}，"
              f"cgroup 配额为 {limits['cpu_quota_cores']}，有效可用 CPU 为 {limits['effective_visible_cpu_cores']} 核。",
              f"整机 CPU 占用 {resources_summary['host_CPU_busy_percent_entire_stage']:.2f}%，"
              f"亲和性集合占用 {resources_summary['affinity_CPU_busy_percent_entire_stage']:.2f}%，包含同期工作负载和等待时段。",
              f"启动前三并发基准 {budget['nominal_wall_hours_before_wait_and_overhead']:.2f} 小时，"
              f"实际总墙钟 {resources_summary['elapsed_seconds'] / 3600:.2f} 小时，"
              f"本队列 GPU 准入等待合计 {result['total_gpu_wait_seconds'] / 60:.2f} 分钟（包含选卡等待）。",
              f"本次接续选定物理 {budget['selected_GPU']['index']} 号卡，UUID `{budget['selected_GPU']['uuid']}`。",
              f"本次实际训练最大并发 {max((w['workers'] for w in waves), default=0)}。权重出处 `{weights['origin']}`，原求解并写出耗时 {weights['seconds']:.2f} 秒。",
              '各次准入、等待、子进程及资源采样见输出日志。', '',
              '|批次|训练组别与种子|本实验进程数|放行时可见外部进程数|从放行到退出（小时）|采样峰值显存 MiB|',
              '|---|---|---:|---:|---:|---:|']
    for wave in waves:
        lines.append(f"|{wave['label']}|{', '.join(wave['jobs'])}|{wave['workers']}|{len(wave['external_GPU_pids_at_release'])}|{wave['GPU_phase_seconds'] / 3600:.3f}|{wave['peak_GPU_memory_MiB']}|")
    lines += [
              '', '阶段 3 已停止，等待用户决定下一步。', '']
    with (out / 'STAGE3_REPORT_ZH.md').open('x', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    return result
