"""CPU reports for V12's predeclared descriptive diagnostics."""
import math
from pathlib import Path

from bc import eval_records, EVAL_POINTS
from common import length_stats, read, sha, utc, write
from science import Project


def rule_1a(score, short_fraction):
    if score < 20 and short_fraction > 0.5:
        return '支持 SA-all 本质上是在模仿源策略这一解释'
    return '按 V12 的预设解读，SA-all 的失败另有原因'


def rule_1b(delta, se):
    if delta > 2 * se:
        return '目标数据规模在当前范围内仍未饱和，支持状态覆盖不足是瓶颈之一'
    if delta < 0 or delta < se:
        return '在当前范围内已饱和'
    return '不确定'


def summarize_runs(seeds):
    import numpy as np
    assert sorted(x['seed'] for x in seeds) == list(range(401, 406))
    last5 = np.asarray([run['last5'] for run in seeds])
    final = np.asarray([run['final'] for run in seeds])
    worst = min(seeds, key=lambda x: (x['last5'], x['seed']))
    return dict(n=len(seeds), last5_mean=float(last5.mean()), last5_sd=float(last5.std(ddof=1)),
                last5_q1=float(np.quantile(last5, .25)), worst_seed=worst['seed'], worst_last5=worst['last5'],
                final_mean=float(final.mean()), final_sd=float(final.std(ddof=1)),
                last5_lengths=length_stats([v for run in seeds for v in run['last5_raw_lengths']]),
                final_lengths=length_stats([v for run in seeds for v in run['final_raw_lengths']]), seeds=seeds)


def welch(a, b):
    return dict(delta=b['last5_mean'] - a['last5_mean'],
                SE=math.sqrt(a['last5_sd'] ** 2 / a['n'] + b['last5_sd'] ** 2 / b['n']))


def cpu_lines(resources):
    def percent(value):
        return '未记录' if value is None else f'{value:.2f}%'
    values = resources['limits']
    quota = values['cpu_quota_cores']
    quota_label = '未见有限配额' if quota is None else f'{quota:.2f} 核'
    return [
        f"`nproc --all` 为 {values['nproc_all']}，`nproc` 为 {values['nproc']}，"
        f"cgroup CPU 配额为 {quota_label}，结合亲和性后的有效可用 CPU 为 "
        f"{values['effective_visible_cpu_cores']:.2f} 核。",
        f"本阶段主机整体 CPU 占用 {percent(resources['host_CPU_busy_percent_entire_stage'])}，"
        f"进程可用 CPU 集合占用 {percent(resources['affinity_CPU_busy_percent_entire_stage'])}。"
        '两项为整个阶段起止的 CPU tick 差，包含同期其他工作负载。',
    ]


def bc_report(stage, base, jobs, resources, waves):
    import numpy as np
    project = Project(jobs[0]['project'])
    groups = {}
    for job in jobs:
        out = Path(job['output'])
        audit = read(out / 'AUDIT.json')
        assert audit['passed'] and audit['completed_sha256'] == sha(out / 'COMPLETED.json')
        events = eval_records(out, job, project.spec(job['task']))
        tail = [length for item in events[-5:] for length in item['episode_lengths']]
        last = events[-1]['episode_lengths']
        row = dict(seed=job['seed'], last5=float(np.mean([e['mean'] for e in events[-5:]])),
                   final=events[-1]['mean'], last5_raw_lengths=tail, final_raw_lengths=last,
                   last5_lengths=length_stats(tail), final_lengths=length_stats(last))
        groups.setdefault(job['task'], {}).setdefault(str(job['trajectories']), []).append(row)
    results = {task: {name: summarize_runs(runs) for name, runs in subsets.items()}
               for task, subsets in groups.items()}
    lines = [f'# V12 阶段 {stage} 诊断报告', '',
             f"tmux 会话 `{resources['tmux_session']}`，运行入口 `{resources['launch']}`。",
             f'输出目录 `{base}`，阶段配置 `{base / "config.json"}`。',
             f'共 {len(jobs)} 项通过 CPU 审计，种子 401–405，每项 200,000 次 actor 更新。',
             '主指标 Last5 取 180k、185k、190k、195k、200k，Final 取 200k。',
             '回合长度在各自窗口内合并后计算。短回合指长度严格小于 200 步。', '', *cpu_lines(resources), '']
    if stage == '1B':
        selections = read(base / 'TRAJECTORY_SUBSETS.json')
        lines += ['## 嵌套训练数据', '', '|任务|轨迹数|轨迹编号|数据行|轨迹平均原始回报|', '|---|---:|---|---:|---:|']
        for task, record in selections.items():
            for k, subset in record['subsets'].items():
                lines.append(f"|{task}|{k}|{subset['trajectory_ids']}|{subset['n_rows']}|{subset['mean_raw_trajectory_return']:.3f}|")
        lines += ['', '用 NumPy RandomState(777) 排列已核对的七条完整轨迹。两个任务均排除 EOF 尾段。', '']
    else:
        lines += ['Hopper 使用 D4RL medium-v2 的 999,998 条原始训练行，沿用 V12 对照所用的 D4RL v2 canonical 行范围。',
                  '训练仅使用状态和动作，评估在 Hopper-morph。', '']
    lines += ['## 组别统计', '', '|任务|轨迹数或数据域|Last5 均值|样本 SD|最差值和种子|Q1|Final 均值|',
              '|---|---|---:|---:|---|---:|---:|']
    for task, subsets in results.items():
        for label, value in subsets.items():
            lines.append(f"|{task}|{label}|{value['last5_mean']:.4f}|{value['last5_sd']:.4f}|"
                         f"{value['worst_last5']:.4f}（{value['worst_seed']}）|{value['last5_q1']:.4f}|{value['final_mean']:.4f}|")
    lines += ['', '## 逐种子得分和回合长度', '',
              '|任务|数据|种子|Last5|Final|Last5 长度均值／中位数／最小值／<200|Final 长度均值／中位数／最小值／<200|',
              '|---|---|---:|---:|---:|---|---|']

    def lengths(value):
        return f"{value['mean']:.2f} / {value['median']:.1f} / {value['minimum']} / {value['fraction_under_200']:.1%}"

    for task, subsets in results.items():
        for label, group in subsets.items():
            for value in group['seeds'] + [dict(seed='合计', last5=group['last5_mean'], final=group['final_mean'],
                                               last5_lengths=group['last5_lengths'], final_lengths=group['final_lengths'])]:
                lines.append(f"|{task}|{label}|{value['seed']}|{value['last5']:.4f}|{value['final']:.4f}|"
                             f"{lengths(value['last5_lengths'])}|{lengths(value['final_lengths'])}|")
    decisions = {}
    lines += ['', '## 预设解读', '']
    if stage == '1A':
        value = results['hopper-morph']['source']
        score, fraction = value['last5_mean'], value['last5_lengths']['fraction_under_200']
        decisions = dict(last5_below_20=score < 20, majority_below_200=fraction > .5, decision=rule_1a(score, fraction),
                         windows_descriptive=dict(SA_all_Last5=14.47, SA_all_mean_length=185.362,
                                                  SA_all_fraction_below_200=.846, BC_all_Last5=52.21))
        lines += [f"Last5 为 {score:.4f}，Last5 窗口中短回合比例 {fraction:.1%}。{decisions['decision']}。",
                  'Windows 对照仅并列描述。V11 SA-all 的 Last5 为 14.47，长度均值 185.362 步，短回合比例 84.6%。'
                  'V9 BC-all 的 Last5 为 52.21。']
    else:
        for task, subsets in results.items():
            d42, d74 = welch(subsets['2'], subsets['4']), welch(subsets['4'], subsets['7'])
            decision = rule_1b(d74['delta'], d74['SE'])
            decisions[task] = dict(four_minus_two=d42, seven_minus_four=d74, decision=decision)
            lines += [f"{task} 的 4−2 为 {d42['delta']:+.4f}，Welch SE 为 {d42['SE']:.4f}。"
                      f"7−4 为 {d74['delta']:+.4f}，Welch SE 为 {d74['SE']:.4f}。{decision}。"]
        lines += ['不同子集同时改变了数据量和轨迹构成。该结果按 V12 作描述性解读。']
    lines += ['', '## 运行吞吐', '']
    for wave in waves:
        lines.append(f"批次 {wave['index']}，{len(wave['worker_rates'])} 进程，"
                     f"整批实际更新率 {wave['wall_updates_per_second']:.2f} 步/秒（含初始化和评估），"
                     f"各进程纯训练速率和 {wave['sum_training_updates_per_second']:.2f} 步/秒。"
                     f"各进程纯训练速率 {', '.join(f'{k}={v:.2f}' for k,v in wave['worker_rates'].items())}。")
    lines += ['', f"总墙钟时间 {resources['elapsed_seconds']:.1f} 秒。", '']
    (base / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(base / 'RESULT.json', dict(stage=stage, groups=results, predeclared_interpretation=decisions,
                                     resource_usage=resources, waves=waves, completed=utc()))
