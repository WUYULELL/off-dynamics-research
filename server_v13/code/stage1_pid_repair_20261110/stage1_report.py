"""CPU aggregation from raw evaluation records and accepted cache evidence."""
import json
from pathlib import Path

import numpy as np

from common import length_stats, read, sha, utc, write
from science import Project
from stage1_core import CONFIG, GIB, PROJECT, TASK, cache_receipt
from v12_bc import eval_records


def summary(values):
    a = np.asarray(values,dtype=np.float64)
    assert len(a)==5 and np.isfinite(a).all()
    return dict(mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),Q1=float(np.percentile(a,25)),min=float(a.min()))


def compare(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape==(5,)
    delta = a-b
    return dict(mean=float(delta.mean()),paired_SE=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                welch_SE=float(np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))),
                seed_differences=delta.tolist(),positive_count=int((delta>0).sum()))


def recover_records(accepted, spec):
    items = {}
    for name, folder in accepted.items():
        if not isinstance(folder,Path):
            continue
        audit = read(folder/'AUDIT.json')
        complete = read(folder/'COMPLETED.json')
        assert audit['passed'] and audit['CPU_only'] and audit['completed_sha256']==sha(folder/'COMPLETED.json')
        assert audit['checkpoint_sha256']==sha(folder/'final_checkpoint.pt')==complete['checkpoint_sha256']
        assert complete['steps_sha256']==sha(folder/'steps.f32')
        assert complete['steps']==CONFIG['bc']['steps']
        for filename,digest in complete['eval_sha256'].items():
            assert sha(folder/filename)==digest
        job = complete['job']
        assert job['job']==name and sha(folder/'JOB.json')==read(folder/'LAUNCH.json')['job_sha256']
        raw = eval_records(folder,job,spec)
        late = raw[-5:]
        assert [x['step'] for x in late]==CONFIG['bc']['last5']
        items[name] = dict(variant=job['variant'],seed=job['seed'],gpu=job['gpu'],attempt=job['attempt'],folder=str(folder),
            last5=float(np.mean([e['mean'] for e in late])),final=raw[-1]['mean'],
            last5_episode_lengths=[n for e in late for n in e['episode_lengths']],final_episode_lengths=raw[-1]['episode_lengths'],
            seconds=complete['wall_seconds'],peak_RSS_bytes=complete['peak_RSS_bytes'],
            audit_sha256=sha(folder/'AUDIT.json'),completed_sha256=audit['completed_sha256'],
            prefix=read(folder/'PREFIX_COMPARISON.json') if (folder/'PREFIX_COMPARISON.json').exists() else None)
    return items


def group_results(items):
    groups = {}
    for variant in CONFIG['bc']['variants']:
        values = sorted([r for r in items.values() if r['variant']==variant],key=lambda r:r['seed'])
        if len(values)!=5:
            continue
        assert [r['seed'] for r in values]==CONFIG['bc']['seeds']
        groups[variant] = dict(last5=summary([r['last5'] for r in values]),final=summary([r['final'] for r in values]),
            last5_worst_seed=min(values,key=lambda r:r['last5'])['seed'],
            last5_lengths=length_stats([n for r in values for n in r['last5_episode_lengths']]),
            final_lengths=length_stats([n for r in values for n in r['final_episode_lengths']]))
    if 'BC-all' in groups:
        def scores(variant):
            return [r['last5'] for r in sorted(items.values(),key=lambda r:r['seed']) if r['variant']==variant]
        baseline = scores('BC-all')
        for variant,record in groups.items():
            if variant!='BC-all':
                record['versus_BC_all']=compare(scores(variant),baseline)
    return groups


def length_line(x):
    return f"{x['mean']:.1f} / {x['median']:.1f} / {x['minimum']} / {100*x['fraction_under_200']:.1f}%"


def render(out, result):
    cs = result['cache_statistics']
    ca = result['cache_acceptance']
    cb = result['cache_RSS']
    bb = result['BC_RSS']
    variants = CONFIG['bc']['variants']
    lines = ['# V13 阶段 1 Hopper 全量动作修正和 BC 诊断','',
             '入口为 `bash ~/v13_action_fix/stage1/run_in_tmux.sh --registered-gpus ...`。输出目录、实际卡号和启动参数见 `PLAN.json` 与 `SESSION_*.json`。',
             '冻结阈值为 e1 ≤ 0.0840，由阶段 0 源残差分布预先确定。筛选只依据 e1。',
             '阶段 0 的报告修复补算了最近邻五档的距离最小值。原表读取统计中缺少的 `min` 字段而退出。修复复用了 5,000 条目标自检和 20,000 条源求解记录，新增仿真和训练为零。',
             '', '## 全量缓存及独立回放','',
             f"共修正 {ca['rows']:,} 行。缓存独立回放 {ca['replay_count']:,} 条，状态逐维最大绝对误差 {ca['state_linf']['max']:.8g}，奖励最大绝对误差 {ca['reward_abs']['max']:.8g}，终止差异 {ca['terminal_disagreement']} 条。",
             f"固定阈值实际保留 {cs['selected_rows']:,} 行，占 {100*cs['selected_fraction']:.4f}%。边界按任一动作维到 ±1 的距离 ≤ 1e-8 计算。",
             f"原动作触边 {100*cs['source_boundary_fraction']:.4f}%，修正动作触边 {100*cs['corrected_boundary_fraction']:.4f}%。由内部变为触边 {cs['new_boundary_count']:,} 行，占全量 {100*cs['new_boundary_fraction_all']:.4f}%。",
             f"原始终止 {cs['source_terminal_count']:,} 行（{100*cs['source_terminal_fraction']:.4f}%），修正后真实终止 {cs['corrected_terminal_count']:,} 行（{100*cs['corrected_terminal_fraction']:.4f}%）。逐行终止标签改变 {cs['terminal_changed_count']:,} 条。",
             f"求解收敛比例为 {100*cs['convergence_fraction']:.4f}%。停止原因计数见结果 JSON。",'']
    if cs['new_boundary_fraction_among_source_interior'] is not None:
        lines += [f"在原动作未触边的子集中，{100*cs['new_boundary_fraction_among_source_interior']:.4f}% 变为触边。",'']
    lines += ['|统计|n|均值|最小值|中位数|P90|P99|最大值|','|---|---:|---:|---:|---:|---:|---:|---:|']
    distributions = {'动作改变量 L2':cs['action_delta'],'修正前 e0':cs['e0'],'修正后 e1':cs['e1']}
    labels = {'interior_converged':'未触边、收敛','interior_not_converged':'未触边、未收敛',
              'boundary_converged':'触边、收敛','boundary_not_converged':'触边、未收敛'}
    distributions.update({'e1 '+labels[key]:val for key,val in cs['e1_by_boundary_and_convergence'].items()})
    for label,row in distributions.items():
        cells = [f"{row[k]:.7g}" if k in row else '—' for k in ('mean','min','median','p90','p99','max')]
        lines.append('|'+label+'|'+str(row['n'])+'|'+'|'.join(cells)+'|')
    lines += ['','## BC 结果','',
              '每个种子 20 万 actor 更新。目标域固定评估每 5,000 步 10 回合。Last5 取 180/185/190/195/200k，Final 取 200k。',
              '长度列为均值／中位数／最短／严格小于 200 步比例。','',
              '|组别|Last5 均值|Last5 样本 SD|最差值／种子|Q1|Final 均值|Last5 长度|Final 长度|',
              '|---|---:|---:|---|---:|---:|---|---|']
    groups = result['groups']
    for v in variants:
        if v not in groups:
            continue
        g = groups[v]
        s = g['last5']
        lines.append(f"|{v}|{s['mean']:.4f}|{s['sample_SD']:.4f}|{s['min']:.4f} / {g['last5_worst_seed']}|{s['Q1']:.4f}|"
                     f"{g['final']['mean']:.4f}|{length_line(g['last5_lengths'])}|{length_line(g['final_lengths'])}|")
    lines += ['','|相对 BC-all（Last5）|均值差|同种子配对 SE|Welch SE（描述）|正差数|',
              '|---|---:|---:|---:|---:|']
    for v in variants:
        if 'versus_BC_all' not in groups.get(v,{}):
            continue
        c = groups[v]['versus_BC_all']
        lines.append(f"|{v}|{c['mean']:.4f}|{c['paired_SE']:.4f}|{c['welch_SE']:.4f}|{c['positive_count']}/5|")
    lines += ['','V12 阶段 1A 的未修正源 BC Last5 为 13.3235，用于历史描述。','',
              '|组别|种子|物理卡|Last5|Final|Last5 长度|Final 长度|尝试编号|',
              '|---|---:|---:|---:|---:|---|---|---:|']
    for row in sorted(result['runs'].values(),key=lambda x:(variants.index(x['variant']),x['seed'])):
        lines.append(f"|{row['variant']}|{row['seed']}|{row['gpu']['index']}|{row['last5']:.4f}|{row['final']:.4f}|"
                     f"{length_line(length_stats(row['last5_episode_lengths']))}|{length_line(length_stats(row['final_episode_lengths']))}|{row['attempt']}|")
    lines += ['','## 预设描述与停点','',result['interpretation'],'',
              '## 资源、等待和中断','',
              f"`nproc` 为 {cb['nproc']}，cgroup 和亲和性有效 CPU 为 {cb['effective_cpu']:.2f} 核。缓存并行上限 {cb['workers']}，单进程 RSS {cb['pilot_RSS_bytes']/GIB:.3f} GiB，预计缓存并行 RSS {cb['predicted_RSS_bytes']/GIB:.3f} GiB。",
              f"全量缓存和独立回放耗时 {ca['wall_seconds']:.1f} 秒。BC 单进程实测峰值 RSS {bb['pilot_peak_RSS_bytes']/GIB:.3f} GiB，按 {bb['maximum_slots']} 个进程预估合计 {bb['projected_total_RSS_bytes']/GIB:.3f} GiB。",
              f"每卡准入上限 {result['per_GPU_limit']}，全卡合计上限 20。队列资源等待计时 {result['resource_wait_seconds']} 秒。外部占用或故障后封锁的卡 {result['blocked_cards']}。",
              '每次启动和退出、GPU 准入样本、设备 UUID 与等待原因见 `queue_events.jsonl`。',
              f"报告收录 {len(result['runs'])}/25 个通过 CPU 审计的训练。外部中断尝试 {len(result['interrupted_attempts'])} 次。",'']
    for record in result['CPU_usage_by_session']:
        busy = record['host_CPU_busy_percent']
        if busy is not None:
            lines.append(f"本轮会话整机 CPU 占用 {busy:.2f}%，包含同期其他用户负载。")
    for row in result['runs'].values():
        if row['prefix']:
            for check in row['prefix']['comparisons']:
                lines.append(f"`{Path(check['original']).name}` 的损失前缀核对 {check['compared_steps']} 步，逐位一致 {check['identical']}，首差步 {check.get('first_differing_step')}。")
    lines += ['', '本次接续复用原全量缓存和已审计的 BC-mix_s601，计划新跑 24 项。原封锁的进程归属未确认。'
              '新批次从空闲卡开始，先逐个核对 CUDA 进程身份，再在同一放行点启动最多三项训练。']
    if result['exhausted_jobs']:
        lines += ['', '达到重跑上限的任务为 '+', '.join(result['exhausted_jobs'])+'。该报告按未完整交付标记。']
    lines += ['','控制器已到阶段 1 停点，阶段 2 等待授权。']
    with (out/'REPORT_ZH.md').open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')


def deliver(out, cache, accepted, plan):
    from continuation import verify_original
    origin = read(out/'CONTINUATION.json')
    assert verify_original()==origin, 'Original cache and pilot evidence changed'
    ca = cache_receipt(cache,complete=True)
    cs = read(cache/'CACHE_STATISTICS.json')
    assert ca['selected_rows']==cs['selected_rows'] and ca['rows']==cs['rows']
    assert set(accepted)==set(plan['bc_jobs'])
    assert read(out/'INPUT_PROTECTION_AFTER.json')['passed']
    items = recover_records(accepted,Project(PROJECT).spec(TASK))
    groups = group_results(items)
    for name,item in items.items():
        assert [item['variant'],item['seed']]==plan['bc_jobs'][name]
    complete = len(items)==25
    condition = None
    if complete:
        assert len(groups)==5
        g = groups['BC-fix']
        condition = g['last5']['mean']<20 and g['last5_lengths']['fraction_under_200']>.5
    interpretation = ('BC-fix 的 Last5 低于 20 且多数回合短于 200 步，符合预设的低表现条件。按本阶段诊断，这组复现源状态变化的动作未提供可用的目标域策略。' if condition else
                      'BC-fix 未同时触发 Last5 低于 20 和多数回合短于 200 步这两个条件。阶段 1 结果作为 BC 描述。')
    if not complete:
        interpretation = '训练组未齐，预设 BC 诊断暂不作完整判定。'
    events = [json.loads(line) for line in (out/'queue_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    result = dict(stage=1,complete=complete,stage2_started=False,formal_H_fix_test=False,continuation=origin,
                  filter=CONFIG['residual_filter'],cache=str(cache),cache_acceptance=ca,cache_statistics=cs,
                  cache_RSS=read(cache/'CACHE_RSS_BEFORE_PARALLEL.json'),BC_RSS=read(out/'BC_RSS_BEFORE_PARALLEL.json'),
                  runs=items,groups=groups,predeclared_low_BC_fix_condition=condition,interpretation=interpretation,
                  exhausted_jobs=[name for name,path in accepted.items() if path=='exhausted'],
                  resource_wait_seconds=sum(e['seconds'] for e in events if e['state']=='resource_wait'),
                  per_GPU_limit=plan['per_GPU_limit'],blocked_cards=[read(p)['physical_gpu'] for p in sorted(out.glob('GPU_BLOCKED_*.json'))],
                  interrupted_attempts=[e for e in events if e['state']=='outcome' and e['status']=='external_interruption'],
                  CPU_usage_by_session=[read(p) for p in sorted(out.glob('CPU_USAGE_*.json'))])
    render(out,result)
    write(out/'RESULT.json',result)
    receipt = dict(completed=complete,CPU_audited_runs=len(items),planned_runs=25,own_GPU_workers_exited=True,
                   accepted_cache=str(cache),cache_acceptance_sha256=sha(cache/'CACHE_ACCEPTANCE.json'),
                   acceptance_status='AWAITING_STAGE2_CONFIRMATION' if complete else 'STAGE1_INCOMPLETE',
                   stage2_started=False,ended_utc=utc(),evidence_sha256={name:sha(out/name) for name in
                       ('REPORT_ZH.md','RESULT.json','PLAN.json','config.json','INPUT_PROTECTION_AFTER.json','CONTINUATION.json')})
    write(out/('STAGE1_COMPLETED.json' if complete else 'STAGE1_INCOMPLETE.json'),receipt)
    print(receipt['acceptance_status'],out/'REPORT_ZH.md',flush=True)
