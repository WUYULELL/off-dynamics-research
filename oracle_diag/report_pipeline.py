"""Refresh REPORT.md from actual artifacts; pending stages stay explicitly pending."""
import json,csv,datetime,math,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results/verified_20261003'
ENVS=['walker2d-kinematic-footjnt','hopper-kinematic-legjnt','halfcheetah-kinematic-footjnt']


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def num(x):return '等待' if x is None else '{:.5g}'.format(x)


def last5(job):
    if job.get('state') not in ['completed','reused']:return None
    folder=Path(job.get('out',''));files=list(folder.rglob('events.out.tfevents*')) if folder.is_dir() else []
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    found=[]
    for file in files:
        path=str(file.resolve());path='\\\\?\\'+path if os.name=='nt' else path
        acc=EventAccumulator(path,size_guidance={'scalars':0});acc.Reload()
        tag=next((t for t in acc.Tags()['scalars'] if t.replace(' ','_')=='test/target_normalized_score'),None)
        if tag:
            s={int(v.step):float(v.value) for v in acc.Scalars(tag)};steps=[460000,470000,480000,490000,500000]
            if all(k in s and math.isfinite(s[k]) for k in steps) and max(s)==500000:
                found.append(dict(mean=sum(s[k] for k in steps)/5,steps=steps,values=[s[k] for k in steps],file=str(file)))
    if len(found)>1:raise RuntimeError('Ambiguous result '+str(folder))
    return found[0] if found else None


def main():
    acceptance=read(OUT/'final_acceptance.json',{});initial=read(OUT/'acceptance.json',{})
    geometry=read(OUT/'diagnostics/stage1_stage2a.json',{})
    status=read(OUT/'queue_status.json',[]);plan=read(OUT/'queue_plan.json',{})
    bellman={}
    for env in ENVS:
        for kind in ['TARGET','MIX_FB']:
            p=OUT/'diagnostics'/(env+'_'+kind+'_bellman.json')
            if p.exists():bellman[(env,kind)]=read(p)
    historic={}
    csvpath=ROOT/'analysis_results/const_w020_vs_bellman/score_by_seed.csv'
    if csvpath.exists():
        with csvpath.open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row['seed']=='0' and int(row['final_step'])==500000:
                    historic[(row['env'],row['label'])]=float(row['last5_avg'])
    scores={};evidence=[]
    for job in status:
        value=last5(job)
        if value:scores[(job['env'],job['method'])]=value['mean'];evidence.append(dict(env=job['env'],method=job['method'],**value))
    done=len([j for j in status if j['state'] in ['completed','reused']]);expected=len(status)
    finished=bool(status and done==expected and len(bellman)==6 and all((e,m) in scores for e in ENVS for m in ['CONST','ORACLE','BACKUP','MIX_FB','TARGET']))
    lines=['# IQL Bellman oracle 诊断', '',
        '**{}**'.format('全部阶段已完成；以下为 seed 0 结果。' if finished else '实验进行中；未完成的训练与诊断不构成结论。'),
        '', '更新时间：'+datetime.datetime.now().astimezone().isoformat(timespec='seconds'), '',
        '## 执行范围与前提修正','',
        '- source=shifted ODRL medium；target=正常 D4RL medium 的 seed 0 / 10% 子集。原数据加载、target 标准化和 kNN 函数均复用。',
        '- 每个 source 原文件 5000 行，原加载器实际输出 4999 行；本诊断全部使用，不重复抽样伪装成 20000 个独立点。',
        '- 原算法、train.py、注册表、旧脚本和旧结果均未修改。新增算法与内存注册适配器位于 oracle_diag；原训练入口的 NTFS 短路径解决 Windows 保存兼容问题。',
        '- source_budget=.20、eta=1、clamp=[.05,1]、batch=128。权重沿用原流程，作用于 source V/Q/actor 三项损失；clamp 后均值可能略高于 .20。',
        '- ORACLE-BACKUP 使用常数 .20，只把 source Q 的回归目标换成 oracle 奖励、下一状态与终止掩码。它是理想转移对照，不是策略回报的数学上界；失败也不能证明所有 Bellman 方法无效。',
        '', '## 阶段 0：回放验收','',
        '使用 CPU。每环境随机检查 5000 条 target、完整 4999 条 source；qvel 截断样本沿同一轨迹恢复。下表是最终 source 回放误差（逐样本最大状态分量误差）：','',
        '| 环境 | source n | 状态误差均值 | 中位数 | p95 | 最大 | 奖励误差均值 | 奖励中位数 | 奖励 p95 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for env in ENVS:
        a=acceptance.get('environments',{}).get(env,{})
        if a:
            s=a['source_state_linf'];r=a['source_reward_abs']
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(env,a['source_n'],*[num(s[k]) for k in ['mean','median','p95','max']],num(r['mean']),num(r['median']),num(r['p95'])))
    lines+=['','| 环境 | target 可恢复样本 | 状态误差均值 / 中位数 / p95 | 奖励误差均值 / 中位数 / p95 | target / source 截断率 |','|---|---:|---|---|---|']
    for env in ENVS:
        a=initial.get('environments',{}).get(env,{})
        if a:
            t=a['target'];s=t['recovered']['state_linf'];r=t['recovered']['reward_abs']
            lines.append('| {} | {} / 5000 | {} | {} | {:.2%} / {:.2%} |'.format(env,t['recovered']['n'],
                ' / '.join(num(s[k]) for k in ['mean','median','p95']),' / '.join(num(r[k]) for k in ['mean','median','p95']),t['clipped']/5000,a['source']['clipped']/4999))
    lines+=['',
        '- shift 由 XML 关节活动范围改变实现，未对动作额外裁剪。Hopper joint range 确实改变，但在本 source 支持集上，两套模拟器的一步转移相同；不能只凭任务名假设存在可识别动力学差异。',
        '- Walker2d 首轮有 20 条 source 回放失败。求解器 warmstart 不是原因；改用 1–64 步候选历史恢复后全部达到原阈值（状态 1e-3，奖励 1e-4）。不同通过验收的历史产生的 target 结果也必须在同样阈值内一致。此前失败结果保留，未放宽阈值。',
        '- Walker2d、Hopper 各 1 条 target 样本在轨迹开头已有速度截断，缺少可恢复历史；明确排除这两条，未将其计为通过。source 则全量通过。',
        '- 这是经过回放验证的近似模拟器 oracle，仍受 float32 观测和有限状态恢复精度限制。诊断没有宣称取得原始未舍入的精确物理状态。',
        '- 证据：results/verified_20261003/{acceptance,warmstart_acceptance,final_acceptance}.json 与每环境 NPZ。',
        '', '## 阶段 1、2a：真实差异与下一状态估计','',
        '误差为 target 特征库状态标准差单位下的 L2；A=邻居下一状态均值，B=当前 source 状态加邻居平均位移，C=原 source 下一状态。','',
        '| 环境 | 物理 shift 均值 | A 误差均值 | B 误差均值 | C 误差均值 | A 差异相关 | B 差异相关 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for env,g in geometry.items():
        a=g['stage2a']['A_knn_next_mean'];b=g['stage2a']['B_displacement'];c=g['stage2a']['C_no_shift']
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(env,num(g['physical_shift_standardized']['mean']),num(a['error']['mean']),num(b['error']['mean']),num(c['error']['mean']),num(a['estimated_vs_true_spearman']),num(b['estimated_vs_true_spearman'])))
    lines+=['','完整分布、近邻距离相关、奖励差异及剔除边界的敏感性见 `results/verified_20261003/diagnostics/stage1_stage2a.json`。',
        '每环境 oracle NPZ 与原 source buffer 的行 i 一一对应，文件 SHA256 与索引说明均已记录。每个 source 加载器有 4 条跨轨迹 next_state 跳变；保留原训练数据，同时另报去除边界的诊断。',
        '', '## 阶段 2b：固定 V 下 Bellman 排序质量','',
        '| 环境 | V | 估计器 | 全局 Spearman | batch 内均值 | top20% 重合 | ESS ratio |',
        '|---|---|---|---:|---:|---:|---:|']
    for env in ENVS:
        for kind in ['TARGET','MIX_FB']:
            data=bellman.get((env,kind))
            if not data:lines.append('| {} | {} | 等待 V / 诊断 | — | — | — | — |'.format(env,kind));continue
            for estimator in ['knn','diff_mean_state','oracle']:
                m=data['metrics'][estimator]
                lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(env,kind,estimator,num(m['global_spearman']),num(m['batch_spearman_mean']),num(m['top20_overlap']),num(m['ess_ratio']['mean'])))
    lines+=['',
        'batch 指 seed 0 打乱后的 39 个完整 128 样本组，余下 7 条只进入全局指标；这些组不是独立训练 seeds。top20% 取 1000 条，平分用原样本索引确定；独立随机排序的期望重合率约 20%。',
        '差分主口径在 B 的均值下一状态上评价 V，奖励及 not_done 取邻居均值；另报逐邻居平移后计算 V 再平均的敏感性，不将两者混为同一估计。',
        'kNN 失配直接调用原 compute_source_weight，并核验权重公式。V 是学到的网络；oracle 的“真实”只指给定 V 下、经回放验证的转移，不代表 V 自身准确。每环境只有约 5 条 source 轨迹，4999 条转移不能当作 4999 个独立样本；相关与重合率为该数据集上的描述指标。',
        '', '## 阶段 3：seed 0 回报训练','',
        '进度：{}/{} 个训练/模型复用任务完成。新训练均为 500k 步，保存最终模型及每 100k 的 V。'.format(done,expected or '计划中'),'',
        '| 环境 | 方法 | Last-5 normalized score | 来源 |','|---|---|---:|---|']
    mapping={'TARGET':'IQL-target','MIX_FB':'IQL_BELLMAN_FB-b0.20','CONST':'IQL_CONST-w0.20','ORACLE':'IQL_BELLMAN_ORACLE-b0.20','BACKUP':'ORACLE-BACKUP'}
    for env in ENVS:
        for method,label in mapping.items():
            v=scores.get((env,method));lines.append('| {} | {} | {} | {} |'.format(env,label,num(v),'本轮新训练/经核验复用' if v is not None else '未完成'))
        lines.append('| {} | IQL-mix | {} | 既有500k seed0，历史对照 |'.format(env,num(historic.get((env,'IQL-mix')))))
    lines+=['','指标统一取 460k、470k、480k、490k、500k 的 `test/target_normalized_score` 均值。仅日志齐备、有限且任务成功时纳入，不选 best checkpoint。归一化保持原项目口径。','',
        '## 当前结论','']
    if bellman:
        for (env,kind),data in bellman.items():
            a=data['metrics']['knn']['batch_spearman_mean'];b=data['metrics']['diff_mean_state']['batch_spearman_mean']
            lines.append('- {} / {}：kNN 的 batch 内相关 {}；差分修正 {}。这是该固定 V、该 source 支持集上的结果。'.format(env,kind,num(a),num(b)))
    else:lines.append('- Bellman 一致性尚待 V 训练；不能由动力学层面误差代替。')
    if finished:
        for env in ENVS:
            delta=scores[env,'ORACLE']-scores[env,'CONST']
            lines.append('- {}：ORACLE−CONST 的 Last-5 差值 {}。单个 seed 的点差不能宣称统计显著或证明总体优势。'.format(env,num(delta)))
    else:lines.append('- oracle 权重是否提高回报仍未得出结论；必须等待所有 seed 0 对照完成。没有自动增加到 5 seeds。')
    lines+=['','## 图表','']
    for (env,kind) in bellman:
        stem=env+'_'+kind+'_bellman'
        lines.append('- [{} / {} 散点图](results/verified_20261003/figures/{}.png)；另存同名 SVG/PDF。'.format(env,kind,stem))
    error=read(OUT/'pipeline_error.json')
    if error:lines+=['','## 执行异常','',str(error.get('error'))]
    lines+=['','## 复现与文件','',
        '- verify_replay_v2.py / verify_warmstart.py / investigate_walker_precision.py / finalize_replay.py：独立回放检查和恢复证据。',
        '- prepare_diagnostics.py：阶段1与2a；bellman_diagnostics.py：阶段2b。',
        '- oracle_methods.py：索引buffer、oracle权重和oracle Q backup；train_entry.py：不改原文件的入口适配。',
        '- run_all_stages.py：单进程 GPU 队列及自动诊断/报告；queue_status.json 记录成功、失败与当前任务。',
        '- 新结果全部位于 oracle_diag/results，既有结果不覆盖；失败 smoke 的目录也保留。',
        '- 当次汇总的机器可读证据见 results/verified_20261003/report_data.json。','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    payload=dict(updated=datetime.datetime.now().isoformat(),all_stages_complete=finished,completed_jobs=done,total_jobs=expected,
        stage0=bool(acceptance.get('all_passed')),stage1_stage2a=len(geometry)==3,stage2b_completed=len(bellman),
        stage3_score_evidence=evidence,queue_plan=plan)
    (OUT/'report_data.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ['stage3_score_evidence','queue_plan']}))


if __name__=='__main__':main()
