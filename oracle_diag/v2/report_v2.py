# -*- coding: utf-8 -*-
"""One evidence-backed report; pending training never becomes a conclusion."""
from support import *
import math,datetime
import numpy as np
def get(p):return read(p) if p.exists() else None
def f(x):return '—' if x is None else '{:.5g}'.format(x)
def d(x):return ' / '.join(f(x[k]) for k in ['mean','median','p95','max'])
def score(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    found=[]
    for p in folder.rglob('events.out.tfevents*'):
        a=EventAccumulator('\\\\?\\'+str(p.resolve()),size_guidance={'scalars':0});a.Reload()
        tag=next((t for t in a.Tags()['scalars'] if t.replace(' ','_')=='test/target_normalized_score'),None)
        if not tag:continue
        s={v.step:float(v.value) for v in a.Scalars(tag)}
        steps=list(range(460000,500001,10000))
        if max(s)==500000 and all(k in s for k in range(10000,500001,10000)):
            vals=[s[k] for k in steps];assert np.isfinite(vals).all();found.append(dict(mean=float(np.mean(vals)),values=vals,file=str(p),evaluation_count=len(s)))
    assert len(found)<=1,'Ambiguous event source'
    return found[0] if found else None
def main():
    a=get(OUT/'stage_a/completed.json');b=get(OUT/'stage_b/completed.json');c=get(OUT/'stage_c/completed.json');extra=get(OUT/'stage_b_new_tasks/completed.json');q=get(OUT/'queue_status.json')
    scores={};evidence=[]
    for j in (q or {}).get('jobs',[]):
        if j['state']=='completed':
            s=score(Path(j['out']))
            if s:scores[j['seed'],j['method']]=s['mean'];evidence.append(dict(seed=j['seed'],method=j['method'],**s))
    methods=['TARGET','CONST','BACKUP']+(['MODEL'] if b and b['selected'] else [])
    done=bool(a and b and c and extra and len(scores)==5*len(methods) and get(OUT/'final_validation.json'))
    text=['# Oracle v2：方案审查、全阶段结果与结论','',
          '**'+('所有阶段完成，以下为 fixed_v2 loader 的正式五种子结果。' if done else '实验尚未全部完成；下列已完成诊断有效，D 的最终结论等待完整五种子结果。')+'**','',
          '更新：'+datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'',
          '## 1. 审查后的协议','',
          '执行 A–D，E 仅估计工作量。用户已授权完成所有阶段。保留原预算 0.20、eta=1、500k 和五种子门槛，修正解释与验证方法。完整审查见 [REVIEW_AND_PROTOCOL.md](<'+(HERE/'REVIEW_AND_PROTOCOL.md').as_posix()+'>)。','',
          '- 单次运行差异不能估计噪声底；checkpoint 波动不能代替种子方差。','- 未通过 D 门槛仅表示本轮证据不足，不能证明没有提升空间。','- 模型按 target 轨迹分割训练/验证，source oracle 不调参数或 gate；候选按事先固定的顺序筛选。','- 首次训练更新统一重置随机种子；检查初始网络、target 子集与首批采样。原文件不改，旧 loader 与 fixed_v2 不混表。','- IQL-target 每步 256 target；其他方法每步 128 source＋128 target。CONST−TARGET 是相同总采样预算下的替换比较，不能解释为固定 target 样本量时添加 source 的纯效应。','',
          '## 2. 阶段完成情况','', '| 阶段 | 状态 |','|---|---|',
          '| A：加载器、Q 目标、规模审计 | '+('完成' if a else '等待')+' |','| B：原 Walker 修正器评测 | '+('完成' if b else '进行中/等待')+' |',
          '| C：四个新任务回放验收 | '+('完成' if c else '进行中/等待')+' |','| B 补充：新任务修正器评测 | '+('完成' if extra else '进行中/等待')+' |',
          '| D：500k 五种子训练 | {}/{} 完成 |'.format(len(scores),5*len(methods)),'| E：下一设定 | 仅估算，未执行 |','']
    if a:
        text+=['## 3. 阶段 A1：loader 修复（fixed_v2）','','next_observations 使用文件的显式下一状态，保留完整 N 行；终止、奖励、dtype 保持原口径。原文件未修改。','',
          '| 环境 | 原条数→修复后 | 改动行 | 新增末行状态误差 | 全量回放状态误差：均值 / 中位 / p95 / 最大 |','|---|---|---|---:|---|']
        for env,r in a['environments'].items():text.append('| {} | {}→{} | {} | {} | {} |'.format(env,r['original_n'],r['fixed_n'],','.join(map(str,r['changed_rows'])),f(r['final_row_error']),d(r['source_replay_error'])))
        text+=['','边界行与上一轮标记逐行对应；新增末行状态≤1e-3、奖励≤1e-4且终止一致。各 oracle NPZ 与完整 source 一一对齐，SHA256 及详细逐行差值见 stage_a/*_loader_audit.json。','',
               '## 4. 阶段 A2：旧运行审计（original，单列）','']
        r=read(OUT/'stage_a/hopper_q_targets_original.json')
        text+=['给定 Hopper CONST 的 500k V，两种 Q 目标绝对差均值 **{}**，最大 **{}**；超过 1e-4 的行数 **{}**，位级相同：**{}**。12 个边界行全部 nd=0。'.format(f(r['difference']['mean']),f(r['difference']['max']),r['over_1e4'],r['bitwise_equal']),
          '', 'BACKUP−CONST 的旧运行 Last-5 差为 **{:+.3f}**。这是一次观测差异，不能据此得出“噪声底为 8 分”；Q 目标也并非完全相同。'.format(r['backup_minus_const']),'',
          '| 环境 | 方法 | Last-5 均值 | 样本 SD | 极差 | 300k–500k 21 点样本 SD |','|---|---|---:|---:|---:|---:|']
        for r in read(OUT/'stage_a/training_variability_original.json')['runs']:text.append('| {} | {} | {} | {} | {} | {} |'.format(r['env'],r['method'],f(r['last5_mean']),f(r['last5_std_ddof1']),f(r['last5_range']),f(r['tail21_std_ddof1'])))
        text+=['','## 5. 阶段 A3：规模事实（fixed_v2）','', '| 环境 | source | 轨迹片段 | seed0 target 10% | source/target | 500k×128/N |','|---|---:|---:|---:|---:|---:|']
        for env,r in a['environments'].items():text.append('| {} | {} | {} | {} | {:.2%} | {:.0f} |'.format(env,r['fixed_n'],r['trajectory_segments'],r['target_n'],r['source_target_ratio'],r['expected_draws_per_row']))
        text+=['','12800 次是均匀有放回采样下每条 source 的期望抽取次数，不是新增独立信息。数据量小与反复重用要一起解释。','']
    if b:
        text+=['## 6. 阶段 B：Walker2d 修正器（fixed_v2）','','C=不修正；A=邻居下一状态均值；B=当前状态＋平均邻居位移；M=五模型集成；M-gate=target 验证分歧 90% 分位门控。','',
            '| 估计器 | 状态误差均值 / 中位 / p95 / 最大 | 比 C 更准的行比例 |','|---|---|---:|']
        for k,m in b['metrics'].items():text.append('| {} | {} | {:.2%} |'.format(k,d(m['state']),m['better_than_C_fraction']))
        text+=['','| 固定 V | 估计器 | 目标误差均值 / 中位 | C 均值 / 中位 | 均值改善 | 中位改善 |','|---|---|---|---|---:|---:|']
        for k,m in b['metrics'].items():
            for v,z in m['bellman'].items():text.append('| {} | {} | {} / {} | {} / {} | {:.1%} | {:.1%} |'.format(v,k,f(z['error']['mean']),f(z['error']['median']),f(z['C_error']['mean']),f(z['C_error']['median']),z['mean_relative_improvement'],z['median_relative_improvement']))
        text+=['','通过全部门槛的候选：**{}**；进入 D 的模型候选：**{}**。'.format(', '.join(b['passed_candidates']) or '无',b['selected'] or '无，D 只跑 oracle 对照'),
          'target 验证状态误差：{}；奖励绝对误差：{}。gate 阈值 {}，source 采用 M 比例 {:.2%}；集成分歧与 source oracle 状态误差 Spearman={}。'.format(d(b['target_validation']['state_error']),d(b['target_validation']['reward_abs']),f(b['gate_threshold']),b['source_gate_fraction'],f(b['disagreement_oracle_state_error_spearman'])),
          '', '误差更小只说明给定固定 V 和支持集上的预测改善；两个 V 均是学习所得，不代表价值函数准确。阈值在 target 验证集固定，source oracle 仅用于候选门槛评估。','']
    if c:
        text+=['## 7. 阶段 C：新任务回放与 shift（fixed_v2）','', '| 数据集 | 通过 / N | 截断率 | 状态误差均值 / 中位 / p95 / 最大 | 奖励误差均值 / 中位 / p95 / 最大 | 修复行 / 剩余失败 |','|---|---:|---:|---|---|---|']
        for k,r in c['datasets'].items():text.append('| {} | {} / {} | {:.2%} | {} | {} | {} / {} |'.format(k,r['n']-len(r['failed_rows']),r['n'],r['clipped_fraction'],d(r['state_error']),d(r['reward_error']),len(r['repaired_rows']),len(r['failed_rows'])))
        text+=['','| 数据集 | shift 均值 / 中位 / p95 / 最大 | 超过 1e-3 比例 |','|---|---|---:|']
        for k,r in c['datasets'].items():text.append('| {} | {} | {:.2%} |'.format(k,d(r['physical_shift']) if r['accepted'] else '未验收，不解释',r.get('shift_over_1e3_fraction',0)))
        text+=['','四个任务均通过时优先考虑 Hopper-morph 与 Walker2d-morph：前者补充独立环境，后者提供另一种较大的 Walker shift；gravity 可作为较小差异对照。回放成功不保证 RL 提升。','',
            'gravity XML 对照（0.5 对应项）：']
        for k,r in c['datasets'].items():
            for x in r.get('repository_xml_comparison',[]):
                if '_0.5.xml' in x['path']:text.append('- {}：字节相同={}；忽略换行表示后的文本 diff 行数={}。'.format(k,x['byte_identical'],x['diff_lines']))
        text+=['']
    if extra:
        text+=['## 8. 新任务上的 B 补充（fixed_v2）','','同一环境复用 target-only 动力学模型，禁止使用新 source oracle 调参。V 仍为上一轮该环境的 TARGET/MIX_FB。Hopper 的隐含速度被观测截断，M 的 healthy 判定只能使用可观测投影，不能宣称完整恢复隐藏速度健康条件；这些补充结果不进入 Walker-kinematic 的 D 候选选择。','',
          '| 任务 | 估计器 | 状态误差均值（C） | TARGET 误差均值 / 中位（C） | MIX_FB 误差均值 / 中位（C） |','|---|---|---|---|---|']
        for task,r in extra['tasks'].items():
            for k,m in r['metrics'].items():
                vals=[]
                for v in ['TARGET','MIX_FB']:
                    x=m['bellman'][v];vals.append('{} / {}（{} / {}）'.format(f(x['error']['mean']),f(x['error']['median']),f(x['C_error']['mean']),f(x['C_error']['median'])))
                text.append('| {} | {} | {}（{}） | {} | {} |'.format(task,k,f(m['state']['mean']),f(m['C_state']['mean']),*vals))
        text+=['']
    text+=['## 9. 阶段 D：Walker2d 五种子训练（fixed_v2）','','主指标：460k、470k、480k、490k、500k 的 test/target_normalized_score 均值。预设主比较 BACKUP−CONST；若有 MODEL 再报告 MODEL−CONST。','',
          '| seed | TARGET | CONST | BACKUP |'+(' MODEL |' if 'MODEL' in methods else ''),'|---|'+'---:|'*len(methods)]
    for seed in range(5):text.append('| {} | {} |'.format(seed,' | '.join(f(scores.get((seed,m))) for m in methods)))
    conclusions=[]
    if len(scores)==5*len(methods):
        text+=['','| 配对比较 | 五个差值 | 均值 | 样本 SD | SE | 正号数 | 门槛 |','|---|---|---:|---:|---:|---:|---|']
        for m,base in [('BACKUP','CONST'),('CONST','TARGET')]+([('MODEL','CONST')] if 'MODEL' in methods else []):
            delta=np.array([scores[s,m]-scores[s,base] for s in range(5)]);mean=float(delta.mean());sd=float(delta.std(ddof=1));se=sd/np.sqrt(5);positive=int((delta>0).sum());passed=mean>2*se and positive>=4
            text.append('| {}−{} | {} | {} | {} | {} | {} | {} |'.format(m,base,', '.join(f(x) for x in delta),f(mean),f(sd),f(se),positive,'通过' if passed else '未通过'))
            conclusions.append(dict(method=m,baseline=base,delta=delta.tolist(),mean=mean,sd=sd,se=se,positive=positive,passed=bool(passed)))
        primary=conclusions[0]
        text+=['','**主比较结论：'+('满足预设的提升证据门槛。' if primary['passed'] else '未满足预设门槛，本轮未取得足够的提升证据；不等于证明效应为零。')+'**','',
            '门槛属于预先约定的工程决策，不冒称严格 95% 显著性检验。五 seed 小样本，单任务，选择同一 source 支持集上的候选，外推需谨慎。']
    else:text+=['','**五种子尚未齐备，暂不判断提升。**']
    text+=['','## 10. 阶段 E 的工作量估计与限制','','E 未执行。若仅改变 target 比例并保留 TARGET/CONST/BACKUP 五种子，至少再需 15 次 500k；计算量约为本轮基础 D 的一轮。改成大量正常 source→少量 shifted target 则需要重做数据方向、评估环境和 oracle 对齐验收，不能直接复用当前结果。','',
        '限制：回放为经过阈值验证的近似状态恢复；Hopper healthy 隐变量不完全可观测；fixed V 未经真实回报校准；单任务五种子和少量 source 轨迹不支持普适优势；旧训练方差只作描述；相同 seed 不代表所有 GPU 运算完全位级确定。','',
        '## 11. 证据与最终交付','','代码：oracle_diag/v2；结果：oracle_diag/results/v2_20261006。preregistered_protocol.json 保存预定标准与旧文件哈希；stage_a、stage_b、stage_b_new_tasks、stage_c 保存逐项数据；stage_d 保存训练日志、模型及快照；queue_status.json 保存队列；final_validation.json 在全部结束后才产生。','',
        '本轮主要问题：loader 修复了显式 next_state 与最后一行；是否存在合格的非模拟器修正器以 B 门槛为准；oracle 是否稳定优于常数权重以 D 完整配对表为准。未完成项不提前生成结论。','']
    (HERE/'REPORT_v2.md').write_text('\n'.join(text),encoding='utf-8')
    write(OUT/'report_data.json',dict(all_stages_complete=done,completed_training=len(scores),total_training=5*len(methods),scores=evidence,paired=conclusions),replace=True)
    print('REPORT',len(scores),'/',5*len(methods),'complete',done,flush=True)
if __name__=='__main__':main()
