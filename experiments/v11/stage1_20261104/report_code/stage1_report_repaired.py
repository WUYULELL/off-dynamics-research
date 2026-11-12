# -*- coding: utf-8 -*-
"""SA-all10 completed: recompute70 raw TB/episode records; unchanged V11 thresholds."""
from common import *
import numpy as np
from audit import raw_scores
from metrics import group, paired

def welch(a,b):
    return dict(mean=float(np.mean(a)-np.mean(b)),
                SE=float(np.sqrt(np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b))),paired=False)

def rule(bc,oracle):
    c1=bc['mean']>=5. and bc['mean']>2.*bc['SE']
    c2=oracle['mean']>2.*oracle['SE'] and oracle['positive']>=7
    negative=bc['mean']<2. or oracle['mean']<=0.
    assert not (c1 and c2 and negative)
    return dict(category='成立' if c1 and c2 else '不成立' if negative else '不确定',
        BC_difference_ge5=bc['mean']>=5.,BC_difference_strict_gt2SE=bc['mean']>2*bc['SE'],
        paired_O1_strict_gt2SE=oracle['mean']>2*oracle['SE'],paired_O1_positive_ge7=oracle['positive']>=7,
        false_BC_difference_lt2=bc['mean']<2.,false_paired_O1_mean_le0=oracle['mean']<=0.)

def main():
    guard();items=list(r.read(STAGE1/'reference_receipts.json')['records'])
    items += [dict(label=j['variant'],seed=j['seed'],out=j['out'],every=10000,last=500000,BC=False,
                   audit=str(Path(j['out'])/'audit.json')) for j in formal_jobs()]
    assert len(items)==70
    records=[];inputs={};diagnostics={}
    for item in items:
        path=Path(item['out']);raw=raw_scores(path,item['seed'],item['every'],item['last'],BC=item['BC'])
        audit=r.read(item['audit']);assert audit['passed']
        assert abs(audit['Last5']-raw['Last5'])<1e-8 and abs(audit['Final']-raw['Final'])<1e-8
        records.append(dict(item,Last5=raw['Last5'],Final=raw['Final']))
        for name in ('evaluations.json','initialization.json','sampling.json','completed.json','config.json'):
            inputs[str(path/name)]=r.sha(path/name)
        inputs[item['audit']]=r.sha(item['audit'])
        for p in (path/'tb').iterdir():
            if p.is_file():inputs[str(p)]=r.sha(p)
        if item['label'] in ('SA-all',):
            # Raw per-step diagnostics were independently recomputed in CPU audit; retain curves.
            diagnostics[Path(item['out']).name]=dict(curve_every1000=audit['diagnostics']['curve_every1000'],
                raw_vectors_sha256=r.sha(path/'diagnostic_vectors.npy'),raw_steps_sha256=r.sha(path/'steps.jsonl'))
            done=r.read(path/'completed.json')
            for p,h in done['sha256'].items():assert r.sha(p)==h,p
            inputs.update(done['sha256'])
    labels=('T','raw1','O1','O.25','BC-all','BC50','SA-all');stats={}
    for metric in ('Last5','Final'):
        groups={}
        for label in labels:
            values=sorted([v for v in records if v['label']==label],key=lambda v:v['seed'])
            seeds=list(range(301,311)) if label in ('BC-all','BC50') else list(range(201,211))
            assert [v['seed'] for v in values]==seeds
            groups[label]=group([v[metric] for v in values],seeds)
        effects={}
        for variant in ('SA-all',):
            a=groups[variant]['values'];effects[variant]={}
            for ref in ('T','O1','raw1'):
                effects[variant][ref]=paired(a,groups[ref]['values'],range(201,211))
            effects[variant]['BC50_Welch']=welch(a,groups['BC50']['values'])
            effects[variant]['descriptive_mean_differences']={ref:groups[variant]['mean']-groups[ref]['mean']
                for ref in ('T','raw1','O1','O.25','BC-all','BC50')}
            if metric=='Last5':effects[variant]['decision']=rule(effects[variant]['BC50_Welch'],effects[variant]['O1'])
        stats[metric]=dict(groups=groups,effects=effects)
    result=dict(passed=True,new_runs=10,raw_records=70,records=records,statistics=stats,diagnostics=diagnostics,
        input_sha256=inputs,audit_values_only_crosscheck_for_scores=True,correction=settings()['actor_loss'],
        no_bootstrap=True,all_seeds_retained=True,stage2_started=False,time=time.time())
    comparison=r.read(OUT/'SEED207_PREFIX_COMPARISON.json')
    assert comparison['comparison_finished'] and not comparison['uses_returns']
    result.update(seed207_prefix_comparison=comparison,accepted_seed207_is_fresh_rerun=True,reused_seeds=list(range(201,207)),new_executed_seeds=[207,208,209,210])
    result['input_sha256'][str(OUT/'SEED207_PREFIX_COMPARISON.json')]=r.sha(OUT/'SEED207_PREFIX_COMPARISON.json')
    r.write(STAGE1/'STAGE1_RESULT.json',result)
    lines=['# V11阶段1：Hopper源动作进入策略的oracle上限检验','',
        '**SA-all10项正式训练全部完成，阶段1结束后停下。阶段2、V9阶段4与T调参未启动。**','',
        '已按用户更正废弃−log π，采用O1实际的优势加权动作MSE：exp(10A)截断上限100；目标批次均值+w_a×源项，w_a=1。V/Q仍为目标均值+源均值，优势来自同一步V更新前target Q最小值−V。',
        '本阶段只训练SA-all，64条源样本加权MSE均值。SA-top分母已经按用户诊断修正为固定64，仅做过seed2012000步检查，未授权正式训练。复用64条V/Q source，不额外抽样。',
        'SA-all为seeds201–210，从0训练500k/cosine500k。每10k十episode，Last5固定460/470/480/490/500k；Final辅助。未按回报修改参数、筛种子或额外搜索。源oracle缓存只读复用V9已验收Hopper缓存，未新增缓存。','',
        '|组|Last5均值|样本SD|最差/seed|Q1|Final均值|Final SD|Final最差/seed|Final Q1|',
        '|---|---:|---:|---|---:|---:|---:|---|---:|']
    for label in labels:
        a=stats['Last5']['groups'][label];b=stats['Final']['groups'][label]
        lines.append('|%s|%.5f|%.5f|%.5f / %s|%.5f|%.5f|%.5f|%.5f / %s|%.5f|'%(label,a['mean'],a['sample_SD'],a['worst'],','.join(map(str,a['worst_seeds'])),a['Q1'],b['mean'],b['sample_SD'],b['worst'],','.join(map(str,b['worst_seeds'])),b['Q1']))
    for metric in ('Last5','Final'):
        lines+=['','## %s比较'%metric,'','|变体/对照|均值差|SE|正号数（仅配对）|','|---|---:|---:|---|']
        for variant in ('SA-all',):
            e=stats[metric]['effects'][variant]
            for ref in ('T','O1','raw1','BC50_Welch'):
                d=e[ref];lines.append('|%s − %s|%+.5f|%.5f|%s|'%(variant,ref,d['mean'],d['SE'],str(d['positive'])+'/10' if 'positive' in d else '非配对'))
        lines+=['','|描述性均值差|T|原始w1|O1|O.25|BC-all|BC50|','|---|---:|---:|---:|---:|---:|---:|']
        for variant in ('SA-all',):
            ds=stats[metric]['effects'][variant]['descriptive_mean_differences']
            lines.append('|%s|'%variant+'|'.join('%+.5f'%ds[k] for k in ('T','raw1','O1','O.25','BC-all','BC50'))+'|')
    lines+=['','## 预设H_act判定（仅Last5）','']
    for variant in ('SA-all',):
        d=stats['Last5']['effects'][variant]['decision']
        lines+=['**%s：%s。**'%(variant,d['category'])]
        for k,v in d.items():
            if k!='category':lines.append('- %s：%s。'%(k,'满足' if v else '不满足'))
    times={j['id']:r.read(Path(j['out'])/'completed.json')['elapsed_seconds'] for j in formal_jobs()}
    r.write(STAGE1/'EXECUTION_SUMMARY.json',dict(passed=True,new_jobs=10,total_updates=5000000,
        training_elapsed_seconds=times,total_training_hours=sum(times.values())/3600.,
        no_automatic_retraining=True,user_authorized_seed207_fresh_rerun=True,new_executed_seeds=[207,208,209,210],reused_completed_seeds=list(range(201,207)),stage0_not_reused_as_formal_runs=True,
        prelaunch_budget=r.read(OUT/'BUDGET_ESTIMATE.json'),stage2_started=False,time=time.time()))
    lines+=['','## 诊断、解释及停点','',
        '各seed逐步A>0比例、原始与归一化的目标/源权重和、目标/源截断比例和ESS（只描述）、损失范围及每1000步曲线见逐项audit.json和STAGE1_RESULT.json。CPU按每一步原始优势/MSE/权重向量复核筛选与损失；原始模型、完整采样流、TB和逐episode回报均保留。',
        'BC50为此前三个BC变体中同批均值最高者，选择/报告偏差须披露；Welch SE没有纠正该选择效应，不是精确95%保证。BC种子301–310不同于方法组201–210，BC比较非配对；O1/T/raw1同seed配对。O.25与BC-all只列描述性均值差。',
        '本结果限定于固定数据、任务、网格和训练seed；oracle标签不等于完美价值函数，源动作通道收益不能直接证明一般机制、实际可部署方法或学到的模型可行性。未改变V11阈值，无新增bootstrap或结果后实验。',
        '实际正式训练累计%.2f小时；与阶段0外推偏差及每项耗时见EXECUTION_SUMMARY.json。Stage0短试跑未充作正式500k训练，不自动恢复失败项，全部10seed保留。'%(sum(times.values())/3600.),
        '命令/科学配置/数据XML及代码hash：launch1.json、FROZEN.json、config.json、每项config.json和完成/审计收据。','',
        '**SA-all完成后停下，等待用户决定；SA-top正式训练、Walker上限检验、V9阶段4、T调参、估计器或额外训练均不启动。** 下一阶段若获授权需单独报告Walker预算，当前未测量其源动作实验吞吐。','']
    lines+=['','## 新增纯描述诊断','',
        '截断比例指批内64条原始权重经clamp(max100)后等于100的比例；SA-all全部源样本均参与。ESS=(sumw)^2/sum(w^2)，全零时记0。所有值由detach权重计算，不额外抽样或改梯度，不参与V11判定。',
        '|seed|目标截断比例|源截断比例|目标ESS|源ESS|','|---|---:|---:|---:|---:|']
    for job in formal_jobs():
        s=r.read(Path(job['out'])/'audit.json')['diagnostics']['statistics']
        lines.append('|%d|%.6f|%.6f|%.6f|%.6f|'%(job['seed'],s['target_capped_fraction']['mean'],s['source_capped_fraction']['mean'],s['target_ESS']['mean'],s['source_ESS']['mean']))
    comparison=r.read(OUT/'SEED207_PREFIX_COMPARISON.json')
    assert comparison['comparison_finished'] and not comparison['uses_returns']
    result['seed207_prefix_comparison']=comparison
    result['accepted_seed207_is_fresh_rerun']=True
    result['reused_seeds']=list(range(201,207));result['new_executed_seeds']=[207,208,209,210]
    # The result was written above. Add provenance before the sole result write instead.
    lines+=['','## 重启后的用户授权接续','',
        '201–206复用原已完成且审计通过的正式训练；207在新目录seed207_rerun从0重跑500k，随后208–210各从0训练500k。未从检查点续训，原207中断目录逐字节保留，不参与正式组统计。六旧加四新构成十个逻辑种子。',
        '重跑207前448506步V/Q/策略损失及全部逐步诊断量与原中断记录逐位比对：%s；首个不一致步数：%s。比对仅使用损失和诊断量（另核对原始诊断向量），不使用回报；不一致不改变参数、判定或后续208–210执行。'%(str(comparison['logs']['parsed_numeric_bits_equal']),str(comparison['logs']['first_mismatch'])),
        '本次新增训练四项共200万更新；原207中断产物不计入十项正式结果。原中断日志有448506完整步，准确断电时更新数未知。再次中断不会自动恢复或重训。',
        '完整比对报告、原文件保护、原CPU验收复用和逐字节科学代码一致性见SEED207_PREFIX_COMPARISON.json、SEED207_PREFIX_COMPARISON_ZH.md、PRELAUNCH_ACCEPTANCE.json、CPU_ACCEPTANCE_REUSE.json、protected_manifest.json。']
    with open(r.winpath(STAGE1/'STAGE1_REPORT_ZH.md'),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    r.write(STAGE1/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,raw_records=70,
        result_sha256=r.sha(STAGE1/'STAGE1_RESULT.json'),report_sha256=r.sha(STAGE1/'STAGE1_REPORT_ZH.md'),time=time.time()))

if __name__=='__main__':main()
