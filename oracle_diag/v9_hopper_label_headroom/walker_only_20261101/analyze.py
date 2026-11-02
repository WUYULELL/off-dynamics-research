from common import *
import statistics
from metrics import group,welch,gate
def main():
    guard();items=[dict(label=j['variant'],seed=j['seed'],task=j['task'],out=j['out'],every=5000,last=200000) for j in jobs()]
    items += [dict(v,every=10000,last=500000) for v in r.read(OUT/'reference_receipts.json')['records']]
    assert len(items)==40;records=[];inputs={}
    for item in items:
        p=Path(item['out']);audit_path=Path(item.get('audit',str(p/'audit.json')));a=r.read(audit_path);done=r.read(p/'completed.json');assert a['passed'] and done['finite']
        assert r.sha(done['final_model'])==done['final_model_sha256']
        v=raw_score_record(p,item,item['every'],item['last']);assert abs(v['Last5']-a['Last5'])<1e-8 and abs(v['Final']-a['Final'])<1e-8
        records.append(dict(label=item['label'],seed=item['seed'],out=item['out'],**v))
        for f in [audit_path,p/'completed.json',p/'evaluations.json',p/'initialization.json',p/'sampling.json',Path(done['final_model'])]+list((p/'tb').iterdir()):inputs[str(f)]=r.sha(f)
        for key in ('derived_receipt','audit_repair_receipt'):
            if key in item:inputs[item[key]]=r.sha(item[key])
    order=['BC_all','BC50','BC25'];refs=['R13_U','R13_G5','R13_R5','R14_w025'];stats={}
    for metric in ('Last5','Final'):
        g={}
        for label in ['T']+order+refs:
            v=sorted([a for a in records if a['label']==label],key=lambda a:a['seed']);g[label]=group([a[metric] for a in v],[a['seed'] for a in v])
        stats[metric]=dict(groups=g,BC_vs_T={k:welch(g[k],g['T']) for k in order},
            descriptive_comparisons={k:{ref:dict(BC_mean=g[k]['mean'],reference_mean=g[ref]['mean'],BC_minus_reference=g[k]['mean']-g[ref]['mean'],no_decision=True) for ref in refs} for k in order})
    g=stats['Last5']['groups'];best=max(order,key=lambda k:g[k]['mean']);e=welch(g[best],g['T']);b=welch(g['BC_all'],g['T'])
    decision=dict(C=best,tied_C=[k for k in order if g[k]['mean']==g[best]['mean']],H=e['mean_difference'],SE_H=e['SE'],headroom=gate(e['mean_difference'],e['SE']),
        T_not_strong=b['mean_difference']>2*b['SE'],BC_all_minus_T=b,C_same_seed_selection_bias_disclosed=True,stage4_started=False,T_retuned=False)
    result=dict(passed=True,Walker_only=True,BC_runs=15,references_reused=25,raw_records=40,statistics=stats,decision=decision,records=records,input_sha256=inputs,
        raw_TB_and_episode_recomputed=True,audit_only_crosscheck=True,descriptive_comparisons_have_no_decisions=True,time=time.time())
    r.write(OUT/'WALKER_RESULT.json',result)
    lines=['# V9阶段3 Walker BC/%BC结果','',
        '**Walker15项完成并审计；到此停止。阶段4与T调参均未启动。**','',
        'BC-all目标5000行含尾段；BC50仅7完整轨迹中的段6/4/7/3（3668行），BC25段6/4（2000行）。段4/6/7的1000步reset边界来自阶段0的连续性推断，原文件无timeouts字段；尾段不参与百分比排名。',
        'BC seeds301–305、200k/cosine200k、target batch64、Adam3e-4、单位优势权重，仅原IQL actor；不训练Q/V、无source。每5k十episode；Last5=180/185/190/195/200k，Final辅助，保留训练RNG。',
        'T为R13 seeds201–205，U/G5/R5同R13正式完成凭据、w=.25为R14原始标签正式结果，均复用500k/temp10；不重跑。BC与T为Welch非配对比较。','',
        '|组|Last5均值|样本SD|最差/seed|Q1|Final均值|Final SD|Final最差/seed|Final Q1|','|---|---:|---:|---|---:|---:|---:|---|---:|']
    for k in ['T']+order+refs:
        a=g[k];bb=stats['Final']['groups'][k]
        lines.append('|%s|%.5f|%.5f|%.5f/%s|%.5f|%.5f|%.5f|%.5f/%s|%.5f|'%(k,a['mean'],a['sample_SD'],a['worst'],a['worst_seeds'],a['Q1'],bb['mean'],bb['sample_SD'],bb['worst'],bb['worst_seeds'],bb['Q1']))
    lines+=['','## V9预设判定','', 'C=%s，H=%+.5f，SE_H=%.5f；按V9判为：**%s**。'%(best,decision['H'],decision['SE_H'],decision['headroom']),
        'BC-all−T=%+.5f，Welch SE=%.5f；T不是强基线：**%s**。后续比较需加入BC-all：%s。'%(b['mean_difference'],b['SE'],'是' if decision['T_not_strong'] else '否','是' if decision['T_not_strong'] else '未触发'),
        '规则原样：H≥10且H>2SE→存在提升空间；H<5→空间不足；5≤H<10→不确定。H≥10但≤2SE为原V9未覆盖分支，只报告请用户解释。C有同批选择偏差，只作门槛；2SE不提供精确95%保证。','',
        '## BC对U/G5/R5/R14w=.25的均值比较（仅描述，不作判定）','',
        '|指标/BC组|BC均值|U均值|BC−U|G5均值|BC−G5|R5均值|BC−R5|R14w.25均值|BC−R14w.25|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for metric in ('Last5','Final'):
        gg=stats[metric]['groups']
        for k in order:
            numbers=[gg[k]['mean']]
            for ref in refs:numbers.extend([gg[ref]['mean'],gg[k]['mean']-gg[ref]['mean']])
            lines.append('|%s/%s|'%(metric,k)+'|'.join('%.5f'%v for v in numbers)+'|')
    lines+=['','## 逐seed BC结果','', '|seed|BC-all Last5|BC50 Last5|BC25 Last5|BC-all Final|BC50 Final|BC25 Final|','|---|---:|---:|---:|---:|---:|---:|']
    for i,seednum in enumerate(range(301,306)):lines.append('|%d|'%seednum+'|'.join('%.5f'%stats[m]['groups'][k]['values'][i] for m in ('Last5','Final') for k in order)+'|')
    lines+=['','## BC变体之间同seed差（描述性）','', '|对比|均值差|样本SD|SE|正号数|','|---|---:|---:|---:|---:|']
    for k in ['BC50','BC25']:
        diff=[a-bb for a,bb in zip(g[k]['values'],g['BC_all']['values'])];sd=statistics.stdev(diff)
        lines.append('|%s−BC-all|%+.5f|%.5f|%.5f|%d/5|'%(k,statistics.mean(diff),sd,sd/math.sqrt(5),sum(a>0 for a in diff)))
    trajectory=r.read(OUT/'config.json')['trajectory_selection'];by={s['segment']:s for s in trajectory['segments']}
    lines+=['','## 实际使用轨迹段及其回报','', '|变体|段|长度|原始回报|归一化回报|结束证据|','|---|---:|---:|---:|---:|---|']
    for k in order:
        for sid in trajectory['variants'][k]['segments']:
            s=by[sid];lines.append('|%s|%d|%d|%.5f|%.5f|%s|'%(k,sid,s['length'],s['raw_return'],s['normalized_return'],s['end_reason']))
    duration=sum(r.read(Path(j['out'])/'completed.json')['elapsed'] for j in jobs())/3600
    lines+=['','## 耗时、异常与停点','',
        '启动前已报告总预算3–4小时；Hopper30项实测均值9.71分钟，直接外推15项约2.43小时。Walker15项实际训练/评估elapsed合计%.3f小时；队列总墙钟另见EXECUTION_SUMMARY.json。'%duration,
        '所有原始输入、复用路径和hash见reference_receipts.json及WALKER_RESULT.json；R5来自最终R13完成凭据指定的恢复目录，未把失败/中断产物当完成结果。',
        'R13 R5 seed203沿用已验收审计修复凭据：原49个TB点保留，100k点只标记为暂停前hash封存的10episode原始回报派生值，绝不称原TB测量。Last5及Final的原TB点全部存在，主指标未受该缺点影响。本次未重新评估或重训旧模型。',
        '未修改任何既有代码、配置、模型或结果，使用Hopper完全相同的冻结train/audit/actor/evaluation；不新增GPU计时探针，不重跑T或其他对照。',
        '**完成后停止，阶段4、T调参与任何追加训练均未启动。**','']
    report=OUT/'WALKER_REPORT_ZH.md'
    with open(r.winpath(report),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    r.write(OUT/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,raw_records=40,report=str(report),report_sha256=r.sha(report),result_sha256=r.sha(OUT/'WALKER_RESULT.json'),time=time.time()))
if __name__=='__main__':main()
