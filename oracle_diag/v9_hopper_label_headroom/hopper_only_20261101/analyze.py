"""Hopper-only corrected report, all30 BC runs plus T/O1/O.25 raw inputs; CPU only."""
from common import *
import statistics
from audit import raw_scores
from metrics import group,welch,gate
def main():
    guard();items=[];inputs={}
    for j in jobs():items.append(dict(variant=j['variant'],seed=j['seed'],out=j['out'],every=5000,last=200000))
    for j in r.read(OLD/'T_reuse.json')['records']:
        if j['task']=='hopper-morph':items.append(dict(variant='T',seed=j['seed'],out=j['out'],every=10000,last=500000))
    for j in r.read(r.STAGE2/'training_plan.json')['jobs']:
        items.append(dict(variant='O1' if j['source_w']==1 else 'O025',seed=j['seed'],out=j['out'],every=10000,last=500000))
    assert len(items)==60
    records=[]
    for item in items:
        p=Path(item['out']);audit=r.read(p/'audit.json');done=r.read(p/'completed.json');assert audit['passed'] and done['finite']
        assert r.sha(done['final_model'])==done['final_model_sha256']
        # The frozen stage3 raw evaluator uses301 for BC,201 for reused T/oracle.
        v=raw_scores(p,dict(task='hopper-morph',seed=item['seed']),item['every'],item['last'])
        assert abs(v['Last5']-audit['Last5'])<1e-8 and abs(v['Final']-audit['Final'])<1e-8
        records.append(dict(item,**v))
        for f in [p/'evaluations.json',p/'audit.json',p/'completed.json',p/'sampling.json',p/'initialization.json',Path(done['final_model'])]+list((p/'tb').iterdir()):inputs[str(f)]=r.sha(f)
    order=['BC_all','BC50','BC25'];stats={}
    for metric in ('Last5','Final'):
        groups={}
        for variant in ['T','O1','O025']+order:
            a=sorted([v for v in records if v['variant']==variant],key=lambda v:v['seed'])
            groups[variant]=group([v[metric] for v in a],[v['seed'] for v in a])
        stats[metric]=dict(groups=groups,BC_vs_T={k:welch(groups[k],groups['T']) for k in order},
            BC_vs_oracle_descriptive={k:{o:dict(BC_mean=groups[k]['mean'],oracle_mean=groups[o]['mean'],BC_minus_oracle=groups[k]['mean']-groups[o]['mean'],no_decision=True) for o in ['O1','O025']} for k in order})
    groups=stats['Last5']['groups'];best=max(order,key=lambda k:groups[k]['mean']);effect=welch(groups[best],groups['T']);b=welch(groups['BC_all'],groups['T'])
    verdict=dict(C=best,tied_C=[k for k in order if groups[k]['mean']==groups[best]['mean']],H=effect['mean_difference'],SE_H=effect['SE'],headroom=gate(effect['mean_difference'],effect['SE']),
        T_not_strong=b['mean_difference']>2*b['SE'],BC_all_minus_T=b,no_T_retuning=True,no_Walker_started=True)
    result=dict(passed=True,Hopper_only=True,BC_runs=30,raw_records=60,statistics=stats,records=records,decision=verdict,input_sha256=inputs,
        raw_TB_and_episode_recomputed=True,audit_only_crosscheck=True,oracle_comparisons_descriptive_only=True,
        complete_trajectory_ranking_n7=True,actual_training_indices_unchanged=True,time=time.time())
    r.write(OUT/'HOPPER_RESULT.json',result)
    lines=['# V9阶段3 Hopper BC/%BC：按用户最新要求汇报并停止','',
        '**Hopper30项完成并审计。Walker0项；到此停止，等待Walker单独确认。**','',
        'BC-all全部5000行含EOF尾段；%BC仅7条完整轨迹按完整回报排序，BC50取前4条、BC25取前2条。此定义已验证与已完成训练所读行号逐位相同，没有重训。',
        '200k、cosine200k、每5k十episode；Last5=180/185/190/195/200k。R14 T seeds201–210复用，BC seeds301–310，Welch非配对SE。',
        '只训练原IQL actor，优势权重1、batch64、Adam3e-4，原始观测；不训练Q/V，无source。','',
        '|组|Last5均值|样本SD|最差/seed|Q1|Final均值|Final SD|Final最差/seed|Final Q1|','|---|---:|---:|---|---:|---:|---:|---|---:|']
    for k in ['T','O1','O025']+order:
        a=groups[k];b2=stats['Final']['groups'][k]
        lines.append('|%s|%.5f|%.5f|%.5f/%s|%.5f|%.5f|%.5f|%.5f/%s|%.5f|'%(k,a['mean'],a['sample_SD'],a['worst'],a['worst_seeds'],a['Q1'],b2['mean'],b2['sample_SD'],b2['worst'],b2['worst_seeds'],b2['Q1']))
    lines+=['','## V9预设判据','', 'C=%s，H=%+.5f，Welch SE_H=%.5f；判定：**%s**。'%(best,verdict['H'],verdict['SE_H'],verdict['headroom']),
        'BC-all−T=%+.5f，SE=%.5f，T不是强基线：**%s**。'%(b['mean_difference'],b['SE'],'是' if verdict['T_not_strong'] else '否'),
        'C是同批seed最高均值，存在选择偏差，只作V9门槛；2SE不提供精确95%或等效性保证。T调参仍需单独确认。','',
        '## BC与oracle均值对比（仅描述，不作判定）','',
        '|指标/BC组|BC均值|O1均值|BC−O1|O.25均值|BC−O.25|','|---|---:|---:|---:|---:|---:|']
    for metric in ['Last5','Final']:
        g=stats[metric]['groups']
        for k in order:lines.append('|%s/%s|%.5f|%.5f|%+.5f|%.5f|%+.5f|'%(metric,k,g[k]['mean'],g['O1']['mean'],g[k]['mean']-g['O1']['mean'],g['O025']['mean'],g[k]['mean']-g['O025']['mean']))
    lines+=['','## 逐seed BC结果','', '|seed|BC-all Last5|BC50 Last5|BC25 Last5|BC-all Final|BC50 Final|BC25 Final|','|---|---:|---:|---:|---:|---:|---:|']
    for i,seed in enumerate(range(301,311)):
        lines.append('|%d|'%seed+'|'.join('%.5f'%stats[m]['groups'][k]['values'][i] for m in ['Last5','Final'] for k in order)+'|')
    lines+=['','## BC变体间同seed配对（描述性）','', '|对比|均值差|差的样本SD|SE|正号数|','|---|---:|---:|---:|---:|']
    for k in ['BC50','BC25']:
        diff=[a-bb for a,bb in zip(groups[k]['values'],groups['BC_all']['values'])];sd=statistics.stdev(diff)
        lines.append('|%s−BC-all|%+.5f|%.5f|%.5f|%d/10|'%(k,statistics.mean(diff),sd,sd/math.sqrt(10),sum(v>0 for v in diff)))
    config=r.read(OUT/'config.json')
    lines+=['','## 轨迹段与完整回报','', 'Hopper为实际训练集；Walker以下仅列待确认的训练集规划，Walker运行数为0。','', '|任务/变体|轨迹段|长度|原始回报|归一化回报|状态|','|---|---:|---:|---:|---:|---|']
    for task,data in config['trajectory_selections'].items():
        by={s['segment']:s for s in data['segments']}
        for k,info in data['variants'].items():
            for sid in info['segments']:
                seg=by[sid];lines.append('|%s/%s|%d|%d|%.5f|%.5f|%s|'%(task,k,sid,seg['length'],seg['raw_return'],seg['normalized_return'],'完整' if seg['complete_boundary'] else 'EOF尾段，仅partial return；只进BC-all'))
    durations=[r.read(Path(j['out'])/'completed.json')['elapsed'] for j in jobs()]
    lines+=['','## 执行差异与停点','',
        '已执行Hopper合计训练/评估elapsed %.2f小时；首项实测9.90分钟，最初外推Hopper总4.95小时。原派发器17340在28项完成后按用户指示停止，仅保留seed309训练子进程；新Hopper-only派发器完成其审计并仅派发尚未开始的seed310，没有重跑。'% (sum(durations)/3600),
        '偏差：先前按8段排序但实际行号相同；先前计划Hopper后自动Walker，现已取消，Walker0项；实测结果没有在推进批量训练前先向用户汇报；原报告仅O1与BC-all比较，现补齐三BC变体对O1/O.25全部描述性比较。原代码、配置、模型、日志不覆盖。',
        '**Hopper结束后停止。Walker、阶段4、T调参及估计器均未执行，须另行确认。**','']
    report=OUT/'HOPPER_REPORT_ZH.md'
    with open(r.winpath(report),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    r.write(OUT/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,raw_records=60,report=str(report),report_sha256=r.sha(report),result_sha256=r.sha(OUT/'HOPPER_RESULT.json'),time=time.time()))
if __name__=='__main__':main()
