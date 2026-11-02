from runtime import *
import argparse
from audit import raw_scores
from metrics import group,welch,gate
def analyze(task):
    check_frozen();jobs=[j for j in read(OUT/'training_plan.json')['jobs'] if j['task']==task]
    refs=[v for v in read(OUT/'T_reuse.json')['records'] if v['task']==task]
    assert len(jobs)==(30 if task=='hopper-morph' else 15) and all((Path(j['out'])/'audit.json').exists() for j in jobs)
    records=[];inputs={}
    for j in jobs:
        p=Path(j['out']);a=read(p/'audit.json');assert a['passed'];v=raw_scores(p,j)
        assert abs(a['Last5']-v['Last5'])<1e-8 and abs(a['Final']-v['Final'])<1e-8
        records.append(dict(variant=j['variant'],seed=j['seed'],out=j['out'],**v))
        for f in [p/'evaluations.json',p/'audit.json',p/'completed.json',p/'initialization.json',p/'sampling.json']+list((p/'tb').iterdir()):inputs[str(f)]=sha(f)
    for j in refs:
        p=Path(j['out']);a=read(p/'audit.json');assert a['passed'];v=raw_scores(p,j,10000,500000)
        assert abs(a['Last5']-v['Last5'])<1e-8
        records.append(dict(variant='T',seed=j['seed'],out=j['out'],**v))
        for f in [p/'evaluations.json',p/'audit.json',p/'completed.json']+list((p/'tb').iterdir()):inputs[str(f)]=sha(f)
    order=['BC_all','BC50','BC25'];stats={}
    for metric in ('Last5','Final'):
        groups={}
        for variant in order+['T']:
            v=sorted([r for r in records if r['variant']==variant],key=lambda r:r['seed']);groups[variant]=group([r[metric] for r in v],[r['seed'] for r in v])
        stats[metric]=dict(groups=groups,comparisons_to_T={k:welch(groups[k],groups['T']) for k in order})
    groups=stats['Last5']['groups'];best=max(order,key=lambda k:groups[k]['mean']);ties=[k for k in order if groups[k]['mean']==groups[best]['mean']]
    effect=welch(groups[best],groups['T']);bc=welch(groups['BC_all'],groups['T'])
    verdict=dict(C=best,tied_best=ties,H=effect['mean_difference'],SE_H=effect['SE'],headroom=gate(effect['mean_difference'],effect['SE']),
        T_not_strong=bc['mean_difference']>2*bc['SE'],BC_all_minus_T=bc,selection_bias_disclosed=True,no_new_T_tuning=True)
    result=dict(passed=True,task=task,records=records,statistics=stats,decision=verdict,input_sha256=inputs,
        raw_TB_and_episode_recomputed=True,audit_only_crosscheck=True,time=time.time())
    prefix='HOPPER' if task=='hopper-morph' else 'WALKER'
    write(OUT/(prefix+'_RESULT.json'),result)
    lines=['# V9阶段3 '+task+' BC/%BC结果','',
        '200k actor-only IQL BC，优势权重1，cosine200k；每5k十episode，Last5=180/185/190/195/200k。T复用R14/R13的500k结果，training seed不同，以Welch非配对SE比较。','',
        '|组别|Last5均值|样本SD|最差/seed|Q1|Final均值|Final SD|','|---|---:|---:|---|---:|---:|---:|']
    for k in ['T']+order:
        a=groups[k];b=stats['Final']['groups'][k]
        lines.append('|%s|%.5f|%.5f|%.5f/%s|%.5f|%.5f|%.5f|'%(k,a['mean'],a['sample_SD'],a['worst'],a['worst_seeds'],a['Q1'],b['mean'],b['sample_SD']))
    lines+=['','C=%s；H=%+.5f；Welch SE_H=%.5f。'%(best,verdict['H'],verdict['SE_H']),
        '按V9判定：**%s**。'%verdict['headroom'],'T不是强基线：**%s**。后续须加入BC-all基线：%s。'%('是' if verdict['T_not_strong'] else '否','是' if verdict['T_not_strong'] else '未触发此规则'),
        'T调参未授权，未执行；C为同批种子中选出的最高均值，存在选择偏差，只作V9门槛，不当作精确可达上限或一般因果证明。','']
    if task=='hopper-morph':
        stage2=read(STAGE2/'LABEL_ATTRIBUTION_RESULT.json');o=stage2['statistics']['Last5']['groups']['oracle_1.0']
        od=welch(dict(n=o['n'],mean=o['mean'],sample_SD=o['sample_SD']),groups['BC_all'])
        result['O1_minus_BC_all_descriptive']=od
        lines+=['O1 Last5=%.5f；O1−BC-all=%+.5f，非配对SE=%.5f（描述性比较，不新增判据）。'%(o['mean'],od['mean_difference'],od['SE']),
            'O1的标签修正对照效应保留，是否超越简单模仿以及强基线意义需结合本表解释。','']
        # Separate receipt carries this descriptive comparison without replacing the immutable result.
        write(OUT/'HOPPER_O1_VS_BC.json',od)
    lines+=['逐seed、配对BC变体差、全部原始输入hash和配置见RESULT.json及各run。BC vs T为非配对，不能报告同seed正号数。',
        'n=8（7完整+1EOF尾段）；BC-all保留5000行。BC50/25排名的尾段未入选，n=7或8选出的数据完全相同。无BC10。',
        '无source、无Q/V训练、无T重跑、无额外调参。下一方法阶段4仍须用户确认。','']
    # Report paired differences only between new BC variants sharing the same seeds.
    lines+=['|BC变体对比|配对均值差|差的SE|正号数|','|---|---:|---:|---:|']
    import statistics,math
    for v in ['BC50','BC25']:
        d=[a-b for a,b in zip(groups[v]['values'],groups['BC_all']['values'])]
        lines.append('|%s−BC_all|%+.5f|%.5f|%d/%d|'%(v,statistics.mean(d),statistics.stdev(d)/math.sqrt(len(d)),sum(a>0 for a in d),len(d)))
    report=OUT/(prefix+'_REPORT_ZH.md')
    with open(winpath(report),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    write(OUT/(prefix+'_ANALYSIS_COMPLETED.json'),dict(passed=True,report=str(report),report_sha256=sha(report),result_sha256=sha(OUT/(prefix+'_RESULT.json')),CPU_only=True,time=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);analyze(p.parse_args().task)
