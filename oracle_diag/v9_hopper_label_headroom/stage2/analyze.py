# -*- coding: utf-8 -*-
"""Final stage2 report from all50 raw TB/episode records; audits only cross-check."""
from runtime import *
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from metrics import group,paired,decision

def main():
    check_frozen()
    jobs=read(OUT/'training_plan.json')['jobs'];assert len(jobs)==20
    items=list(read(OUT/'reuse_receipts.json')['runs'])
    items += [dict(task=j['task'],seed=j['seed'],w=j['source_w'],label='oracle',out=j['out'],audit=str(Path(j['out'])/'audit.json')) for j in jobs]
    assert len(items)==50 and all((Path(j['out'])/'audit.json').exists() for j in jobs)
    records=[];inputs={};spec=task_spec('hopper-morph');steps=list(range(10000,500001,10000))
    for item in items:
        path=Path(item['out']);events=read(path/'evaluations.json')['events']
        assert [e['step'] for e in events]==steps
        tb=EventAccumulator(str(Path(shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
        assert [s.step for s in tb]==steps
        values=[]
        for e,s in zip(events,tb):
            assert e['xml_sha256']==spec['xml_sha256'] and e['timestep']==.002
            assert e['env_seeds']==list(range(500000+10*(item['seed']-201),500010+10*(item['seed']-201)))
            scores=(np.asarray(e['raw_returns'],np.float64)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100
            assert len(scores)==10 and np.isfinite(scores).all() and abs(s.value-float(scores.mean()))<1e-4
            values.append(float(scores.mean()))
        record=dict(item,Last5=float(np.mean(values[-5:])),Final=values[-1]);records.append(record)
        audit=read(item['audit']);assert audit['passed']
        assert abs(audit['Last5']-record['Last5'])<1e-8 and abs(audit['Final']-record['Final'])<1e-8
        for n in ('evaluations.json','sampling.json','initialization.json','completed.json','audit.json'):inputs[str(path/n)]=sha(path/n)
        for p in (path/'tb').iterdir():
            if p.is_file():inputs[str(p)]=sha(p)
    seeds=list(range(201,211))
    def values(label,w,metric):
        rs=sorted([r for r in records if r['label']==label and r['w']==w],key=lambda r:r['seed'])
        assert [r['seed'] for r in rs]==seeds
        return [r[metric] for r in rs]
    stats={}
    for metric in ('Last5','Final'):
        groups={'T':group(values('T',0.,metric),seeds)}
        for w in (.25,1.):
            groups['raw_'+str(w)]=group(values('raw',w,metric),seeds)
            groups['oracle_'+str(w)]=group(values('oracle',w,metric),seeds)
        effects={}
        for w in (.25,1.):
            L=paired(values('oracle',w,metric),values('raw',w,metric),seeds)
            E=paired(values('oracle',w,metric),values('T',0.,metric),seeds)
            effects[str(w)]=dict(L=L,E=E)
            if metric=='Last5':effects[str(w)]['decision']=decision(L,E)
        stats[metric]=dict(groups=groups,effects=effects)
    result=dict(stage=2,passed=True,all20_new_runs_audited=True,records=records,statistics=stats,input_sha256=inputs,
        independent_raw_TB_and_episode_recompute=True,audit_values_only_cross_check=True,
        no_bootstrap_reported=True,no_threshold_changes=True,no_stage3_or_4_started=True,no_estimator=True,time=time.time())
    write(OUT/'LABEL_ATTRIBUTION_RESULT.json',result)
    names={'T':'T（R14复用）','raw_0.25':'原始w=.25（R14复用）','raw_1.0':'原始w=1（R14复用）','oracle_0.25':'O.25（oracle）','oracle_1.0':'O1（oracle）'}
    lines=['# V9阶段2：Hopper源数据标签归因','',
        '**20项新训练与审计全部完成；停止，等待下一阶段确认。T及原始标签两组全部复用R14。**','',
        'IQL、temp10、500k、每步64target+64source，source仅进V/Q，actor只用target；V/Q损失target.mean+w*source.mean，不额外除1+w。唯一受控变化为source的next-state/reward/terminal换成阶段1已验收的morph动力学标签。',
        '两组各seed201–210；同seed初始化、完整target/source抽样流与R14一致。每10k十episode，Last5固定460/470/480/490/500k；Final辅助，不据回报选模型/seed。','',
        '## 各组表现','', '|组|Last5均值|样本SD|最差分数/seed|Q1|Final均值|Final SD|Final最差/seed|Final Q1|',
        '|---|---:|---:|---|---:|---:|---:|---|---:|']
    for key in ('T','raw_0.25','oracle_0.25','raw_1.0','oracle_1.0'):
        a=stats['Last5']['groups'][key];b=stats['Final']['groups'][key]
        lines.append('|%s|%.5f|%.5f|%.5f / %s|%.5f|%.5f|%.5f|%.5f / %s|%.5f|'%(names[key],a['mean'],a['sample_SD'],a['worst'],','.join(map(str,a['worst_seeds'])),a['Q1'],b['mean'],b['sample_SD'],b['worst'],','.join(map(str,b['worst_seeds'])),b['Q1']))
    for metric in ('Last5','Final'):
        lines+=['','## %s同seed配对比较'%metric,'','|w / 对比量|均值差|差的样本SD|SE|正号数|','|---|---:|---:|---:|---:|']
        for w in ('0.25','1.0'):
            for key in ('L','E'):
                a=stats[metric]['effects'][w][key]
                lines.append('|%s / %s|%+.5f|%.5f|%.5f|%d/10|'%(w,'L=O−原始' if key=='L' else 'E=O−T',a['mean'],a['sample_SD'],a['SE'],a['positive']))
        lines+=['','|seed|T|原始.25|O.25|L(.25)|E(.25)|原始1|O1|L(1)|E(1)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for i,s in enumerate(seeds):
            g=stats[metric]['groups'];fx=stats[metric]['effects']
            nums=[g['T']['values'][i],g['raw_0.25']['values'][i],g['oracle_0.25']['values'][i],fx['0.25']['L']['differences'][i],fx['0.25']['E']['differences'][i],g['raw_1.0']['values'][i],g['oracle_1.0']['values'][i],fx['1.0']['L']['differences'][i],fx['1.0']['E']['differences'][i]]
            lines.append('|%d|'%s+'|'.join('%.5f'%v for v in nums)+'|')
    lines+=['','## 按V9预设规则逐项判定（仅Last5）','']
    for w in ('0.25','1.0'):
        a=stats['Last5']['effects'][w];d=a['decision']
        lines+=['### w='+w,'', '|条件|结果|','|---|---|']
        labels={'L_mean_strict_gt_2SE':'L均值严格>2SE_L','L_positive_ge7':'L正号≥7/10','E_mean_strict_gt_minus2SE':'E均值严格>−2SE_E',
            'E_mean_strict_lt_minus2SE':'E均值严格<−2SE_E','E_mean_strict_gt_2SE':'E均值严格>2SE_E','E_positive_ge7':'E正号≥7/10'}
        for k,v in d['conditions'].items():lines.append('|%s|%s|'%(labels[k],'是' if v else '否'))
        lines+=['','按预设名称判定：**%s：%s**。'%(d['category'],d['label']),
            '进一步满足修正后source相对T收益条件：**%s**。'%('是' if d['corrected_source_gain'] else '否'),'']
    lines+=['','## 解释、执行与停点','',
        'A/B名称与不确定分支严格采用用户v9规则，每个w单独判定；未修改阈值、未将其中一个w的结论替代另一个。2SE门槛是预设描述性判定，不是精确95%保证或等效检验。没有报告bootstrap区间，避免把SD比值区间误认作均值差区间。',
        '本轮方差是固定数据下训练seed方差；oracle依赖MuJoCo与轨迹历史恢复，不能据回报差直接证明独立物理真值、价值MSE或一般因果机制。所有失败/恢复证据须独立列示；不删除低分seed。',
        '各新作业命令、配置、代码/data/XML/hash位于jobs/、runs/*/config.json、launch1.json及engineering_delta.json；源缓存来源与验收见阶段1STAGE1_COMPLETED.json。',
        '全部50份原始TB/逐episode回报独立重算；audit值只交叉核验。组均值/SD/最差/Q1、逐seed L/E与原始输入hash见LABEL_ATTRIBUTION_RESULT.json。',
        '实际耗时、对原48–55小时估计的偏差、失败/恢复记录另见EXECUTION_SUMMARY.json。本阶段不额外运行吞吐探针、环境诊断或重跑R14 T/原始组。',
        '**阶段2完成后停止，阶段3 BC、阶段4近/远筛选和估计器开发均未启动。** 阶段3启动前须明确末尾不完整轨迹的处理、报告单次实测及总耗时，再由用户确认。','']
    report=DOC/'STAGE2_REPORT_ZH.md'
    with open(winpath(report),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    write(OUT/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,raw_records=50,new_training_runs=20,
        report=str(report),report_sha256=sha(report),result_sha256=sha(OUT/'LABEL_ATTRIBUTION_RESULT.json'),time=time.time()))

if __name__=='__main__':main()
