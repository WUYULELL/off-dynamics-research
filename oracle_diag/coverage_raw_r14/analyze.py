"""Independent CPU aggregation from raw TB/episode data; no audit values used as inputs."""
from runtime import *
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from metrics import group,paired,sd_ratio,best_weight

def main():
    check_frozen()
    items=list(read(OUT/'reuse_receipts.json')['runs'])
    items += [dict(task=j['task'],seed=j['seed'],w=j['source_w'],label='raw',out=j['out'],
        audit=str(Path(j['out'])/'audit.json'),audit_sha256=sha(Path(j['out'])/'audit.json')) for j in read(OUT/'training_plan.json')['jobs']]
    assert len(items)==50
    records=[];input_hashes={}
    for item in items:
        path=Path(item['out']);spec=task_spec(item['task']);events=read(path/'evaluations.json')['events']
        assert len(events)==50 and [v['step'] for v in events]==list(range(10000,500001,10000))
        tb=EventAccumulator(str(Path(shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
        assert [v.step for v in tb]==[v['step'] for v in events]
        values=[]
        for scalar,event in zip(tb,events):
            score=(np.asarray(event['raw_returns'])-spec['score_min'])/(spec['score_max']-spec['score_min'])*100.
            assert len(score)==10 and np.isfinite(score).all() and abs(scalar.value-score.mean())<1e-4
            assert event['xml_sha256']==spec['xml_sha256'] and event['timestep']==.002
            values.append(float(score.mean()))
        assert [v['step'] for v in events[-5:]]==[460000,470000,480000,490000,500000]
        record=dict(item,Last5=float(np.mean(values[-5:])),Final=values[-1]);records.append(record)
        old=read(item['audit']);assert old['passed'] and abs(record['Last5']-old['Last5'])<1e-8 and abs(record['Final']-old['Final'])<1e-8
        for file in ('evaluations.json','sampling.json','initialization.json','completed.json'):
            input_hashes[str(path/file)]=sha(path/file)
        for file in (path/'tb').iterdir():
            if file.is_file():input_hashes[str(file)]=sha(file)
    tasks={}
    for name in ('walker2d-morph','hopper-morph'):
        seeds=list(range(201,206 if name=='walker2d-morph' else 211))
        order=sorted([r for r in records if r['task']==name],key=lambda r:r['seed'])
        def values(w,label,metric):
            chosen=[r for r in order if r['w']==w and r['label']==label]
            assert [r['seed'] for r in chosen]==seeds
            return [r[metric] for r in chosen]
        result={}
        for metric in ('Last5','Final'):
            groups={str(w):group(values(w,'raw',metric),seeds) for w in (0.,.25,1.)}
            base=groups['0.0']['values']
            result[metric]=dict(groups=groups,paired_vs_T={str(w):paired(groups[str(w)]['values'],base,seeds) for w in (.25,1.)},
                SD_ratios={str(w):sd_ratio(groups[str(w)]['values'],base,name+'|'+metric+'|'+str(w)) for w in (.25,1.)})
            if name=='walker2d-morph':
                oracle=values(1.,'R13_oracle',metric)
                result[metric]['R13_U_oracle']=dict(group=group(oracle,seeds),paired_vs_T=paired(oracle,base,seeds),
                    SD_ratio=sd_ratio(oracle,base,name+'|'+metric+'|oracle'))
        result['best_w_Last5']=best_weight(result['Last5']['groups']);tasks[name]=result
    best_different=tasks['walker2d-morph']['best_w_Last5']['selected_w']!=tasks['hopper-morph']['best_w_Last5']['selected_w']
    result=dict(stage='R14_precheck_only',tasks=tasks,records=records,best_w_different=best_different,
        original_labels_only_in_new_source_runs=True,no_estimator_developed=True,no_threshold_changes=True,
        variance_object='fixed dataset, training random seeds; not target dataset sampling variance',
        independent_recomputed_from_raw_TB_and_episodes=True,input_sha256=input_hashes,time=time.time())
    write(OUT/'PRECHECK_RESULT.json',result)
    lines=['# R14：原始标签 source 强度前置验证','',
        '**40项新训练完成；停止，不开发估计器、不追加其他实验。**','',
        '固定数据下的训练种子方差，不是重新抽取target数据导致的统计估计方差。',
        'IQL，temp10、500k；混合64target+64source，source只进V/Q、actor只用target；source使用原始next_state/reward/terminal，无oracle。',
        'w=0/.25/1分别对应相对权重λ=0/.2/.5；损失为target均值+w×source均值，未额外除以1+w。这一λ换算不意味着与归一化混合目标在固定学习率下的完整优化轨迹等价。',
        'Hopper沿用temp10，未单独调参；T温度原网格上界和算法族未必最强的限制保留。','']
    for name,res in tasks.items():
        lines += ['## '+name,'','|组|Last-5均值|样本SD|最差种子分数|最差seed|下四分位数|Final均值|',
            '|---|---:|---:|---:|---|---:|---:|']
        for w in ('0.0','0.25','1.0'):
            a=res['Last5']['groups'][w];f=res['Final']['groups'][w]
            lines.append('|w=%s|%.5f|%.5f|%.5f|%s|%.5f|%.5f|'%(w,a['mean'],a['sample_SD'],a['worst'],','.join(map(str,a['worst_seeds'])),a['lower_quartile'],f['mean']))
        if 'R13_U_oracle' in res['Last5']:
            a=res['Last5']['R13_U_oracle']['group'];f=res['Final']['R13_U_oracle']['group']
            lines.append('|R13 U（oracle，w=1）|%.5f|%.5f|%.5f|%s|%.5f|%.5f|'%(a['mean'],a['sample_SD'],a['worst'],','.join(map(str,a['worst_seeds'])),a['lower_quartile'],f['mean']))
        lines+=['','|与T比较|Last-5配对均值差|差的样本SD|SE|正号|SD(w)/SD(T)|描述性95% bootstrap区间|',
            '|---|---:|---:|---:|---:|---:|---|']
        rows=[(str(w),res['Last5']['paired_vs_T'][str(w)],res['Last5']['SD_ratios'][str(w)]) for w in (.25,1.)]
        if 'R13_U_oracle' in res['Last5']:
            o=res['Last5']['R13_U_oracle'];rows.append(('R13 U oracle',o['paired_vs_T'],o['SD_ratio']))
        for label,p,v in rows:
            interval=v['bootstrap_percentile_interval'];ci='不可识别' if interval is None else '[%.5f, %.5f]'%tuple(interval)
            lines.append('|%s|%+.5f|%.5f|%.5f|%d/%d|%s|%s|'%(label,p['mean'],p['sample_SD'],p['SE'],p['positive'],p['n'],'不可识别' if v['point'] is None else '%.5f'%v['point'],ci))
        lines+=['','### 逐seed Last-5','', '|seed|T|w=.25|w=1|'+('R13 U oracle|' if name=='walker2d-morph' else ''),
            '|---|---:|---:|---:|'+('---:|' if name=='walker2d-morph' else '')]
        for i,s in enumerate(res['Last5']['groups']['0.0']['seeds']):
            row='|%d|'%s+'|'.join('%.5f'%res['Last5']['groups'][w]['values'][i] for w in ('0.0','0.25','1.0'))+'|'
            if name=='walker2d-morph':row+='%.5f|'%res['Last5']['R13_U_oracle']['group']['values'][i]
            lines.append(row)
        lines+=['','按Last-5均值选出的有限网格最佳w：**%s**（λ=%s）。'%(res['best_w_Last5']['selected_w'],res['best_w_Last5']['selected_lambda']),'']
    lines+=['## 解释与边界','',
        '两个任务的网格最佳w是否不同：**%s**。'%('是' if best_different else '否'),
        '这只是已测试{0,.25,1}三点的描述性结果，按同批seed选出并报告，存在选择偏差；不是连续最优权重或其一致估计。最佳点相同也不能证明自适应普遍无必要，最佳点不同也不能证明估计器已可行。',
        'SD下降、最差值或Q1改善不能单独证明价值估计MSE下降或偏差—方差因果机制。Hopper10seed与Walker5seed的下尾统计精度不同。',
        'Q1为线性插值25%分位；SD用ddof1。bootstrap按同seed联合重采样T与比较组，20000个非退化样本，百分位[2.5%,97.5%]；少量seed下只作描述，不声称精确95%或新正式门槛。',
        'Walker复用的T/U来自已看过结果的R13；本阶段是在看到这些历史结果及文献核查后提出，协议在任何本阶段新结果前固定。',
        '原≥5分等后续门槛不调整，任何结果均不自动触发估计器开发。完成后停止，由用户决定后续。',
        '全部逐seed配对差值、Final辅助统计、bootstrap种子与拒绝数、原始输入哈希见PRECHECK_RESULT.json。','']
    report=DOC/'DELIVERY_REPORT_R14.md'
    with open(report,'x',encoding='utf-8') as file:file.write('\n'.join(lines))
    write(OUT/'ANALYSIS_COMPLETED.json',dict(passed=True,CPU_only=True,raw_runs=50,new_training_runs=40,
        estimator_development=False,report=str(report),report_sha256=sha(report),result_sha256=sha(OUT/'PRECHECK_RESULT.json'),time=time.time()))

if __name__=='__main__':main()
