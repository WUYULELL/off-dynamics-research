"""Independent final CPU verification, no simulator or training calls."""
from runtime import *
import math
import statistics
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-9)

def main():
    check_frozen()
    done=read(OUT/'STAGE_COMPLETED.json');analysis=read(OUT/'ANALYSIS_COMPLETED.json')
    assert done['passed'] and done['all20_audited'] and done['status']=='AWAITING_STAGE3_CONFIRMATION'
    assert not done['stage3_started'] and not done['stage4_started'] and not done['estimator_development']
    assert not list(OUT.glob('queue_failure*'))
    assert sha(analysis['report'])==analysis['report_sha256']==done['report_sha256']
    assert sha(OUT/'LABEL_ATTRIBUTION_RESULT.json')==analysis['result_sha256']
    result=read(OUT/'LABEL_ATTRIBUTION_RESULT.json');assert len(result['records'])==50
    cfg=read(OUT/'protocol.json');spec=task_spec('hopper-morph');raw_values={};audits={}
    for r in result['records']:
        path=Path(r['out']);events=read(path/'evaluations.json')['events']
        assert [e['step'] for e in events]==list(range(10000,500001,10000))
        tb=EventAccumulator(str(Path(shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
        scores=[]
        for e,t in zip(events,tb):
            assert e['step']==t.step and len(e['raw_returns'])==10
            v=statistics.mean([(float(a)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100 for a in e['raw_returns']])
            assert abs(v-t.value)<1e-4;scores.append(v)
        vals=dict(Last5=statistics.mean(scores[-5:]),Final=scores[-1]);key=(r['label'],r['w'],r['seed']);raw_values[key]=vals
        a=read(r['audit']);d=read(path/'completed.json');assert a['passed'] and d['finite'] and d['steps']==500000
        assert sha(d['final_model'])==d['final_model_sha256']==a['final_model_sha256']
        assert sha(path/'completed.json')==a['completed_sha256'] and sha(path/'evaluations.json')==a['evaluations_sha256']
        for metric,v in vals.items():close(v,r[metric]);close(v,a[metric])
        audits[str(path)]=sha(r['audit'])
        if r['label']=='oracle':
            for q in result['records']:
                if q['seed']!=r['seed'] or q['label']=='oracle':continue
                assert read(path/'initialization.json')==read(Path(q['out'])/'initialization.json')
                s=read(path/'sampling.json');old=read(Path(q['out'])/'sampling.json')
                assert s['target_stream_sha256']==old['target_stream_sha256']
                if q['w']>0:assert s['source_stream_sha256']==old['source_stream_sha256']
    seeds=list(range(201,211))
    mapping={'T':('T',0.),'raw_0.25':('raw',.25),'raw_1.0':('raw',1.),'oracle_0.25':('oracle',.25),'oracle_1.0':('oracle',1.)}
    for metric in ('Last5','Final'):
        by_group={name:[raw_values[(label,w,s)][metric] for s in seeds] for name,(label,w) in mapping.items()}
        for name,v in by_group.items():
            saved=result['statistics'][metric]['groups'][name]
            close(saved['values'],v);close(saved['mean'],statistics.mean(v));close(saved['sample_SD'],statistics.stdev(v))
            close(saved['worst'],min(v));close(saved['Q1'],np.quantile(v,.25,interpolation='linear'))
            assert saved['worst_seeds']==[s for s,x in zip(seeds,v) if x==min(v)]
        for w in ('0.25','1.0'):
            effects=result['statistics'][metric]['effects'][w]
            computed={}
            for key,ref in [('L','raw_'+w),('E','T')]:
                d=[a-b for a,b in zip(by_group['oracle_'+w],by_group[ref])]
                m=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(10);pos=sum(x>0 for x in d)
                saved=effects[key];close(saved['differences'],d);close(saved['mean'],m);close(saved['SE'],se)
                assert saved['positive']==pos;computed[key]=(m,se,pos)
            if metric=='Last5':
                lm,ls,lp=computed['L'];em,es,ep=computed['E']
                expected='A' if lm>2*ls and lp>=7 and em>-2*es else 'B' if not(lm>2*ls) and em<-2*es else 'uncertain'
                assert effects['decision']['category']==expected
                assert effects['decision']['corrected_source_gain']==(expected=='A' and em>2*es and ep>=7)
    for p,h in result['input_sha256'].items():assert sha(p)==h,p
    protected=read(OUT/'protected_manifest.json')['sha256']
    for p,h in protected.items():assert sha(p)==h,p
    pre=read(OUT/'PRE_RESULT_COMMITMENT.json')
    assert sha(OUT/'protocol.json')==pre['protocol_sha256'] and sha(OUT/'training_plan.json')==pre['training_plan_sha256']
    write(OUT/'DELIVERY_VERIFIED_STAGE2.json',dict(passed=True,CPU_only=True,new_training_jobs_verified=20,R14_reuse_verified=30,
        independent_raw_records_recomputed=50,group_paired_and_V9_decisions_recomputed=True,all_input_hashes_verified=True,
        protected_old_files_verified=len(protected),report_sha256=sha(analysis['report']),result_sha256=sha(OUT/'LABEL_ATTRIBUTION_RESULT.json'),
        audit_sha256=audits,code_sha256=sha(__file__),no_new_environment_steps=True,no_new_training=True,time=time.time()))
    print('DELIVERY VERIFIED; no further training authorized',flush=True)

if __name__=='__main__':main()
