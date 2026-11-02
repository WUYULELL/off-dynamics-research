"""CPU-only audits from original event files and model artifacts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v4 import *
import argparse
import math
import statistics
import numpy as np
import torch
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


def paired(values):
    assert len(values)==5 and all(math.isfinite(v) for v in values)
    mean=statistics.mean(values); sd=statistics.stdev(values); se=sd/math.sqrt(5)
    positive=sum(v>0 for v in values)
    return dict(values=values,mean=mean,sample_sd=sd,se=se,positive=positive,
                passed=mean>2*se and positive>=4,n=5)


def events(folder):
    files=list(Path(folder).rglob('events.out.tfevents.*'))
    assert len(files)==1, files
    values={}
    for event in LegacyEventFileLoader(winpath(files[0])).Load():
        for value in event.summary.value:
            if value.WhichOneof('value')!='simple_value':
                continue
            assert math.isfinite(value.simple_value),value.tag
            tag=value.tag.replace(' ','_')
            series=values.setdefault(tag,{})
            assert event.step not in series,(tag,event.step)
            series[event.step]=float(value.simple_value)
    return files[0],values


def finite(obj):
    if torch.is_tensor(obj):assert torch.isfinite(obj).all()
    elif isinstance(obj,dict):
        for value in obj.values():finite(value)
    elif isinstance(obj,(list,tuple)):
        for value in obj:finite(value)
    elif isinstance(obj,float):assert math.isfinite(obj)


def audit_job(job):
    folder=Path(job['out']); completed=read(folder/'completed.json')
    assert all(completed[k]==job[k] for k in ('method','seed','steps'))
    cfg=read(folder/'effective_config.json'); c=cfg['config']; method=job['method']; seed=job['seed']
    assert cfg['method']==method and cfg['seed']==seed and cfg['purpose']==job['kind']
    assert cfg['protocol_sha256']==sha(OUT/'protocol.json')
    assert cfg['oracle_receipt_sha256']==sha(oracle_receipt())
    base=read(V3_OUT/('s1/t0' if job['kind']=='smoke' else 'b1/t'+str(seed))/'effective_config.json')['config']
    expected=dict(base,seed=seed,target_only=method!='ORACLE-W1',source_budget=1. if method=='ORACLE-W1' else .2)
    assert c==expected,(c,expected)
    draws=DRAW_COUNTS[method]
    assert cfg['total_batch']==sum(draws) and (cfg['source_draws'],cfg['target_draws'])==draws
    progress=read(folder/'progress.json')
    assert progress['step']==job['steps']
    assert (progress['source_draws'],progress['target_draws'])==tuple(n*job['steps'] for n in draws)
    assert read(folder/'initialization.json')==read(V3_OUT/('b1/t'+str(seed))/'initialization.json')
    first=read(folder/'first_sample.json'); rng=np.random.RandomState(seed)
    si=rng.randint(0,999322,draws[0]).tolist() if draws[0] else []
    ti=rng.randint(0,5000,draws[1]).tolist() if draws[1] else []
    assert first==dict(loader='explicit_next_v3',source_ids=si,target_ids=ti,
                       actual_source_draws=draws[0],actual_target_draws=draws[1])
    data=read(folder/'data_verified.json')
    assert data['all_source_rows_aligned'] and data['routed_data']['all_rows_equal']
    cache=read(oracle_receipt()); assert data['source_dataset_sha256']==cache['source_dataset_sha256']
    if method in SINGLE_SOURCE:
        assert data['training_buffer_n']==999322 and data['target_dataset_sha256'] is None
        assert data['loader_calls']==dict(target_dataset_calls=0,source_route_calls=1)
        assert data['routed_data']['target_dataset_used'] is False
    else:
        from task_env import paths
        assert data['training_buffer_n']==5000 and data['target_dataset_sha256']==sha(paths('walker2d-morph')[1])
    file,scalars=events(folder)
    scores=scalars['test/target_normalized_score']; raw=scalars['test/target_return']
    steps=[5000] if job['kind']=='smoke' else list(range(10000,500001,10000))
    assert sorted(scores)==sorted(raw)==steps
    evaluation=read(folder/'evaluation_verified.json')
    assert [e['step'] for e in evaluation['events']]==steps
    for row in evaluation['events']:
        step=row['step']
        calculated=(raw[step]-10.079455055289959)/(4398.43-10.079455055289959)*100
        assert abs(calculated-scores[step])<2e-5
        assert abs(row['normalized_score']-scores[step])<2e-5
        assert row['confirmed_target_env_object'] and row['time_limit']==1000
        assert row['env_class']=='gym.envs.mujoco.walker2d_v3'
        assert row['xml_sha256']==cache['xml_sha256'] and row['lookup_key']=='walker2d-morph-medium'
    if method=='ORACLE-W1':
        weights=scalars['train/source_weight_mean']
        assert sorted(weights)==[1]+list(range(5000,job['steps']+1,5000))
        assert all(w==1.0 for w in weights.values()) and progress['q_calls']==2
    early=read(folder/'early_losses.json')['events']
    assert [(e['step'],e['loss']) for e in early]==[(s,k) for s in range(1,11) for k in ('V','Q','actor')]
    assert all(math.isfinite(e['value']) for e in early)
    model_files=[p for p in folder.rglob('model*') if os.path.isfile(winpath(p))]
    value_files=list((folder/'values').glob('V_*.pt'))
    assert len(model_files)==7 and len(value_files)==(1 if job['kind']=='smoke' else 5)
    hashes={}
    for file_path in model_files+value_files:
        finite(torch.load(winpath(file_path),map_location='cpu'))
        hashes[str(file_path)]=sha(file_path)
    row=dict(method=method,seed=seed,steps=job['steps'],out=str(folder),tb_file=str(file),tb_sha256=sha(file),
        elapsed=progress['elapsed'],scores=scores,final_score=scores[job['steps']],artifact_sha256=hashes,
        last5=[scores[k] for k in read(OUT/'protocol.json')['last5_steps']] if job['kind']=='formal' else None)
    if row['last5'] is not None:row['last5_mean']=statistics.mean(row['last5'])
    assert not torch.cuda.is_initialized()
    return row


def target_reuse(job):
    assert job['method']=='TARGET' and job['kind']=='smoke'
    current=Path(job['out']); historical=V3_OUT/'s1/t0'
    assert read(current/'initialization.json')==read(historical/'initialization.json')
    assert read(current/'first_sample.json')==read(historical/'first_sample.json')
    assert read(current/'effective_config.json')['config']==read(historical/'effective_config.json')['config']
    tolerances=read(OUT/'protocol.json')['target_reuse']
    comparisons=[]
    for file in historical.rglob('model*'):
        if not os.path.isfile(winpath(file)):continue
        matches=list(current.rglob(file.name));assert len(matches)==1
        a=torch.load(winpath(file),map_location='cpu'); b=torch.load(winpath(matches[0]),map_location='cpu')
        differences=[]
        def compare(x,y):
            if torch.is_tensor(x):
                assert torch.is_tensor(y) and x.shape==y.shape and x.dtype==y.dtype
                assert torch.allclose(x,y,**tolerances['model_at_5000']),file.name
                differences.append(float((x-y).abs().max()) if x.numel() else 0.)
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for key in x:compare(x[key],y[key])
            elif isinstance(x,(tuple,list)):
                assert len(x)==len(y)
                for xx,yy in zip(x,y):compare(xx,yy)
            else:assert x==y,(x,y)
        compare(a,b)
        comparisons.append(dict(file=file.name,max_absolute_tensor_difference=max(differences,default=0.)))
    assert len(comparisons)==7
    old_score=read(historical/'evaluation_verified.json')['events'][0]['normalized_score']
    new_score=read(current/'evaluation_verified.json')['events'][0]['normalized_score']
    assert abs(old_score-new_score)<=tolerances['evaluation_absolute_score_tolerance']
    ready=read(OUT/'cpu_ready.json')
    reference=read(Path(ready['cpu_attempt'])/'target_reference_losses_cpu.json')
    observed=read(current/'early_losses.json')['events']
    assert reference['v4_cpu_bitwise_equal'] and len(observed)==len(reference['events'])==30
    for a,b in zip(reference['events'],observed):
        assert (a['step'],a['loss'])==(b['step'],b['loss'])
        assert np.isclose(a['value'],b['value'],rtol=1e-4,atol=1e-5),(a,b)
    return dict(passed=True,reference=str(historical),current=str(current),initialization_exact=True,
        first_indices_exact=True,effective_config_exact=True,model_checks=comparisons,
        original_score=old_score,repeated_score=new_score,score_absolute_difference=abs(old_score-new_score),
        early_losses='CPU reconstruction of frozen v3 matched v4 CPU exactly, new GPU within frozen tolerance; historical early trace unavailable',
        thresholds=tolerances,checked_at=time.time())


def baselines():
    """Re-read all v3 formal raw events, never infer baseline values from a report."""
    rows=[]
    for record in read(V3_OUT/'formal_audited.json')['runs']:
        file=Path(record['tb_file']);assert sha(file)==record['tb_sha256']
        scores={}
        for event in LegacyEventFileLoader(winpath(file)).Load():
            for value in event.summary.value:
                if value.tag.replace(' ','_')=='test/target_normalized_score':
                    assert event.step not in scores and math.isfinite(value.simple_value)
                    scores[event.step]=float(value.simple_value)
        assert sorted(scores)==list(range(10000,500001,10000))
        last5=[scores[s] for s in (460000,470000,480000,490000,500000)]
        assert last5==record['last5'] and statistics.mean(last5)==record['last5_mean']
        rows.append(dict(method=record['method'],seed=record['seed'],last5=last5,
                         last5_mean=statistics.mean(last5),tb_file=str(file),tb_sha256=sha(file)))
    assert len(rows)==15
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('smoke','formal','interim'),required=True);a=p.parse_args()
    plan=read(OUT/'training_plan.json')
    kind='formal' if a.kind=='interim' else a.kind
    jobs=[j for j in plan['jobs'] if j['kind']==kind and (a.kind!='interim' or j['method'] in SINGLE_SOURCE)]
    assert len(jobs)=={'smoke':5,'formal':20,'interim':10}[a.kind]
    runs=[audit_job(j) for j in jobs]
    if kind=='smoke':
        reuse=target_reuse(next(j for j in jobs if j['method']=='TARGET'))
        write(OUT/'target_reuse_verified.json',reuse)
    else:
        assert read(OUT/'target_reuse_verified.json')['passed']
    result=dict(passed=True,kind=a.kind,runs=runs,baseline_runs=baselines(),
                device='cpu',protected_files_unchanged=protected_check())
    if kind=='smoke':
        result['estimated_formal_hours_from_smoke']=sum(r['elapsed'] for r in runs if r['method']!='TARGET')*500/3600
        result['estimate_limitation']='Linear estimate includes startup and excludes final evaluation; formal learning trajectories can change evaluation time'
    write(OUT/(a.kind+'_audited.json'),result)
    print(a.kind.upper()+' AUDIT PASSED',flush=True)


if __name__=='__main__':main()
