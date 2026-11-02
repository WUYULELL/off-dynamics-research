"""CPU audit: full model/Adam/cosine, full draw streams, raw episode and actor algebra."""
from common import *
import argparse
import numpy as np
import torch
from learner import initialize as original_initialize

def raw_scores(path,seed,every,last,BC=False,require_tb=True):
    events=r.read(path/'evaluations.json')['events'];steps=list(range(every,last+1,every))
    assert [e['step'] for e in events]==steps
    spec=r.task_spec(TASK);values=[]
    for e in events:
        base=301 if BC else 201
        assert e['env_seeds']==list(range(500000+10*(seed-base),500010+10*(seed-base)))
        assert e['xml_sha256']==spec['xml_sha256'] and abs(e['timestep']-.002)<1e-12
        assert e['preserve_training_RNG'] and e['actual_shifted_object']
        assert e['normalization_min']==spec['score_min'] and e['normalization_max']==spec['score_max']
        a=(np.asarray(e['raw_returns'],np.float64)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100.
        assert a.shape==(10,) and np.isfinite(a).all()
        assert np.allclose(a,e['scores'],rtol=0,atol=1e-9) and abs(float(a.mean())-e['mean'])<1e-9
        assert all(1<=v<=1000 for v in e['episode_lengths'])
        values.append(float(a.mean()))
    if require_tb:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tb=EventAccumulator(str(Path(r.shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
        assert [e.step for e in tb]==steps and np.allclose([e.value for e in tb],values,rtol=0,atol=1e-4)
    return dict(Last5=float(np.mean(values[-5:])),Final=values[-1],values=values)

def diagnose(path,variant,steps,wa,validate=True):
    vectors=np.load(r.winpath(path/'diagnostic_vectors.npy'),mmap_mode='r') if wa else None
    if wa:assert vectors.shape==(steps,64,6) and vectors.dtype==np.float32
    accum={};curves=[];block={};count=0;zero_selected_batches=0
    keys=('V','Q','actor','source_A_positive_fraction','target_weight_sum','source_weight_sum',
          'target_normalized_weight_sum','source_normalized_weight_sum','source_selected','target_actor_term','source_actor_term')
    with open(r.winpath(path/'steps.jsonl'),encoding='utf-8') as f:
        for i,line in enumerate(f):
            row=json.loads(line);assert row['step']==i+1
            assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
            if not wa:
                assert row['losses_bitwise_equal'] and all(row[k]==row['O1_'+k] for k in ('V','Q','actor'))
            else:
                a=vectors[i];assert np.isfinite(a).all()
                ta,sa,tmse,smse,tw,sw=a.T
                assert np.all(tmse>=0) and np.all(smse>=0)
                expected_t=np.exp(10.*ta,dtype=np.float32).clip(max=100.)
                raw_s=np.exp(10.*sa,dtype=np.float32).clip(max=100.)
                selected=np.ones(64,dtype=bool)
                if variant=='SA-top':
                    selected[:]=False
                    ix=np.lexsort((np.arange(64),-sa))[:16];selected[ix]=sa[ix]>0
                denom=64 if variant=='SA-all' else int(selected.sum())
                expected_s=raw_s*selected
                assert np.allclose(tw,expected_t,rtol=3e-6,atol=1e-7)
                assert np.allclose(sw,expected_s,rtol=3e-6,atol=1e-7)
                independent=dict(source_A_positive_fraction=float((sa>0).mean()),target_weight_sum=float(tw.sum()),
                    source_weight_sum=float(sw.sum()),target_normalized_weight_sum=float(tw.mean()),
                    source_normalized_weight_sum=float(sw.sum()/denom) if denom else 0.,
                    source_selected=int(selected.sum()),target_actor_term=float(np.mean(tw*tmse)),
                    source_actor_term=float(np.sum(sw*smse)/denom) if denom else 0.)
                independent['actor']=independent['target_actor_term']+independent['source_actor_term']
                assert row['source_denominator']==denom and row['source_coefficient']==1.
                for k,v in independent.items():assert np.isclose(row[k],v,rtol=3e-5,atol=2e-6),(i,k,row[k],v)
            if wa and row['source_selected']==0:zero_selected_batches+=1
            for k in keys:
                if k not in row:continue
                v=float(row[k])
                s=accum.setdefault(k,dict(sum=0.,min=v,max=v))
                s['sum']+=v;s['min']=min(s['min'],v);s['max']=max(s['max'],v)
                block[k]=block.get(k,0.)+v
            count+=1
            if count%1000==0:
                curves.append(dict(step=count,**{k:v/1000. for k,v in block.items()}));block={}
    assert count==steps
    return dict(steps=count,statistics={k:dict(mean=v['sum']/count,min=v['min'],max=v['max']) for k,v in accum.items()},
        curve_every1000=curves,all_steps_source_weight_and_MSE_recomputed=bool(wa),
        zero_selected_batches=zero_selected_batches,zero_selected_batch_fraction=zero_selected_batches/count if wa else None)

def main(trial_id,formal):
    guard();r.configure_cpu();assert not torch.cuda.is_initialized()
    if formal:
        job=next(j for j in formal_jobs() if j['id']==trial_id)
        path=Path(job['out']);seed,steps=job['seed'],500000
    else:
        entry=next(j for j in TRIALS if j[0]==trial_id)
        path=OUT/'runs'/trial_id;seed,steps=201,entry[3]
    config=r.read(path/'config.json');done=r.read(path/'completed.json')
    assert done['passed'] and done['steps']==steps and done['finite']
    for p,h in done['sha256'].items():assert r.sha(p)==h,p
    expected=dict(settings()['learner'],seed=seed);assert config['learner']==expected
    d=torch.load(r.winpath(path/'model_final.pt'),map_location='cpu')
    assert full_finite(d) and d['config']==expected and d['step']==steps
    assert d['actor_scheduler']['last_epoch']==steps and d['actor_scheduler']['T_max']==500000
    for key in ('V_optimizer','Q_optimizer','actor_optimizer'):
        assert d[key]['state'] and all(float(s['step'])==steps for s in d[key]['state'].values())
    original=original_initialize(expected,'cpu')
    fixture=r.read(r.OUT/'PAIRING_EXPECTATIONS.json')['seeds'][str(seed)]['initialization']
    assert original.digest()==fixture==r.read(path/'initialization.json')['sha256']
    state={k:v for k,v in d.items() if k!='config'};assert tree_digest(state)==done['scientific_state_sha256']
    sampling=r.read(path/'sampling.json')
    for purpose,n in (('target',5000),('source',999998)):
        assert sampling[purpose+'_draws']==steps*64
        assert sampling[purpose+'_stream_sha256']==r.stream_hash(seed,purpose+'_sample',n,steps,TASK)
    if not config['actor_source_w']:
        p=r.read(path/'PARITY_RESULT.json');assert p['passed']
        reference=torch.load(r.winpath(p['original_model']),map_location='cpu')
        assert r.sha(p['original_model'])==p['original_model_sha256']
        assert tree_digest({k:v for k,v in reference.items() if k!='config'})==tree_digest(state)
    raw=raw_scores(path,seed,10000,steps,require_tb=formal)
    diag=diagnose(path,config['variant'],steps,config['actor_source_w'])
    r.write(path/'audit.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,steps=steps,
        all_four_networks_three_Adam_cosine_finite=True,full_sampling_and_initialization_passed=True,
        actor_algebra_all_steps_passed=True,diagnostics=diag,Last5=raw['Last5'] if formal else None,
        Final=raw['Final'],completed_sha256=r.sha(path/'completed.json'),model_sha256=r.sha(path/'model_final.pt'),time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',required=True);p.add_argument('--formal',action='store_true')
    a=p.parse_args();main(a.trial,a.formal)
