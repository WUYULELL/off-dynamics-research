"""CPU-only per-run audit, including full sampling streams and raw evaluation returns."""
from runtime import *
import argparse
import numpy as np
import torch
from learner import initialize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def finite_tree(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return not isinstance(value,float) or bool(np.isfinite(value))

def audit(path):
    configure_cpu();path=Path(path);done=read(path/'completed.json');job=done['job'];check_job(job,False)
    assert done['steps']==500000 and done['finite'] and sha(done['final_model'])==done['final_model_sha256']
    config=read(path/'config.json');spec=task_spec(job['task'])
    assert config['learner']==model_config(10,500000,job['seed'],job['task'],job['source_w'])
    assert config['actor_source'] is False and config['source_labels']==job['source_labels']
    saved=torch.load(done['final_model'],map_location='cpu')
    assert finite_tree(saved) and saved['step']==500000 and saved['config']==config['learner']
    assert saved['actor_scheduler']['last_epoch']==500000
    for key in ('V_optimizer','Q_optimizer','actor_optimizer'):
        states=saved[key]['state'];assert states and all(float(v['step'])==500000 for v in states.values())
    initial=initialize(config['learner'],'cpu').digest()
    assert initial==read(path/'initialization.json')['sha256']
    events=read(path/'evaluations.json')['events'];expected=list(range(10000,500001,10000))
    assert [v['step'] for v in events]==expected
    tb=EventAccumulator(str(Path(shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
    assert [v.step for v in tb]==expected
    for scalar,event in zip(tb,events):
        assert event['env_seeds']==list(range(500000+10*(job['seed']-201),500010+10*(job['seed']-201)))
        assert event['xml_sha256']==spec['xml_sha256'] and event['actual_shifted_object'] and event['preserve_training_RNG']
        assert abs(event['timestep']-.002)<1e-12
        assert event['normalization_min']==spec['score_min'] and event['normalization_max']==spec['score_max']
        scores=(np.asarray(event['raw_returns'])-spec['score_min'])/(spec['score_max']-spec['score_min'])*100
        assert len(scores)==10 and np.isfinite(scores).all()
        assert np.allclose(scores,event['scores'],rtol=0,atol=1e-10)
        assert abs(scores.mean()-event['mean'])<1e-10 and abs(scalar.value-scores.mean())<1e-4
    sample=read(path/'sampling.json')
    assert sample['target_draws']==32000000 and sample['target_stream_sha256']==stream_hash(job['seed'],'target_sample',5000,500000,job['task'])
    assert sample['source_draws']==(32000000 if job['arm']=='M' else 0)
    if job['arm']=='M':
        assert sample['source_stream_sha256']==stream_hash(job['seed'],'source_sample',spec['source_rows'],500000,job['task'])
        alignment=read(path/'source_alignment.json');assert alignment['source_labels']=='original_dataset' and alignment['oracle_loaded'] is False
        arrays,ids=source_data(job['task'])
        assert alignment['canonical_raw_ids_sha256']==hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
        assert alignment['array_sha256']=={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in zip(('state','action','next_state','reward','not_done'),arrays)}
    assert [v['step'] for v in events[-5:]]==[460000,470000,480000,490000,500000]
    result=dict(passed=True,job=job,Last5=float(np.mean([v['mean'] for v in events[-5:]])),Final=events[-1]['mean'],
        sampling=sample,initialization_sha256=initial,final_model_sha256=done['final_model_sha256'],
        evaluations_sha256=sha(path/'evaluations.json'),completed_sha256=sha(path/'completed.json'),
        raw_TB_and_episode_returns_checked=True,all_model_optimizer_scheduler_finite=True,
        complete_target_source_sampling_streams_rebuilt=True,CPU_only=True,time=time.time())
    write(path/'audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args().out)
