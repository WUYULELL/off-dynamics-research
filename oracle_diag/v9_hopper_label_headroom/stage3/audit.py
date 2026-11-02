from runtime import *
import argparse
import numpy as np
import torch
from learner import Actor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def finite_tree(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
    if isinstance(x,(tuple,list)):return all(finite_tree(v) for v in x)
    return not isinstance(x,float) or math.isfinite(x)
def raw_scores(path,job,every=5000,last=200000):
    events=read(path/'evaluations.json')['events'];spec=task_spec(job['task']);expected=list(range(every,last+1,every))
    tb=EventAccumulator(str(Path(shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
    assert [e['step'] for e in events]==[e.step for e in tb]==expected
    points=[]
    for e,s in zip(events,tb):
        start=500000+10*(job['seed']-(301 if every==5000 else 201))
        assert e['env_seeds']==list(range(start,start+10)) and e['xml_sha256']==spec['xml_sha256'] and e['timestep']==.002
        assert e['actual_shifted_object'] and e['preserve_training_RNG']
        v=(np.asarray(e['raw_returns'],dtype=np.float64)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100
        assert len(v)==10 and np.isfinite(v).all() and abs(v.mean()-s.value)<1e-4
        assert abs(v.mean()-e['mean'])<1e-10;points.append(float(v.mean()))
    return dict(Last5=float(np.mean(points[-5:])),Final=points[-1])
def audit(path):
    configure_cpu();path=Path(path);done=read(path/'completed.json');job=done['job'];validate(job)
    assert done['finite'] and done['steps']==200000 and sha(done['final_model'])==done['final_model_sha256']
    model=torch.load(winpath(done['final_model']),map_location='cpu')
    assert set(model)=={'policy','actor_optimizer','actor_scheduler','step','job'} and finite_tree(model)
    assert model['step']==model['actor_scheduler']['last_epoch']==200000 and model['actor_scheduler']['T_max']==200000
    states=model['actor_optimizer']['state'];assert states and all(float(v['step'])==200000 for v in states.values())
    assert read(path/'initialization.json')['sha256']==Actor(job,'cpu').digest()
    sampling=read(path/'sampling.json');assert sampling['draws']==12800000 and sampling['sampling_sha256']==sample_hash(job)
    config=read(path/'config.json');assert not config['Q_or_V_constructed'] and not config['Q_or_V_optimized'] and config['advantage_weight']==1
    result=dict(passed=True,CPU_only=True,job=job,actor_optimizer_scheduler_finite=True,initialization_and_full_sampling_checked=True,
        final_model_sha256=done['final_model_sha256'],raw_TB_and_episode_returns_checked=True,time=time.time(),**raw_scores(path,job))
    write(path/'audit.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
