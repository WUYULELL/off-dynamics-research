"""Full-state recovery for the deterministic target-only R11 learner."""
import hashlib
import numpy as np
import torch

def restore_training(model, checkpoint):
    d=torch.load(str(checkpoint),map_location=model.device)
    assert d['config']==model.config
    for attr,key in (('policy','policy'),('v_func','V'),('q_funcs','Q'),('target_q_funcs','target_Q')):
        getattr(model,attr).load_state_dict(d[key])
    for attr,key in (('v_optimizer','V_optimizer'),('q_optimizer','Q_optimizer'),('policy_optimizer','actor_optimizer')):
        getattr(model,attr).load_state_dict(d[key])
    model.policy_lr_schedule.load_state_dict(d['actor_scheduler'])
    model.total_it=d['step']
    assert model.policy_lr_schedule.T_max==1000000
    assert model.policy_lr_schedule.last_epoch==model.total_it
    assert model.policy_optimizer.param_groups[0]['lr']==d['actor_optimizer']['param_groups'][0]['lr']
    assert all(float(state['step'])==model.total_it for opt in
        (model.v_optimizer,model.q_optimizer,model.policy_optimizer) for state in opt.state.values())
    return d

def advance_sampler(sampler,updates):
    assert sampler.draws==0 and len(sampler.ids)==5000
    for start in range(0,updates,4096):
        count=min(4096,updates-start)
        ids=sampler.ids[sampler.rng.randint(len(sampler.ids),size=count*64)]
        if sampler.first is None:sampler.first=ids[:64].tolist()
        sampler.hash.update(ids.astype('<i8').tobytes())
        sampler.draws+=len(ids)
    assert sampler.draws==updates*64
    return sampler

def tensor_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensor_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tensor_equal(x,y) for x,y in zip(a,b))
    return a==b
