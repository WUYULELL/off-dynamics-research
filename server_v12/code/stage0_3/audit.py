"""CPU audit of a completed 0.3 benchmark; never create a learner or optimizer."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback
from common import read,write,sha,utc,length_stats


def expected_network_shapes():
    def mlp(prefix,inputs,outputs):
        return {prefix+'.0.weight':(256,inputs),prefix+'.0.bias':(256,),
                prefix+'.2.weight':(256,256),prefix+'.2.bias':(256,),
                prefix+'.4.weight':(outputs,256),prefix+'.4.bias':(outputs,)}
    q=dict(mlp('network1.network',14,1),**mlp('network2.network',14,1))
    return dict(policy=mlp('network.network',11,6),V=mlp('network.network',11,1),Q=q,target_Q=q)


def state_digest(value):
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            assert bool(torch.isfinite(x).all()), 'Non-finite checkpoint tensor'
            a=x.detach().cpu().contiguous().numpy()
            h.update(b'tensor'+str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda z:(type(z).__name__,repr(z))):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode()+str(len(x)).encode())
            for y in x:visit(y)
        elif isinstance(x,float):
            assert math.isfinite(x);h.update(b'float'+struct.pack('>d',x))
        elif isinstance(x,(int,str,bool)) or x is None:
            h.update(type(x).__name__.encode()+repr(x).encode())
        else:raise TypeError(type(x))
    visit(value)
    return h.hexdigest()


def audit(run):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    c=read(run/'COMPLETED.json');j=c['job'];assert j['steps']==c['training_updates']==20000
    assert c['counts']['training_updates']==20000 and c['evaluation_episodes']==10
    for f,k in (('losses.npy','losses_sha256'),('losses.jsonl','JSONL_sha256'),('final_checkpoint.pt','checkpoint_sha256')):
        assert sha(run/f)==c[k]
    loss=np.load(str(run/'losses.npy'),allow_pickle=False)
    assert loss.dtype==np.float32 and loss.shape==(20000,3 if j['algorithm']=='IQL' else 1) and np.isfinite(loss).all()
    lines=0
    with (run/'losses.jsonl').open(encoding='utf-8') as f:
        for lines,line in enumerate(f,1):
            x=json.loads(line);assert lines<=20000 and x['step']==lines
            a=np.asarray(x['losses'],dtype=np.float32)
            assert np.array_equal(a.view(np.uint32),loss[lines-1].view(np.uint32))
    assert lines==20000
    ck=torch.load(str(run/'final_checkpoint.pt'),map_location='cpu')
    assert ck['step']==20000
    shapes=expected_network_shapes()
    network_names=('policy','V','Q','target_Q') if j['algorithm']=='IQL' else ('policy',)
    if j['algorithm']=='BC':assert all(k not in ck for k in ('V','Q','target_Q','V_optimizer','Q_optimizer'))
    for n in network_names:
        assert set(ck[n])==set(shapes[n])
        for key,shape in shapes[n].items():assert tuple(ck[n][key].shape)==shape and ck[n][key].dtype==torch.float32
    full_state=state_digest(ck)
    h=hashlib.sha256()
    for n in network_names:
        for x in ck[n].values():h.update(x.detach().cpu().numpy().tobytes())
    assert h.hexdigest()==c['parameter_sha256']
    optimizers=(('V_optimizer','V'),('Q_optimizer','Q'),('actor_optimizer','policy')) if j['algorithm']=='IQL' else (('actor_optimizer','policy'),)
    for name,network in optimizers:
        opt=ck[name];params=list(ck[network].values());ids=[p for g in opt['param_groups'] for p in g['params']]
        assert len(ids)==len(params)==len(opt['state'])
        for pid,param in zip(ids,params):
            state=opt['state'][pid];step=state['step'];assert int(step.item() if hasattr(step,'item') else step)==20000
            assert state['exp_avg'].shape==state['exp_avg_sq'].shape==param.shape
    period=500000 if j['algorithm']=='IQL' else 200000
    assert ck['actor_scheduler']['T_max']==period and ck['actor_scheduler']['last_epoch']==20000
    lr=3e-4*(1+math.cos(math.pi*20000/period))/2
    assert math.isclose(ck['actor_optimizer']['param_groups'][0]['lr'],lr,rel_tol=1e-10,abs_tol=1e-15)
    if j['algorithm']=='IQL':assert ck['config']==read(run/'config.json')['learner']
    else:assert ck['job']==j
    prefix='20261016' if j['algorithm']=='IQL' else 'V9-stage3-20261031'
    seed=int.from_bytes(hashlib.sha256(('%s|hopper-morph|%d|target_sample'%(prefix,j['seed'])).encode()).digest()[:4],'little')
    rng=np.random.RandomState(seed);stream=hashlib.sha256()
    for _ in range(20000):stream.update(rng.randint(5000,size=64).astype('<i8').tobytes())
    sample=read(run/'sampling.json')
    assert sample['target_draws']==1280000 and sample['source_draws']==0 and sample['target_stream_sha256']==stream.hexdigest()
    ev=read(run/'evaluation.json');r=read(run/'EVALUATION_RNG.json')
    assert r['identical'] and r['before']==r['after']
    assert ev['step']==20000 and len(ev['episode_lengths'])==10 and ev['length_statistics']==length_stats(ev['episode_lengths'])
    offset=201 if j['algorithm']=='IQL' else 301
    assert ev['env_seeds']==[500000+10*(j['seed']-offset)+e for e in range(10)]
    scores=(np.asarray(ev['raw_returns'])+26.3360015397715)/(3152.75+26.3360015397715)*100
    assert np.array_equal(scores,np.asarray(ev['scores'])) and np.isfinite(scores).all()
    assert sum(ev['episode_lengths'])==c['counts']['evaluation_environment_steps']
    assert c['counts']['constructor_environment_steps']==1
    assert c['counts']['Adam_updates']==20000*(3 if j['algorithm']=='IQL' else 1)
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,CUDA_initialized=False,algorithm=j['algorithm'],
        raw_loss_records=20000,all_model_optimizer_scheduler_finite=True,complete_sampling_stream_verified=True,
        evaluation_RNG_and_lengths_verified=True,full_checkpoint_state_sha256=full_state,
        completed_sha256=sha(run/'COMPLETED.json'),checkpoint_sha256=sha(run/'final_checkpoint.pt'),
        new_optimization_updates=0,new_environment_steps=0,time=utc())
    write(run/'AUDIT.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    try:print(json.dumps(audit(a.run.resolve()),ensure_ascii=False,indent=2));return 0
    except BaseException:
        error=traceback.format_exc();write(a.run/'AUDIT_FAILED.json',dict(time=utc(),traceback=error))
        print(error,file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())


