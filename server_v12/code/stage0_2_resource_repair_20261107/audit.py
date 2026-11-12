"""CPU-only checkpoint, raw-loss, optimizer, scheduler, and full-stream audit."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback
from common import read,write,sha,utc,compare_loss_arrays,length_stats


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
    completed=read(run/'COMPLETED.json');job=completed['job'];steps=job['steps']
    for file,key in [('final_checkpoint.pt','checkpoint_sha256'),('losses.npy','losses_npy_sha256'),
                     ('losses.jsonl','losses_jsonl_sha256'),('sampling.json','sampling_sha256')]:
        assert sha(run/file)==completed[key], 'Completed artifact changed: '+file
    a=np.load(str(run/'losses.npy'),allow_pickle=False)
    assert a.dtype==np.float32 and a.shape==(steps,3) and np.isfinite(a).all()
    scalar=np.empty_like(a)
    with (run/'losses.jsonl').open(encoding='utf-8') as f:
        n=0
        for n,line in enumerate(f,1):
            x=json.loads(line);assert n<=steps and x['step']==n
            scalar[n-1]=[x[k] for k in ('V','Q','actor')]
    assert n==steps and compare_loss_arrays(a,scalar)['bitwise_equal']
    checkpoint=torch.load(str(run/'final_checkpoint.pt'),map_location='cpu')
    assert checkpoint['step']==steps and checkpoint['config']['max_step']==500000
    assert checkpoint['config']==read(run/'config.json')['learner']
    for name,shapes in expected_network_shapes().items():
        assert set(checkpoint[name])==set(shapes),'Network state keys changed: '+name
        for key,shape in shapes.items():
            assert tuple(checkpoint[name][key].shape)==shape and checkpoint[name][key].dtype==torch.float32
    full_digest=state_digest(checkpoint)
    ph=hashlib.sha256()
    for name in ('policy','V','Q','target_Q'):
        for value in checkpoint[name].values():ph.update(value.detach().cpu().numpy().tobytes())
    assert ph.hexdigest()==completed['parameter_sha256']
    optimizers={}
    for name in ('V_optimizer','Q_optimizer','actor_optimizer'):
        optimizer=checkpoint[name]
        network={'V_optimizer':'V','Q_optimizer':'Q','actor_optimizer':'policy'}[name]
        parameters=list(checkpoint[network].values())
        ids=[pid for g in optimizer['param_groups'] for pid in g['params']]
        assert len(optimizer['state'])==len(ids)==len(parameters)
        counters=[]
        for pid,parameter in zip(ids,parameters):
            state=optimizer['state'][pid]
            value=state['step'];value=int(value.item()) if hasattr(value,'item') else int(value)
            assert value==steps;counters.append(value)
            assert state['exp_avg'].shape==state['exp_avg_sq'].shape==parameter.shape
        optimizers[name]=dict(state_entries=len(counters),all_steps=steps,finite=True)
    scheduler=checkpoint['actor_scheduler']
    assert scheduler['T_max']==500000 and scheduler['last_epoch']==steps
    expected_lr=3e-4*(1+math.cos(math.pi*steps/500000))/2
    assert math.isclose(checkpoint['actor_optimizer']['param_groups'][0]['lr'],expected_lr,rel_tol=1e-10,abs_tol=1e-15)
    sampling=read(run/'sampling.json')
    seed=int.from_bytes(hashlib.sha256(b'20261016|hopper-morph|201|target_sample').digest()[:4],'little')
    rng=np.random.RandomState(seed);h=hashlib.sha256();first=None
    # Same randint size=64 call sequence as original Sampler.sample.
    for _ in range(steps):
        ids=rng.randint(5000,size=64).astype('<i8')
        if first is None:first=ids.tolist()
        h.update(ids.tobytes())
    assert h.hexdigest()==sampling['target_stream_sha256'] and first==sampling['target_first']
    assert sampling['target_draws']==64*steps and sampling['source_draws']==0
    events=read(run/'evaluations.json')['events']
    assert len(events)==(2 if job['evaluate'] else 0)
    if events:
        assert [x['step'] for x in events]==[10000,20000]
        total=0
        for e in events:
            assert e['length_statistics']==length_stats(e['episode_lengths'])
            assert e['env_seeds']==list(range(500000,500010))
            assert len(e['raw_returns'])==len(e['scores'])==len(e['episode_lengths'])==10
            assert np.isfinite(e['raw_returns']).all() and np.isfinite(e['scores']).all()
            scores=(np.asarray(e['raw_returns'])+26.3360015397715)/(3152.75+26.3360015397715)*100
            assert np.array_equal(scores,np.asarray(e['scores']))
            assert e['steps']==sum(e['episode_lengths'])
            total+=e['steps']
        assert total==completed['counts']['evaluation_environment_steps']
        checks=read(run/'EVALUATION_RNG.json')['checks']
        assert len(checks)==2 and all(x['before']==x['after'] and x['identical'] for x in checks)
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,CUDA_initialized=False,step=steps,
        raw_loss_records=steps,raw_loss_vector_and_JSONL_bits_equal=True,
        all_four_networks_three_Adam_scheduler_finite=True,optimizers=optimizers,
        cosine_period=500000,complete_target_sampling_stream_verified=True,
        full_checkpoint_state_sha256=full_digest,parameter_sha256=ph.hexdigest(),
        evaluation_events=len(events),episode_lengths_verified=bool(events),
        completed_sha256=sha(run/'COMPLETED.json'),checkpoint_sha256=sha(run/'final_checkpoint.pt'),
        new_optimization_updates=0,new_environment_steps=0,time=utc())
    write(run/'AUDIT.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    try:
        print(json.dumps(audit(args.run.resolve()),ensure_ascii=False,indent=2));return 0
    except BaseException:
        failure=traceback.format_exc();write(args.run/'AUDIT_FAILED.json',dict(traceback=failure,time=utc()))
        print(failure,file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
