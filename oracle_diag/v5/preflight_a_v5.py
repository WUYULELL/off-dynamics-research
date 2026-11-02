"""Real-data CPU contract tests, no rollout or GPU initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v5 import *
from methods_v5 import build, IndexedReplayBuffer, SourceBackup
from methods_v4 import OracleW1IQL
from data_v4 import load_source_cpu, verify_source
from task_env import call_target_dataset
from instrument_v4 import install_loss_trace
import argparse, copy
import numpy as np
import torch

def buffer(data):
    b=IndexedReplayBuffer(17,6,'cpu',max_size=1);b.convert_D4RL(data);return b

def state(m):return {k:{n:v.detach().clone() for n,v in getattr(m,k).state_dict().items()} for k in ('policy','v_func','q_funcs')}
def equal(a,b):return all(torch.equal(a[k],b[k]) for k in a)

def fixed_sample(b,ids):
    def sample(n):
        assert n==64;b.last_indices=ids.copy();b.draw_count+=n
        return tuple(torch.as_tensor(getattr(b,k)[ids],dtype=torch.float32) for k in ('state','action','next_state','reward','not_done'))
    b.sample=sample

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'cpu_a1');a=p.parse_args();fresh(a.out)
    torch.set_num_threads(2)
    source=load_source_cpu();target=call_target_dataset('walker2d-morph');src=buffer(source);verify_source(src,oracle_receipt())
    SourceBackup('cpu',oracle_receipt()) # checks every saved cache array hash
    cfg=read(V4_OUT/'s1/w0/effective_config.json')['config']
    checks={}
    for method in A_METHODS+('ORACLE-W1',):
        snapshots=[];traces=[]
        for variant in (0,1):
            torch.manual_seed(0);m=build(method,cfg,'cpu',oracle_receipt());before=state(m)
            s=buffer(source);t=buffer(target)
            fixed_sample(s,np.random.RandomState(101+variant).choice(s.size,64,replace=False))
            fixed_sample(t,np.random.RandomState(303).choice(t.size,64,replace=False))
            trace=install_loss_trace(m);m.train(s,t,128)
            assert (s.draw_count,t.draw_count)==(64,64) and m._q_calls==2
            after=state(m);assert all(not equal(before[k],after[k]) for k in before)
            snapshots.append(after);traces.append(trace)
        invariance={k:equal(snapshots[0][k],snapshots[1][k]) for k in snapshots[0]}
        expected=dict(policy=method!='ORACLE-W1',v_func=method=='Q-ORACLE',q_funcs=False)
        assert invariance==expected,(method,invariance)
        checks[method]=dict(changed_source_invariance=invariance,expected=expected,traces=traces)
    # Reusing the original updater through the v5 factory must be exact on CPU.
    reference=[]
    for mode in ('old','new'):
        torch.manual_seed(0)
        m=OracleW1IQL(cfg,'cpu',SourceBackup('cpu',oracle_receipt())) if mode=='old' else build('ORACLE-W1',cfg,'cpu',oracle_receipt())
        init=model_hash(m);trace=install_loss_trace(m);s=buffer(source);t=buffer(target)
        np.random.seed(0);torch.manual_seed(0)
        for _ in range(10):m.train(s,t,128)
        reference.append(dict(initial=init,final=model_hash(m),trace=trace,draws=[s.draw_count,t.draw_count]))
    assert reference[0]==reference[1]
    assert reference[0]['initial']==read(V4_OUT/'s1/w0/initialization.json')['sha256']
    # Test direct loss exclusion with independent gradients, not only outputs.
    gradient_checks={}
    for method in A_METHODS:
        torch.manual_seed(0);m=build(method,cfg,'cpu',oracle_receipt());m._loss_calls=0
        rows=[]
        for kind in ('V','Q','actor'):
            tl=torch.randn(64,1,requires_grad=True);sl=torch.randn(64,1,requires_grad=True)
            loss=m.combine_target_source_loss(tl,sl,torch.ones_like(tl));loss.backward()
            excluded=kind=='actor' or (kind=='V' and method=='Q-ORACLE')
            assert torch.equal(tl.grad,torch.full_like(tl,1/64))
            assert sl.grad is None if excluded else torch.equal(sl.grad,torch.full_like(sl,1/64))
            rows.append(dict(loss=kind,source_gradient_absent=excluded))
        gradient_checks[method]=rows
    assert not torch.cuda.is_initialized()
    write(a.out/'completed.json',dict(passed=True,device='cpu',source_n=src.size,target_n=len(target['observations']),
      source_cache_verified=True,one_step_tests=checks,gradient_tests=gradient_checks,reuse_reference=reference[0],
      old_new_cpu_ten_steps_exact=True,gpu_reuse='pending',scope='CPU implementation tests; no performance evidence'))
    print('A CPU contract tests PASSED',flush=True)

if __name__=='__main__':main()
