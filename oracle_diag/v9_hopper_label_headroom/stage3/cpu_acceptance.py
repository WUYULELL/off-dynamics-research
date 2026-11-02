from runtime import *
import ast,copy
import numpy as np
import torch
from learner import Actor,Sampler
from offline_offline.iql_dyn import IQL_DYN
from metrics import group,welch,gate

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    configure_cpu();assert not torch.cuda.is_initialized()
    jobs=read(OUT/'training_plan.json')['jobs'];expected={};updates=0
    for task in ('hopper-morph','walker2d-morph'):
        selected=[j for j in jobs if j['task']==task];first=selected[0];spec=task_spec(task)
        actor=Actor(first,'cpu')
        config=dict(state_dim=spec['state_dim'],action_dim=spec['action_dim'],hidden_sizes=256,max_action=1.,gamma=.99,tau=.005,
            update_interval=2,lam=.7,temp=0,critic_lr=.0003,actor_lr=.0003,max_step=200000,source_budget=1.,source_loss_weight=1.)
        reference=IQL_DYN(config,'cpu');reference.policy.load_state_dict(copy.deepcopy(actor.policy.state_dict()))
        before=[copy.deepcopy(v.state_dict()) for v in (reference.q_funcs,reference.v_func,reference.target_q_funcs)]
        gen=torch.Generator().manual_seed(20261031)
        for _ in range(2):
            state=torch.randn(64,spec['state_dim'],generator=gen);action=torch.tanh(torch.randn(64,spec['action_dim'],generator=gen));adv=torch.randn(64,1,generator=gen)
            loss=actor.step(state,action);updates+=1
            ref_loss=reference.policy_loss_per_sample(adv,state,action).mean()
            reference.policy_optimizer.zero_grad();ref_loss.backward();reference.policy_optimizer.step();reference.policy_lr_schedule.step();updates+=1
            assert torch.equal(loss,ref_loss.detach()) and equal(actor.policy.state_dict(),reference.policy.state_dict())
            assert equal(actor.optimizer.state_dict(),reference.policy_optimizer.state_dict())
            assert equal(actor.scheduler.state_dict(),reference.policy_lr_schedule.state_dict())
        for original,net in zip(before,(reference.q_funcs,reference.v_func,reference.target_q_funcs)):assert equal(original,net.state_dict())
        assert not reference.q_optimizer.state and not reference.v_optimizer.state
        for seednum in sorted(set(j['seed'] for j in selected)):
            variants=[j for j in selected if j['seed']==seednum]
            digests=[Actor(j,'cpu').digest() for j in variants];assert len(set(digests))==1
            expected[task+'|'+str(seednum)]=digests[0]
        arrays=target_data(task)
        for variant in ('BC_all','BC50','BC25'):
            j=next(j for j in selected if j['variant']==variant)
            sampler=Sampler(arrays,j,'cpu');sampler.sample();sampler.sample()
            assert sampler.hash.hexdigest()==sample_hash(j,2) and sampler.draws==128
    a=group([10.,12.,14.,16.,18.],[1,2,3,4,5]);b=group([0.,2.,4.,6.,8.],[6,7,8,9,10])
    assert a['mean']==14 and a['Q1']==12 and welch(a,b)['mean_difference']==10
    assert gate(10,4)=='估分员是瓶颈，存在提升空间'
    assert gate(4.999,1)=='该结构下提升空间不足' and gate(5,1)=='不确定' and gate(9.999,1)=='不确定'
    assert gate(10,5).startswith('V9未覆盖')
    # Evaluation only changes the independent training-seed offset from201 to301.
    old=(ROOT/'oracle_diag/coverage_raw_r14/train.py').read_text(encoding='utf-8-sig')
    new=(HERE/'train.py').read_text(encoding='utf-8-sig')
    def extract(s):
        n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        return '\n'.join(s.splitlines()[n.lineno-1:n.end_lineno])
    assert ast.dump(ast.parse(extract(old).replace("job['seed']-201","job['seed']-301")),include_attributes=False)==ast.dump(ast.parse(extract(new)),include_attributes=False)
    assert updates==8 and not torch.cuda.is_initialized()
    write(OUT/'INITIALIZATION_EXPECTATIONS.json',dict(sha256=expected))
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,synthetic_CPU_actor_updates=updates,
        real_data_optimizer_updates=0,real_environment_steps=0,CUDA_initialized=False,
        IQL_beta0_actor_policy_optimizer_scheduler_bitwise_equal=True,Q_V_never_optimized=True,
        all_variant_initializations_identical_per_seed=True,sampling_indices_rebuilt=True,
        evaluation_AST_R14_equal_except_seed_offset=True,Welch_and_strict_boundary_fixtures=True,time=time.time()))
    write(OUT/'CPU_ACCEPTANCE_COMMITMENT.json',dict(sha256=sha(OUT/'cpu_acceptance.json'),time=time.time()))
    write(OUT/'code_frozen.json',dict(sha256=code_hashes(),time=time.time()))
    check_frozen();print(json.dumps(dict(passed=True,synthetic_CPU_actor_updates=updates,real_data_updates=0,environment_steps=0,CUDA_initialized=False)))

if __name__=='__main__':main()
