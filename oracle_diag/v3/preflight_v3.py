"""CPU forward-only contract tests. No backward(), optimizer step, or RL training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v3 import *
from task_env import TASKS,paths,patch_normalization
from methods_v3 import *
import numpy as np,torch,yaml,ast,inspect

def digest(model):
    h=hashlib.sha256()
    for net in [model.policy,model.v_func,model.q_funcs]:
        for value in net.state_dict().values():h.update(value.detach().numpy().tobytes())
    return h.hexdigest()

def main():
    dest=OUT/'preflight1';fresh(dest);torch.set_num_threads(2)
    config=yaml.safe_load((ROOT/'config/mujoco/iql/walker2d.yaml').read_text())
    config.update(state_dim=17,action_dim=6,max_action=1,source_budget=.20,source_loss_weight=1.,bellman_eta=1.,seed=0)
    models={}
    for method in ['TARGET','MIX-CONST','MIX-NAIVE']:
        torch.manual_seed(0);c=dict(config,target_only=method=='TARGET');models[method]=build(method,c,'cpu')
    hashes={k:digest(m) for k,m in models.items()};assert len(set(hashes.values()))==1
    rng=np.random.default_rng(2);n=257
    data=dict(observations=rng.normal(size=(n,17)).astype('float32'),actions=rng.normal(size=(n,6)).astype('float32'),
        next_observations=rng.normal(size=(n,17)).astype('float32'),rewards=rng.normal(size=n).astype('float32'),terminals=(np.arange(n)%3==0))
    src=IndexedReplayBuffer(17,6,'cpu',max_size=1);src.convert_D4RL(data)
    old=ReplayBuffer(17,6,'cpu',max_size=1);old.convert_D4RL(data)
    np.random.seed(0);x=src.sample(64);np.random.seed(0);ref=old.sample(64)
    assert all(torch.equal(a,b) for a,b in zip(x,ref))
    s,a,ns,r,nd=x;constant=models['MIX-CONST'];naive=models['MIX-NAIVE']
    for model in [constant,naive]:model.source_buffer=src;model.source_backup.buffer=src;model.compute_source_weight(*x)
    torch.manual_seed(0)
    def oracle_provider(ids):return ns+0.5,r+2.,1.-nd
    oracle=MixedIQL(dict(config,target_only=False),'cpu',oracle_provider)
    oracle.source_buffer=src;oracle.compute_source_weight(*x)
    assert digest(oracle)==hashes['MIX-CONST']
    with torch.no_grad():
        qc=constant.q_loss_per_sample(s,a,r,ns,nd)[0];qn=naive.q_loss_per_sample(s,a,r,ns,nd)[0]
        baseline=IQL_BELLMAN_FB.q_loss_per_sample(constant,s,a,r,ns,nd)[0]
        assert torch.equal(qc,qn) and torch.equal(qc,baseline)
        qo=oracle.q_loss_per_sample(s,a,r,ns,nd)[0]
        y=(r+2.)+(1.-nd)*.99*oracle.v_func(ns+.5);q1,q2=oracle.q_funcs(s,a)
        assert torch.equal(qo,(q1-y).square()+(q2-y).square())
        assert not torch.equal(qo,qc)
        for model in [constant,oracle]:
            qt=model.q_loss_per_sample(s,a,r,ns,nd)[0]
            assert torch.equal(qt,IQL_BELLMAN_FB.q_loss_per_sample(model,s,a,r,ns,nd)[0])
        for func,arguments in [('v_loss_per_sample',(s,a)),('policy_loss_per_sample',(torch.ones_like(r),s,a))]:
            vc=getattr(constant,func)(*arguments);vo=getattr(oracle,func)(*arguments)
            if isinstance(vc,tuple):assert all(torch.equal(j,k) for j,k in zip(vc,vo))
            else:assert torch.equal(vc,vo)
        for loss in [qc,torch.ones_like(qc)]:
            assert torch.equal(constant.combine_target_source_loss(loss,loss,torch.full_like(loss,.2)),loss.mean()+(.2*loss).mean())
    # Exercise adapter delegation without invoking the original learner.
    saved_target=IQL.train;saved_mixed=IQL_BELLMAN_FB.train
    calls=[]
    def fake_target(self,source,target,batch_size,writer):
        assert batch_size==64;target.sample(batch_size*2);calls.append([0,batch_size*2])
    def fake_mixed(self,source,target,batch_size,writer):
        assert batch_size==64;source.sample(batch_size);target.sample(batch_size);self._q_calls=2;calls.append([batch_size,batch_size])
    IQL.train=fake_target;IQL_BELLMAN_FB.train=fake_mixed
    try:
        for model in [models['TARGET'],constant,oracle]:model.train(src,old,128)
    finally:IQL.train=saved_target;IQL_BELLMAN_FB.train=saved_mixed
    assert calls==[[0,128],[64,64],[64,64]]
    assert all(not m.q_optimizer.state and not m.v_optimizer.state and not m.policy_optimizer.state for m in [*models.values(),oracle])
    for bad in ['walker2d-morph-leg','walker2d','halfcheetah-morph']:
        try:paths(bad)
        except ValueError:pass
        else:raise AssertionError('Unsupported task accepted')
    from envs.infos import get_normalized_score
    normalization={}
    for task in TASKS:
        c=patch_normalization(task);normalization[task]=c
        assert get_normalized_score(c['min'],task+'-medium')==0
        assert get_normalized_score(c['max'],task+'-medium')==100
    train=(ROOT/'train.py').read_text(encoding='utf-8')
    assert "tar_eval_return = eval_policy(policy, tar_eval_env" in train
    assert 'get_normalized_score(tar_eval_return, ref_env_name)' in train
    assert "writer.add_scalar('test/target normalized score', eval_normalized_score" in train
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    from train_entry_v3 import check_approval
    assert not (OUT/'gpu_authorization.json').exists()
    try:check_approval()
    except RuntimeError:approval_gate=True
    else:raise AssertionError('Unapproved training allowed')
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,device='cpu',rl_training_steps=0,optimizer_steps=0,cuda_initialized=False,
        initialization_hashes=hashes,naive_const_identity=True,source_q_formula_exact=True,target_q_unchanged=True,
        source_v_actor_formula_unchanged=True,sampling_matches_original=True,adapter_draws=calls,
        unsupported_tasks_rejected=True,normalization_endpoints_passed=True,normalization=normalization,
        train_target_score_flow_static_verified=True,runtime_evaluation_evidence_pending_gpu_smoke=True,
        approval_gate_enforced=approval_gate,fixture='Forward-only Q tests use synthetic arrays, not experimental results')
    write(dest/'completed.json',result);print('PREFLIGHT PASSED; no RL updates',flush=True)

if __name__=='__main__':main()
