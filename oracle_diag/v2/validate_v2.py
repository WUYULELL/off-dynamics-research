"""Formula, replacement, matching initialization and update-path checks on GPU."""
from support import *
import numpy as np,torch,time
from common import load_policy_config,load_target_buffer
from oracle_methods import IndexedReplayBuffer,IQL_ORACLE_BACKUP
from offline_offline.iql_unc import IQL_UNC
from fixed_loader import load_source_fixed

def main():
    torch.set_num_threads(2);device=torch.device('cuda');assert torch.cuda.is_available()
    cfg=load_policy_config(ENVS[0],overrides=dict(source_budget=.2,bellman_eta=1.,const_source_weight=.2,max_step=500000))
    src=IndexedReplayBuffer(17,6,device);src.convert_D4RL(load_source_fixed(ENVS[0],'medium','medium'))
    tar,_=load_target_buffer(ENVS[0],0,device=device)
    oracle=dict(np.load(OUT/'stage_a'/(ENVS[0]+'_source_oracle_fixed.npz')))
    identity=dict(oracle,oracle_next=src.next_state,oracle_reward=src.reward.ravel(),oracle_not_done=src.not_done.ravel())
    models=[]
    for c in [IQL_UNC,IQL_ORACLE_BACKUP]:
        torch.manual_seed(0);models.append(c(cfg,device) if c==IQL_UNC else c(cfg,device,identity))
    for step in range(5):
        for m in models:np.random.seed(100+step);torch.manual_seed(100+step);m.train(src,tar,128)
    maxdiff=max(float((a-b).abs().max()) for net in ['v_func','q_funcs','policy','target_q_funcs'] for a,b in zip(getattr(models[0],net).parameters(),getattr(models[1],net).parameters()))
    assert maxdiff<1e-7,('Identity backup must match CONST',maxdiff)
    actual=IQL_ORACLE_BACKUP(cfg,device,oracle);actual.source_buffer=src;batch=src.sample(128);ob=actual.oracle_tensors()
    actual.compute_source_weight(*batch);loss,*_=actual.q_loss_per_sample(batch[0],batch[1],batch[3],batch[2],batch[4])
    from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
    expected,*_=IQL_BELLMAN_FB.q_loss_per_sample(actual,batch[0],batch[1],ob[1].reshape(-1,1),ob[0],ob[2].reshape(-1,1))
    assert torch.equal(loss,expected)
    targetbatch=tar.sample(128);got,*_=actual.q_loss_per_sample(targetbatch[0],targetbatch[1],targetbatch[3],targetbatch[2],targetbatch[4])
    expected,*_=IQL_BELLMAN_FB.q_loss_per_sample(actual,targetbatch[0],targetbatch[1],targetbatch[3],targetbatch[2],targetbatch[4])
    assert torch.equal(got,expected)
    start=time.time()
    for _ in range(300):actual.train(src,tar,128)
    torch.cuda.synchronize()
    write(OUT/'gpu_validation.json',dict(passed=True,identity_backup_vs_const_max_parameter_difference=maxdiff,
        source_q_replacement_exact=True,target_q_unchanged=True,steps=300,seconds=time.time()-start,
        max_memory_allocated_mb=torch.cuda.max_memory_allocated()/2**20,max_memory_reserved_mb=torch.cuda.max_memory_reserved()/2**20))
    print('GPU VALIDATION PASS',flush=True)
if __name__=='__main__':main()
