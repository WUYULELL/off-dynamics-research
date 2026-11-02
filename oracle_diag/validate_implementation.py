"""Bounded GPU test of indexing, oracle formula, backup, and throughput."""
import time,json
from common import *
from oracle_methods import IndexedReplayBuffer,IQL_BELLMAN_ORACLE,IQL_ORACLE_BACKUP
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
from offline_offline.iql_unc import IQL_UNC
from offline_offline.iql import IQL


def main():
    out=RESULTS/'verified_20261003';dest=out/'gpu_smoke.json'
    if dest.exists():raise RuntimeError('Exists')
    assert torch.cuda.is_available();torch.set_num_threads(2)
    device=torch.device('cuda');name=ENVS[0]
    oracle=dict(np.load(out/'diagnostics'/(name+'_source_oracle.npz')))
    src=IndexedReplayBuffer(17,6,device,max_size=1)
    src.convert_D4RL(dict(observations=oracle['source_state'],actions=oracle['source_action'],
        next_observations=oracle['source_next'],rewards=oracle['source_reward'],terminals=1-oracle['source_not_done']))
    tar,full=load_target_buffer(name,0,device=device)
    cfg=load_policy_config(name,overrides=dict(source_budget=.2,bellman_eta=1.,seed=0,max_step=500000))
    m=IQL_BELLMAN_ORACLE(cfg,device,oracle);m.source_buffer=src
    batch=src.sample(128);stats=m.compute_source_weight(*batch);ids=src.last_indices.copy()
    with torch.no_grad():
        original_y=batch[3]+.99*batch[4]*m.v_func(batch[2])
        oracle_y=torch.as_tensor(oracle['oracle_reward'][ids],device=device,dtype=torch.float32).reshape(-1,1)
        oracle_y=oracle_y+.99*torch.as_tensor(oracle['oracle_not_done'][ids],device=device,dtype=torch.float32).reshape(-1,1)*m.v_func(torch.as_tensor(oracle['oracle_next'][ids],device=device,dtype=torch.float32))
        b=(original_y-oracle_y).abs();raw=(-b/(b.mean()+1e-6)).exp();w=(.2*raw/(raw.mean()+1e-6)).clamp(.05,1)
    torch.testing.assert_close(stats['weight'],w,rtol=0,atol=0)
    assert not stats['weight'].requires_grad
    # A backup is exactly the original Q loss fed oracle next/reward/done.
    backup=IQL_ORACLE_BACKUP(cfg,device,oracle);backup.source_buffer=src
    backup.compute_source_weight(*batch)
    got=backup.q_loss_per_sample(batch[0],batch[1],batch[3],batch[2],batch[4])[0]
    on,orr,ond=backup.oracle_tensors()
    expected=IQL_BELLMAN_FB.q_loss_per_sample(backup,batch[0],batch[1],orr.reshape(-1,1),on,ond.reshape(-1,1))[0]
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    second=backup.q_loss_per_sample(batch[0],batch[1],batch[3],batch[2],batch[4])[0]
    original=IQL_BELLMAN_FB.q_loss_per_sample(backup,batch[0],batch[1],batch[3],batch[2],batch[4])[0]
    torch.testing.assert_close(second,original,rtol=0,atol=0)
    del m,backup
    result=dict(passed=True,device=torch.cuda.get_device_name(),index_formula_check=True,
        source_backup_exact=True,target_backup_unchanged=True,methods={})
    for method in ['TARGET','MIX_FB','CONST','ORACLE','BACKUP']:
        config=dict(cfg)
        if method=='TARGET':config=load_policy_config(name,'iql',dict(target_only=True,seed=0,max_step=500000))
        if method=='CONST':config=load_policy_config(name,'iql_unc',dict(const_source_weight=.2,source_budget=.2,seed=0,max_step=500000))
        torch.manual_seed(0);np.random.seed(0)
        if method=='TARGET':model=IQL(config,device)
        elif method=='MIX_FB':model=IQL_BELLMAN_FB(config,device)
        elif method=='CONST':model=IQL_UNC(config,device)
        elif method=='ORACLE':model=IQL_BELLMAN_ORACLE(config,device,oracle)
        else:model=IQL_ORACLE_BACKUP(config,device,oracle)
        for _ in range(20):model.train(src,tar)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(300):model.train(src,tar)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        assert all(torch.isfinite(p).all() for net in [model.v_func,model.q_funcs,model.policy] for p in net.parameters())
        result['methods'][method]=dict(timed_steps=300,seconds=seconds,estimated_500k_training_hours=seconds/300*500000/3600)
        print(method,seconds,flush=True);del model
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
