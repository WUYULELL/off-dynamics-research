"""Recover clipped source velocity from verified predecessors, retaining failed attempts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v3 import *
from stage_a_v3 import step,normal_file,array_hash
from task_env import call_shifted_env,paths
import numpy as np,h5py,time,argparse,gc

def physical_step(env,p,v,a):
    ns,r,d=step(env,p,v,a)
    return ns,r,d,env.unwrapped.sim.data.qpos.copy(),env.unwrapped.sim.data.qvel.copy()

def recover(raw,env,rows):
    nq=env.unwrapped.model.nq;obs=raw['observations'];actions=raw['actions']
    clipped=(np.abs(obs[:,nq-1:])>=10-1e-6).any(axis=1)
    end=raw['terminals'].astype(bool)|raw['timeouts'].astype(bool)
    p=raw['infos/qpos'][rows].copy();v=raw['infos/qvel'][rows].copy();p[:,0]=0
    status=np.full(len(rows),2,np.int8);error=np.full(len(rows),np.nan)
    for ix,row in enumerate(rows):
        if not clipped[row]:status[ix]=0;error[ix]=0
        elif ix>0 and rows[ix-1]==row-1 and not end[row-1] and status[ix-1]!=2:
            ns,_,_,qp,qv=physical_step(env,p[ix-1],v[ix-1],actions[row-1])
            error[ix]=np.max(np.abs(ns-obs[row]))
            if error[ix]<=1e-4:
                qp[0]=0;p[ix]=qp;v[ix]=qv;status[ix]=1
    return p,v,status,error,clipped

def history_repair(row,raw,env,shifted,rows,p,v,status):
    """Use the previous 1--64 step recovery gate; no tolerance relaxation."""
    end=raw['terminals'].astype(bool)|raw['timeouts'].astype(bool)
    obs=raw['observations']
    lo=row
    while lo>max(0,row-64) and not end[lo-1]:lo-=1
    candidates=[];u=env.unwrapped
    for back in [1,2,4,8,16,32,64]:
        start=max(lo,row-back);ix=int(np.searchsorted(rows,start))
        if ix==len(rows) or rows[ix]!=start:ix=None
        if start==row:continue
        if ix is not None and status[ix]!=2:qp,qv=p[ix],v[ix]
        elif not (np.abs(obs[start,u.model.nq-1:])>=10-1e-6).any():
            qp=raw['infos/qpos'][start].copy();qp[0]=0;qv=raw['infos/qvel'][start]
        else:continue
        u.sim.reset();u.set_state(qp,qv);u.sim.data.qacc_warmstart[:]=0;hist=0.
        for j in range(start,row+1):
            hist=max(hist,float(np.abs(u._get_obs()-obs[j]).max()))
            if hist>1e-4:break
            if j==row:pp=u.sim.data.qpos.copy();vv=u.sim.data.qvel.copy()
            ns,r,d,_=u.step(raw['actions'][j].astype(np.float64))
        if hist<=1e-4 and np.max(np.abs(ns-raw['next_observations'][row]))<=1e-3 and abs(r-raw['rewards'][row])<=1e-4 and d==raw['terminals'][row]:
            on,orr,od=step(shifted,pp,vv,raw['actions'][row])
            candidates.append((pp,vv,ns.copy(),r,d,on,orr,od,back,hist))
    if not candidates:return None
    ns_spread=float(np.ptp([c[5] for c in candidates],axis=0).max());r_spread=float(np.ptp([c[6] for c in candidates]).max())
    if ns_spread>1e-3 or r_spread>1e-4 or len(set(c[7] for c in candidates))!=1:return None
    return candidates[0],dict(raw_row=int(row),candidates=len(candidates),back=candidates[0][8],
        history_error=candidates[0][9],shifted_next_spread=ns_spread,shifted_reward_spread=r_spread)

def sample_gate(kind,dest):
    import gym,d4rl
    env=gym.make(kind+'-medium-v2');shifted=call_shifted_env(dict(env_name=kind+'-morph'))
    with h5py.File(str(normal_file(kind)),'r') as f:
        raw={k:f[k][:] for k in ['observations','next_observations','actions','rewards','terminals','timeouts','infos/qpos','infos/qvel']}
    old=read(OUT/'stage_a1'/(kind+'_source_acceptance.json'))
    with np.load(str(OUT/'stage_a1'/(kind+'_normal_source_replay.npz'))) as z:ids=z['raw_indices'];pick=z['buffer_indices']
    assert np.array_equal(np.random.default_rng(0).choice(old['retained_count'],5000,replace=False),pick)
    clipped=(np.abs(raw['observations'][:,env.unwrapped.model.nq-1:])>=10-1e-6).any(axis=1)
    end=raw['terminals'].astype(bool)|raw['timeouts'].astype(bool);needed=set()
    for row in ids:
        row=int(row)
        while row not in needed:
            needed.add(row)
            if not clipped[row] or row==0 or end[row-1]:break
            row-=1
    rows=np.array(sorted(needed));loc={int(r):i for i,r in enumerate(rows)}
    p,v,status,error,clipped=recover(raw,env,rows)
    pred=[];reward=[];done=[];good=[];repairs=[];initial=[]
    for row in ids:
        ix=loc[int(row)];ns,r,d=step(env,p[ix],v[ix],raw['actions'][row])
        passed=status[ix]!=2 and abs(ns-raw['next_observations'][row]).max()<=1e-3 and abs(r-raw['rewards'][row])<=1e-4 and d==raw['terminals'][row]
        if not passed:
            initial.append(int(row));candidate=history_repair(int(row),raw,env,shifted,rows,p,v,status)
            if candidate is not None:
                c,receipt=candidate;p[ix],v[ix],ns,r,d=c[:5];status[ix]=3;passed=True;repairs.append(receipt)
        pred.append(ns);reward.append(r);done.append(d);good.append(passed)
    pred=np.array(pred);reward=np.array(reward);done=np.array(done);good=np.array(good)
    se=np.max(np.abs(pred-raw['next_observations'][ids]),axis=1);re=np.abs(reward-raw['rewards'][ids])
    result={k:val for k,val in old.items() if k not in ['loader','full_state','replay','replay_sha256','replay_seconds']}
    file=dest/(kind+'_normal_source_replay.npz')
    np.savez_compressed(str(file),loader=np.array('explicit_next_v3'),buffer_indices=pick,raw_indices=ids,passed=good,
        replay_next=pred,replay_reward=reward,replay_done=done,state_error=se,reward_error=re,
        recovery_raw_rows=rows,qpos=p,qvel=v,recovery_status=status,recovery_error=error,clipped=clipped[ids])
    result.update(accepted=bool(good.all()),replay=str(file),replay_sha256=sha(file),
        state_error=stats(se),reward_error=stats(re),done_disagreements=int((done!=raw['terminals'][ids]).sum()),
        failed_raw_rows=ids[~good].tolist(),failed_buffer_rows=pick[~good].tolist(),
        recovery='Verified chronological predecessors and previous v2 1--64-step recovery; infos/qvel is clipped',
        initial_history_failed_raw_rows=initial,short_history_repairs=repairs,
        full_state_claim_corrected=True,clipped_full_dataset_fraction=float(clipped.mean()),
        hidden_state_from_history=True,unchanged_sample_and_thresholds=True)
    write(dest/(kind+'_source_acceptance.json'),result)
    print(kind,'RECOVERED SAMPLE',good.sum(),'/',len(good),'failed',ids[~good].tolist(),'maxstate',se.max(),flush=True)
    env.close();shifted.close();del raw;gc.collect()
    return result

def main():
    args=argparse.ArgumentParser();args.add_argument('--attempt',default='stage_a2');a=args.parse_args();dest=OUT/a.attempt;fresh(dest)
    write(dest/'protocol_amendment.json',dict(original_protocol_sha256=sha(OUT/'protocol.json'),
        evidence='stage_a1: infos/qvel max abs=10; it is clipped even though stored float64',
        correction='Replace false full-state assumption with already validated chronological state recovery',
        fixed=['Same 5000 raw rows for each environment, seed0','State<=1e-3 reward<=1e-4 done exact','All sampled rows must pass','No RL/GPU training'],
        history_consistency_tolerance=1e-4,short_history_steps=[1,2,4,8,16,32,64]))
    results={k:sample_gate(k,dest) for k in ['walker2d','hopper']}
    write(dest/'sample_completed.json',dict(complete=True,all_passed=all(r['accepted'] for r in results.values()),sources=results))

if __name__=='__main__':main()
