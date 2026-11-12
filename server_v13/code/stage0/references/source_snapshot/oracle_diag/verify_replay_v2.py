"""Independent cold-reset replay gate. Does not overwrite the earlier Stage 0."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, time, inspect, hashlib
from pathlib import Path
import numpy as np
import h5py
from common import (ROOT, ENVS, load_source_buffer, load_target_buffer,
    make_target_env, make_shifted_env, source_hdf5_path, build_bellman_agent,
    is_qvel_clipped, obs_to_qpos_qvel, sha256_file, target_subsample_indices,
    summarize, torch, d4rl)
torch.set_num_threads(2)
OUT = Path(__file__).parent / 'results' / 'verified_20261003'
TOL_STATE, TOL_REWARD = 1e-3, 1e-4


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def step(env, qp, qv, action):
    u = env.unwrapped
    u.sim.reset()
    u.set_state(np.asarray(qp,dtype=np.float64), np.asarray(qv,dtype=np.float64))
    u.sim.data.qacc_warmstart[:] = 0
    ns,r,done,_ = u.step(np.asarray(action,dtype=np.float64))
    return np.asarray(ns).copy(), float(r), bool(done), u.sim.data.qpos.copy(), u.sim.data.qvel.copy()


def recover(env, obs, actions, starts, name):
    """Only a verified chronological predecessor may supply hidden velocity."""
    clipped = is_qvel_clipped(env,name,obs,tol=1e-6)
    qp=np.empty((len(obs),env.unwrapped.model.nq)); qv=np.empty((len(obs),env.unwrapped.model.nv))
    status=np.full(len(obs),2,np.int8); err=np.full(len(obs),np.nan)
    for i in range(len(obs)):
        qp[i],qv[i]=obs_to_qpos_qvel(env,obs[i])
        if not clipped[i]:
            status[i]=0;err[i]=0
        elif i>0 and not starts[i] and status[i-1]!=2:
            ns,_,_,p,v=step(env,qp[i-1],qv[i-1],actions[i-1])
            err[i]=np.max(np.abs(ns-obs[i]))
            if err[i]<=1e-4:
                p[0]=0
                qp[i]=p;qv[i]=v;status[i]=1
    return qp,qv,status,err,clipped


def replay(env, qp,qv,actions):
    rows=[step(env,p,v,a)[:3] for p,v,a in zip(qp,qv,actions)]
    return np.array([r[0] for r in rows]),np.array([r[1] for r in rows]),np.array([r[2] for r in rows])


def errors(pred,ref,pr,rr,pd,rd,mask):
    return dict(n=int(mask.sum()), state_linf=summarize(np.max(np.abs(pred-ref),axis=1)[mask]),
        state_l2=summarize(np.linalg.norm(pred-ref,axis=1)[mask]), reward_abs=summarize(np.abs(pr-rr)[mask]),
        done_disagreements=int(np.count_nonzero((pd!=rd)[mask])))


def main():
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite '+str(OUT))
    OUT.mkdir(parents=True)
    protected=[ROOT/'train.py',ROOT/'dataset/call_dataset.py',ROOT/'algo/utils.py',ROOT/'envs/mujoco/call_mujoco_env.py']
    protected+=list((ROOT/'algo/offline_offline').glob('iql*.py'))
    protected+=list((ROOT/'config/mujoco/iql_bellman_fb').glob('*.yaml'))
    protected+=list(Path(__file__).parent.glob('*.py'))
    hashes={str(p):sha256_file(p) for p in protected}
    save('plan.json',dict(seed=0,device='cpu',target_check_n=5000,source_check='all loaded rows',
        state_linf_tolerance=TOL_STATE,reward_tolerance=TOL_REWARD,
        gate='All source rows must be recoverable and reproduce explicit source next_observations/reward/done within tolerance; target recoverable rows must pass.',
        code_sha256=hashes,source_files={e:sha256_file(source_hdf5_path(e)) for e in ENVS}))
    results={}
    for name in ENVS:
        begin=time.time();print('CHECK',name,flush=True)
        src=load_source_buffer(name);tar,full=load_target_buffer(name,0)
        se,te=make_shifted_env(name),make_target_env(name)
        with h5py.File(source_hdf5_path(name),'r') as f:
            raw={k:f[k][:] for k in ['observations','next_observations','actions','rewards','terminals']}
            keys=[];f.visit(keys.append)
        assert np.array_equal(src.state,raw['observations'][:src.size].astype(np.float32))
        boundary=np.max(np.abs(src.next_state-raw['next_observations'][:src.size]),axis=1)>1e-6
        starts=np.r_[True,boundary[:-1]]
        qp,qv,st,ve,cl=recover(se,src.state,src.action,starts,name)
        pred,pr,pd=replay(se,qp,qv,src.action)
        ref=raw['next_observations'][:src.size];rr=src.reward.ravel();rd=~src.not_done.ravel().astype(bool)
        source_pass=(np.max(np.abs(pred-ref),axis=1)<=TOL_STATE)&(np.abs(pr-rr)<=TOL_REWARD)&(pd==rd)&(st!=2)
        # The same recovered source physical state is used in both dynamics.
        oracle,orr,odd=replay(te,qp,qv,src.action)
        np.savez_compressed(OUT/(name+'_source_replay.npz'),qpos=qp,qvel=qv,status=st,
            recovery_error=ve,clipped=cl,boundary=boundary,passed=source_pass,
            shifted_next=pred,shifted_reward=pr,shifted_done=pd,
            candidate_target_next=oracle,candidate_target_reward=orr,candidate_target_done=odd)
        # Match target rows through the original subsample, and check explicit labels.
        ds=te.get_dataset(); keep=np.flatnonzero(~ds['timeouts'][:-1].astype(bool))
        assert len(keep)==full
        subset=target_subsample_indices(full,.1,0,src.state.shape[1],src.action.shape[1])
        pick=np.random.default_rng(0).choice(tar.size,5000,replace=False);rawids=keep[subset[pick]]
        assert np.array_equal(ds['observations'][rawids].astype(np.float32),tar.state[pick])
        needed=set()
        end=ds['terminals'].astype(bool)|ds['timeouts'].astype(bool)
        allclip=is_qvel_clipped(te,name,ds['observations'],tol=1e-6)
        for row in rawids:
            row=int(row)
            while row not in needed:
                needed.add(row)
                if not allclip[row] or row==0 or end[row-1]:break
                row-=1
        rows=np.array(sorted(needed));pos={int(r):i for i,r in enumerate(rows)}
        start=np.array([r==0 or end[r-1] or r-1 not in pos for r in rows])
        p,v,s,e,c=recover(te,ds['observations'][rows],ds['actions'][rows],start,name)
        loc=np.array([pos[int(i)] for i in rawids]);good=s[loc]!=2
        pp,rv,dd=replay(te,p[loc],v[loc],ds['actions'][rawids])
        truth=ds['next_observations'][rawids];rewards=ds['rewards'][rawids].ravel();done=ds['terminals'][rawids].ravel().astype(bool)
        target_pass=(np.max(np.abs(pp-truth),axis=1)<=TOL_STATE)&(np.abs(rv-rewards)<=TOL_REWARD)&(dd==done)
        pob=[];vob=[]
        for ob in tar.state[pick]:
            po,vo=obs_to_qpos_qvel(te,ob);pob.append(po);vob.append(vo)
        naive,nr,nd=replay(te,pob,vob,tar.action[pick]);nc=is_qvel_clipped(te,name,tar.state[pick],tol=1e-6)
        np.savez_compressed(OUT/(name+'_target_replay.npz'),raw_indices=rawids,clipped=nc,
            recoverable=good,passed=target_pass,next=pp,reward=rv,done=dd,truth_next=truth,truth_reward=rewards,
            truth_done=done,naive_next=naive,naive_reward=nr)
        modeldiff={}
        for field in ['jnt_range','jnt_limited','dof_damping','body_mass','geom_friction','actuator_ctrlrange','actuator_gear']:
            a=np.asarray(getattr(se.unwrapped.model,field));b=np.asarray(getattr(te.unwrapped.model,field))
            modeldiff[field]=dict(max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))),different_entries=int(np.count_nonzero(a!=b)))
        result=dict(env=name,source_size=src.size,target_size=tar.size,source_hdf5_keys=keys,
            shift='XML joint range change; actions unchanged',model_differences=modeldiff,
            target_class=type(te.unwrapped).__module__,source_class=type(se.unwrapped).__module__,
            frame_skip=[se.unwrapped.frame_skip,te.unwrapped.frame_skip],dt=[se.unwrapped.dt,te.unwrapped.dt],
            source=dict(clipped=int(cl.sum()),recoverable=int(np.count_nonzero(st!=2)),boundaries=np.flatnonzero(boundary).tolist(),
                fail_rows=np.flatnonzero(~source_pass).tolist(),all=errors(pred,ref,pr,rr,pd,rd,np.ones(src.size,bool)),
                unclipped=errors(pred,ref,pr,rr,pd,rd,~cl),source_target_next_difference=summarize(np.linalg.norm(pred-oracle,axis=1)),
                source_target_reward_difference=summarize(np.abs(pr-orr))),
            target=dict(clipped=int(nc.sum()),unrecoverable_raw_ids=rawids[~good].tolist(),
                recoverable_fail_raw_ids=rawids[good&~target_pass].tolist(),
                recovered=errors(pp,truth,rv,rewards,dd,done,good),
                from_observation=errors(naive,truth,nr,rewards,nd,done,np.ones(len(pick),bool)),
                unclipped=errors(naive,truth,nr,rewards,nd,done,~nc)),
            gate_passed=bool(source_pass.all() and target_pass[good].all()),seconds=time.time()-begin)
        results[name]=result;save(name+'_acceptance.json',result)
        print(json.dumps(dict(env=name,gate=result['gate_passed'],source_fail=result['source']['fail_rows'],target_fail=result['target']['recoverable_fail_raw_ids'],seconds=result['seconds'])),flush=True)
        se.close();te.close()
    save('acceptance.json',dict(all_passed=all(r['gate_passed'] for r in results.values()),environments=results,
        old_files_unchanged=all(sha256_file(p)==h for p,h in hashes.items()),cuda_initialized=torch.cuda.is_initialized()))


if __name__=='__main__':main()
