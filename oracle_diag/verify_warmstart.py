"""Investigate residual Walker replay errors using trajectory solver history."""
from verify_replay_v2 import *


def with_warm(env,p,v,a,warm):
    u=env.unwrapped;u.sim.reset();u.set_state(p,v)
    u.sim.data.qacc_warmstart[:]=warm
    ns,r,d,_=u.step(np.asarray(a,dtype=np.float64))
    return np.asarray(ns).copy(),float(r),bool(d)


def main():
    dest=OUT/'warmstart_acceptance.json'
    if dest.exists():raise RuntimeError('Existing results')
    results={}
    for name in ENVS:
        src=load_source_buffer(name);se=make_shifted_env(name);te=make_target_env(name);u=se.unwrapped
        with h5py.File(source_hdf5_path(name),'r') as f:ref=f['next_observations'][:src.size]
        boundary=np.max(np.abs(src.next_state-ref),axis=1)>1e-6
        starts=np.r_[True,boundary[:-1]];cl=is_qvel_clipped(se,name,src.state,tol=1e-6)
        p=[];v=[];w=[];pred=[];rewards=[];done=[];current_errors=[]
        for i in range(src.size):
            if starts[i]:u.sim.reset()
            warm=u.sim.data.qacc_warmstart.copy()
            if cl[i] and not starts[i]:
                qp=u.sim.data.qpos.copy();qv=u.sim.data.qvel.copy();qp[0]=0
            else:qp,qv=obs_to_qpos_qvel(se,src.state[i])
            u.set_state(qp,qv);u.sim.data.qacc_warmstart[:]=warm
            current_errors.append(float(np.max(np.abs(u._get_obs()-src.state[i]))))
            p.append(qp.copy());v.append(qv.copy());w.append(warm.copy())
            ns,r,d,_=u.step(src.action[i].astype(np.float64));pred.append(ns);rewards.append(r);done.append(d)
        p=np.array(p);v=np.array(v);w=np.array(w);pred=np.array(pred);rewards=np.array(rewards);done=np.array(done)
        rr=src.reward.ravel();rd=~src.not_done.ravel().astype(bool)
        stateerr=np.max(np.abs(pred-ref),axis=1);rewarderr=np.abs(rewards-rr)
        passed=(stateerr<=TOL_STATE)&(rewarderr<=TOL_REWARD)&(done==rd)&(np.array(current_errors)<=1e-4)
        cold,cr,cd=replay(te,p,v,src.action)
        warmrows=[with_warm(te,qp,qv,a,wa) for qp,qv,a,wa in zip(p,v,src.action,w)]
        wn=np.array([row[0] for row in warmrows]);wr=np.array([row[1] for row in warmrows]);wd=np.array([row[2] for row in warmrows])
        sensitivity=np.max(np.abs(cold-wn),axis=1);rsensitivity=np.abs(cr-wr)
        result=dict(env=name,source_all=errors(pred,ref,rewards,rr,done,rd,np.ones(src.size,bool)),
            current_obs_error=summarize(current_errors),fail_rows=np.flatnonzero(~passed).tolist(),
            target_warm_vs_cold_state_linf=summarize(sensitivity),target_warm_vs_cold_reward=summarize(rsensitivity),
            target_warm_vs_cold_done_disagreements=int(np.count_nonzero(wd!=cd)),
            gate_passed=bool(passed.all() and sensitivity.max()<=TOL_STATE and rsensitivity.max()<=TOL_REWARD and np.array_equal(wd,cd)))
        np.savez_compressed(OUT/(name+'_recovered_full_state.npz'),qpos=p,qvel=v,qacc_warmstart=w,
            passed=passed,boundary=boundary,clipped=cl,source_next=pred,source_reward=rewards,source_done=done,
            target_next=cold,target_reward=cr,target_done=cd,target_warm_next=wn,target_warm_reward=wr,target_warm_done=wd)
        results[name]=result;print(json.dumps(result),flush=True);se.close();te.close()
    save('warmstart_acceptance.json',dict(all_passed=all(r['gate_passed'] for r in results.values()),environments=results,
        definition='Sequential source solver warmstart reconstructed; canonical target cold reset checked against transferred source warmstart.'))


if __name__=='__main__':main()
