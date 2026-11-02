"""Resolve rounded observations by validated short trajectory histories."""
from verify_warmstart import *


def main():
    if (OUT/'final_acceptance.json').exists():raise RuntimeError('Existing acceptance')
    prior=json.loads((OUT/'warmstart_acceptance.json').read_text());answer={}
    for name in ENVS:
        src=load_source_buffer(name);se=make_shifted_env(name);te=make_target_env(name);u=se.unwrapped
        dat=np.load(OUT/(name+'_recovered_full_state.npz'))
        arrays={k:dat[k].copy() for k in dat.files};repairs=[]
        with h5py.File(source_hdf5_path(name),'r') as f:truth=f['next_observations'][:src.size]
        for i in np.flatnonzero(~arrays['passed']):
            candidates=[]
            for back in [1,2,4,8,16,32,64]:
                start=max(int(i//1000*1000),int(i-back))
                u.sim.reset();u.set_state(arrays['qpos'][start],arrays['qvel'][start]);u.sim.data.qacc_warmstart[:]=0
                history_error=0.
                for j in range(start,i+1):
                    history_error=max(history_error,float(np.max(np.abs(u._get_obs()-src.state[j]))))
                    if j==i:p=u.sim.data.qpos.copy();v=u.sim.data.qvel.copy();w=u.sim.data.qacc_warmstart.copy()
                    ns,r,d,_=u.step(src.action[j].astype(np.float64))
                e=float(np.max(np.abs(ns-truth[i])));re=abs(float(r)-float(src.reward[i,0]))
                if history_error<=1e-4 and e<=TOL_STATE and re<=TOL_REWARD and d==bool(1-src.not_done[i,0]):
                    tn,tr,td=with_warm(te,p,v,src.action[i],np.zeros_like(w))
                    candidates.append(dict(back=back,p=p,v=v,w=w,ns=ns.copy(),r=r,d=d,tn=tn,tr=tr,td=td,
                        history_error=history_error,source_state_error=e,source_reward_error=re))
            if not candidates:raise RuntimeError('No validated history for row '+str(i))
            spread=float(np.ptp(np.array([c['tn'] for c in candidates]),axis=0).max())
            rspread=float(np.ptp([c['tr'] for c in candidates]))
            if spread>TOL_STATE or rspread>TOL_REWARD or len(set(c['td'] for c in candidates))!=1:
                raise RuntimeError('Target counterfactual ambiguous across valid histories: '+str(i))
            c=candidates[0]
            for key,source in [('qpos','p'),('qvel','v'),('qacc_warmstart','w'),('source_next','ns'),
                               ('source_reward','r'),('source_done','d'),('target_next','tn'),('target_reward','tr'),('target_done','td')]:arrays[key][i]=c[source]
            arrays['passed'][i]=True
            repairs.append(dict(row=int(i),selected_history_steps=c['back'],validated_histories=len(candidates),
                target_state_spread=spread,target_reward_spread=rspread,history_error=c['history_error'],
                source_state_error=c['source_state_error'],source_reward_error=c['source_reward_error']))
        stateerr=np.max(np.abs(arrays['source_next']-truth),axis=1);rewarderr=np.abs(arrays['source_reward']-src.reward.ravel())
        passed=bool(arrays['passed'].all() and stateerr.max()<=TOL_STATE and rewarderr.max()<=TOL_REWARD)
        if not passed:raise RuntimeError('Final gate failed '+name)
        np.savez_compressed(OUT/(name+'_validated_states.npz'),**arrays)
        answer[name]=dict(gate_passed=passed,source_n=src.size,repaired_rows=repairs,
            source_state_linf=summarize(stateerr),source_reward_abs=summarize(rewarderr),
            source_done_disagreements=int(np.count_nonzero(arrays['source_done']!=(1-src.not_done.ravel()).astype(bool))),
            source_sha256=sha256_file(source_hdf5_path(name)),states_sha256=sha256_file(OUT/(name+'_validated_states.npz')))
        print(name,'PASS','repaired',len(repairs),'max_state_error',stateerr.max(),flush=True)
        se.close();te.close()
    save('final_acceptance.json',dict(all_passed=True,environments=answer,device='cpu',
        tolerance=dict(state_linf=TOL_STATE,reward_abs=TOL_REWARD,history_observation_linf=1e-4),
        note='Validated approximate oracle from rounded observations; not exact latent-state truth. Earlier failed acceptance retained. Target dataset has one unrecoverable clipped episode-start sample in Walker and Hopper, excluded from recovery check.'))


if __name__=='__main__':main()
