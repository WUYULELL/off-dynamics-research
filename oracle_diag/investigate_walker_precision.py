"""Trace whether recent trajectory history resolves rounded-state residuals."""
from verify_replay_v2 import *


def main():
    name=ENVS[0];src=load_source_buffer(name);env=make_shifted_env(name);u=env.unwrapped
    with h5py.File(source_hdf5_path(name),'r') as f:truth=f['next_observations'][:src.size]
    old=json.loads((OUT/(name+'_acceptance.json')).read_text());ids=old['source']['fail_rows']
    saved=np.load(OUT/(name+'_recovered_full_state.npz'));results=[]
    for i in ids:
        trials=[]
        for back in [1,2,4,8,16,32,64]:
            start=max(i//1000*1000,i-back)
            u.sim.reset();u.set_state(saved['qpos'][start],saved['qvel'][start]);u.sim.data.qacc_warmstart[:]=0
            max_before=0.
            for j in range(start,i+1):
                max_before=max(max_before,float(np.max(np.abs(u._get_obs()-src.state[j]))))
                if j==i:qp=u.sim.data.qpos.copy();qv=u.sim.data.qvel.copy()
                ns,r,d,_=u.step(src.action[j].astype(np.float64))
            trials.append(dict(back=back,current_and_history_error=max_before,
                next_error=float(np.max(np.abs(ns-truth[i]))),reward_error=abs(float(r)-float(src.reward[i,0]))))
        results.append(dict(row=i,trials=trials))
    target=OUT/'walker_precision_investigation.json'
    if target.exists():raise RuntimeError('Exists')
    save(target.name,results)
    for row in results:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
