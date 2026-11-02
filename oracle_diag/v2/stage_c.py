"""All-row OTDF XML replay acceptance. A failed replay is a reported exclusion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support import *
import numpy as np,difflib
from fixed_loader import load_file
from verify_replay_v2 import step,recover,make_target_env
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv

def main():
    dest=OUT/'stage_c';fresh(dest);results={}
    for kind in ['hopper','walker2d']:
      base=kind.replace('walker2d','walker')
      refenv=ENVS[0] if kind=='walker2d' else ENVS[1]
      std=np.load(PREV/'diagnostics'/(refenv+'_source_oracle.npz'))['target_state_std']
      for shift in ['gravity','morph']:
        name=kind+'_'+shift;print('REPLAY C',name,flush=True)
        xml=ROOT/'reproductions/OTDF/envs/assets'/(base+'_'+shift+'.xml')
        path=ROOT/'reproductions/OTDF/datasets'/(kind+'_'+('gravity_0.5' if shift=='gravity' else shift)+'_medium.hdf5')
        data=load_file(path);n=len(data['rewards'])
        boundary=np.r_[np.max(np.abs(data['next_observations'][:-1]-data['observations'][1:]),axis=1)>1e-6,False]
        starts=np.r_[True,(boundary|data['terminals'])[:-1]]
        se=(Walker2dEnv if kind=='walker2d' else HopperEnv)(xml_file=str(xml));te=make_target_env(refenv)
        p,v,status,he,cl=recover(se,data['observations'],data['actions'],starts,refenv)
        pred=[];reward=[];done=[];tn=[];tr=[];td=[]
        for i in range(n):
            a,b,c,_,_=step(se,p[i],v[i],data['actions'][i]);pred.append(a);reward.append(b);done.append(c)
            a,b,c,_,_=step(te,p[i],v[i],data['actions'][i]);tn.append(a);tr.append(b);td.append(c)
        pred=np.array(pred);reward=np.array(reward);done=np.array(done);tn=np.array(tn);tr=np.array(tr);td=np.array(td)
        err=np.max(np.abs(pred-data['next_observations']),axis=1);re=np.abs(reward-data['rewards'])
        passed=(status!=2)&(err<=1e-3)&(re<=1e-4)&(done==data['terminals'])
        initial_fail=np.flatnonzero(~passed).tolist();repaired=[]
        # Apply the previous short-history validation, respecting actual episodes.
        for i in initial_fail:
            candidates=[];lo=int(np.flatnonzero(starts[:i+1])[-1]);u=se.unwrapped
            for back in [1,2,4,8,16,32,64]:
                start=max(lo,i-back)
                if start==i or status[start]==2:continue
                u.sim.reset();u.set_state(p[start],v[start]);u.sim.data.qacc_warmstart[:]=0;hist=0.
                for j in range(start,i+1):
                    hist=max(hist,float(np.max(np.abs(u._get_obs()-data['observations'][j]))))
                    if hist>1e-4:break
                    if j==i:pp=u.sim.data.qpos.copy();vv=u.sim.data.qvel.copy()
                    ns,r,d,_=u.step(data['actions'][j].astype(np.float64))
                if hist<=1e-4 and np.max(np.abs(ns-data['next_observations'][i]))<=1e-3 and abs(r-data['rewards'][i])<=1e-4 and d==data['terminals'][i]:
                    on,orr,od,_,_=step(te,pp,vv,data['actions'][i]);candidates.append((pp,vv,ns.copy(),r,d,on,orr,od))
            if candidates and np.ptp([c[5] for c in candidates],axis=0).max()<=1e-3 and np.ptp([c[6] for c in candidates])<=1e-4 and len(set(c[7] for c in candidates))==1:
                p[i],v[i],pred[i],reward[i],done[i],tn[i],tr[i],td[i]=candidates[0];passed[i]=True;repaired.append(i)
        err=np.max(np.abs(pred-data['next_observations']),axis=1);re=np.abs(reward-data['rewards'])
        np.savez_compressed(dest/(name+'_replay.npz'),passed=passed,clipped=cl,recovery_status=status,boundary=boundary,
            state_error=err,reward_error=re,source_physical_next=pred,source_physical_reward=reward,
            candidate_oracle_next=tn,candidate_oracle_reward=tr,candidate_oracle_done=td)
        result=dict(n=n,source_sha256=sha(path),xml=str(xml),xml_sha256=sha(xml),state_error=stats(err),reward_error=stats(re),
            clipped_fraction=float(cl.mean()),unrecoverable_initial_rows=np.flatnonzero(status==2).tolist(),
            initial_failed_rows=initial_fail,repaired_rows=repaired,failed_rows=np.flatnonzero(~passed).tolist(),
            done_disagreements=int((done!=data['terminals']).sum()),accepted=bool(passed.all()),
            source_class=type(se).__module__,target_class=type(te.unwrapped).__module__)
        if passed.all():
            file=dest/(name+'_source_oracle_fixed.npz')
            np.savez_compressed(file,indices=np.arange(n),source_state=data['observations'],source_action=data['actions'],
                source_next=data['next_observations'],source_reward=data['rewards'],source_not_done=1.-data['terminals'],
                oracle_next=tn,oracle_reward=tr,oracle_not_done=1.-td,source_physical_next=pred,
                source_physical_reward=reward,boundary=boundary,clipped=cl,target_state_std=std)
            physical=np.linalg.norm((pred-tn)/std,axis=1)
            result.update(oracle_path=str(file),oracle_sha256=sha(file),physical_shift=stats(physical),shift_over_1e3_fraction=float((physical>1e-3).mean()))
        else:result['exclusion']='All-row source replay gate failed; candidate counterfactuals are diagnostic only, not accepted oracle.'
        if shift=='gravity':
            candidates=list((ROOT/'envs/mujoco/assets').glob(kind+'*gravity*'))
            result['repository_xml_comparison']=[]
            for other in candidates:
                diff=list(difflib.unified_diff(xml.read_text().splitlines(),other.read_text().splitlines(),fromfile=str(xml),tofile=str(other),lineterm=''))
                (dest/(name+'__'+other.stem+'.diff.txt')).write_text('\n'.join(diff),encoding='utf-8')
                result['repository_xml_comparison'].append(dict(path=str(other),sha256=sha(other),byte_identical=sha(xml)==sha(other),diff_lines=len(diff)))
        write(dest/(name+'_acceptance.json'),result);results[name]=result
        print('RESULT C',name,'accepted',result['accepted'],'failed',len(result['failed_rows']),flush=True);se.close();te.close()
    write(dest/'completed.json',dict(completed=True,datasets=results,accepted=[k for k,v in results.items() if v['accepted']]))
if __name__=='__main__':main()
