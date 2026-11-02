import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support import *
import numpy as np,h5py,torch
from fixed_loader import load_source_fixed,source_path
from verify_replay_v2 import step,recover,obs_to_qpos_qvel,is_qvel_clipped,make_target_env,make_shifted_env
from common import load_source_buffer,load_policy_config
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    dest=OUT/'stage_a';fresh(dest);torch.set_num_threads(2);rows={}
    for env in ENVS:
        print('AUDIT A',env,flush=True)
        raw=load_source_fixed(env,'medium','medium');old=load_source_buffer(env)
        cached=dict(np.load(PREV/'diagnostics'/(env+'_source_oracle.npz')))
        physical=dict(np.load(PREV/(env+'_validated_states.npz')))
        n=len(raw['rewards']);assert n==5000 and old.size==4999
        for attr,key in [('state','observations'),('action','actions'),('reward','rewards')]:
            assert np.array_equal(getattr(old,attr).ravel(),raw[key][:-1].ravel())
        assert np.array_equal(old.not_done.ravel(),1.-raw['terminals'][:-1])
        delta=np.max(np.abs(old.next_state-raw['next_observations'][:-1]),axis=1)
        boundary=delta>1e-6;assert np.array_equal(boundary,cached['boundary'])
        starts=np.r_[True,boundary|raw['terminals'][:-1]]
        se,te=make_shifted_env(env),make_target_env(env)
        p,v,s,err,cl=recover(se,raw['observations'][-2:],raw['actions'][-2:],starts[-2:],env)
        # The validated penultimate state also recovers a clipped last state.
        if cl[-1] and not starts[-1]:
            ns,_,_,p[-1],v[-1]=step(se,physical['qpos'][-1],physical['qvel'][-1],raw['actions'][-2])
            assert np.max(np.abs(ns-raw['observations'][-1]))<=1e-4
            p[-1,0]=0;s[-1]=1
        assert s[-1]!=2,'Unrecoverable final source row'
        sn,sr,sd,_,_=step(se,p[-1],v[-1],raw['actions'][-1])
        assert np.max(np.abs(sn-raw['next_observations'][-1]))<=1e-3
        assert abs(sr-raw['rewards'][-1])<=1e-4 and sd==raw['terminals'][-1]
        on,orr,od,_,_=step(te,p[-1],v[-1],raw['actions'][-1])
        result=dict(indices=np.arange(n),source_state=raw['observations'],source_action=raw['actions'],
            source_next=raw['next_observations'],source_reward=raw['rewards'],source_not_done=(1.-raw['terminals']).astype(np.float32),
            oracle_next=np.vstack([cached['oracle_next'],on]),oracle_reward=np.r_[cached['oracle_reward'],orr],
            oracle_not_done=np.r_[cached['oracle_not_done'],1.-od],
            source_physical_next=np.vstack([cached['source_physical_next'],sn]),
            source_physical_reward=np.r_[cached['source_physical_reward'],sr],boundary=np.r_[boundary,False],
            clipped=np.r_[cached['clipped'],cl[-1]],target_state_std=cached['target_state_std'])
        file=dest/(env+'_source_oracle_fixed.npz');np.savez_compressed(file,**result)
        with h5py.File(source_path(env)) as f:
            keys=[];f.visit(keys.append)
        g=read(PREV/'diagnostics/stage1_stage2a.json')[env]
        rows[env]=dict(source_path=str(source_path(env)),source_sha256=sha(source_path(env)),hdf5_keys=keys,
            original_n=old.size,fixed_n=n,changed_rows=np.flatnonzero(boundary).tolist(),
            changed_row_max_abs=delta[boundary].tolist(),changed_rows_terminal=raw['terminals'][:-1][boundary].tolist(),
            exactly_matches_previous_boundary=True,source_replay_error=stats(np.max(np.abs(result['source_next']-result['source_physical_next']),axis=1)),
            final_row_error=float(np.max(np.abs(sn-raw['next_observations'][-1]))),final_row_reward_error=abs(sr-float(raw['rewards'][-1])),
            trajectory_segments=int(starts.sum()),terminal_count=int(raw['terminals'].sum()),
            target_n=g['target_subset_size'],source_target_ratio=n/g['target_subset_size'],expected_draws_per_row=500000*128/n,
            oracle_path=str(file),oracle_sha256=sha(file),alignment='row i equals raw hdf5 row i; all N rows; explicit next_observations; original terminal semantics')
        write(dest/(env+'_loader_audit.json'),rows[env]);se.close();te.close()
    env=ENVS[1];a=np.load(PREV/'diagnostics'/(env+'_source_oracle.npz'))
    checkpoint=OLD/'results/training_20261003/hopper_CONST/values/V_500000.pt'
    m=IQL_BELLMAN_FB(load_policy_config(env,overrides={'source_budget':.2}),torch.device('cpu'))
    m.v_func.load_state_dict(torch.load(str(checkpoint),map_location='cpu'));m.v_func.eval()
    t=lambda x:torch.as_tensor(x,dtype=torch.float32)
    with torch.no_grad():
        y=t(a['source_reward'])+.99*t(a['source_not_done'])*m.v_func(t(a['source_next'])).ravel()
        yo=t(a['oracle_reward'])+.99*t(a['oracle_not_done'])*m.v_func(t(a['oracle_next'])).ravel()
    diff=(y-yo).abs().numpy();tb=[]
    for r in read(PREV/'report_data.json')['stage3_score_evidence']:
        acc=EventAccumulator('\\\\?\\'+str(Path(r['file']).resolve()),size_guidance={'scalars':0});acc.Reload()
        tag=next(k for k in acc.Tags()['scalars'] if k.replace(' ','_')=='test/target_normalized_score')
        vals={x.step:float(x.value) for x in acc.Scalars(tag)}
        last=np.array([vals[k] for k in range(460000,500001,10000)])
        tail=np.array([vals[k] for k in range(300000,500001,10000)])
        tb.append(dict(env=r['env'],method=r['method'],last5_mean=float(last.mean()),last5_std_ddof1=float(last.std(ddof=1)),
            last5_range=float(np.ptp(last)),tail21_std_ddof1=float(tail.std(ddof=1)),last5=last.tolist(),tail21=tail.tolist(),file=r['file']))
    hc=next(r for r in tb if r['env']==env and r['method']=='CONST');hb=next(r for r in tb if r['env']==env and r['method']=='BACKUP')
    write(dest/'hopper_q_targets_original.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        difference=stats(diff),over_1e4=int((diff>1e-4).sum()),bitwise_equal=bool(np.array_equal(y.numpy(),yo.numpy())),
        boundary_count=int(a['boundary'].sum()),all_boundary_terminal=bool((a['source_not_done'][a['boundary']]==0).all()),
        backup_minus_const=hb['last5_mean']-hc['last5_mean'],interpretation='Observed difference between two runs; not an estimate of a universal noise floor; checkpoint variability is not seed variability'),loader='original')
    write(dest/'training_variability_original.json',dict(runs=tb),loader='original')
    write(dest/'completed.json',dict(passed=True,environments=rows,cuda_initialized=torch.cuda.is_initialized()))
    print('STAGE A COMPLETE',flush=True)
if __name__=='__main__':main()
