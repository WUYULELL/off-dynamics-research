"""Independent raw-history restoration and >=10000 target replay checks, CPU only."""
from stage1_runtime import *
import numpy as np

def main():
    check_frozen();cfg=read(OUT/'config.json');generated=read(OUT/'CACHE_GENERATED.json');start=time.time()
    assert generated['passed'] and generated['canonical_rows']==999998
    raw=load_raw();ids=np.load(wp(OUT/'cache/raw_indices.npy'))
    sample=np.load(wp(OUT/'acceptance/sample_buffer_indices.npy'));rows=np.load(wp(OUT/'acceptance/sample_raw_indices.npy'))
    assert np.array_equal(sample,np.random.RandomState(cfg['independent_sample_seed']).choice(len(ids),10000,replace=False))
    assert np.array_equal(rows,ids[sample]) and len(np.unique(rows))==10000
    for r in generated['arrays'].values():assert sha(r['path'])==r['sha256']
    labels={k:np.load(wp(v['path']),mmap_mode='r') for k,v in generated['arrays'].items()}
    # These are read only after independent reconstruction to measure agreement.
    genp=np.load(wp(OUT/'cache/recovery/qpos.npy'),mmap_mode='r')
    genv=np.load(wp(OUT/'cache/recovery/qvel.npy'),mmap_mode='r')
    source,target=make_envs()
    count=dict(source_replays=0,target_replays=0,environment_constructor_steps=2)
    def independent_step(environment,qp,qv,action,domain):
        # Independent implementation; generator's cold_step/recovery are never called.
        u=environment.unwrapped;u.sim.reset()
        u.set_state(qp.astype(np.float64),qv.astype(np.float64));u.sim.data.qacc_warmstart[:]=0
        obs=u._get_obs().copy();next_obs,reward,terminal,_=u.step(action.astype(np.float64))
        count[domain+'_replays']+=1
        return obs,next_obs.copy(),float(reward),bool(terminal),u.sim.data.qpos.copy(),u.sim.data.qvel.copy()
    clipped=np.any(np.abs(raw['observations'][:,5:])>=10-1e-6,axis=1)
    end=raw['terminals']|raw['timeouts']
    names=['restored_observation','source_next_state','source_reward','target_cache_next_state','target_cache_reward','restoration_qpos_agreement','restoration_qvel_agreement']
    errors={n:np.zeros(len(rows),np.float64) for n in names}
    source_disagreement=np.zeros(len(rows),bool);target_disagreement=np.zeros(len(rows),bool)
    backsteps=np.zeros(len(rows),np.int32);failures=[];bitwise_next=0;bitwise_reward=0
    replay_next=np.zeros((len(rows),11),np.float64);replay_reward=np.zeros(len(rows),np.float64);replay_terminal=np.zeros(len(rows),np.uint8)
    try:
        for index,(row,cache_ix) in enumerate(zip(rows,sample)):
            row=int(row);cache_ix=int(cache_ix);anchor=row
            while clipped[anchor] and anchor>0 and not end[anchor-1]:anchor-=1
            if clipped[anchor]:
                failures.append(dict(raw_row=row,reason='No unclipped in-episode anchor'));continue
            pp=raw['infos/qpos'][anchor].copy();vv=raw['infos/qvel'][anchor].copy();backsteps[index]=row-anchor
            history_bad=False
            for current in range(anchor,row+1):
                pp[0]=0
                if current==row:restored_p=pp.copy();restored_v=vv.copy()
                obs,ns,reward,terminal,afterp,afterv=independent_step(source,pp,vv,raw['actions'][current],'source')
                if np.max(np.abs(obs-raw['observations'][current]))>cfg['history_consistency_tolerance'] or np.max(np.abs(ns-raw['next_observations'][current]))>cfg['factual_state_Linf_tolerance'] or abs(reward-raw['rewards'][current])>cfg['factual_reward_abs_tolerance'] or terminal!=raw['terminals'][current]:history_bad=True;break
                pp=afterp;vv=afterv
            if history_bad:
                failures.append(dict(raw_row=row,reason='Independent source history/factual replay gate failed'));continue
            errors['restored_observation'][index]=np.max(np.abs(obs-raw['observations'][row]))
            errors['source_next_state'][index]=np.max(np.abs(ns-raw['next_observations'][row]))
            errors['source_reward'][index]=abs(reward-raw['rewards'][row]);source_disagreement[index]=terminal!=raw['terminals'][row]
            _,ons,orr,od,_,_=independent_step(target,restored_p,restored_v,raw['actions'][row],'target')
            replay_next[index]=ons;replay_reward[index]=orr;replay_terminal[index]=od
            errors['target_cache_next_state'][index]=np.max(np.abs(ons-labels['next_state'][cache_ix]))
            errors['target_cache_reward'][index]=abs(orr-float(labels['reward'][cache_ix,0]));target_disagreement[index]=od!=bool(labels['terminal'][cache_ix,0])
            errors['restoration_qpos_agreement'][index]=np.max(np.abs(restored_p-genp[row]))
            errors['restoration_qvel_agreement'][index]=np.max(np.abs(restored_v-genv[row]))
            bitwise_next+=int(np.array_equal(ons.astype(np.float32),labels['next_state'][cache_ix]))
            bitwise_reward+=int(np.float32(orr)==labels['reward'][cache_ix,0])
            if errors['restored_observation'][index]>cfg['restored_observation_Linf_tolerance'] or errors['target_cache_next_state'][index]>cfg['target_replay_state_Linf_tolerance'] or errors['target_cache_reward'][index]>cfg['target_replay_reward_abs_tolerance'] or target_disagreement[index]:
                failures.append(dict(raw_row=row,reason='Target cache replay/current observation gate failed',target_state_error=float(errors['target_cache_next_state'][index]),target_reward_error=float(errors['target_cache_reward'][index]),target_done_disagreement=bool(target_disagreement[index])))
            if (index+1)%1000==0:
                write(OUT/'acceptance_progress.json',dict(pid=os.getpid(),sampled=index+1,total=10000,failed=len(failures),elapsed_seconds=time.time()-start,counts=count),replace=True)
        np.savez_compressed(wp(OUT/'acceptance/independent_replay.npz'),raw_indices=rows,buffer_indices=sample,
            target_next=replay_next,target_reward=replay_reward,target_terminal=replay_terminal,
            source_terminal_disagreement=source_disagreement,target_terminal_disagreement=target_disagreement,backsteps=backsteps,**errors)
        result=dict(passed=not failures,CPU_only=True,sample_n=10000,sample_seed=cfg['independent_sample_seed'],sampled_without_replacement=True,
            independently_recovered_from_raw=True,generator_saved_qpos_qvel_used_as_recovery_input=False,
            errors={k:stats(v) for k,v in errors.items()},state_recovery_failed_count=sum('anchor' in r['reason'] or 'history' in r['reason'] for r in failures),
            failed_count=len(failures),failures=failures,source_terminal_disagreements=int(source_disagreement.sum()),target_terminal_disagreements=int(target_disagreement.sum()),
            bitwise_float32_next_matches=bitwise_next,bitwise_float32_reward_matches=bitwise_reward,history_steps=stats(backsteps),
            counts=count,elapsed_seconds=time.time()-start,replay_file_sha256=sha(OUT/'acceptance/independent_replay.npz'),
            generated_receipt_sha256=sha(OUT/'CACHE_GENERATED.json'),time=time.time(),training_updates=0,
            limitation='Independent re-execution in the same MuJoCo target model verifies restoration and cache consistency, not independent physical ground truth; observed-state recovery errors do not certify hidden velocity truth')
        write(OUT/'INDEPENDENT_ACCEPTANCE.json',result)
        print(json.dumps(result),flush=True)
        assert result['passed'],'Independent replay acceptance failed; preserve evidence, no training'
    finally:source.close();target.close()

if __name__=='__main__':main()
