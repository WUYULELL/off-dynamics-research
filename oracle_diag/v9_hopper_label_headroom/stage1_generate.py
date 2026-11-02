"""Generate all canonical normal-Hopper -> morph labels, using Walker recovery gates."""
from stage1_runtime import *
import numpy as np

def main():
    check_frozen();cfg=read(OUT/'config.json');start=time.time()
    assert not (OUT/'CACHE_GENERATED.json').exists()
    raw=load_raw();ids=np.load(wp(OUT/'cache/raw_indices.npy'))
    assert np.array_equal(ids,np.flatnonzero(~raw['timeouts'][:-1].astype(bool)))
    source,target=make_envs()
    counts=dict(source_replays=0,target_replays=0,environment_constructor_steps=2)
    def observed(e,p,v,a,domain):
        counts[domain+'_replays']+=1
        return cold_step(e,p,v,a)
    write(OUT/'generation_started.json',dict(pid=os.getpid(),time=time.time(),source_class=type(source.unwrapped).__module__,
        target_class=type(target.unwrapped).__module__,XML_sha256=sha(XML),CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    nraw=len(raw['rewards']);n=len(ids);folder=OUT/'cache/recovery';folder.mkdir(exist_ok=False)
    def mmap(name,dtype,shape):return np.lib.format.open_memmap(wp(folder/(name+'.npy')),mode='w+',dtype=dtype,shape=shape)
    p=mmap('qpos',np.float64,(nraw,6));v=mmap('qvel',np.float64,(nraw,6))
    p[:]=raw['infos/qpos'];p[:,0]=0.;v[:]=raw['infos/qvel']
    replay=mmap('source_replay_next',np.float64,(nraw,11));rewards=mmap('source_replay_reward',np.float64,(nraw,))
    done=mmap('source_replay_terminal',np.uint8,(nraw,))
    status=mmap('recovery_status',np.int8,(nraw,));restore=mmap('restored_observation_error',np.float64,(nraw,))
    se=mmap('source_replay_state_error',np.float64,(nraw,));re=mmap('source_replay_reward_error',np.float64,(nraw,))
    clipped=(np.abs(raw['observations'][:,5:])>=10-1e-6).any(axis=1)
    end=raw['terminals'].astype(bool)|raw['timeouts'].astype(bool)
    status[:]=np.where(clipped,2,0)
    arrays={k:np.lib.format.open_memmap(wp(OUT/'cache'/(k+'.npy')),mode='w+',dtype=dt,shape=shape)
        for k,dt,shape in [('next_state',np.float32,(n,11)),('reward',np.float32,(n,1)),('not_done',np.float32,(n,1)),('terminal',np.uint8,(n,1))]}
    repairs=[];initial_failures=[];excluded_failures=[];last=None;ix=0

    def repair(row):
        lo=row
        while lo>max(0,row-64) and not end[lo-1]:lo-=1
        candidates=[]
        for back in cfg['fallback_history_steps']:
            anchor=max(lo,row-back)
            if anchor==row or status[anchor]==2:continue
            pp=p[anchor].copy();vv=v[anchor].copy();history_error=0.
            for j in range(anchor,row+1):
                pp[0]=0
                result=observed(source,pp,vv,raw['actions'][j],'source')
                history_error=max(history_error,float(np.max(np.abs(result[5]-raw['observations'][j]))))
                if history_error>cfg['history_consistency_tolerance']:break
                if j==row:state_p=pp.copy();state_v=vv.copy()
                pp=result[3].copy();vv=result[4].copy()
            else:
                if np.max(np.abs(result[0]-raw['next_observations'][row]))<=cfg['factual_state_Linf_tolerance'] and abs(result[1]-raw['rewards'][row])<=cfg['factual_reward_abs_tolerance'] and result[2]==raw['terminals'][row]:
                    counter=observed(target,state_p,state_v,raw['actions'][row],'target')
                    candidates.append((state_p,state_v,result,counter,back,history_error))
        if not candidates:return None
        if np.ptp([c[3][0] for c in candidates],axis=0).max()>cfg['target_replay_state_Linf_tolerance']:return None
        if np.ptp([c[3][1] for c in candidates])>cfg['target_replay_reward_abs_tolerance']:return None
        if len(set(c[3][2] for c in candidates))!=1:return None
        return candidates[0],dict(raw_row=int(row),candidates=len(candidates),back=candidates[0][4],history_error=candidates[0][5])

    try:
        for row in range(nraw):
            if clipped[row] and row>0 and not end[row-1] and status[row-1]!=2 and last is not None:
                if np.max(np.abs(last[0]-raw['observations'][row]))<=cfg['history_consistency_tolerance']:
                    p[row]=last[3];p[row,0]=0;v[row]=last[4];status[row]=1
            actual=observed(source,p[row],v[row],raw['actions'][row],'source')
            good=status[row]!=2 and np.max(np.abs(actual[0]-raw['next_observations'][row]))<=cfg['factual_state_Linf_tolerance'] and abs(actual[1]-raw['rewards'][row])<=cfg['factual_reward_abs_tolerance'] and actual[2]==raw['terminals'][row] and np.max(np.abs(actual[5]-raw['observations'][row]))<=cfg['restored_observation_Linf_tolerance']
            if not good:
                initial_failures.append(row);candidate=repair(row)
                if candidate is not None:
                    c,receipt=candidate;p[row]=c[0];v[row]=c[1];actual=c[2];status[row]=3;repairs.append(receipt);good=True
            last=actual;replay[row]=actual[0];rewards[row]=actual[1];done[row]=actual[2]
            restore[row]=float(np.max(np.abs(actual[5]-raw['observations'][row])))
            se[row]=float(np.max(np.abs(actual[0]-raw['next_observations'][row])));re[row]=abs(actual[1]-raw['rewards'][row])
            if not good:status[row]=2
            retained=ix<n and int(ids[ix])==row
            if retained:
                if not good:
                    write(OUT/'RECOVERY_FAILED.json',dict(raw_row=row,buffer_index=ix,reason='Factual source recovery/replay or current observation gate failed; no row removed',
                        status=int(status[row]),state_error=float(se[row]),reward_error=float(re[row]),restored_error=float(restore[row]),
                        done_disagreement=bool(done[row]!=raw['terminals'][row]),counts=counts,time=time.time()))
                    raise RuntimeError('Canonical source recovery failed at raw row %d; preserve artifacts and stop'%row)
                oracle=observed(target,p[row],v[row],raw['actions'][row],'target')
                assert np.isfinite(oracle[0]).all() and np.isfinite(oracle[1])
                arrays['next_state'][ix]=oracle[0];arrays['reward'][ix,0]=oracle[1]
                arrays['terminal'][ix,0]=oracle[2];arrays['not_done'][ix,0]=1.-oracle[2];ix+=1
            elif not good:excluded_failures.append(row)
            if (row+1)%10000==0 or row+1==nraw:
                for a in arrays.values():a.flush()
                elapsed=time.time()-start
                progress=dict(stage='cache_generation',pid=os.getpid(),raw_rows=row+1,canonical_rows=ix,total_canonical=n,
                    elapsed_seconds=elapsed,raw_rows_per_second=(row+1)/elapsed,repaired_rows=len(repairs),source_replays=counts['source_replays'],target_replays=counts['target_replays'])
                write(OUT/'progress.json',progress,replace=True)
                if (row+1)%100000==0:print(json.dumps(progress),flush=True)
        assert ix==n and not np.any(status[ids]==2)
        for a in (p,v,replay,rewards,done,status,restore,se,re):a.flush()
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert np.array_equal(arrays['not_done'],1.-arrays['terminal'])
        receipt=dict(passed=True,task='hopper-morph',direction='normal_source_to_morph_target',canonical_rows=n,raw_rows=nraw,
            source_sha256=sha(SOURCE),XML_sha256=sha(XML),raw_indices_sha256=sha(OUT/'cache/raw_indices.npy'),
            arrays={k:dict(path=str(OUT/'cache'/(k+'.npy')),sha256=sha(OUT/'cache'/(k+'.npy')),shape=list(a.shape),dtype=str(a.dtype)) for k,a in arrays.items()},
            restored_observation_error=stats(restore[ids]),source_replay_state_error=stats(se[ids]),source_replay_reward_error=stats(re[ids]),
            source_terminal_disagreements=int((done[ids]!=raw['terminals'][ids]).sum()),initial_factual_failures=initial_failures,
            final_canonical_recovery_failures=[],short_history_repairs=repairs,excluded_raw_row_failures=excluded_failures,
            recovery_status_counts={str(s):int((status[ids]==s).sum()) for s in (0,1,2,3)},
            source_terminal_count=int(raw['terminals'][ids].sum()),oracle_terminal_count=int(arrays['terminal'].sum()),
            terminal_changes=int((arrays['terminal'].ravel()!=raw['terminals'][ids]).sum()),
            counts=counts,elapsed_seconds=time.time()-start,time=time.time(),CPU_only=True,training_updates=0,
            independent_acceptance_passed=False,no_RL_training_authorized=True)
        assert receipt['source_terminal_disagreements']==0
        write(OUT/'CACHE_GENERATED.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('arrays','short_history_repairs','initial_factual_failures')}),flush=True)
    finally:source.close();target.close()

if __name__=='__main__':main()
