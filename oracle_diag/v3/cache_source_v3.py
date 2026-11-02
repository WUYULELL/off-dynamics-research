"""Build a full, indexed Walker source counterfactual cache on CPU after sample gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v3 import *
from recover_source_v3 import physical_step,history_repair,normal_file
from stage_a_v3 import step
from task_env import call_shifted_env,paths
import numpy as np,h5py,time,argparse,traceback

def main():
    p0=argparse.ArgumentParser();p0.add_argument('--attempt',default='source_oracle1');args=p0.parse_args()
    dest=OUT/args.attempt;fresh(dest)
    gate=read(OUT/'stage_a2/sample_completed.json');assert gate['all_passed']
    source=gate['sources']['walker2d'];begin=time.time()
    write(dest/'plan.json',dict(sample_gate=str(OUT/'stage_a2/sample_completed.json'),
        sample_gate_sha256=sha(OUT/'stage_a2/sample_completed.json'),task='walker2d-morph',device='cpu',
        repair='Same chronological and 1--64-step recovery as sample gate',
        cache_gate='Every retained row must have a recovered state and pass the original factual thresholds; otherwise refuse training cache',
        thresholds=dict(state=1e-3,reward=1e-4,history=1e-4),source_file_sha256=source['dataset_sha256']))
    import gym,d4rl
    env=gym.make('walker2d-medium-v2');shifted=call_shifted_env(dict(env_name='walker2d-morph'))
    with h5py.File(str(normal_file('walker2d')),'r') as f:
        raw={k:f[k][:] for k in ['observations','next_observations','actions','rewards','terminals','timeouts','infos/qpos','infos/qvel']}
    nraw=len(raw['rewards']);ids=np.flatnonzero(~raw['timeouts'][:-1].astype(bool));n=len(ids)
    assert n==source['retained_count'];np.save(str(dest/'raw_indices.npy'),ids)
    p=raw['infos/qpos'].copy();p[:,0]=0;v=raw['infos/qvel'].copy();rows=np.arange(nraw)
    clip=(np.abs(raw['observations'][:,8:])>=10-1e-6).any(axis=1)
    end=raw['terminals'].astype(bool)|raw['timeouts'].astype(bool)
    status=np.where(clip,2,0).astype(np.int8);se=np.full(nraw,np.nan);re=np.full(nraw,np.nan);dd=np.zeros(nraw,bool)
    output={k:np.lib.format.open_memmap(str(dest/(k+'.npy')),mode='w+',dtype=np.float32,shape=(n,dim))
            for k,dim in [('next_state',17),('reward',1),('not_done',1)]}
    failures=[];repairs=[];out_index=0;last=None;normal_terminal=0;oracle_terminal=0;terminal_changes=0
    try:
        for row in range(nraw):
            if clip[row] and row>0 and not end[row-1] and last is not None and status[row-1]!=2:
                ns,_,_,qp,qv=last
                if np.max(np.abs(ns-raw['observations'][row]))<=1e-4:
                    p[row]=qp;p[row,0]=0;v[row]=qv;status[row]=1
            current=physical_step(env,p[row],v[row],raw['actions'][row])
            ns,r,d=current[:3]
            good=status[row]!=2 and np.max(np.abs(ns-raw['next_observations'][row]))<=1e-3 and abs(r-raw['rewards'][row])<=1e-4 and d==raw['terminals'][row]
            if not good:
                candidate=history_repair(row,raw,env,shifted,rows,p,v,status)
                if candidate is not None:
                    c,receipt=candidate;p[row],v[row],ns,r,d=c[:5];status[row]=3;good=True;repairs.append(receipt)
                    current=physical_step(env,p[row],v[row],raw['actions'][row])
            last=current
            se[row]=np.max(np.abs(ns-raw['next_observations'][row]));re[row]=abs(r-raw['rewards'][row]);dd[row]=d!=raw['terminals'][row]
            if not good:status[row]=2
            retained=out_index<n and ids[out_index]==row
            if retained:
                if not good:
                    failures.append(row)
                    for a in output.values():a[out_index]=np.nan
                else:
                    on,orr,od=step(shifted,p[row],v[row],raw['actions'][row])
                    assert np.isfinite(on).all() and np.isfinite(orr)
                    output['next_state'][out_index]=on;output['reward'][out_index,0]=orr;output['not_done'][out_index,0]=1.-od
                    normal_terminal+=int(raw['terminals'][row]);oracle_terminal+=int(od);terminal_changes+=int(od!=raw['terminals'][row])
                out_index+=1
            if (row+1)%10000==0 or row+1==nraw:
                for a in output.values():a.flush()
                elapsed=time.time()-begin
                write(dest/'progress.json',dict(raw_rows=row+1,retained_rows=out_index,total_raw=nraw,
                    elapsed_seconds=elapsed,rows_per_second=(row+1)/elapsed,failed=len(failures),repaired=len(repairs)),replace=True)
                if (row+1)%100000==0:print('SOURCE ORACLE',row+1,'fail',len(failures),'repair',len(repairs),'seconds',round(elapsed,1),flush=True)
        assert out_index==n
        np.savez_compressed(str(dest/'recovery_evidence.npz'),loader=np.array('explicit_next_v3'),qpos=p,qvel=v,
            recovery_status=status,state_error=se,reward_error=re,done_disagreement=dd,clipped=clip)
        receipt=dict(complete=not failures,task='walker2d-morph',n=n,failed_raw_rows=failures,
            source_dataset_sha256=source['dataset_sha256'],xml_sha256=sha(paths('walker2d-morph')[0]),
            source_array_sha256=source['source_array_sha256'],raw_indices_path=str(dest/'raw_indices.npy'),
            raw_indices_sha256=sha(dest/'raw_indices.npy'),
            arrays={k:dict(path=str(dest/(k+'.npy')),sha256=sha(dest/(k+'.npy'))) for k in output},
            seconds=time.time()-begin,state_error=stats(se[ids]),reward_error=stats(re[ids]),
            done_disagreements=int(dd[ids].sum()),clipped_fraction=float(clip[ids].mean()),
            short_history_repairs=repairs,normal_terminals=normal_terminal,oracle_terminals=oracle_terminal,
            terminal_changes=terminal_changes,independently_validated_counterfactual=False,
            recovery='Verified chronological history for hidden clipped velocity; shifted cold reset/zero warmstart',
            gpu_used=False,recovery_evidence_sha256=sha(dest/'recovery_evidence.npz'))
        write(dest/('completed.json' if not failures else 'blocked.json'),receipt)
        write(OUT/'status.json',dict(state='stage_a_cache_ready' if not failures else 'stage_a_full_cache_gate_failed',
            attempt=str(dest),gpu_training_authorized=False),replace=True)
        print('CACHE FINISHED','accepted',not failures,'failures',len(failures),flush=True)
    except Exception:
        write(dest/'failed.json',dict(traceback=traceback.format_exc()));raise
    finally:env.close();shifted.close()

if __name__=='__main__':main()
