"""CPU-only, immutable Stage A evidence; no policy learning takes place here."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from support_v3 import *
from task_env import TASKS,paths,normalization,call_target_dataset,call_shifted_env
import argparse,time,gc,traceback
import numpy as np,h5py

def step(env,qp,qv,action):
    # Same cold-reset convention as the independently checked v2 replay.
    u=env.unwrapped
    u.sim.reset();u.set_state(np.asarray(qp,np.float64),np.asarray(qv,np.float64))
    u.sim.data.qacc_warmstart[:]=0
    ns,r,d,_=u.step(np.asarray(action,np.float64))
    return np.asarray(ns).copy(),float(r),bool(d)

def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def normal_file(kind):return Path.home()/'.d4rl/datasets'/(kind+'_medium-v2.hdf5')

def target_gate(task,dest):
    from utils import ReplayBuffer
    xml,file=paths(task);key=task.replace('-','_')
    previous=V2/'stage_c'/(key+'_acceptance.json')
    old=read(previous);cache=V2/'stage_c'/(key+'_replay.npz')
    assert old['accepted'] and old['done_disagreements']==0 and not old['failed_rows']
    assert sha(xml)==old['xml_sha256'] and sha(file)==old['source_sha256']
    data=call_target_dataset(task)
    buf=ReplayBuffer(data['observations'].shape[1],data['actions'].shape[1],'cpu',max_size=1)
    buf.convert_D4RL(data)
    with h5py.File(str(file),'r') as f:
        for attr,k in [('state','observations'),('action','actions'),('next_state','next_observations'),('reward','rewards')]:
            assert np.array_equal(getattr(buf,attr).reshape(f[k].shape),f[k][:].astype(np.float32)),k
        assert np.array_equal(buf.not_done.ravel(),1.-f['terminals'][:].ravel())
        assert buf.size==len(f['rewards'])==5000
    with np.load(str(cache)) as z:
        pred=z['source_physical_next'];reward=z['source_physical_reward']
        se=np.abs(pred-data['next_observations']).max(axis=1);re=np.abs(reward-data['rewards'])
        assert (se<=1e-3).all() and (re<=1e-4).all() and z['passed'].all()
        assert np.allclose(se,z['state_error'],atol=1e-12,rtol=0)
        replay=dest/(key+'_target_replay.npz')
        # v2 saved aggregate done agreement, not its per-row factual done array.
        # Preserve this provenance; never relabel inferred masks as new observations.
        np.savez_compressed(str(replay),loader=np.array('explicit_next_v3'),indices=np.arange(buf.size),
            replay_next=pred,replay_reward=reward,recorded_done=data['terminals'],
            passed=z['passed'],clipped=z['clipped'],state_error=se,reward_error=re,boundary=z['boundary'])
        trajectories=int(1+np.count_nonzero((z['boundary']|data['terminals'])[:-1]))
    env=call_shifted_env(dict(env_name=task,shift_level='medium'))
    assert env._max_episode_steps==1000
    env_info=dict(class_name=type(env.unwrapped).__module__,time_limit=env._max_episode_steps,
        gravity=env.unwrapped.model.opt.gravity.tolist(),nq=env.unwrapped.model.nq,nv=env.unwrapped.model.nv)
    env.close()
    result=dict(task=task,accepted=True,n=buf.size,trajectories=trajectories,buffer_exact_all_rows=True,
        explicit_next=True,source='Reused v2 factual shifted replay; no new replay',
        previous_acceptance=str(previous),previous_acceptance_sha256=sha(previous),
        previous_replay=str(cache),previous_replay_sha256=sha(cache),dataset_sha256=sha(file),
        xml=str(xml),xml_sha256=sha(xml),replay_file=str(replay),replay_sha256=sha(replay),
        done_evidence='v2 acceptance has zero disagreements; individual replay done values were not stored',
        state_error=stats(se),reward_error=stats(re),done_disagreements=0,
        clipped_fraction=old['clipped_fraction'],failed_rows=[],repaired_rows=old['repaired_rows'],
        normalization=normalization(task),environment=env_info)
    write(dest/(key+'_acceptance.json'),result)
    print('TARGET',task,'accepted',buf.size,flush=True)
    return result

def source_gate(kind,dest):
    import gym,d4rl
    source=normal_file(kind);env=gym.make(kind+'-medium-v2')
    keys=['observations','next_observations','actions','rewards','terminals','timeouts','infos/qpos','infos/qvel']
    with h5py.File(str(source),'r') as f:raw={k:f[k][:] for k in keys}
    n=len(raw['rewards']);keep=np.flatnonzero(~raw['timeouts'][:-1].astype(bool))
    # Independent row mapping for the unchanged, installed source loader.
    loaded=d4rl.qlearning_dataset(env,dataset=raw)
    mapping={k:raw[k][keep].astype(np.float32 if k!='terminals' else bool) for k in ['observations','actions','rewards','terminals']}
    mapping['next_observations']=raw['observations'][keep+1].astype(np.float32)
    for k in mapping:assert np.array_equal(loaded[k],mapping[k]),k
    source_hashes={k:array_hash(v) for k,v in loaded.items()}
    projection=np.concatenate([raw['infos/qpos'][:,1:],np.clip(raw['infos/qvel'],-10,10)],axis=1)
    projection_error=float(np.max(np.abs(projection-raw['observations'])))
    assert projection_error<=1e-7
    pick=np.random.default_rng(0).choice(len(keep),5000,replace=False);ids=keep[pick]
    begin=time.time();rows=[step(env,raw['infos/qpos'][i],raw['infos/qvel'][i],raw['actions'][i]) for i in ids]
    pred=np.array([r[0] for r in rows]);reward=np.array([r[1] for r in rows]);done=np.array([r[2] for r in rows])
    se=np.max(np.abs(pred-raw['next_observations'][ids]),axis=1);re=np.abs(reward-raw['rewards'][ids])
    passed=(se<=1e-3)&(re<=1e-4)&(done==raw['terminals'][ids])
    velocity_start=env.unwrapped.model.nq-1
    clipped=(np.abs(raw['observations'][ids,velocity_start:])>=10-1e-6).any(axis=1)
    replay=dest/(kind+'_normal_source_replay.npz')
    np.savez_compressed(str(replay),loader=np.array('explicit_next_v3'),buffer_indices=pick,raw_indices=ids,
        replay_next=pred,replay_reward=reward,replay_done=done,state_error=se,reward_error=re,passed=passed,clipped=clipped)
    np.save(str(dest/(kind+'_source_raw_indices.npy')),keep)
    terminal=raw['terminals'].astype(bool);timeouts=raw['timeouts'].astype(bool)
    source_difference=np.max(np.abs(loaded['next_observations']-raw['next_observations'][keep]),axis=1)
    result=dict(kind=kind,accepted=bool(passed.all()),n=5000,seed=0,raw_count=n,retained_count=len(keep),
        trajectories=int(1+np.count_nonzero((terminal|timeouts)[:-1])),terminals=int(terminal.sum()),timeouts=int(timeouts.sum()),
        replay=str(replay),replay_sha256=sha(replay),dataset=str(source),dataset_sha256=sha(source),
        source_loader='Unchanged d4rl.qlearning_dataset: observations[i+1], skip timeouts and final raw row',
        source_alignment_exact=True,source_array_sha256=source_hashes,projection_max_error=projection_error,
        full_state='Stored infos/qpos and infos/qvel, privileged simulator state',
        source_next_vs_explicit_different_rows=int((source_difference>1e-6).sum()),
        source_next_vs_explicit_nonterminal_different=int(((source_difference>1e-6)&~loaded['terminals']).sum()),
        state_error=stats(se),reward_error=stats(re),done_disagreements=int((done!=raw['terminals'][ids]).sum()),
        clipped_fraction=float(clipped.mean()),failed_raw_rows=ids[~passed].tolist(),failed_buffer_rows=pick[~passed].tolist(),
        replay_seconds=time.time()-begin,environment=type(env.unwrapped).__module__,gpu_used=False,
        tolerance=dict(state=1e-3,reward=1e-4,done='exact'),no_failed_rows_removed=True)
    write(dest/(kind+'_source_acceptance.json'),result);env.close()
    print('SOURCE',kind,'accepted',result['accepted'],'failed',len(result['failed_raw_rows']),
        'max state',float(se.max()),'max reward',float(re.max()),flush=True)
    del loaded,mapping,raw,projection;gc.collect()
    return result

def source_oracle(dest,source_result):
    task='walker2d-morph';kind='walker2d';n=source_result['retained_count']
    assert source_result['accepted']
    folder=dest/'walker2d_morph_source_oracle';fresh(folder)
    ids=np.load(str(dest/(kind+'_source_raw_indices.npy')))
    env=call_shifted_env(dict(env_name=task,shift_level='medium'));env.seed(0)
    nexts=np.lib.format.open_memmap(str(folder/'next_state.npy'),mode='w+',dtype=np.float32,shape=(n,17))
    rewards=np.lib.format.open_memmap(str(folder/'reward.npy'),mode='w+',dtype=np.float32,shape=(n,1))
    masks=np.lib.format.open_memmap(str(folder/'not_done.npy'),mode='w+',dtype=np.float32,shape=(n,1))
    begin=time.time();terminal_changes=0;normal_terminal=0;oracle_terminal=0
    with h5py.File(str(normal_file(kind)),'r') as f:
        qp=f['infos/qpos'][:];qv=f['infos/qvel'][:];actions=f['actions'][:];terms=f['terminals'][:]
    for i,rawid in enumerate(ids):
        ns,r,d=step(env,qp[rawid],qv[rawid],actions[rawid])
        assert np.isfinite(ns).all() and np.isfinite(r),'Nonfinite oracle row '+str(rawid)
        nexts[i]=ns;rewards[i,0]=r;masks[i,0]=1.-d
        normal_terminal+=int(terms[rawid]);oracle_terminal+=int(d);terminal_changes+=int(d!=terms[rawid])
        if (i+1)%10000==0 or i+1==n:
            nexts.flush();rewards.flush();masks.flush()
            elapsed=time.time()-begin
            write(folder/'progress.json',dict(rows=i+1,total=n,elapsed_seconds=elapsed,rows_per_second=(i+1)/elapsed),replace=True)
            if (i+1)%100000==0:print('ORACLE',i+1,'/',n,'seconds',round(elapsed,1),flush=True)
    env.close();del nexts,rewards,masks
    result=dict(task=task,complete=True,n=n,gpu_used=False,seconds=time.time()-begin,
        source_dataset_sha256=source_result['dataset_sha256'],xml_sha256=sha(paths(task)[0]),
        source_array_sha256=source_result['source_array_sha256'],raw_indices_path=str(dest/(kind+'_source_raw_indices.npy')),
        raw_indices_sha256=sha(dest/(kind+'_source_raw_indices.npy')),
        arrays={k:dict(path=str(folder/(k+'.npy')),sha256=sha(folder/(k+'.npy'))) for k in ['next_state','reward','not_done']},
        normal_terminals=normal_terminal,oracle_terminals=oracle_terminal,terminal_changes=terminal_changes,
        convention='Each row cold reset, full normal-source qpos/qvel, zero qacc_warmstart, float64 action, shifted XML',
        independently_validated_counterfactual=False)
    write(folder/'completed.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='stage_a1');args=parser.parse_args()
    dest=OUT/args.attempt;fresh(dest);begin=time.time()
    write(dest/'started.json',dict(started=begin,device='cpu',protocol_sha256=sha(OUT/'protocol.json'),seed=0))
    try:
        targets={t:target_gate(t,dest) for t in TASKS}
        sources={k:source_gate(k,dest) for k in ['walker2d','hopper']}
        scales=[]
        for task in TASKS:
            s=sources[task.split('-')[0]];t=targets[task];n=s['retained_count']
            scales.append(dict(task=task,source_n=n,target_n=t['n'],source_to_target=n/t['n'],
                source_trajectories=s['trajectories'],target_trajectories=t['trajectories'],
                expected_reuse_source_mixed=500000*64/n,expected_reuse_target_mixed=500000*64/t['n'],
                expected_reuse_target_only=500000*128/t['n']))
        write(dest/'scale.json',dict(rows=scales,steps=500000,total_batch=128))
        if not all(s['accepted'] for s in sources.values()):
            write(dest/'blocked.json',dict(reason='Normal-source replay gate failed',sources=sources))
            write(OUT/'status.json',dict(state='stage_a_replay_gate_failed',attempt=str(dest),gpu_training_authorized=False),replace=True)
            return
        oracle=source_oracle(dest,sources['walker2d'])
        protected_check()
        write(dest/'completed.json',dict(completed=True,targets=targets,sources=sources,oracle=oracle,
            scales=scales,seconds=time.time()-begin,protected_files_unchanged=True))
        write(OUT/'status.json',dict(state='stage_a_ready_for_review',attempt=str(dest),gpu_training_authorized=False),replace=True)
        print('STAGE A COMPLETE',flush=True)
    except Exception:
        write(dest/'failed.json',dict(traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
