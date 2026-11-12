"""Final-policy CPU fitting errors, saved-loss audit, and 200 new CPU episodes."""
import ast
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from core import A_LABELS,SEEDS,VARIANTS,stats,bins,csv_write,write,sha,utc,interpret_a
from context import CODE,PROJECT,load_policies,actions


def fit(policy, obs, reference):
    errors = []
    for start in range(0,len(obs),1024):
        prediction = actions(policy,obs[start:start+1024])
        errors.extend(np.mean((prediction.astype('float64')-reference[start:start+1024])**2,axis=1))
    return np.asarray(errors)


def sampling_check(inputs, folder, completed):
    path = PROJECT/'source_snapshot/oracle_diag/v9_hopper_label_headroom/stage3/runtime.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='seed')
    ns = dict(hashlib=hashlib)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    streams = {}
    for domain,n in (('target',5000),('source',999998)):
        rng = np.random.RandomState(ns['seed'](domain+'_sample',completed['job']['seed'],'hopper-morph'))
        h = hashlib.sha256()
        for step in range(0,200000,4096):
            count = min(4096,200000-step)*64
            ids = rng.randint(n,size=count).astype('<i8')
            h.update(ids.tobytes())
        value = dict(draws=200000*64,sha256=h.hexdigest())
        assert value==completed['streams'][domain], (domain,folder)
        streams[domain] = value
    cfg = __import__('core').read(folder/'config.json')
    assert cfg['training']['batch_each_domain']==64
    assert cfg['training']['mixed_loss']=='mean target action MSE + mean source action MSE'
    return streams


def run(inputs, out, heartbeat):
    from core import read
    policies = load_policies(inputs)
    target = inputs.data['target']
    source_obs = inputs.array('source_observations')
    source_actions = inputs.array('action_fixed')
    target_bins = bins(inputs.target_steps)
    fit_rows,fit_bin_rows,logs = [],[],[]
    for variant in VARIANTS:
        for seed in SEEDS:
            p = policies[variant,seed]
            te = fit(p,target['observations'],target['actions'])
            se = fit(p,source_obs[inputs.fit_positions],source_actions[inputs.fit_positions])
            fit_rows.append(dict(variant=variant,seed=seed,target_n=5000,target_MSE=float(te.mean()),
                                 corrected_source_n=20000,corrected_source_MSE=float(se.mean())))
            np.savez_compressed(out/f'{variant}_s{seed}_A1_errors.npz',target_row=np.arange(5000),
                                target_MSE_each=te,source_position=inputs.fit_positions,source_MSE_each=se)
            for j,label in enumerate(A_LABELS):
                fit_bin_rows.append(dict(variant=variant,seed=seed,step_bin=label,**stats(te[target_bins==j])))
            if variant=='BC-mix':
                folder = inputs.folder(variant,seed)
                done = read(folder/'COMPLETED.json')
                assert sha(folder/'steps.f32')==done['steps_sha256']
                loss = np.fromfile(folder/'steps.f32',dtype='<f4').reshape(-1,3)
                assert loss.shape==(200000,3) and np.isfinite(loss).all()
                assert np.array_equal(loss[:,0]+loss[:,1],loss[:,2])
                streams = sampling_check(inputs,folder,done)
                tail = loss[-5000:].astype('float64')
                logs.append(dict(seed=seed,steps_first=195001,steps_last=200000,
                    target_loss_mean=float(tail[:,0].mean()),source_loss_mean=float(tail[:,1].mean()),
                    total_loss_mean=float(tail[:,2].mean()),final_target_full_MSE=float(te.mean()),
                    final_source_sample_MSE=float(se.mean()),streams=streams,
                    audit=read(folder/'AUDIT.json'),log_sha256=sha(folder/'steps.f32')))
            heartbeat('A1_A2',variant=variant,seed=seed)
    csv_write(out/'A1_models.csv',fit_rows)
    csv_write(out/'A1_target_step_bins.csv',fit_bin_rows)
    write(out/'A2_log_check.json',dict(runs=logs,code_sources={
        str(inputs.folder('BC-mix',601)/'config.json'):sha(inputs.folder('BC-mix',601)/'config.json'),
        str(__import__('context').V13/'stage1_shared60_20261110/worker.py'):sha(__import__('context').V13/'stage1_shared60_20261110/worker.py')},
        distinction='Saved losses use changing pre-update policies and sampled batches; A1 uses final 200k policy and full target/fixed source sample. Exact equality is not required.',
        sampler_order='target 64 then source 64, independent RandomState streams',
        optimizer_evidence='Code: zero_grad, backward(total target+source), optimizer.step, scheduler.step. Existing CPU audits check full 200k Adam state and cosine; no optimization repeated.'))
    source_tree = cKDTree(np.asarray(source_obs[inputs.nn_positions],dtype='float64')/inputs.scale)
    target_tree = cKDTree(np.asarray(target['observations'],dtype='float64')/inputs.scale)
    metric_rows,action_rows,episodes,pooled = [],[],[],[]
    env = inputs.new_env()
    environment_steps = 1
    try:
        for variant in VARIANTS:
            for seed in SEEDS:
                all_steps,all_dt,all_ds,all_pair = [],[],[],[]
                for episode in range(10):
                    eval_seed = 500000+10*(seed-301)+episode
                    env.seed(eval_seed)
                    obs = env.reset()
                    observations,predicted,nexts,rewards,terminals = [],[],[],[],[]
                    for t in range(1000):
                        action = actions(policies[variant,seed],obs.reshape(1,-1))[0]
                        nxt,reward,done,info = env.step(action)
                        environment_steps += 1
                        if not np.isfinite(nxt).all() or not np.isfinite(reward):
                            raise FloatingPointError('Nonfinite A3 environment output')
                        observations.append(obs.copy());predicted.append(action.copy())
                        nexts.append(nxt.copy());rewards.append(float(reward));terminals.append(bool(done))
                        obs = nxt
                        if done:
                            break
                    observations = np.asarray(observations)
                    step = np.arange(len(observations))
                    scaled = observations.astype('float64')/inputs.scale
                    dt = target_tree.query(scaled)[0]
                    ds = source_tree.query(scaled)[0]
                    pairs = np.empty((len(step),0))
                    if variant=='BC-mix':
                        pm = actions(policies['BC-mix',seed],observations).astype('float64')
                        pa = actions(policies['BC-all',seed],observations).astype('float64')
                        pf = actions(policies['BC-fix',seed],observations).astype('float64')
                        pairs = np.stack([np.linalg.norm(pm-pa,axis=1),np.linalg.norm(pm-pf,axis=1),np.linalg.norm(pa-pf,axis=1)],axis=1)
                        all_pair.append(pairs)
                    np.savez_compressed(out/f'{variant}_s{seed}_episode{episode:02d}.npz',step=step,
                        observations=observations,actions=predicted,next_observations=nexts,rewards=rewards,
                        terminal=terminals,target_neighbor_distance=dt,source_neighbor_distance=ds,
                        mix_all_fix_pairwise_L2=pairs)
                    all_steps.append(step);all_dt.append(dt);all_ds.append(ds)
                    raw_return = float(np.sum(rewards,dtype='float64'))
                    episodes.append(dict(variant=variant,seed=seed,episode=episode,eval_seed=eval_seed,
                        length=len(step),physical_terminal=bool(terminals[-1] and not info.get('TimeLimit.truncated',False)),raw_return=raw_return,
                        normalized_score=(raw_return-inputs.spec['score_min'])/(inputs.spec['score_max']-inputs.spec['score_min'])*100))
                    heartbeat('A3',variant=variant,seed=seed,episode=episode+1,episodes=10)
                step,dt,ds = map(np.concatenate,(all_steps,all_dt,all_ds))
                masks = bins(step)
                pair = np.concatenate(all_pair) if all_pair else None
                for j,label in enumerate(A_LABELS):
                    m = masks==j
                    for domain,value in (('target',dt),('source_sample_100000',ds)):
                        metric_rows.append(dict(variant=variant,seed=seed,step_bin=label,neighbor_domain=domain,**stats(value[m])))
                    if pair is not None:
                        for column,name in enumerate(('mix_vs_all','mix_vs_fix','all_vs_fix')):
                            action_rows.append(dict(seed=seed,step_bin=label,models=name,**stats(pair[m,column])))
    finally:
        env.close()
    csv_write(out/'A3_neighbor_bins.csv',metric_rows)
    csv_write(out/'A3_actions_on_mix_states.csv',action_rows)
    csv_write(out/'A3_episodes.csv',episodes)
    for variant in VARIANTS:
        paths = sorted(out.glob(variant+'_s*_episode*.npz'))
        assert len(paths)==50
        values = {key:[] for key in ('step','target_neighbor_distance','source_neighbor_distance')}
        pair_values = []
        for path in paths:
            with np.load(path,allow_pickle=False) as f:
                for key in values:
                    values[key].append(f[key])
                if variant=='BC-mix':
                    pair_values.append(f['mix_all_fix_pairwise_L2'])
        values = {key:np.concatenate(value) for key,value in values.items()}
        pair = np.concatenate(pair_values) if pair_values else None
        for j,label in enumerate(A_LABELS):
            m = bins(values['step'])==j
            n = int(m.sum())
            dt,ds = values['target_neighbor_distance'][m],values['source_neighbor_distance'][m]
            pooled.append(dict(variant=variant,step_bin=label,n=n,target_NN_mean=stats(dt)['mean'],
                source_NN_mean=stats(ds)['mean'],source_NN_closer_fraction=float((ds<dt).mean()) if n else None,
                mix_all_L2=stats(pair[m,0])['mean'] if pair is not None else None,
                mix_fix_L2=stats(pair[m,1])['mean'] if pair is not None else None,
                all_fix_L2=stats(pair[m,2])['mean'] if pair is not None else None,
                mix_action_closer_to_fix_fraction=float((pair[m,1]<pair[m,0]).mean()) if pair is not None and n else None))
    csv_write(out/'A3_pooled_step_bins.csv',pooled)
    groups = {}
    for variant in VARIANTS:
        values = [r for r in fit_rows if r['variant']==variant]
        groups[variant] = dict(target_MSE_mean=float(np.mean([r['target_MSE'] for r in values])),
            target_MSE_seed_SD=float(np.std([r['target_MSE'] for r in values],ddof=1)),
            source_MSE_mean=float(np.mean([r['corrected_source_MSE'] for r in values])))
    mix,all_ = groups['BC-mix']['target_MSE_mean'],groups['BC-all']['target_MSE_mean']
    interpretation = interpret_a(mix,all_)
    result = dict(completed=True,groups=groups,A1=fit_rows,A1_bins=fit_bin_rows,A2=logs,
        A3_neighbor_bins=metric_rows,A3_action_bins=action_rows,A3_pooled=pooled,episodes=episodes,
        interpretation=interpretation,mix_to_all_MSE_ratio=mix/all_ if all_ else None,
        ratio_undefined_if_denominator_zero=all_==0,environment_steps=environment_steps,
        new_training_updates=0,optimizer_instances=0,CUDA_initialized=False,
        rollout_note='New CPU diagnostic evaluations; not substitutes for V13 formal score records.',
        neighbor_note='Uncentered SD-normalized Euclidean distance; target 5000 vs fixed canonical source sample 100000, so nearest-distance comparisons are descriptive and sample-size-dependent.')
    import torch
    assert not torch.cuda.is_initialized()
    write(out/'RESULT.json',result)
    write(out/'COMPLETED.json',dict(completed=True,result_sha256=sha(out/'RESULT.json'),episodes=200,
        new_training_updates=0,optimizer_instances=0,CUDA_initialized=False,ended_utc=utc()))
    return result
