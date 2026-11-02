"""CPU-only delivery audit. Re-read raw events; preserve all training artifacts."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from support_v3 import *
import argparse, math, time, shutil, runpy
import numpy as np
import torch
import h5py
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from audit_runs_v3_attempt3 import audit


def moments(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(values=x.tolist(), n=len(x), mean=float(x.mean()),
                sample_sd=float(x.std(ddof=1)), se=float(x.std(ddof=1)/np.sqrt(len(x))))


def finite(obj):
    if torch.is_tensor(obj):
        assert torch.isfinite(obj).all()
    elif isinstance(obj, dict):
        for x in obj.values(): finite(x)
    elif isinstance(obj, (list, tuple)):
        for x in obj: finite(x)
    elif isinstance(obj, float):
        assert math.isfinite(obj)


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); dest = a.out.resolve()
    assert OUT in dest.parents
    fresh(dest); torch.set_num_threads(2)
    for f in [HERE/'REPORT_v3.md', OUT/'active_execution.json', OUT/'status.json']:
        shutil.copy2(winpath(f), winpath(dest/(f.name+'.before')))
    active = read(OUT/'active_execution.json')
    assert read(active['queue_status'])['state'] == 'completed'
    plan = read(active['training_plan']); original = read(OUT/'training_plan.json')
    assert len(plan['jobs']) == 18
    changed = []
    for old, new in zip(original['jobs'], plan['jobs']):
        if old != new:
            assert new['kind']=='formal' and new['method']=='MIX-ORACLE-BACKUP' and new['seed']==3
            expected = dict(old, out=str(OUT/'b2/o3'), command=old['command'][:-1]+[str(OUT/'b2/o3')])
            assert new == expected; changed.append(new['out'])
    assert changed == [str(OUT/'b2/o3')]
    assert not (OUT/'b1/o3/completed.json').exists()
    interrupted = read(OUT/'recovery2/interrupted_evidence.json')['files']
    for f, meta in interrupted.items(): assert sha(f)==meta['sha256'], f
    repair = read(OUT/'recovery2/repair.json')
    for key in ['original_queue', 'new_queue', 'original_audit', 'new_audit']:
        assert sha(repair[key]['path']) == repair[key]['sha256']
    assert sha(active['training_plan']) == repair['new_plan_sha256']
    assert sha(OUT/'training_plan.json') == repair['original_plan_sha256']
    repair1=read(OUT/'recovery1/repair.json')
    assert sha(repair1['preserved_error_log'])==repair1['preserved_error_sha256']
    stage_a=read(OUT/'stage_a_completed.json')
    for f,h in stage_a['code_sha256'].items(): assert sha(f)==h, f
    protocol_sha=sha(OUT/'protocol.json')
    assert read(OUT/'gpu_authorization.json')['protocol_sha256']==protocol_sha
    print('Protocol, frozen code and recovery evidence verified', flush=True)

    # Re-run the actual designated audit without replacing its immutable receipt.
    for kind in ['smoke','formal']:
        recomputed=audit(kind); recorded=read(OUT/(kind+'_audited.json'))
        for row in recomputed['runs']:
            row['scores']={str(k):v for k,v in row['scores'].items()}
        assert dict(loader='explicit_next_v3', **recomputed)==recorded
    print('Actual smoke/formal audits reproduced (18 runs, 126 checkpoint files)', flush=True)

    # Independent statistics and exact tag selection, not the audit's paired helper.
    norm=runpy.run_path(str(ROOT/'reproductions/OTDF/info.py'))
    lo=float(norm['REF_MIN_SCORE']['walker2d-morph']); hi=float(norm['REF_MAX_SCORE']['walker2d-morph'])
    assert (lo,hi)==(10.079455055289959,4398.43)
    xml_sha=sha(ROOT/'reproductions/OTDF/envs/assets/walker_morph.xml')
    formal=read(OUT/'formal_audited.json')
    official={(r['method'],r['seed']):r for r in formal['runs']}
    rows=[]; values_checked=[]; common_configs=[]; evaluation_count=0; max_norm_error=0.
    for job in plan['jobs']:
        folder=Path(job['out']); cfg=read(folder/'effective_config.json'); c=cfg['config']
        assert cfg['protocol_sha256']==protocol_sha and cfg['oracle_receipt_sha256']==stage_a['oracle_receipt_sha256']
        assert cfg['normalization']['min']==lo and cfg['normalization']['max']==hi
        assert sha(cfg['normalization']['file'])==cfg['normalization']['sha256']
        assert c['device']=='cuda' and c['source_budget']==.2 and c['bellman_eta']==1. and c['gamma']==.99
        assert c['batch_size']==128 and c['eval_episode']==10 and c['seed']==job['seed']
        assert c['max_step']==job['steps'] and c['target_only']==(job['method']=='TARGET')
        if job['kind']=='formal':
            assert c['eval_freq']==10000
            common_configs.append({k:v for k,v in c.items() if k not in ['seed','target_only']})
        first=read(folder/'first_sample.json'); rng=np.random.RandomState(job['seed'])
        src=[] if job['method']=='TARGET' else rng.randint(0,999322,64).tolist()
        tar=rng.randint(0,5000,128 if job['method']=='TARGET' else 64).tolist()
        assert src==first['source_ids'] and tar==first['target_ids']
        snapshots=sorted((folder/'values').glob('V_*.pt'))
        expected_steps=[5000] if job['kind']=='smoke' else list(range(100000,500001,100000))
        assert sorted(int(x.stem.split('_')[1]) for x in snapshots)==expected_steps
        for f in snapshots:
            finite(torch.load(winpath(f),map_location='cpu'))
            values_checked.append(dict(path=str(f),sha256=sha(f)))
        events=list(folder.rglob('events.out.tfevents.*')); assert len(events)==1
        scores={}; raw={}; weight_events=[]; scalar_count=0
        for e in LegacyEventFileLoader(winpath(events[0])).Load():
            for v in e.summary.value:
                if v.WhichOneof('value')!='simple_value': continue
                assert math.isfinite(v.simple_value), (job['method'],job['seed'],v.tag,e.step)
                scalar_count+=1
                if v.tag=='test/target_normalized_score':
                    assert e.step not in scores; scores[e.step]=float(v.simple_value)
                elif v.tag=='test/target_return':
                    assert e.step not in raw; raw[e.step]=float(v.simple_value)
                elif v.tag=='train/source_weight_mean': weight_events.append(float(v.simple_value))
                elif v.tag=='train/source_weight_ess_ratio': assert abs(v.simple_value-1.)<1e-6
        if job['method']!='TARGET':
            assert len(weight_events)==(2 if job['kind']=='smoke' else 101)
            assert all(abs(x-.2)<1e-6 for x in weight_events)
        ev=read(folder/'evaluation_verified.json')
        for e in ev['events']:
            assert e['confirmed_target_env_object'] and e['env_class']=='gym.envs.mujoco.walker2d_v3'
            assert e['xml_sha256']==xml_sha and e['lookup_key']=='walker2d-morph-medium'
            assert abs(e['raw_return']-raw[e['step']])<.001
            err=abs((raw[e['step']]-lo)/(hi-lo)*100-scores[e['step']])
            max_norm_error=max(max_norm_error,err); assert err<2e-5
            evaluation_count+=1
        if job['kind']=='formal':
            steps=[460000,470000,480000,490000,500000]
            last=[scores[s] for s in steps]; last_raw=[raw[s] for s in steps]
            mean=float(np.asarray(last,dtype=np.float64).mean())
            assert abs(mean-official[job['method'],job['seed']]['last5_mean'])<1e-12
            progress=read(folder/'progress.json'); done=read(folder/'completed.json')
            rows.append(dict(method=job['method'],seed=job['seed'],out=str(folder),tb_file=str(events[0]),
                tb_sha256=sha(events[0]),last5=last,last5_mean=mean,last5_raw_return_mean=float(np.mean(last_raw)),
                progress_elapsed=progress['elapsed'],completed_at=done['ended'],gpu_peak_bytes=progress['gpu_peak_bytes'],
                scalar_events_finite=scalar_count))
    assert all(x==common_configs[0] for x in common_configs)
    vals={(r['method'],r['seed']):r['last5_mean'] for r in rows}
    arms={m:moments([vals[m,s] for s in range(5)]) for m in ['TARGET','MIX-CONST','MIX-ORACLE-BACKUP']}
    comparisons={}
    for left,right in [('MIX-ORACLE-BACKUP','TARGET'),('MIX-CONST','TARGET'),('MIX-ORACLE-BACKUP','MIX-CONST')]:
        k=left+' minus '+right; v=[vals[left,s]-vals[right,s] for s in range(5)]; rec=moments(v)
        rec.update(positive=int(np.count_nonzero(np.asarray(v)>0)),two_se=2*rec['se'])
        rec['gate_passed']=bool(rec['mean']>rec['two_se'] and rec['positive']>=4)
        for key in ['mean','sample_sd','se']:
            assert abs(rec[key]-formal['comparisons'][k][key])<1e-12
        assert rec['positive']==formal['comparisons'][k]['positive']; comparisons[k]=rec
    print('Independent raw-event statistics, shifted evaluation, config and V snapshots verified', flush=True)

    cache=read(stage_a['oracle_receipt']); assert sha(stage_a['oracle_receipt'])==stage_a['oracle_receipt_sha256']
    sample=read(OUT/'stage_a2/sample_completed.json')
    source=sample['sources']['walker2d']['dataset']; assert sha(source)==cache['source_dataset_sha256']
    assert sha(cache['raw_indices_path'])==cache['raw_indices_sha256']
    ids=np.load(cache['raw_indices_path'])
    with h5py.File(source,'r') as f:
        expected_ids=np.flatnonzero(~f['timeouts'][:-1].astype(bool)); assert np.array_equal(ids,expected_ids)
        for key in ['observations','actions','next_observations','rewards','terminals']:
            arr=f['observations' if key=='next_observations' else key][:]
            arr=arr[ids+1 if key=='next_observations' else ids].astype(bool if key=='terminals' else np.float32)
            assert array_hash(arr)==cache['source_array_sha256'][key], key
        normal_term=f['terminals'][:][ids].astype(bool)
    arrays={}
    for k,meta in cache['arrays'].items():
        assert sha(meta['path'])==meta['sha256']
        arr=np.load(meta['path'],mmap_mode='r'); assert len(arr)==999322 and np.isfinite(arr).all()
        arrays[k]=arr
    assert arrays['next_state'].shape==(999322,17)
    assert arrays['reward'].shape==arrays['not_done'].shape==(999322,1)
    assert np.isin(arrays['not_done'],[0,1]).all()
    oracle_term=arrays['not_done'].ravel()==0
    assert int(oracle_term.sum())==506 and int(normal_term.sum())==513 and int((normal_term!=oracle_term).sum())==17
    evidence=OUT/'source_oracle1/recovery_evidence.npz'
    assert sha(evidence)==cache['recovery_evidence_sha256']
    with np.load(str(evidence)) as z:
        assert not (z['recovery_status'][ids]==2).any()
        assert (z['state_error'][ids]<=1e-3).all() and (z['reward_error'][ids]<=1e-4).all()
        assert not z['done_disagreement'][ids].any()
    # Use prior replay as evidence, without claiming this read is a new replay.
    target_meta=read(OUT/'stage_a1/walker2d_morph_acceptance.json')
    target_path=ROOT/'reproductions/OTDF/datasets/walker2d_morph_medium.hdf5'
    assert sha(target_path)==target_meta['dataset_sha256'] and xml_sha==target_meta['xml_sha256']
    assert sha(target_meta['previous_replay'])==target_meta['previous_replay_sha256']
    with np.load(target_meta['previous_replay']) as z,h5py.File(str(target_path),'r') as f:
        segments=int(1+np.count_nonzero((z['boundary']|f['terminals'][:].astype(bool))[:-1]))
        assert segments==8
        for key in ['observations','actions','next_observations','rewards','terminals']:
            assert len(f[key])==5000 and np.isfinite(f[key][:]).all()
    for job in plan['jobs']:
        data=read(Path(job['out'])/'data_verified.json')
        assert data['source_dataset_sha256']==cache['source_dataset_sha256']
        assert data['target_dataset_sha256']==target_meta['dataset_sha256']
    print('Full source/cache row alignment and data hashes verified; checking protected files',flush=True)
    protected_check()
    receipt=dict(passed=True,device='cpu',cuda_initialized=torch.cuda.is_initialized(),reviewed_at=time.time(),
        protocol_sha256=protocol_sha,actual_plan=str(active['training_plan']),formal_runs=15,smoke_runs=3,
        actual_audit_reproduced=True,independent_stats='NumPy float64 mean and sample std ddof=1 from exact raw event tags',
        last5_steps=[460000,470000,480000,490000,500000],runs=rows,arms=arms,comparisons=comparisons,
        evaluation_events_verified=evaluation_count,max_normalization_float32_error=max_norm_error,
        same_configs_except_seed_and_target_only=True,initialization_and_first_batch_verified=True,
        model_optimizer_checkpoints_finite=126,value_snapshots_finite=len(values_checked),value_snapshots=values_checked,
        source_cache_rows_verified=len(ids),target_rows=5000,target_segments=segments,
        source_oracle_terminal_changes=17,protected_files_unchanged=read(OUT/'protected_manifest.json')['count'],
        frozen_v3_code_unchanged=len(stage_a['code_sha256']),interrupted_evidence_unchanged=len(interrupted),
        source_replay_reused_not_rerun=True,stage_c_executed=False,
        sum_formal_progress_hours=sum(r['progress_elapsed'] for r in rows)/3600,
        formal_wall_hours=(max(r['completed_at'] for r in rows)-active['formal_started'])/3600,
        interrupted_steps=40000,review_script_sha256=sha(__file__))
    assert not receipt['cuda_initialized']
    write(dest/'review.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['passed','arms','comparisons','protected_files_unchanged','value_snapshots_finite','sum_formal_progress_hours','formal_wall_hours']},indent=2),flush=True)


if __name__=='__main__':main()
