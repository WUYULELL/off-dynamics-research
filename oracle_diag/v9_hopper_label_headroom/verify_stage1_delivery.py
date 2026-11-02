"""Offline post-completion verification: no new simulator or optimization steps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import datetime
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261030/stage1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open('\\\\?\\'+str(Path(p).resolve()),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    done=read(OUT/'STAGE1_COMPLETED.json');cfg=read(OUT/'config.json')
    assert done['passed'] and done['status']=='AWAITING_STAGE2_CONFIRMATION'
    assert not done['stage2_started'] and not done['stage3_started'] and not done['stage4_started']
    assert done['training_updates']==0
    assert not list(OUT.glob('STAGE1_FAILURE*')) and not (OUT/'RECOVERY_FAILED.json').exists()
    assert sha(done['report'])==done['report_sha256']
    assert sha(OUT/'config.json')==done['config_sha256']
    assert sha(OUT/'INDEPENDENT_ACCEPTANCE.json')==done['acceptance_sha256']
    assert sha(OUT/'CACHE_GENERATED.json')==done['generated_receipt_sha256']
    commit=read(OUT/'PRE_RESULT_COMMITMENT.json')
    assert commit['before_any_new_simulator_steps'] and commit['config_sha256']==sha(OUT/'config.json')
    assert commit['code_frozen_sha256']==sha(OUT/'code_frozen.json')
    for name in ('code_frozen.json','dependency_hashes.json','protected_manifest.json'):
        for p,h in read(OUT/name)['sha256'].items():assert sha(p)==h,p
    g=read(OUT/'CACHE_GENERATED.json');a=read(OUT/'INDEPENDENT_ACCEPTANCE.json')
    assert g['passed'] and a['passed'] and a['sample_n']==10000
    assert not g['final_canonical_recovery_failures'] and not a['failures']
    assert a['state_recovery_failed_count']==a['source_terminal_disagreements']==a['target_terminal_disagreements']==0
    assert a['independently_recovered_from_raw'] and not a['generator_saved_qpos_qvel_used_as_recovery_input']
    ids=np.load(OUT/'cache/raw_indices.npy');sample=np.load(OUT/'acceptance/sample_buffer_indices.npy')
    with h5py.File(cfg['source'],'r') as f:
        assert np.array_equal(ids,np.flatnonzero(~np.asarray(f['timeouts'][:-1],dtype=bool)))
    assert len(ids)==999998 and np.array_equal(sample,np.random.RandomState(20261030).choice(999998,10000,replace=False))
    assert sha(OUT/'acceptance/sample_raw_indices.npy')==commit['sample_sha256']
    arrays={k:np.load(v['path'],mmap_mode='r') for k,v in g['arrays'].items()}
    for k,v in g['arrays'].items():
        assert sha(v['path'])==v['sha256'] and np.isfinite(arrays[k]).all()
    assert np.array_equal(arrays['not_done'],1.-arrays['terminal'])
    with np.load(OUT/'acceptance/independent_replay.npz') as z:
        assert np.array_equal(z['raw_indices'],ids[sample])
        comparisons={'target_cache_next_state':np.abs(z['target_next']-arrays['next_state'][sample]).max(axis=1),
            'target_cache_reward':np.abs(z['target_reward']-arrays['reward'][sample].ravel())}
        assert np.array_equal(z['target_terminal'],arrays['terminal'][sample].ravel())
        for name,values in comparisons.items():
            assert np.array_equal(values,z[name])
        for name,s in a['errors'].items():
            values=z[name]
            measured=dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p99=float(np.percentile(values,99)),maximum=float(np.max(values)))
            for k,v in measured.items():assert abs(s[k]-v)<1e-14
    protected=read(OUT/'protected_manifest.json')['count']
    outputs={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    receipt=dict(passed=True,stage=1,CPU_only=True,new_training_updates=0,new_verification_simulator_steps=0,
        generated_canonical_rows=999998,independent_replay_sample=10000,raw_sample_indices_verified=True,
        error_summaries_recomputed=True,all_cache_arrays_finite_and_hash_verified=True,protected_old_files_verified=protected,
        frozen_config_code_and_dependencies_verified=True,pre_result_commitment_verified=True,
        no_next_stage_started=True,report=done['report'],report_sha256=sha(done['report']),
        verifier_sha256=sha(__file__),output_sha256=outputs,
        verified_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    with open(OUT/'DELIVERY_VERIFIED_STAGE1.json','x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in receipt.items() if k!='output_sha256'}))
if __name__=='__main__':main()
