"""Stage2 prelaunch CPU acceptance: no real-data optimizer updates or env steps."""
from runtime import *
import ast
import subprocess
import statistics
import numpy as np
import torch
from learner import initialize
from metrics import group,paired,decision

def main():
    configure_cpu();check_frozen()
    old=ROOT/'oracle_diag/coverage_raw_r14_finish1'
    assert sha(HERE/'learner.py')==sha(old/'learner.py')
    delta=read(OUT/'engineering_delta.json')
    assert delta['training_loop_AST_identical'] and delta['evaluate_AST_identical']
    base=read(OUT/'reuse_receipts.json')['runs'];assert len(base)==30
    for r in base:
        assert sha(r['audit'])==r['audit_sha256'] and read(r['audit'])['passed']
        for p,h in r['files_sha256'].items():assert sha(p)==h
    target=target_data('hopper-morph');arrays,ids=source_data('hopper-morph')
    assert len(ids)==999998 and all(len(a)==5000 for a in target)
    raw=next(r for r in base if r['seed']==201 and r['w']==1.)
    align=read(Path(raw['out'])/'source_alignment.json')
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==align['canonical_raw_ids_sha256']
    for key,a in zip(('state','action'),arrays[:2]):assert hashlib.sha256(a.tobytes()).hexdigest()==align['array_sha256'][key]
    receipt=read(CACHE_RECEIPT)
    for key,a in zip(('next_state','reward','not_done'),arrays[2:]):
        expected=np.load(winpath(receipt['cache_arrays'][key]['path']),mmap_mode='r')
        assert np.array_equal(a,expected) and a.dtype==np.float32
    full={}
    for s in range(201,211):
        t=next(r for r in base if r['seed']==s and r['w']==0.)
        initialization=initialize(model_config(10,500000,s,'hopper-morph',1.),'cpu').digest()
        assert initialization==read(Path(t['out'])/'initialization.json')['sha256']
        th=stream_hash(s,'target_sample',5000,500000,'hopper-morph')
        sh=stream_hash(s,'source_sample',999998,500000,'hopper-morph')
        for r in base:
            if r['seed']!=s:continue
            assert read(Path(r['out'])/'initialization.json')['sha256']==initialization
            sampling=read(Path(r['out'])/'sampling.json')
            assert sampling['target_stream_sha256']==th
            if r['w']>0:assert sampling['source_stream_sha256']==sh
        full[str(s)]=dict(initialization=initialization,target=th,source=sh)
    del arrays,target
    for w in (.25,1.):
        results=[]
        for tag,code in [('R14',old),('V9',HERE)]:
            p=OUT/('CPU_PROBE_%s_w%s.json'%(tag,str(w)))
            envp=os.environ.copy();envp['CUDA_VISIBLE_DEVICES']='-1'
            subprocess.run([str(PYTHON),'-B',str(HERE/'cpu_probe.py'),'--code',str(code),'--w',str(w),'--out',str(p)],cwd=str(ROOT),env=envp,check=True)
            results.append(read(p))
        assert results[0]==results[1]
    seeds=list(range(201,211));vals=list(range(10))
    g=group(vals,seeds);assert g['mean']==4.5 and g['Q1']==2.25 and abs(g['sample_SD']-statistics.stdev(vals))<1e-12
    p=paired(np.arange(10)+1,np.arange(10),seeds);assert p['mean']==1. and p['positive']==10 and p['SE']==0.
    def v(mean,se,pos):return dict(mean=mean,SE=se,positive=pos)
    fixtures=[(v(4,1,8),v(0,1,5),'A',False),(v(4,1,8),v(3,1,8),'A',True),
        (v(0,1,5),v(-3,1,2),'B',False),(v(3,1,6),v(1,1,6),'uncertain',False),
        (v(2,1,10),v(-2,1,0),'uncertain',False)]
    for L,E,label,gain in fixtures:
        d=decision(L,E);assert d['category']==label and d['corrected_source_gain']==gain
    assert not torch.cuda.is_initialized() and not (OUT/'runs').exists()
    write(OUT/'PAIRING_EXPECTATIONS.json',dict(task='hopper-morph',seeds=full,full_draws_per_stream=32000000))
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        synthetic_CPU_optimizer_updates=8,real_data_optimizer_updates=0,environment_steps=0,
        both_weights_R14_complete_state_bitwise_equal=True,learner_byte_identical=True,
        evaluation_and_training_loop_AST_identical=True,source_state_action_indices_exact_R14=True,
        labels_exact_accepted_stage1=True,same_seed_initialization_and_complete_target_source_streams_match_R14=True,
        all30_R14_reuse_receipts_verified=True,decision_strict_boundaries_and_gain_fixture_passed=True,time=time.time()))
    write(OUT/'CPU_ACCEPTANCE_COMMITMENT.json',dict(cpu_acceptance_sha256=sha(OUT/'cpu_acceptance.json'),
        protocol_sha256=sha(OUT/'protocol.json'),code_frozen_sha256=sha(OUT/'code_frozen.json'),time=time.time()))
    protect('CPU_acceptance')
    print(json.dumps(dict(passed=True,synthetic_CPU_updates=8,real_data_updates=0,environment_steps=0,CUDA_initialized=False)))

if __name__=='__main__':main()
