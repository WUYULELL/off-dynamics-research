"""CPU-only recovery acceptance; no real-data optimization or environment steps."""
from runtime import *
import ast
import subprocess
import numpy as np
import torch
from learner import initialize
from metrics import group,paired,sd_ratio,best_weight

def main():
    configure_cpu();check_frozen()
    old_code=ROOT/'oracle_diag/coverage_raw_r14';old=ROOT/'oracle_diag/results/coverage_raw_20261026_r14'
    for n in ('learner.py','train.py','audit.py','metrics.py','analyze.py'): assert sha(HERE/n)==sha(old_code/n)
    def functions(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    assert functions(HERE/'runtime.py')==functions(old_code/'runtime.py')
    old_cpu=read(old/'cpu_acceptance.json');old_commit=read(old/'CPU_ACCEPTANCE_COMMITMENT.json')
    assert old_cpu['passed'] and sha(old/'cpu_acceptance.json')==old_commit['cpu_acceptance_sha256']
    assert sha(OUT/'protocol.json')==sha(old/'protocol.json')==old_commit['protocol_sha256']
    jobs=read(OUT/'training_plan.json')['jobs']; assert len(jobs)==7
    receipts=read(OUT/'reuse_receipts.json')['runs'];assert len(receipts)==43
    for r in receipts:
        assert sha(r['audit'])==r['audit_sha256'] and read(r['audit'])['passed']
        for p,h in r['files_sha256'].items():assert sha(p)==h
    raw=[r for r in receipts if r['label']=='raw'];assert len(raw)==38
    for r in raw:
        if not r['w']:continue
        base=next(b for b in raw if b['task']==r['task'] and b['seed']==r['seed'] and not b['w'])
        assert read(Path(r['out'])/'initialization.json')==read(Path(base['out'])/'initialization.json')
        s=read(Path(r['out'])/'sampling.json');b=read(Path(base['out'])/'sampling.json')
        assert s['target_stream_sha256']==b['target_stream_sha256']
        other=next((x for x in raw if x['task']==r['task'] and x['seed']==r['seed'] and x['w']>0 and x['w']!=r['w']),None)
        if other:assert s['source_stream_sha256']==read(Path(other['out'])/'sampling.json')['source_stream_sha256']
    streams={}
    for j in jobs:
        validate_spec(j)
        oldj=next(x for x in read(old/'training_plan.json')['jobs'] if x['id']==j['id'])
        assert {k:v for k,v in j.items() if k!='out'}=={k:v for k,v in oldj.items() if k!='out'}
        initial=initialize(model_config(10,500000,j['seed'],j['task'],1.),'cpu').digest()
        base=next(r for r in raw if r['task']==j['task'] and r['seed']==j['seed'] and not r['w'])
        assert initial==read(Path(base['out'])/'initialization.json')['sha256']
        target=stream_hash(j['seed'],'target_sample',5000,500000,j['task'])
        source=stream_hash(j['seed'],'source_sample',999998,500000,j['task'])
        assert target==read(Path(base['out'])/'sampling.json')['target_stream_sha256']
        quarter=next(r for r in raw if r['task']==j['task'] and r['seed']==j['seed'] and r['w']==.25)
        assert source==read(Path(quarter['out'])/'sampling.json')['source_stream_sha256']
        streams[j['id']]=dict(initialization=initial,target_stream_sha256=target,source_stream_sha256=source)
    interrupted=old/'runs/hopp_w1_s204';pr=read(interrupted/'progress.json')
    assert streams['hopp_w1_s204']['initialization']==read(interrupted/'initialization.json')['sha256']
    for purpose in ('target_sample','source_sample'):
        key='target_stream_sha256' if purpose=='target_sample' else 'source_stream_sha256'
        assert stream_hash(204,purpose,5000 if purpose=='target_sample' else 999998,3000,'hopper-morph')==pr[key]
    arrays,ids=source_data('hopper-morph');target_data('hopper-morph')
    alignment=read(interrupted/'source_alignment.json')
    assert alignment['oracle_loaded'] is False and alignment['source_labels']=='original_dataset'
    assert alignment['canonical_raw_ids_sha256']==hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    assert alignment['array_sha256']=={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in zip(('state','action','next_state','reward','not_done'),arrays)}
    probes=[]
    for label,path in [('original',old_code),('recovery',HERE)]:
        dest=OUT/('CPU_PROBE_'+label+'.json')
        environ=os.environ.copy();environ['CUDA_VISIBLE_DEVICES']='-1'
        subprocess.run([str(PYTHON),'-B',str(HERE/'cpu_probe.py'),'--code',str(path),'--out',str(dest)],cwd=str(ROOT),env=environ,check=True)
        probes.append(read(dest))
    assert probes[0]==probes[1]
    assert group([0,1,2,3,4],list(range(5)))['lower_quartile']==1.
    assert paired([1,2,3],[0,0,0],[1,2,3])['positive']==3
    assert sd_ratio([1,2,3,4,5],[1,2,3,4,5],'recovery-acceptance',100)['bootstrap_percentile_interval']==[1.,1.]
    assert best_weight({str(w):{'mean':1} for w in (0.,.25,1.)})['selected_w']==0.
    assert not torch.cuda.is_initialized() and not (OUT/'runs').exists()
    write(OUT/'RESTORE_ACCEPTANCE_EXPECTATIONS.json',dict(fresh_attempts=True,full_streams=streams,
        interrupted_attempt_3000_progress=pr,claim_GPU_checkpoint_restore=False,claim_GPU_loss_overlap=False))
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        synthetic_CPU_optimizer_updates=4,real_data_optimizer_updates=0,simulator_steps=0,
        seven_job_configs_unchanged=True,all43_reuse_receipts_verified=True,
        original_and_recovery_CPU_two_step_complete_state_bitwise_equal=True,
        same_seed_T_initialization_and_full_target_streams_match=True,quarter_source_full_streams_match=True,
        partial3000_double_streams_match=True,source_alignment_and_original_labels_match=True,
        original_R14_CPU_acceptance_sha256=sha(old/'cpu_acceptance.json'),time=time.time()))
    write(OUT/'CPU_ACCEPTANCE_COMMITMENT.json',dict(cpu_acceptance_sha256=sha(OUT/'cpu_acceptance.json'),
        code_frozen_sha256=sha(OUT/'code_frozen.json'),protocol_sha256=sha(OUT/'protocol.json'),time=time.time()))
    protect('CPU_acceptance')
    print(json.dumps(dict(passed=True,CPU_only=True,synthetic_updates=4,no_real_data_updates=True,no_environment_steps=True,no_CUDA_initialization=True)))

if __name__=='__main__':main()
