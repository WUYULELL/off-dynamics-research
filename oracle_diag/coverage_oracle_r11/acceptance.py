"""CPU-only R11 delta review; do not rerun previously accepted training."""
from runtime import *
from decision import choose_temperature,choose_horizon
import ast
import runtime
import torch
from learner import initialize,Sampler

OLD=ROOT/'oracle_diag/results/coverage_oracle_20261016_r09'

def fixture(temp,steps,values):
    windows=[steps-40000,steps-30000,steps-20000,steps-10000,steps]
    return [dict(job=dict(seed=s,temp=temp,steps=steps),Last5=v,last5_steps=windows)
        for s,v in zip((101,102,103),values)]

def main():
    configure_cpu()
    assert not torch.cuda.is_initialized()
    assert read(OLD/'cpu_acceptance.json')['passed']
    for p,h in read(OLD/'code_frozen.json')['sha256'].items():assert sha(p)==h
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    copied=read(OUT/'science_copy_provenance.json')['copied_unchanged']
    assert all(sha(HERE/n)==x['from_sha256']==sha(x['from_path']) for n,x in copied.items())
    oldrun=OLD/'runs/tune_T_t1_h500000_s101'
    a=read(oldrun/'audit.json');done=read(oldrun/'completed.json')
    assert a['passed'] and a['raw_TensorBoard_independently_checked'] and a['finite_cpu_checked']
    assert sha(oldrun/'completed.json')==a['completed_sha256']
    assert sha(oldrun/'evaluations.json')==a['evaluations_sha256']
    assert sha(done['final_model'])==a['final_model_sha256']==done['final_model_sha256']
    assert model_config(1.,500000,101)==read(oldrun/'config.json')['learner']
    model=initialize(model_config(1.,500000,101),'cpu')
    assert model.digest()==read(oldrun/'initialization.json')['sha256']
    sample=Sampler(target_data(),101,'target_sample','cpu')
    sample.sample()
    assert sample.first==a['sampling']['target_first']
    tsmoke=OLD/'runs/smoke_T_t3_h5000_s7001'
    assert read(tsmoke/'audit.json')['passed']
    reuse=dict(T500_temp1_seed101=dict(audit=str(oldrun/'audit.json'),audit_sha256=sha(oldrun/'audit.json'),
        completed_sha256=sha(oldrun/'completed.json'),final_model_sha256=a['final_model_sha256'],
        rationale='same parameters, exact science code, initialization and sample seed; previously CPU-audited'),
        T_smoke=dict(audit=str(tsmoke/'audit.json'),audit_sha256=sha(tsmoke/'audit.json'),passed=True))
    write(OUT/'reuse_receipts.json',reuse)
    write(OUT/'T_smoke_audited.json',dict(passed=True,reused=True,receipt=reuse['T_smoke']))
    checks=[]
    short=fixture(3.,500000,[30.,31.,32.])
    for delta,expected in (([1.,2.,3.],1000000),([-1.,1.,1.],500000),([0.,0.,0.],500000),([-1.,-1.,-1.],500000)):
        long=fixture(3.,1000000,[x['Last5']+d for x,d in zip(short,delta)])
        choice=choose_horizon(short,long,3.)
        assert choice['steps']==expected
        checks.append(dict(differences=delta,selected=expected))
    candidates=[dict(temp=t,mean=31.,runs=fixture(t,500000,[30.,31.,32.])) for t in (1.,3.,10.)]
    assert choose_temperature(candidates)['temp']==3.
    wrong=fixture(10.,1000000,[35.,36.,37.])
    try:choose_horizon(short,wrong,3.)
    except AssertionError:checks.append('reject 1M additional temp search')
    else:raise AssertionError('Wrong 1M temp accepted')
    original_check,original_read=runtime.check_frozen,runtime.read
    runtime.check_frozen=lambda:None
    def fake(p):
        if Path(p).name in ('TEMP_500K_SELECTED.json','T_selected.json'):return dict(temp=3.,steps=500000)
        return dict(passed=True)
    runtime.read=fake
    base=dict(task='walker2d-morph',phase='tune',arm='T',temp=3.,seed=101,steps=1000000,
        out=str(OUT/'runs/never_created'))
    try:
        runtime.check_job(base)
        for delta in (dict(phase='screen',arm='U',steps=500000),dict(phase='screen',arm='G',steps=500000,ratio=.2),
            dict(phase='formal'),dict(task='hopper-morph'),dict(temp=1.),dict(seed=201)):
            try:runtime.check_job(dict(base,**delta))
            except AssertionError:checks.append(dict(rejected=delta))
            else:raise AssertionError('Unauthorized configuration accepted')
    finally:runtime.check_frozen,runtime.read=original_check,original_read
    protected=dict(read(OLD/'protected_manifest_final.json')['sha256'])
    for base_path in (ROOT/'oracle_diag/coverage_oracle_r09',OLD,
                      ROOT/'outputs/research_idea_protocol_20261016_r10'):
        for path in base_path.rglob('*'):
            if path.is_file() and '__pycache__' not in str(path):protected[str(path)]=sha(path)
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,
        extends_R09_3140=True,includes_completed_R09_outputs=True))
    assert not torch.cuda.is_initialized()
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        reused_mathematics_acceptance=str(OLD/'cpu_acceptance.json'),science_copy_hashes_exact=True,
        initialization_and_first_target_batch_exact=True,tests=checks,
        reuse_verified_without_retraining=True,protected_count=len(protected),
        H_windows_are_different_training_stages=True,no_U_G_validation_dispatch=True))
    write(OUT/'code_frozen.json',dict(sha256=code_hashes(),cpu_acceptance_sha256=sha(OUT/'cpu_acceptance.json')))
    print('R11 CPU delta acceptance passed; protected',len(protected),flush=True)

if __name__=='__main__':main()
