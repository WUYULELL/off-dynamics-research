"""Wait for verified Walker; Stage0 acceptance then authorized Hopper Stage1; stop."""
from common import *
import traceback

def status(stage,**kw):
    r.write(OUT/'queue_status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),
        scope='V11_Stage0_then_Hopper_Stage1_then_stop',**kw),replace=True)

def main():
    guard()
    recovery_auth=r.read(OUT/'RECOVERY_AUTHORIZED.json')
    assert recovery_auth['authorized'] and recovery_auth['recovery_plan_sha256']==r.sha(OUT/'RECOVERY_PLAN.json')
    r.write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time()))
    status('WAITING_WALKER_COMPLETION',GPU_initialized=False)
    while True:
        guard()
        assert not list(WALKER.glob('queue_failure_*.json')), 'Walker failed; preserve evidence, no retry'
        if (WALKER/'WALKER_ONLY_COMPLETED.json').exists() and (WALKER/'DELIVERY_VERIFIED.json').exists():
            assert r.read(WALKER/'WALKER_ONLY_COMPLETED.json')['passed']
            assert r.read(WALKER/'DELIVERY_VERIFIED.json')['passed']
            competitors=competing_processes()
            if not competitors: break
        time.sleep(30)
    r.write(OUT/'WALKER_GATE_VERIFIED.json',dict(passed=True,completed_sha256=r.sha(WALKER/'WALKER_ONLY_COMPLETED.json'),
        delivery_verified_sha256=r.sha(WALKER/'DELIVERY_VERIFIED.json'),no_competing_training_queue=True,time=time.time()))
    status('CPU_PROTECTION');protect('before_stage0',create=True)
    status('CPU_ACCEPTANCE_REUSE')
    reuse=r.read(OUT/'CPU_ACCEPTANCE_REUSE.json')
    assert reuse['passed'] and r.sha(reuse['original_receipt'])==reuse['original_sha256']
    assert r.sha(OUT/'CPU_ACCEPTANCE.json')==reuse['original_sha256']
    assert r.read(OUT/'CPU_ACCEPTANCE.json')['passed']
    r.write(OUT/'CPU_ACCEPTANCE_COMMITMENT.json',dict(cpu_acceptance_sha256=r.sha(OUT/'CPU_ACCEPTANCE.json'),time=time.time()))
    r.write(OUT/'PRE_GPU_COMMITMENT.json',dict(config_sha256=r.sha(OUT/'config.json'),frozen_sha256=r.sha(OUT/'FROZEN.json'),
        CPU_commitment_sha256=r.sha(OUT/'CPU_ACCEPTANCE_COMMITMENT.json'),
        protected_manifest_sha256=r.sha(OUT/'protected_manifest.json'),trial_order=[x[0] for x in TRIALS],
        new_learner_updates=50000,original_O1_reference_updates=10000,total_learner_updates=60000,
        formal_stage1_started=False,automatic_Stage1_after_pass=True,time=time.time()))
    for trial_id,_,_,_ in TRIALS:
        guard();assert not (OUT/'runs'/trial_id).exists(),'Existing output: never overwrite or auto retry'
        assert not competing_processes(),'Another training queue exists; do not terminate it'
        status('GPU_TRIAL',trial=trial_id);child('trial.py',['--trial',trial_id],trial_id,False)
        status('CPU_AUDIT',trial=trial_id);child('audit.py',['--trial',trial_id],trial_id+'_audit',True)
        assert r.read(OUT/'runs'/trial_id/'audit.json')['passed']
    status('CPU_REPORT');child('report.py',[],'Stage0_report',True)
    status('CPU_DELIVERY_VERIFICATION');child('verify.py',[],'Stage0_verify',True)
    assert r.read(OUT/'DELIVERY_VERIFIED.json')['passed'];protect('stage_end')
    r.write(OUT/'STAGE0_COMPLETED.json',dict(passed=True,status='PROCEED_TO_AUTHORIZED_STAGE1',
        stage1_started=False,stage2_started=False,stage4_started=False,T_retuned=False,
        delivery_sha256=r.sha(OUT/'DELIVERY_VERIFIED.json'),report_sha256=r.sha(OUT/'STAGE0_REPORT_ZH.md'),time=time.time()))
    r.write(STAGE1/'PRE_GPU_COMMITMENT.json',dict(stage0_verified_sha256=r.sha(OUT/'DELIVERY_VERIFIED.json'),
        budget_sha256=r.sha(OUT/'MEASURED_BUDGET.json'),training_plan_sha256=r.sha(STAGE1/'training_plan.json'),
        config_sha256=r.sha(OUT/'config.json'),new_jobs=20,updates=10000000,stage2_authorized=False,time=time.time()))
    for index,job in enumerate(formal_jobs()):
        guard();assert not Path(job['out']).exists(),'No automatic retry, resume or overwrite'
        assert not competing_processes(),'Another experimental queue is active'
        status('STAGE1_TRAIN',job=job['id'],audited=index,total=20)
        child('trial.py',['--trial',job['id'],'--formal'],job['id']+'_formal',False)
        status('STAGE1_CPU_AUDIT',job=job['id'],audited=index,total=20)
        child('audit.py',['--trial',job['id'],'--formal'],job['id']+'_formal_audit',True)
        assert r.read(Path(job['out'])/'audit.json')['passed']
    status('STAGE1_CPU_ANALYSIS');child('stage1_report.py',[],'Stage1_report',True)
    status('STAGE1_CPU_DELIVERY_VERIFICATION');child('stage1_verify.py',[],'Stage1_verify',True)
    assert r.read(STAGE1/'DELIVERY_VERIFIED.json')['passed'];protect('stage1_end')
    r.write(STAGE1/'STAGE1_COMPLETED.json',dict(passed=True,status='AWAITING_USER_DECISION',new_runs=20,
        stage2_started=False,V9_stage4_started=False,T_retuned=False,report_sha256=r.sha(STAGE1/'STAGE1_REPORT_ZH.md'),time=time.time()))
    status('AWAITING_USER_DECISION',passed=True,Stage1_audited=20,stage2_started=False)

if __name__=='__main__':
    try: main()
    except BaseException:
        r.write(OUT/('queue_failure_%d.json'%time.time()),dict(time=time.time(),traceback=traceback.format_exc(),
            action='CPU diagnosis only; no auto retry/retraining; preserve all files'))
        status('FAILED');raise
