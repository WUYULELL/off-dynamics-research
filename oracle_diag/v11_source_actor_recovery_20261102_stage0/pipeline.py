"""Revised authorization: execute exactly three Stage0 trials, verify, stop and exit."""
from common import *
import traceback

def status(stage, **kw):
    r.write(OUT/'queue_status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),
        scope='V11_revision_Stage0_only_then_stop',**kw),replace=True)

def main():
    guard()
    authorization=r.read(OUT/'RECOVERY_AUTHORIZED.json')
    assert authorization['authorized'] and authorization['stop_after_Stage0']
    assert authorization['recovery_plan_sha256']==r.sha(OUT/'RECOVERY_PLAN.json')
    assert not competing_processes(),'Another experimental queue is active; do not terminate it'
    assert r.read(WALKER/'WALKER_ONLY_COMPLETED.json')['passed']
    assert r.read(WALKER/'DELIVERY_VERIFIED.json')['passed']
    r.write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time()))
    status('CPU_PROTECTION');protect('before_stage0',create=True)
    reuse=r.read(OUT/'CPU_ACCEPTANCE_REUSE.json')
    assert reuse['passed'] and reuse['actor_py_byte_identical'] and reuse['learner_config_identical']
    assert r.sha(reuse['original_receipt'])==reuse['original_sha256']==r.sha(OUT/'CPU_ACCEPTANCE.json')
    assert r.read(OUT/'CPU_ACCEPTANCE.json')['passed']
    r.write(OUT/'PRE_GPU_COMMITMENT.json',dict(config_sha256=r.sha(OUT/'config.json'),
        frozen_sha256=r.sha(OUT/'FROZEN.json'),protected_manifest_sha256=r.sha(OUT/'protected_manifest.json'),
        CPU_reuse_sha256=r.sha(OUT/'CPU_ACCEPTANCE_REUSE.json'),trial_order=[x[0] for x in TRIALS],
        new_learner_updates=50000,original_O1_reference_updates=10000,total_learner_updates=60000,
        automatic_Stage1_after_pass=False,stage1_authorized=False,stage2_authorized=False,time=time.time()))
    for trial_id,_,_,_ in TRIALS:
        guard();assert not (OUT/'runs'/trial_id).exists(),'Never retry or overwrite existing output'
        assert not competing_processes(),'Another training queue exists; do not terminate it'
        status('GPU_TRIAL',trial=trial_id)
        child('trial.py',['--trial',trial_id],trial_id,False)
        status('CPU_AUDIT',trial=trial_id)
        child('audit.py',['--trial',trial_id],trial_id+'_audit',True)
        assert r.read(OUT/'runs'/trial_id/'audit.json')['passed']
    status('CPU_REPORT');child('report.py',[],'Stage0_report',True)
    status('CPU_DELIVERY_VERIFICATION');child('verify.py',[],'Stage0_verify',True)
    assert r.read(OUT/'DELIVERY_VERIFIED.json')['passed']
    protect('stage_end')
    r.write(OUT/'STAGE0_COMPLETED.json',dict(passed=True,status='AWAITING_USER_STAGE_ORDER',
        stage1_started=False,stage2_started=False,stage1_authorized=False,stage2_authorized=False,
        report_sha256=r.sha(OUT/'STAGE0_REPORT_ZH.md'),delivery_sha256=r.sha(OUT/'DELIVERY_VERIFIED.json'),time=time.time()))
    status('AWAITING_USER_STAGE_ORDER',passed=True,stage1_started=False,stage2_started=False)

if __name__=='__main__':
    try:main()
    except BaseException:
        r.write(OUT/('queue_failure_%d.json'%time.time()),dict(time=time.time(),traceback=traceback.format_exc(),
            action='Stop and report; no automatic recovery, retry or retraining; preserve all evidence'))
        status('FAILED');raise
