"""Final Stage0 CPU evidence/receipt verification before automatic Stage1."""
from common import *
import torch
from audit import diagnose,raw_scores

def main():
    guard();assert not torch.cuda.is_initialized()
    result=r.read(OUT/'STAGE0_RESULT.json');analysis=r.read(OUT/'ANALYSIS_COMPLETED.json')
    assert analysis['result_sha256']==r.sha(OUT/'STAGE0_RESULT.json')
    assert analysis['report_sha256']==r.sha(OUT/'STAGE0_REPORT_ZH.md')
    for p,h in result['input_sha256'].items():assert r.sha(p)==h,p
    for trial_id,variant,wa,steps in TRIALS:
        path=OUT/'runs'/trial_id
        assert r.read(path/'audit.json')['passed']
        assert diagnose(path,variant,steps,wa)==result['trials'][trial_id]['diagnostics']
        raw_scores(path,201,10000,steps,require_tb=False)
        d=torch.load(r.winpath(path/'model_final.pt'),map_location='cpu');assert full_finite(d)
        if wa==0.:
            other=torch.load(r.winpath(path/'original_O1_model_final.pt'),map_location='cpu')
            assert tree_digest({k:v for k,v in d.items() if k!='config'})==tree_digest({k:v for k,v in other.items() if k!='config'})
    receipt=protect('Stage0_delivery')
    assert result['total_GPU_learner_updates']==60000 and settings()['stage1_authorized']
    assert not torch.cuda.is_initialized()
    r.write(OUT/'DELIVERY_VERIFIED.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        three_audits_and_raw_actor_diagnostics_rechecked=True,bitwise_wa0_full_state_rechecked=True,
        frozen_and_old_protected_passed=True,protected_count=receipt['count'],
        result_sha256=r.sha(OUT/'STAGE0_RESULT.json'),report_sha256=r.sha(OUT/'STAGE0_REPORT_ZH.md'),time=time.time()))

if __name__=='__main__':main()
