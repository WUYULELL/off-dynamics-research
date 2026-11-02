"""CPU acceptance of R12 guard/decision delta; no training, no new simulator sampling."""
from runtime import *
from decision import decide,prospective_interpretation
import copy
import runpy
import numpy as np
import torch

def main():
    configure_cpu();check_frozen()
    plan=read(OUT/'training_plan.json');assert len(plan['jobs'])==12
    for j in plan['jobs']:validate_spec(j)
    for change in (dict(arm='C'),dict(arm='R'),dict(arm='D'),dict(phase='formal'),dict(phase='tune'),
                   dict(task='hopper-morph'),dict(seed=201),dict(steps=1000000),dict(temp=3),dict(ratio=.1)):
        j=next(x for x in plan['jobs'] if x['arm']=='G').copy();j.update(change)
        try:validate_spec(j)
        except AssertionError:pass
        else:raise AssertionError('Unauthorized spec accepted '+str(change))
    def synthetic(arm,vals,ratio=None):
        runs=[]
        for seed,val in zip((101,102,103),vals):
            j=dict(seed=seed,arm=arm,temp=10,steps=500000)
            if ratio is not None:j['ratio']=ratio
            runs.append(dict(job=j,Last5=val,last5_steps=[460000,470000,480000,490000,500000]))
        return runs
    t=synthetic('T',[40,40,40]);u=synthetic('U',[39,39,39])
    def groups(vals):return [dict(ratio=p,mean=v,runs=synthetic('G',[v]*3,p)) for p,v in zip((.05,.2,.5),vals)]
    assert not decide(t,u,groups([40,39,38]))['passed_validation_stop']
    assert decide(t,u,groups([41,39,38]))['passed_validation_stop']
    assert decide(t,u,groups([41,41,41]))['G_ratio']==.5
    assert not decide(t,synthetic('U',[42]*3),groups([41,40,39]))['passed_validation_stop']
    for p in (.05,.2,.5):
        d=prospective_interpretation(p)
        assert d['coverage_estimator_eligible_after_future_four_rules']==(p==.05)
        assert not d['coverage_estimator_authorized_now'] and d['user_decision_required']
        assert d['novelty_review_required_before_further_development']==(p!=.05)
    assert decide(t,u,groups([45,39,38]))['T_boundary_limitation']['current_validation_gap_in_interval']
    assert not decide(t,u,groups([48,39,38]))['T_boundary_limitation']['current_validation_gap_in_interval']
    hashes=read(OUT/'engineering_delta.json')['science_copies']
    assert all(sha(HERE/n)==r['before_sha256']==r['after_sha256'] for n,r in hashes.items())
    loaded=runpy.run_path(str(HERE/'train.py'))
    assert Path(loaded['evaluate'].__code__.co_filename)==HERE/'train.py'
    prior=read(PRIOR/'STAGE_DELIVERY_RECEIPT_20261017.json')
    assert read(prior['verification'])['passed'] and sha(prior['verification'])==prior['verification_sha256']
    from learner import initialize
    for a in read(OUT/'T_selected.json')['runs']:
        job=a['job'];model=initialize(model_config(10,500000,job['seed']),'cpu')
        assert model.digest()==read(Path(job['out'])/'initialization.json')['sha256']
        rng=np.random.RandomState(seed('target_sample',job['seed']))
        assert rng.randint(5000,size=64).tolist()==a['sampling']['target_first']
    for name,receipt in read(OUT/'scores/completed.json')['masks'].items():
        path=OUT/'scores'/name;ids=np.load(path)
        assert sha(path)==receipt['sha256'] and len(ids)==receipt['n'] and len(np.unique(ids))==len(ids)
    commitment=read(OUT/'PRE_RESULT_COMMITMENT.json')
    assert commitment['protocol_sha256']==sha(OUT/'protocol.json')
    assert commitment['contract_sha256']==sha(commitment['contract'])
    assert not (OUT/'runs').exists() and not (OUT/'queue_started.json').exists()
    assert not torch.cuda.is_initialized()
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        science_byte_identical=True,guard_rejections_passed=True,all12_specs_valid=True,
        validation_stop_equality_and_ties_tested=True,all_ratio_interpretation_branches_tested=True,
        future_gap_3_to_7_boundary_caveat_tested=True,R11_initialization_and_first_batch_reproduced=True,
        static_masks_reused_hashes_pass=True,prior_full_science_acceptance_reused=True,
        zero_new_training_updates=True,zero_new_diagnostic_steps=True,
        pre_result_commitment_sha256=sha(OUT/'PRE_RESULT_COMMITMENT.json'),time=time.time()))
    print('R12 CPU acceptance passed; CUDA uninitialized; no training or diagnosis launched',flush=True)

if __name__=='__main__':main()
