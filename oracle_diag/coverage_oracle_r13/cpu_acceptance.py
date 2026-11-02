from runtime import *
from decision import decide
import numpy as np
import torch,runpy,types
def main():
    configure_cpu();check_frozen()
    for j in read(OUT/'training_plan.json')['jobs']:validate_spec(j)
    template=next(j for j in read(OUT/'training_plan.json')['jobs'] if j['arm']=='G')
    for change in (dict(arm='C'),dict(arm='D'),dict(seed=101),dict(steps=1000000),dict(temp=3),dict(ratio=.2),dict(phase='formal'),dict(task='hopper-morph')):
        j=template.copy();j.update(change)
        try:validate_spec(j)
        except AssertionError:pass
        else:raise AssertionError(change)
    def runs(arm,values):return [dict(job=dict(seed=s,arm=arm),Last5=v,Final=v) for s,v in zip(range(201,206),values)]
    def case(t,u,g,r):return decide({a:runs(a,v) for a,v in zip(('T','U','G','R'),(t,u,g,r))})
    x=case([0]*5,[0]*5,[1]*5,[0]*5);assert x['all_three_passed'] and not x['estimator_development_triggered']
    x=case([0]*5,[0]*5,[1]*5,[1]*5);assert not x['all_three_passed'] and x['direction2_ended_by_predefined_decision'] and '同数量' in x['conclusion']
    x=case([1]*5,[0]*5,[1]*5,[0]*5);assert not x['all_three_passed'] and x['notes'] and '随机' in x['notes'][0]
    x=case([0]*5,[1]*5,[1]*5,[0]*5);assert not x['all_three_passed'] and '等效' in x['notes'][0]
    for s,m in read(OUT/'R_mask_receipts.json')['masks'].items():
        ids=np.sort(np.random.RandomState(seed('R_mask',int(s))).choice(999322,size=49966,replace=False))
        assert np.array_equal(ids,np.load(m['path'])) and sha(m['path'])==m['sha256']
    delta=read(OUT/'engineering_delta.json')['science_copies']
    assert all(sha(HERE/n)==delta[n]['before_sha256']==delta[n]['after_sha256'] for n in ('learner.py','audit.py'))
    # The entire update loop is byte-for-text identical; only evaluator and R mask acquisition differ.
    before=Path(delta['train.py']['source']).read_text(encoding='utf-8');after=(HERE/'train.py').read_text(encoding='utf-8')
    assert before[before.index('        for step in range(1,'):]==after[after.index('        for step in range(1,'):]
    loaded=runpy.run_path(str(HERE/'train.py'))
    class Walker2dEnv:
        model=types.SimpleNamespace(opt=types.SimpleNamespace(timestep=.002))
    class FakeEnv:
        _max_episode_steps=1000
        unwrapped=Walker2dEnv()
        def seed(self,s):self.s=s
        def reset(self):return np.zeros(17)
        def step(self,a):return np.zeros(17),1.,True,{}
    fake_model=types.SimpleNamespace(deterministic_action=lambda s:np.zeros(6))
    for index,s in enumerate(range(201,206)):
        ev=loaded['evaluate'](fake_model,FakeEnv(),'confirmation',s)
        assert ev['env_seeds']==list(range(500000+10*index,500010+10*index))
    assert not torch.cuda.is_initialized() and not (OUT/'runs').exists()
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        all20_specs_valid=True,forbidden_jobs_rejected=True,all_result_branches_tested=True,
        R_masks_exact_single_uniform_draw_per_seed=True,training_update_loop_unchanged=True,
        learner_and_audit_byte_identical=True,eval_namespace_mock_checked=True,
        no_new_training_updates=True,no_actual_simulator_diagnostic_steps=True,time=time.time()))
    print('CPU acceptance passed; no CUDA initialized or training',flush=True)
if __name__=='__main__':main()
