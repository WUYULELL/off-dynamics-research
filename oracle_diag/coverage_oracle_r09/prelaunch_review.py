"""Targeted review of guards, Windows paths and score numerics added after CPU algebra acceptance."""
from runtime import *
import ast
import numpy as np
import runtime

def main():
    assert read(OUT/'cpu_acceptance.json')['passed']
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    # No GPU work, no fake authorization persisted: test guards against local read stubs.
    original_check, original_read = runtime.check_frozen, runtime.read
    runtime.check_frozen=lambda:None
    def fake_read(p):
        name=Path(p).name
        if name=='T_selected.json':return dict(temp=3.,steps=500000)
        return dict(passed=True)
    runtime.read=fake_read
    rejected=[]
    base=dict(task='walker2d-morph',arm='G',phase='screen',seed=101,temp=3.,steps=500000,
        ratio=.2,out=str(OUT/'runs/dummy_never_created'))
    try:
        runtime.check_job(base)
        for change in (dict(arm='C'),dict(phase='formal'),dict(task='hopper-morph'),dict(seed=201),dict(ratio=.1)):
            try:runtime.check_job(dict(base,**change))
            except AssertionError:rejected.append(change)
            else:raise AssertionError('Scope guard accepted unauthorized job '+str(change))
    finally:
        runtime.check_frozen, runtime.read=original_check,original_read
    # Stable logarithms must agree with the direct formula in its non-underflow range.
    rng=np.random.RandomState(88)
    t,n,p=rng.uniform(0,4,(3,100))
    lr=np.logaddexp(-.5*p*p,-.5*n*n)-np.log(2.)
    lg=np.log(np.maximum(-np.expm1(-.5*t*t),1e-12))
    direct=np.log(.5*np.exp(-.5*p*p)+.5*np.exp(-.5*n*n))+np.log(np.maximum(1-np.exp(-.5*t*t),1e-12))
    assert np.allclose(lr+lg,direct,rtol=0,atol=1e-11)
    # Recover long historical paths omitted by a plain Path.exists in the first snapshot.
    initial=read(OUT/'protected_manifest.json')
    expected=dict(initial['sha256'])
    legacy=read(ROOT/'oracle_diag/results/v5_20261013/protected_manifest.json')['sha256']
    missing=[]
    recovered=[]
    for path,h in legacy.items():
        if path in expected:continue
        if os.path.exists(winpath(path)):
            expected[path]=sha(path)
            recovered.append(dict(path=path,previous=h,current=expected[path]))
        else:missing.append(path)
    write(OUT/'protected_manifest_final.json',dict(sha256=expected,count=len(expected),
        recovered_long_paths=recovered,already_missing_before_R09=missing,
        prior_snapshot_sha256=sha(OUT/'protected_manifest.json')))
    write(OUT/'prelaunch_review.json',dict(passed=True,rejected_unauthorized_jobs=rejected,
        stable_score_formula_checked=True,old_long_path_handling=True,
        old_current_snapshot_count=len(expected),initial_CPU_receipt_sha256=sha(OUT/'cpu_acceptance.json'),
        reviewed_changes='authorization guards, CPU C dtype, reference hash, short Windows TB paths, queue orchestration',
        scientific_learner_sha256=sha(HERE/'learner.py'), code_sha256=code_hashes()))
    write(OUT/'code_frozen.json',dict(sha256=code_hashes(),
        reference_CPU_receipt=sha(OUT/'cpu_acceptance.json'),prelaunch_review=sha(OUT/'prelaunch_review.json')))
    print('prelaunch passed; protected',len(expected),'recovered long paths',len(recovered),'already absent',len(missing),flush=True)

if __name__=='__main__':main()
