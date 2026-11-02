"""CPU tooling checks; fixtures are explicitly synthetic and never formal data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v4 import *
from audit_v4 import events, paired, baselines
from run_queue_v4 import plan
import ast
import subprocess
import numpy as np
import torch
from tensorboardX import SummaryWriter


def main():
    dest=OUT/'tooling1';fresh(dest)
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    queue=plan();jobs=queue['jobs']
    assert len(jobs)==25 and len({j['out'] for j in jobs})==25
    formal=[j for j in jobs if j['kind']=='formal']
    assert [(j['method'],j['seed']) for j in formal]==[(m,s) for m in METHODS for s in range(5)]
    assert all(j['steps']==500000 for j in formal)
    assert [j['method'] for j in jobs[:5]]==list(ALL_METHODS)
    for j in jobs:
        assert OUT in Path(j['out']).parents and not Path(j['out']).exists()
    paths={}
    for name,duplicate,nan in [('valid',False,False),('duplicate',True,False),('nan',False,True)]:
        folder=dest/('synthetic_'+name)
        writer=SummaryWriter(str(folder))
        writer.add_scalar('test/target normalized score',float('nan') if nan else 7.,5000)
        if duplicate:writer.add_scalar('test/target normalized score',8.,5000)
        writer.close()
        if name=='valid':
            _,values=events(folder);assert values['test/target_normalized_score']=={5000:7.}
            paths[name]=True
        else:
            try:events(folder)
            except AssertionError:paths[name+'_rejected']=True
            else:raise AssertionError('Invalid event fixture accepted')
    tests=[]
    for values in ([1.,2.,3.,4.,5.],[-1.,-2.,-3.,-4.,-5.],[1.,1.,1.,1.,-20.],[0.,0.,0.,0.,0.]):
        calculated=paired(values); x=np.asarray(values)
        assert np.isclose(calculated['mean'],x.mean())
        assert np.isclose(calculated['sample_sd'],x.std(ddof=1))
        assert np.isclose(calculated['se'],x.std(ddof=1)/np.sqrt(5))
        assert calculated['passed']==(x.mean()>2*x.std(ddof=1)/np.sqrt(5) and (x>0).sum()>=4)
        tests.append(calculated)
    baseline=baselines();assert len(baseline)==15
    historical,scalars=events(V3_OUT/'s1/t0')
    assert list(scalars['test/target_normalized_score'])==[5000]
    refusals={}
    for kind in ('smoke','formal'):
        run=subprocess.run([str(PYTHON),'-B',str(HERE/'run_queue_v4.py'),'--kind',kind],
            cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8',errors='replace')
        assert run.returncode!=0 and 'Explicit v4 '+kind+' GPU authorization is required' in run.stderr
        assert not (OUT/'queue.lock').exists()
        refusals[kind]=dict(rejected=True,exit_code=run.returncode)
    assert all(not Path(j['out']).exists() for j in jobs)
    assert not torch.cuda.is_initialized()
    write(dest/'completed.json',dict(passed=True,device='cpu',cuda_initialized=False,
        syntax_files=len(list(HERE.glob('*.py'))),queue_jobs=25,formal_group_order_verified=True,
        tensorboard_fixtures=paths,statistics_cases=tests,v3_baseline_runs_recomputed=15,
        guard_rejections=refusals,synthetic_fixtures_excluded_from_results=True))
    print('Tooling checks passed; 15 historical baseline event files re-read; both GPU gates refuse',flush=True)


if __name__=='__main__':main()
