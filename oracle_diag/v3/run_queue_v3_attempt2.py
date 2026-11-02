"""Create the reviewable queue; execute it only after user authorization is recorded."""
from support_v3 import *
import argparse,subprocess,time
PYTHON=Path('D:/ANACONDA/envs/offdynamics/python.exe')
METHODS=[('t','TARGET'),('c','MIX-CONST'),('o','MIX-ORACLE-BACKUP')]

def plan():
    jobs=[]
    for short,method in METHODS:
        jobs.append(dict(kind='smoke',method=method,seed=0,steps=5000,out=str(OUT/'s1'/(short+'0'))))
    for seed in range(5):
        for short,method in METHODS:
            jobs.append(dict(kind='formal',method=method,seed=seed,steps=500000,out=str(OUT/'b1'/(short+str(seed)))))
    for j in jobs:
        j['command']=[str(PYTHON),'-B',str(HERE/'train_entry_v3.py'),'--method',j['method'],'--seed',str(j['seed']),
            '--steps',str(j['steps']),'--out',j['out']]
    return dict(jobs=jobs,formal_runs=15,formal_gradient_steps=7500000,smoke_runs=3,
        batch=128,source_weight=.20,task='walker2d-morph',seeds=list(range(5)),
        execution='Single training subprocess at a time, CUDA GPU0, CPU threads2; memory-mapped CPU oracle',
        runtime_estimate='Planning range 25--40 hours using v2 38.3h reference; revise using actual smoke before formal launch',
        missing_authorization='Explicit stage B GPU/RL confirmation required; no authorized receipt is created by this script',
        stage_c='not executed',MIX_NAIVE='Exact alias of MIX-CONST; no duplicate runs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:
        write(OUT/'training_plan.json',plan());print('Training plan written; no training started');return
    from train_entry_v3 import check_approval
    check_approval();queue=read(OUT/'training_plan.json')
    lock=OUT/'queue.lock'
    with open(str(lock),'x') as f:f.write(str(os.getpid()))
    try:
        for job in queue['jobs']:
            if job['kind']=='formal' and not (OUT/'smoke_audited.json').exists():
                # Independent scores/model/sampler audit before the first long job.
                audit=subprocess.run([str(PYTHON),'-B',str(HERE/'audit_runs_v3_attempt2.py'),'--kind','smoke'],cwd=str(ROOT))
                if audit.returncode:raise RuntimeError('GPU smoke audit failed; formal runs were not started')
            out=Path(job['out']);completed=out/'completed.json'
            if completed.exists():
                saved=read(completed)
                assert all(saved[k]==job[k] for k in ['method','seed','steps']);continue
            if out.exists() and any(out.iterdir()):raise RuntimeError('Preserving incomplete attempt; choose a new attempt after diagnosis: '+str(out))
            out.parent.mkdir(parents=True,exist_ok=True)
            logfile=out.parent/(out.name+'.stdout.log');errfile=out.parent/(out.name+'.stderr.log')
            with open(str(logfile),'x',encoding='utf-8') as stdout,open(str(errfile),'x',encoding='utf-8') as stderr:
                child=subprocess.Popen(job['command'],cwd=str(ROOT),stdout=stdout,stderr=stderr,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                write(OUT/'queue_status.json',dict(state='running',pid=child.pid,job=job,started=time.time()),replace=True)
                code=child.wait()
            if code!=0 or not completed.exists():raise RuntimeError('Job failed; evidence preserved at '+str(out))
        audit=subprocess.run([str(PYTHON),'-B',str(HERE/'audit_runs_v3_attempt2.py'),'--kind','formal'],cwd=str(ROOT))
        if audit.returncode:raise RuntimeError('Formal results require audit repair; training output is preserved')
        write(OUT/'queue_status.json',dict(state='completed',ended=time.time()),replace=True)
    finally:lock.unlink()

if __name__=='__main__':main()
