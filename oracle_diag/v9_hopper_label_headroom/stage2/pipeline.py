"""Exactly20 oracle-label jobs, one GPU serial; CPU audits/analysis then stop."""
from runtime import *
import subprocess
import traceback

class Queue:
    def __init__(self):
        check_frozen();assert read(OUT/'cpu_acceptance.json')['passed']
        assert sha(OUT/'cpu_acceptance.json')==read(OUT/'CPU_ACCEPTANCE_COMMITMENT.json')['cpu_acceptance_sha256']
        write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv))
        self.completed=[];self.dispatched=[];self.started=time.time()
        self.base=read(OUT/'reuse_receipts.json')['runs'];assert len(self.base)==30
        self.status('STARTING')
    def status(self,stage,**kw):
        write(OUT/'queue_status.json',dict(stage=stage,queue_pid=os.getpid(),updated=time.time(),total_new_jobs=20,
            dispatched_new_jobs=len(self.dispatched),audited_new_jobs=len(self.completed),stage3_or_4=False,**kw),replace=True)
    def child(self,script,args,name,cpu=False):
        command=[str(PYTHON),'-B',str(HERE/script)]+[str(v) for v in args]
        environ=os.environ.copy();environ['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';environ['PYTHONDONTWRITEBYTECODE']='1'
        environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
        with open(winpath(OUT/'logs'/(name+'.stdout.log')),'xb') as stdout,open(winpath(OUT/'logs'/(name+'.stderr.log')),'xb') as stderr:
            p=subprocess.Popen(command,cwd=str(ROOT),env=environ,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.status(script,active_pid=p.pid,active_command=command,log_prefix=str(OUT/'logs'/name));code=p.wait()
        if code:raise RuntimeError('%s failed exit%d; preserve evidence; no automatic retraining'%(script,code))
    def main(self):
        protect('before_gpu');jobs=read(OUT/'training_plan.json')['jobs'];assert len(jobs)==20
        for job in jobs:
            check_job(job);self.dispatched.append(job)
            write(OUT/'dispatch_progress.json',dict(jobs=self.dispatched),replace=True)
            self.child('train.py',['--job',OUT/'jobs'/(job['id']+'.json')],job['id'])
            self.child('audit.py',['--out',job['out']],job['id']+'_audit',True)
            p=Path(job['out']);audit=read(p/'audit.json');assert audit['passed']
            for r in self.base:
                if r['seed']!=job['seed']:continue
                assert read(p/'initialization.json')==read(Path(r['out'])/'initialization.json')
                s=read(Path(r['out'])/'sampling.json')
                assert s['target_stream_sha256']==audit['sampling']['target_stream_sha256']
                if r['w']>0:assert s['source_stream_sha256']==audit['sampling']['source_stream_sha256']
            self.completed.append(job);self.status('job_audited',last_audited=job['id'])
        self.child('analyze.py',[],'final_CPU_analysis',True)
        assert read(OUT/'ANALYSIS_COMPLETED.json')['passed'];protect('stage_end')
        durations={j['id']:read(Path(j['out'])/'completed.json')['elapsed'] for j in jobs}
        write(OUT/'EXECUTION_SUMMARY.json',dict(passed=True,new_jobs=20,R14_comparisons_reused=30,individual_training_seconds=durations,
            total_training_seconds=sum(durations.values()),queue_wall_seconds=time.time()-self.started,quoted_hours=[48,55],
            all_jobs_completed_without_automatic_rerun=True,stage3_or_4_started=False,time=time.time()))
        analysis=read(OUT/'ANALYSIS_COMPLETED.json')
        write(OUT/'STAGE_COMPLETED.json',dict(passed=True,status='AWAITING_STAGE3_CONFIRMATION',all20_audited=True,
            analysis_completed_sha256=sha(OUT/'ANALYSIS_COMPLETED.json'),report=analysis['report'],report_sha256=analysis['report_sha256'],
            stage3_started=False,stage4_started=False,estimator_development=False,time=time.time()))
        self.status('AWAITING_STAGE3_CONFIRMATION',report=analysis['report'])

if __name__=='__main__':
    q=None
    try:q=Queue();q.main()
    except BaseException:
        p=OUT/('queue_failure_%d.json'%int(time.time()))
        write(p,dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc(),action='CPU diagnosis; no automatic retraining'))
        if q:q.status('FAILED',failure=str(p))
        raise
