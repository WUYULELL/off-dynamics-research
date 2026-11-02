"""Exactly forty new jobs, sequential on one GPU; never dispatch a next stage."""
from runtime import *
import subprocess
import traceback

class Queue:
    def __init__(self):
        check_frozen()
        assert read(OUT/'cpu_acceptance.json')['passed']
        assert sha(OUT/'cpu_acceptance.json')==read(OUT/'CPU_ACCEPTANCE_COMMITMENT.json')['cpu_acceptance_sha256']
        write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv))
        self.results=[];self.dispatched=[]
        for r in read(OUT/'reuse_receipts.json')['runs']:
            assert sha(r['audit'])==r['audit_sha256'];a=read(r['audit']);assert a['passed']
            if r['label']=='raw':self.results.append(dict(task=r['task'],seed=r['seed'],w=r['w'],audit=a,out=r['out']))
        assert len(self.results)==38
        self.status('STARTING')
    def status(self,stage,**kw):
        write(OUT/'queue_status.json',dict(stage=stage,queue_pid=os.getpid(),updated=time.time(),
            dispatched_new_jobs=33+len(self.dispatched), recovery_dispatched_jobs=len(self.dispatched),audited_new_jobs=max(0,len(self.results)-5),total_new_jobs=40,
            estimator_development_authorized=False,extra_training_authorized=False,**kw),replace=True)
    def child(self,script,args,name,cpu=False):
        command=[str(PYTHON),'-B',str(HERE/script)]+[str(a) for a in args]
        environ=os.environ.copy();environ['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';environ['PYTHONDONTWRITEBYTECODE']='1'
        with open(OUT/'logs'/(name+'.stdout.log'),'xb') as stdout,open(OUT/'logs'/(name+'.stderr.log'),'xb') as stderr:
            child=subprocess.Popen(command,cwd=str(ROOT),env=environ,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.status(script,active_pid=child.pid,active_command=command,log_prefix=str(OUT/'logs'/name))
            code=child.wait()
        if code:raise RuntimeError('%s exited %d; preserve evidence; no automatic retraining'%(script,code))
    def main(self):
        protect('before_gpu')
        jobs=read(OUT/'training_plan.json')['jobs'];assert len(jobs)==7
        for job in jobs:
            check_job(job);self.dispatched.append(job)
            write(OUT/'dispatch_progress.json',dict(jobs=self.dispatched),replace=True)
            self.child('train.py',['--job',OUT/'jobs'/(job['id']+'.json')],job['id'])
            self.child('audit.py',['--out',job['out']],job['id']+'_audit',True)
            a=read(Path(job['out'])/'audit.json');assert a['passed']
            if job['arm']=='M':
                base=next(r for r in self.results if r['task']==job['task'] and r['seed']==job['seed'] and r['w']==0.)
                assert a['sampling']['target_stream_sha256']==base['audit']['sampling']['target_stream_sha256']
                assert read(Path(job['out'])/'initialization.json')==read(Path(base['out'])/'initialization.json')
                other=next((r for r in self.results if r['task']==job['task'] and r['seed']==job['seed'] and r['w']>0),None)
                if other:assert a['sampling']['source_stream_sha256']==other['audit']['sampling']['source_stream_sha256']
            self.results.append(dict(task=job['task'],seed=job['seed'],w=job['source_w'],audit=a,out=job['out']))
            self.status('job_audited',last_audited=job['id'])
        self.child('analyze.py',[],'final_CPU_analysis',True)
        assert read(OUT/'ANALYSIS_COMPLETED.json')['passed'];protect('stage_end')
        write(OUT/'STAGE_COMPLETED.json',dict(status='AWAITING_USER_DECISION',new_GPU_jobs=40,recovery_GPU_jobs=7,reused_R14_jobs=33,all_audited=True,
            estimator_development=False,no_additional_training=True,report=str(DOC/'DELIVERY_REPORT_R14.md'),time=time.time()))
        self.status('AWAITING_USER_DECISION',report=str(DOC/'DELIVERY_REPORT_R14.md'))

if __name__=='__main__':
    queue=None
    try:queue=Queue();queue.main()
    except BaseException:
        path=OUT/('queue_failure_%d.json'%int(time.time()))
        write(path,dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc(),action='CPU diagnosis; no automatic retraining'))
        if queue:queue.status('FAILED',failure=str(path))
        raise
