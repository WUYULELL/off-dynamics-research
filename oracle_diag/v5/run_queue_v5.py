"""One locked queue per stack, immutable job evidence, CPU audit after each GPU job."""
from common_v5 import *
import argparse, subprocess, msvcrt, traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--stack',choices=('A','B'),required=True)
    p.add_argument('--kind',choices=('smoke','formal'),required=True)
    p.add_argument('--phase',choices=('standalone','concurrent','formal'),required=True)
    p.add_argument('--launch',type=Path,required=True);a=p.parse_args();check_approval(a.kind)
    assert (a.kind=='formal')==(a.phase=='formal')
    folder=OUT if a.stack=='A' else B_OUT
    lock=open(folder/'queue.lock','a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    fresh(a.launch);state=folder/'queue_status.json'
    jobs=[j for j in read(OUT/'training_plan.json')['jobs'] if (j['stack'],j['kind'],j['phase'])==(a.stack,a.kind,a.phase)]
    status=dict(pid=os.getpid(),stack=a.stack,kind=a.kind,phase=a.phase,launch=str(a.launch),completed=[],started=time.time())
    try:
        for j in jobs:
            audit_path=Path(j['out'])/'audit.json'
            if audit_path.exists():assert read(audit_path)['passed'];status['completed'].append(j['id']);continue
            if a.kind=='formal' and a.stack=='B':
                while (OUT/'defer_B_until_A10.json').exists():
                    aj=[x for x in read(OUT/'training_plan.json')['jobs'] if x['stack']=='A' and x['kind']=='formal'][:10]
                    if all((Path(x['out'])/'audit.json').exists() for x in aj):break
                    status.update(status='deferred_at_job_boundary',current=j['id']);write(state,status,replace=True);time.sleep(30)
            command=[j['python'],'-B','-u',j['entry'],'--method',j['method'],'--seed',str(j['seed']),'--steps',str(j['steps']),'--out',j['out']]
            write(a.launch/(j['id']+'_command.json'),dict(command=command,started=time.time()))
            with open(a.launch/(j['id']+'.stdout.log'),'x',encoding='utf-8') as stdout,open(a.launch/(j['id']+'.stderr.log'),'x',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=str(ROOT),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                status.update(status='running',current=j['id'],child_pid=child.pid);write(state,status,replace=True)
                while child.poll() is None:
                    if a.kind=='formal':
                        from progress_v5 import milestones
                        milestones()
                    time.sleep(30)
                rc=child.returncode
            if rc:raise RuntimeError('Job failed, preserve output and restart same seed in a new attempt: '+j['id']+' exit '+str(rc))
            status['status']='cpu_audit';write(state,status,replace=True)
            with open(a.launch/(j['id']+'.audit.log'),'x',encoding='utf-8') as f:
                subprocess.run([str(PY_A),'-B','-u',str(HERE/'audit_v5.py'),'--job-id',j['id'],'--out',str(audit_path)],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=True)
            status['completed'].append(j['id']);write(state,status,replace=True)
            if a.kind=='formal':
                from progress_v5 import milestones
                milestones()
        status.update(status='completed',ended=time.time(),current=None);write(state,status,replace=True)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),ended=time.time());write(state,status,replace=True)
        write(a.launch/'failure.json',status);raise
    finally:lock.close()

if __name__=='__main__':main()
