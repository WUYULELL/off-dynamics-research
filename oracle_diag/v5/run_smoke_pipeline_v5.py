"""Authorized smoke sequence only. Stops at failures and never starts formal jobs."""
from common_v5 import *
import argparse, subprocess, traceback, msvcrt

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    check_approval('smoke');out=args.out.resolve();assert OUT in out.parents
    lock=open(OUT/'smoke_pipeline.lock','a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);fresh(out)
    state=dict(pid=os.getpid(),scope='GPU smoke/reuse/concurrent throughput only',started=time.time(),completed=[])
    def save():write(OUT/'smoke_pipeline_status.json',state,replace=True)
    def launch(stack,phase):
        name=stack+'_'+phase
        command=[str(PY_A),'-B','-u',str(HERE/'run_queue_v5.py'),'--stack',stack,'--kind','smoke','--phase',phase,'--launch',str(out/name)]
        stdout=open(out/(name+'.stdout.log'),'x',encoding='utf-8');stderr=open(out/(name+'.stderr.log'),'x',encoding='utf-8')
        child=subprocess.Popen(command,cwd=str(ROOT),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        return name,child,stdout,stderr
    def wait(item):
        name,child,stdout,stderr=item;rc=child.wait();stdout.close();stderr.close()
        if rc:raise RuntimeError(name+' failed with exit '+str(rc)+'; inspect logs, preserve outputs')
        state['completed'].append(name);save()
    try:
        for stack in ('A','B'):
            item=launch(stack,'standalone');state.update(status='running',phase=item[0],children=[item[1].pid]);save();wait(item)
        items=[launch(stack,'concurrent') for stack in ('A','B')]
        state.update(status='running',phase='concurrent',children=[x[1].pid for x in items]);save()
        # Wait for both authorized tests even when one fails; do not kill another process.
        failures=[]
        for item in items:
            try:wait(item)
            except RuntimeError as e:failures.append(str(e))
        if failures:raise RuntimeError('; '.join(failures))
        state.update(status='cpu_audit',phase='all_smoke_audit');save()
        with open(out/'all_smoke_audit.log','x',encoding='utf-8') as f:
            subprocess.run([str(PY_A),'-B','-u',str(HERE/'audit_smoke_v5.py')],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=True)
        state.update(status='completed',ended=time.time(),next='Deliver smoke report; await separate formal authorization');save()
        write(out/'completed.json',state)
    except BaseException:
        state.update(status='failed',ended=time.time(),error=traceback.format_exc());save();write(out/'failure.json',state);raise
    finally:lock.close()

if __name__=='__main__':main()
