from common import *
import traceback
def status(stage,**kw):r.write(OUT/'queue_status.json',dict(stage=stage,pid=os.getpid(),scope='Walker15_only_then_stop',time=time.time(),**kw),replace=True)
def main():
    guard();assert r.read(OUT/'CPU_ACCEPTANCE.json')['passed'];r.write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time()));start=time.time();completed=[]
    protect('before_GPU')
    for j in jobs():
        guard();assert not Path(j['out']).exists(),'Do not overwrite, rerun or resume an existing run automatically'
        status('TRAIN',job=j['id'],audited=len(completed),total=15)
        child(r.HERE/'train.py',['--job',OLD/'jobs'/(j['id']+'.json')],j['id'],False)
        status('CPU_AUDIT',job=j['id'],audited=len(completed),total=15)
        child(r.HERE/'audit.py',['--out',j['out']],j['id']+'_audit',True)
        assert r.read(Path(j['out'])/'audit.json')['passed'];completed.append(j['id'])
    status('CPU_ANALYSIS',audited=15);child(HERE/'analyze.py',[],'Walker_analysis',True)
    status('CPU_VERIFICATION',audited=15);child(HERE/'verify.py',[],'Walker_verify',True)
    assert r.read(OUT/'DELIVERY_VERIFIED.json')['passed'];protect('stage_end')
    elapsed={j['id']:r.read(Path(j['out'])/'completed.json')['elapsed'] for j in jobs()}
    r.write(OUT/'EXECUTION_SUMMARY.json',dict(passed=True,new_jobs=15,all15_audited=True,training_elapsed_seconds=elapsed,
        total_training_hours=sum(elapsed.values())/3600,queue_wall_hours=(time.time()-start)/3600,quoted_hours=[3,4],
        no_automatic_retraining=True,stage4_started=False,T_retuned=False,time=time.time()))
    r.write(OUT/'WALKER_ONLY_COMPLETED.json',dict(passed=True,Walker_audited=15,status='AWAITING_USER_DECISION',stage4_started=False,T_retuned=False,time=time.time()))
    status('AWAITING_USER_DECISION',audited=15,total=15)
if __name__=='__main__':
    try:main()
    except BaseException:
        r.write(OUT/('queue_failure_%d.json'%time.time()),dict(time=time.time(),traceback=traceback.format_exc(),action='CPU diagnosis; no automatic retraining'))
        status('FAILED');raise
