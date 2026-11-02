from common import *
import traceback
def status(stage,**kw):r.write(OUT/'queue_status.json',dict(stage=stage,pid=os.getpid(),scope='Hopper30_only_then_stop_before_Walker',time=time.time(),**kw),replace=True)
def main():
    guard();r.write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time()))
    saved=r.read(OUT/'BEFORE_QUEUE_CORRECTION.json');oldq=saved['original_queue_status'];active_pid=oldq['active_pid']
    active_job=r.read(oldq['active_command'][-1]);assert active_job in jobs()
    for j in jobs():
        guard();p=Path(j['out'])
        if (p/'audit.json').exists():assert r.read(p/'audit.json')['passed'];continue
        if p.exists():
            assert j['id']==active_job['id'],'Existing unaudited run cannot be automatically retrained'
            while not (p/'completed.json').exists():
                assert not (p/'failed.json').exists(),'Current run failed; preserve, no automatic retry'
                ps="Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq %d } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"%active_pid
                value=subprocess.check_output(['powershell','-NoProfile','-Command',ps],universal_newlines=True).strip()
                assert value and j['id']+'.json' in json.loads(value)['CommandLine'],'Current child disappeared before completed marker'
                status('WAIT_CURRENT_HOPPER_CHILD',job=j['id'],active_pid=active_pid);time.sleep(15)
        else:
            assert j['id']=='hopp_BC25_s310','Only the single unstarted Hopper job is authorized for dispatch'
            status('TRAIN_REMAINING_HOPPER',job=j['id']);child(r.HERE/'train.py',['--job',OLD/'jobs'/(j['id']+'.json')],j['id'],False)
        status('CPU_AUDIT',job=j['id']);child(r.HERE/'audit.py',['--out',p],j['id']+'_audit',True)
        assert r.read(p/'audit.json')['passed']
    assert len(list((OLD/'runs').glob('hopp_*/audit.json')))==30
    status('CPU_ANALYSIS');child(HERE/'analyze.py',[],'Hopper_analysis',True)
    status('CPU_VERIFICATION');child(HERE/'verify.py',[],'Hopper_verify',True)
    assert r.read(OUT/'DELIVERY_VERIFIED.json')['passed']
    r.write(OUT/'HOPPER_ONLY_COMPLETED.json',dict(passed=True,Hopper_audited=30,Walker_started=False,status='AWAITING_WALKER_CONFIRMATION',time=time.time()))
    status('AWAITING_WALKER_CONFIRMATION',Hopper_audited=30,Walker_runs=0)
if __name__=='__main__':
    try:main()
    except BaseException:
        r.write(OUT/('queue_failure_%d.json'%time.time()),dict(time=time.time(),traceback=traceback.format_exc(),action='CPU diagnosis; no automatic retry'))
        status('FAILED');raise
