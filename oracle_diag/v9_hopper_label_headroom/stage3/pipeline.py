from runtime import *
import subprocess,traceback
def processes():
    cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*v9_hopper_label_headroom*stage2*' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    s=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],universal_newlines=True).strip();v=json.loads(s) if s else []
    return v if isinstance(v,list) else [v]
class Queue:
    def __init__(self):
        check_frozen();write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv));self.completed=[];self.status('WAITING_FOR_STAGE2')
    def status(self,stage,**kw):write(OUT/'queue_status.json',dict(stage=stage,pid=os.getpid(),audited_jobs=len(self.completed),total_jobs=45,**kw),replace=True)
    def child(self,script,args,name,cpu=False,external=False):
        command=[str(PYTHON),'-B',str(script if external else HERE/script)]+[str(a) for a in args]
        en=os.environ.copy();en['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';en.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
        with open(winpath(OUT/'logs'/(name+'.stdout.log')),'xb') as so,open(winpath(OUT/'logs'/(name+'.stderr.log')),'xb') as se:
            p=subprocess.Popen(command,cwd=str(ROOT),env=en,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
            self.status(name,active_pid=p.pid,active_command=command);code=p.wait()
        if code:raise RuntimeError('%s failed exit%d; preserve evidence, no automatic retry'%(name,code))
    def main(self):
        while True:
            check_frozen()
            failures=list(STAGE2.glob('queue_failure_*.json'));assert not failures,failures
            p=processes()
            if (STAGE2/'STAGE_COMPLETED.json').exists() and not p:break
            self.status('WAITING_FOR_STAGE2',stage2_status=read(STAGE2/'queue_status.json'),stage2_processes=p)
            time.sleep(60)
        assert read(STAGE2/'STAGE_COMPLETED.json')['passed']
        if not (STAGE2/'DELIVERY_VERIFIED_STAGE2.json').exists():
            self.child(ROOT/'oracle_diag/v9_hopper_label_headroom/stage2/verify_delivery.py',[],'verify_stage2',True,True)
        assert read(STAGE2/'DELIVERY_VERIFIED_STAGE2.json')['passed']
        manifest=dict(read(STAGE2/'protected_manifest.json')['sha256'])
        for p in STAGE2.rglob('*'):
            if p.is_file():manifest[str(p)]=sha(p)
        for p in (ROOT/'outputs/v9_hopper_label_headroom_20261030/interim_stage0_stage1_O1_20261031').rglob('*'):
            if p.is_file():manifest[str(p)]=sha(p)
        write(OUT/'protected_manifest.json',dict(sha256=manifest,time=time.time(),stage2_completed_and_verified=True));protect('before_GPU')
        jobs=read(OUT/'training_plan.json')['jobs'];assert len(jobs)==45
        for index,job in enumerate(jobs):
            self.child('train.py',['--job',OUT/'jobs'/(job['id']+'.json')],job['id'])
            self.child('audit.py',['--out',job['out']],job['id']+'_audit',True)
            assert read(Path(job['out'])/'audit.json')['passed'];self.completed.append(job['id'])
            if index==0:
                elapsed=read(Path(job['out'])/'completed.json')['elapsed']
                write(OUT/'MEASURED_BUDGET.json',dict(first_formal_run=job['id'],elapsed_seconds=elapsed,
                    first_run_counts_in_45=True,additional_probe_updates=0,remaining_Hopper_jobs=29,remaining_Walker_jobs=15,
                    estimated_Hopper_total_hours=30*elapsed/3600,estimated_all45_total_hours=45*elapsed/3600,
                    caveat='single formal run measured; evaluation duration and Walker morphology may change wall time',time=time.time()))
                with open(winpath(OUT/'FIRST_RUN_MEASURED_AND_BUDGET.md'),'x',encoding='utf-8') as f:
                    f.write('首个正式Hopper BC-all seed301的200k训练（含评估）实测%.2f分钟。计入45项预算，无额外GPU探针。按单项外推Hopper30项约%.2f小时，全部45项约%.2f小时；Walker和不同策略的评估长度可导致偏差。其余44项在此实测报告落盘后启动。'%(elapsed/60,30*elapsed/3600,45*elapsed/3600))
            if index==29:
                self.child('analyze.py',['--task','hopper-morph'],'HOPPER_analysis',True)
                self.child('verify_delivery.py',['--task','hopper-morph'],'HOPPER_verify',True);protect('Hopper_end')
                write(OUT/'HOPPER_PRIORITY_COMPLETED.json',dict(passed=True,jobs=30,Walker_follows=True,time=time.time()))
                if read(OUT/'HOPPER_RESULT.json')['decision']['headroom'].startswith('V9未覆盖'):
                    self.status('AWAITING_RULE_CLARIFICATION');return
        self.child('analyze.py',['--task','walker2d-morph'],'WALKER_analysis',True)
        self.child('verify_delivery.py',['--task','walker2d-morph'],'WALKER_verify',True);protect('stage_end')
        write(OUT/'STAGE_COMPLETED.json',dict(passed=True,status='AWAITING_USER_STAGE4_DECISION',new_jobs=45,Hopper_first=True,
            Hopper_audited=30,Walker_audited=15,stage4_started=False,estimator_development=False,time=time.time()))
        self.status('AWAITING_USER_STAGE4_DECISION')
if __name__=='__main__':
    q=None
    try:q=Queue();q.main()
    except BaseException:
        write(OUT/('queue_failure_%d.json'%time.time()),dict(time=time.time(),traceback=traceback.format_exc(),action='CPU diagnosis; no automatic retry'))
        if q:q.status('FAILED')
        raise
