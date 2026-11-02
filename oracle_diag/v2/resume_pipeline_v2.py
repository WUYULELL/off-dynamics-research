"""Fresh coordinator reuses completed A/B/C/smoke and starts a clean formal queue."""
from support import *
import subprocess,time,traceback
from run_pipeline_v2 import check_protected,verify_pairing

RES=OUT/'resume1'
def refresh():
    with (RES/'report_refresh.log').open('a',encoding='utf-8') as f:
        code=subprocess.call([sys.executable,'-B','-u',str(HERE/'report_resume_v2.py')],stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT))
    assert code==0,'Report update failed'
def status(state,**kwargs):write(RES/'status.json',dict(state=state,pid=os.getpid(),updated=time.time(),**kwargs),replace=True)
def main():
    fresh(RES)
    with (RES/'pipeline.lock').open('x') as f:f.write(str(os.getpid()))
    child=None;extra=None;handles=[]
    try:
        check_protected();plan=read(OUT/'training_plan.json')
        for p,h in plan['code_sha256'].items():assert sha(p)==h,'Core training code changed'
        assert read(OUT/'stage_a/completed.json')['passed']
        assert read(OUT/'stage_c/completed.json')['completed']
        assert read(OUT/'stage_b/completed.json')['selected'] is None
        assert read(OUT/'entry_smoke_verified.json')['passed'] and read(OUT/'gpu_validation.json')['passed']
        benchmark_files=list((OUT/'stage_b_new_tasks').glob('*'))
        write(RES/'resume_manifest.json',dict(started=time.time(),reason='User resumed after reboot',
            protocol_sha256=sha(OUT/'preregistered_protocol.json'),core_sha256=plan['code_sha256'],
            retained_prior_partial_files={str(p):sha(p) for p in benchmark_files if p.is_file()},
            reused_stages=['A','B','C','GPU validation','entry smoke'],
            restarted_formal_job='seed0_TARGET; no saved training checkpoint',
            changed_scientific_hyperparameters=False))
        write(OUT/'active_execution.json',dict(directory=str(RES),coordinator=str(HERE/'resume_pipeline_v2.py'),started=time.time()))
        extra_log=(RES/'benchmark.log').open('w',encoding='utf-8');handles.append(extra_log)
        extra=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'resume_benchmark_v2.py')],cwd=str(ROOT),stdout=extra_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs=[dict(seed=s,method=m,state='pending',out=str(RES/'stage_d'/('seed%d_%s'%(s,m)))) for s in range(5) for m in ['TARGET','CONST','BACKUP']]
        write(RES/'queue_status.json',dict(jobs=jobs),replace=True);status('running',supplement_pid=extra.pid);refresh()
        for job in jobs:
            check_protected()
            for p,h in plan['code_sha256'].items():assert sha(p)==h,'Core training code changed'
            while True:
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                if free>=1000:break
                status('waiting_gpu',free_mb=free,required_mb=1000);time.sleep(30)
            logfile=RES/('seed%d_%s.log'%(job['seed'],job['method']))
            with logfile.open('w',encoding='utf-8') as f:
                child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'train_entry_v2.py'),'--method',job['method'],
                    '--seed',str(job['seed']),'--out',job['out']],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                job.update(state='running',pid=child.pid,started=time.time())
                write(RES/'queue_status.json',dict(jobs=jobs,updated=time.time()),replace=True)
                status('training',method=job['method'],seed=job['seed'],child_pid=child.pid,supplement_pid=extra.pid)
                while child.poll() is None:
                    if extra.poll() not in [None,0]:
                        # A CPU supplement failure does not invalidate ongoing RL; finish it and report the issue.
                        write(RES/'supplement_failure.json',dict(returncode=extra.returncode),replace=True)
                    time.sleep(30)
                code=child.returncode
            job.update(returncode=code,ended=time.time(),state='completed' if code==0 else 'failed')
            write(RES/'queue_status.json',dict(jobs=jobs,updated=time.time()),replace=True)
            assert code==0,'Training failed '+str(job)
            assert read(Path(job['out'])/'completed.json')['steps']==500000
            verify_pairing(jobs,job['seed']);refresh()
        status('waiting_supplement')
        assert extra.wait()==0,'Supplement benchmark failed'
        assert read(RES/'stage_b_new_tasks/completed.json')['completed']
        status('validating')
        import torch
        from report_v2 import score
        torch.set_num_threads(1)
        def finite(obj):
            if torch.is_tensor(obj):assert torch.isfinite(obj).all()
            elif isinstance(obj,dict):
                for value in obj.values():finite(value)
            elif isinstance(obj,(list,tuple)):
                for value in obj:finite(value)
        count=0
        for job in jobs:
            folder=Path(job['out']);assert score(folder) is not None,'Missing complete evaluations'
            files=list(folder.rglob('model_*'));assert len(files)==7
            files+=list((folder/'values').glob('V_*.pt'));assert len(files)==12
            for file in files:finite(torch.load(str(file),map_location='cpu'));count+=1
        for seed in range(5):verify_pairing(jobs,seed)
        check_protected()
        manifest=read(RES/'resume_manifest.json')
        for p,h in manifest['retained_prior_partial_files'].items():assert sha(p)==h,'Prior partial artifact changed'
        write(OUT/'final_validation.json',dict(passed=True,training_jobs=len(jobs),model_and_snapshot_files=count,
            protected_unchanged=True,pairing_verified=True,execution_directory=str(RES)))
        status('complete');refresh();assert read(OUT/'report_data.json')['all_stages_complete']
        write(OUT/'pipeline_completed.json',dict(completed=True,ended=time.time(),jobs=len(jobs),execution_directory=str(RES)))
        print('RESUMED V2 ALL STAGES COMPLETE',flush=True)
    except BaseException:
        status('failed',error=traceback.format_exc())
        write(RES/'pipeline_error.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
    finally:
        for handle in handles:handle.close()
if __name__=='__main__':main()
