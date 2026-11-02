"""Durable authorized pipeline: await CPU models, entry smoke, serial GPU trials, verify/report."""
from support import *
import subprocess,time,traceback,ast

def run(script,args,log):
    with log.open('w',encoding='utf-8') as f:
        return subprocess.call([sys.executable,'-B','-u',str(HERE/script)]+list(args),cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
def refresh():
    if run('report_v2.py',[],OUT/'report_refresh.log'):raise RuntimeError('Report refresh failed')
def wait_gpu():
    while True:
        x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        free=int(x.splitlines()[0])
        if free>=1000:return
        write(OUT/'resource_wait.json',dict(free_mb=free,required_mb=1000,checked=time.time()),replace=True);time.sleep(30)
def check_protected():
    for p,h in read(OUT/'preregistered_protocol.json')['protected_sha256'].items():assert sha(p)==h,'Protected file changed '+p
def verify_pairing(jobs,seed):
    group={j['method']:Path(j['out']) for j in jobs if j['seed']==seed and j['state']=='completed'}
    if len(group)<3:return
    hashes=[read(p/'initialization.json')['sha256'] for p in group.values()];assert len(set(hashes))==1,'Initialization mismatch'
    hashes=[read(p/'data_verified.json')['target_sha256'] for p in group.values()];assert len(set(hashes))==1,'Target dataset mismatch'
    draws=[read(p/'first_sample.json') for m,p in group.items() if m!='TARGET']
    assert all(x==draws[0] for x in draws),'Paired sampler mismatch'

def main():
    lock=OUT/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        check_protected()
        assert read(OUT/'stage_a/completed.json')['passed'] and read(OUT/'stage_c/completed.json')['completed']
        while not (OUT/'stage_b/completed.json').exists():
            write(OUT/'pipeline_status.json',dict(state='waiting_stage_b',pid=os.getpid(),updated=time.time()),replace=True);time.sleep(30)
        assert read(OUT/'gpu_validation.json')['passed']
        methods=['TARGET','CONST','BACKUP']+(['MODEL'] if read(OUT/'stage_b/completed.json')['selected'] else [])
        extra_log=(OUT/'benchmark_new_tasks.log').open('w',encoding='utf-8')
        extra=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'benchmark_new_tasks.py')],cwd=str(ROOT),stdout=extra_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        scientific=[HERE/p for p in ['support.py','fixed_loader.py','train_entry_v2.py','validate_v2.py']]
        codehashes={str(p):sha(p) for p in scientific}
        write(OUT/'training_plan.json',dict(created=time.time(),seeds=list(range(5)),methods=methods,steps=500000,
            estimated_hours=[40,60],gpu='One RTX 2060 SUPER serial queue; minimum 1000 MB free before starting child',
            code_sha256=codehashes,supplemental_pid=extra.pid))
        # Check the real train.py route, saved models and seed parameter before long training.
        for method in methods:
            wait_gpu();path=OUT/'entry_smoke'/method
            code=run('train_entry_v2.py',['--method',method,'--seed','3','--steps','20','--check-eval','10','--out',str(path)],OUT/('entry_smoke_'+method+'.log'))
            if code:raise RuntimeError('Entry smoke failed '+method)
            assert read(path/'completed.json')['steps']==20
        smokejobs=[dict(seed=3,method=m,out=str(OUT/'entry_smoke'/m),state='completed') for m in methods]
        verify_pairing(smokejobs,3)
        write(OUT/'entry_smoke_verified.json',dict(passed=True,methods=methods,seed=3))
        jobs=[dict(seed=s,method=m,state='pending',out=str(OUT/'stage_d'/('seed%d_%s'%(s,m)))) for s in range(5) for m in methods]
        write(OUT/'queue_status.json',dict(jobs=jobs,created=time.time()),replace=True);refresh()
        for job in jobs:
            check_protected()
            for p,h in codehashes.items():assert sha(p)==h,'Training implementation changed'
            wait_gpu();job.update(state='running',started=time.time());write(OUT/'queue_status.json',dict(jobs=jobs,updated=time.time()),replace=True)
            log=OUT/('train_seed%d_%s.log'%(job['seed'],job['method']))
            code=run('train_entry_v2.py',['--method',job['method'],'--seed',str(job['seed']),'--out',job['out']],log)
            job.update(returncode=code,ended=time.time(),state='completed' if code==0 else 'failed')
            write(OUT/'queue_status.json',dict(jobs=jobs,updated=time.time()),replace=True)
            if code:raise RuntimeError('Training failed '+str(job))
            assert read(Path(job['out'])/'completed.json')['steps']==500000
            verify_pairing(jobs,job['seed']);refresh()
        code=extra.wait();extra_log.close()
        if code:raise RuntimeError('Supplemental CPU benchmark failed')
        # Read every final model/optimizer state and snapshot on CPU; no new training.
        import torch
        torch.set_num_threads(1)
        def finite(obj):
            if torch.is_tensor(obj):assert torch.isfinite(obj).all()
            elif isinstance(obj,dict):
                for v in obj.values():finite(v)
            elif isinstance(obj,(list,tuple)):
                for v in obj:finite(v)
        count=0
        for job in jobs:
            folder=Path(job['out']);files=list(folder.rglob('model_*'));assert len(files)==7
            files+=list((folder/'values').glob('V_*.pt'));assert len(files)==12
            for f in files:finite(torch.load(str(f),map_location='cpu'));count+=1
        check_protected()
        write(OUT/'final_validation.json',dict(passed=True,training_jobs=len(jobs),model_and_snapshot_files=count,protected_unchanged=True,pairing_verified=True))
        refresh();assert read(OUT/'report_data.json')['all_stages_complete']
        write(OUT/'pipeline_completed.json',dict(completed=True,ended=time.time(),jobs=len(jobs)))
        write(OUT/'pipeline_status.json',dict(state='complete',pid=os.getpid(),updated=time.time()),replace=True)
        print('V2 ALL STAGES COMPLETE',flush=True)
    except BaseException:
        write(OUT/'pipeline_error.json',dict(error=traceback.format_exc(),time=time.time()),replace=True)
        raise
if __name__=='__main__':main()
