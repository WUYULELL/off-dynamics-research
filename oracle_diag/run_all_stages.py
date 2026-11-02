"""Durable single-GPU queue for every authorized seed-0 stage, then report."""
import os,sys,json,time,hashlib,ast,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results/verified_20261003'
ENVS=['walker2d-kinematic-footjnt','hopper-kinematic-legjnt','halfcheetah-kinematic-footjnt']
OLD=Path(r'C:\Users\TXin\Documents\Codex\2026-09-15\referenced-chatgpt-conversation-this-is-an-2\outputs\ot_bellman')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(str(tmp),str(path))


def run_tool(script,args=(),logname=None):
    cmd=[sys.executable,'-B','-u',str(HERE/script)]+list(args)
    if logname:
        with (OUT/logname).open('a',encoding='utf-8') as f:
            return subprocess.call(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    return subprocess.call(cmd,cwd=str(ROOT),stdin=subprocess.DEVNULL)


def report():
    code=run_tool('report_pipeline.py',logname='report_refresh.log')
    if code:raise RuntimeError('Report refresh failed')


def verify_reuse():
    path=OLD/'formal_walker_seed0_0917/IQL_BELLMAN_FB/walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium/r0/models/model_value'
    if not path.is_file():return None
    log=path.parent.parent.parent/'r0log.txt';cfg={}
    for line in log.read_text().splitlines():
        if line.strip().startswith('('):
            k,v=ast.literal_eval(line.strip());cfg[k]=v
    for key,value in dict(source_budget=.2,bellman_eta=1.,seed=0,max_step=500000,gamma=.99,batch_size=128,knn_k=5,target_bank_size=20000,hidden_sizes=256,lam=.7,temp=3.).items():
        if cfg.get(key)!=value:raise RuntimeError('Reuse configuration mismatch '+key)
    old_project=OLD/'project'
    for name in ['dataset/mujoco/walker2d_kinematic_footjnt_medium_medium.hdf5',
                 'envs/mujoco/assets/walker2d_kinematic_footjnt_medium.xml','algo/utils.py','dataset/call_dataset.py',
                 'algo/offline_offline/iql_dyn.py','algo/offline_offline/iql_unc.py']:
        if sha(ROOT/name)!=sha(old_project/name):raise RuntimeError('Reuse dependency mismatch '+name)
    # Earlier copy added optional logging; compare the scientific functions themselves.
    trees=[ast.parse((p/'algo/offline_offline/iql_bellman_fb.py').read_text()) for p in [ROOT,old_project]]
    for method in ['train','compute_knn_indices','initialize_target_feature_bank']:
        nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==method) for tree in trees]
        assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False),method
    # The old copy extracted a target-reference helper. Independently check
    # that this refactor preserves weights and actual parameter updates.
    import importlib.util
    from common import load_source_buffer,load_target_buffer,load_policy_config,torch,np
    from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('legacy_bellman_compare',str(old_project/'algo/offline_offline/iql_bellman_fb.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=load_policy_config(ENVS[0],overrides=dict(seed=0,source_budget=.2,bellman_eta=1.,max_step=500000))
    src=load_source_buffer(ENVS[0]);tar,_=load_target_buffer(ENVS[0],0)
    models=[]
    for cls in [IQL_BELLMAN_FB,module.IQL_BELLMAN_FB]:
        torch.manual_seed(0);m=cls(cfg,torch.device('cpu'));m.initialize_target_feature_bank(tar)
        m.v_func.load_state_dict(torch.load(str(path),map_location='cpu'));models.append(m)
    np.random.seed(0)
    for _ in range(8):
        batch=src.sample(128);results=[m.compute_source_weight(*batch) for m in models]
        for key in ['weight','bellman_mismatch','bellman_weight_raw']:
            torch.testing.assert_close(results[0][key],results[1][key],atol=0,rtol=0)
    for step in range(3):
        for m in models:np.random.seed(100+step);m.train(src,tar)
        for net in ['v_func','q_funcs','policy','target_q_funcs']:
            for a,b in zip(getattr(models[0],net).parameters(),getattr(models[1],net).parameters()):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
    write(OUT/'checkpoint_reuse_verification.json',dict(passed=True,paired_weight_samples=1024,
        identical_training_updates=3,bitwise_equal=True,checkpoint_sha256=sha(path)))
    old_status=json.loads((OLD/'formal_walker_seed0_0917/status.json').read_text())
    assert any(j['state']=='completed' and 'IQL_BELLMAN_FB' in j['label'] and j['exit_code']==0 for j in old_status)
    return dict(value=str(path),value_sha256=sha(path),out=str(path.parent.parent),configuration=cfg,
        verification='Same source bytes, XML, replay/loader, IQL cores and training/weight functions; completed seed0 500k .20 log')


def main():
    if (OUT/'queue_plan.json').exists():raise RuntimeError('Existing queue; do not double-launch')
    assert json.loads((OUT/'final_acceptance.json').read_text())['all_passed']
    assert json.loads((OUT/'gpu_smoke.json').read_text())['passed']
    for method in ['TARGET','MIX_FB','CONST','ORACLE','BACKUP']:
        assert (HERE/'results'/('entry_v2_'+method)/'completed.json').is_file(),method
    reuse=verify_reuse();jobs=[]
    order=[(e,m) for e in ENVS for m in ['TARGET','MIX_FB']]+[(e,m) for e in ENVS for m in ['CONST','ORACLE','BACKUP']]
    for env,method in order:
        label=env.split('-')[0]+'_'+method
        job=dict(label=label,env=env,method=method,state='pending',steps=500000,
                 out=str(HERE/'results/training_20261003'/label))
        if env==ENVS[0] and method=='MIX_FB' and reuse:
            job.update(state='reused',**reuse)
        jobs.append(job)
    protected=[ROOT/'train.py',ROOT/'algo/call_algo.py',ROOT/'algo/utils.py',ROOT/'dataset/call_dataset.py']
    protected+=list((ROOT/'algo/offline_offline').glob('iql*.py'))
    protected+=list(HERE.glob('*.py'))
    hashes={str(p):sha(p) for p in protected}
    bench=json.loads((OUT/'gpu_smoke.json').read_text())['methods']
    hours=sum(bench[j['method']]['estimated_500k_training_hours'] for j in jobs if j['state']=='pending')
    write(OUT/'queue_plan.json',dict(created=time.time(),runner_pid=os.getpid(),jobs=jobs,
        estimated_training_hours=hours,estimate_excludes='evaluation, I/O, other environments, load changes',
        protected_sha256=hashes,source_budget=.2,eta=1.,seed=0,max_step=500000,
        user_authorization='2026-10-03: complete this experiment and all stages; no expansion to five seeds',
        protocol='Stages 0/1/2a verified. TARGET and MIX_FB V for all three environments, then CONST/ORACLE/BACKUP seed0. Automatic fixed-V diagnostics and final Last5 report.',
        early_stop='Stop on exceptions, nonfinite models, wrong alignment, or changed protected code; never stop on observed return.'))
    write(OUT/'queue_status.json',jobs);report()
    envvars=dict(os.environ)
    envvars.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    envvars.pop('CUDA_VISIBLE_DEVICES',None)
    for job in jobs:
        if job['state']=='pending':
            changed=[name for name,value in hashes.items() if sha(name)!=value]
            if changed:raise RuntimeError('Protected code changed '+repr(changed))
            job.update(state='running',started=time.time())
            cmd=[sys.executable,'-B','-u',str(HERE/'train_entry.py'),'--env',job['env'],'--method',job['method'],'--out',job['out']]
            print('START',job['label'],flush=True)
            child=None
            try:
                with (OUT/(job['label']+'.console.log')).open('x',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,cwd=str(ROOT),env=envvars,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    job['pid']=child.pid;write(OUT/'queue_status.json',jobs);report()
                    while child.poll() is None:
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:pass
                        job['heartbeat']=time.time()
                        progress=Path(job['out'])/'progress.json'
                        if progress.exists():
                            try:job['progress']=json.loads(progress.read_text())
                            except ValueError:pass
                        write(OUT/'queue_status.json',jobs)
                if child.returncode or not (Path(job['out'])/'completed.json').exists():
                    raise RuntimeError('Training exit '+str(child.returncode))
                job.update(state='completed',ended=time.time(),returncode=0,value=str(Path(job['out'])/'values/V_500000.pt'))
            except BaseException as exc:
                if child and child.poll() is None:child.terminate();child.wait()
                job.update(state='failed',ended=time.time(),error=repr(exc));write(OUT/'queue_status.json',jobs);report();raise
            write(OUT/'queue_status.json',jobs);print('DONE',job['label'],flush=True)
        if job['method'] in ['TARGET','MIX_FB']:
            diagnostic=OUT/'diagnostics'/(job['env']+'_'+job['method']+'_bellman.json')
            if not diagnostic.exists():
                code=run_tool('bellman_diagnostics.py',['--env',job['env'],'--kind',job['method'],'--checkpoint',job['value']],job['label']+'.diagnostic.log')
                if code:raise RuntimeError('Fixed V diagnostic failed '+job['label'])
        if run_tool('verify_saved_diagnostics.py',logname='independent_checks.log'):
            raise RuntimeError('Independent diagnostic verification failed')
        report()
        plot_runtime=Path(r'C:\Users\TXin\.codex\environments\scientific-figures\.venv\Scripts\python.exe')
        with (OUT/'figures.log').open('a',encoding='utf-8') as log:
            rc=subprocess.call([str(plot_runtime),str(HERE/'plot_diagnostics.py')],stdout=log,stderr=subprocess.STDOUT)
        if rc:raise RuntimeError('Figure generation failed')
    assert all(sha(p)==v for p,v in hashes.items())
    write(OUT/'pipeline_completed.json',dict(ended=time.time(),jobs=len(jobs),old_files_unchanged=True))
    report();print('ALL STAGES COMPLETED',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(OUT/'pipeline_error.json',dict(time=time.time(),error=repr(error),traceback=traceback.format_exc()))
        try:report()
        except Exception:pass
        raise
