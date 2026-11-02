"""Single hidden queue. Hard stop after task-1 U/G screening, pass or fail."""
from runtime import *
import subprocess
import traceback

class Queue:
    def __init__(self):
        check_frozen()
        assert read(OUT/'cpu_acceptance.json')['passed']
        write(OUT/'queue_started.json',dict(pid=os.getpid(),started=time.time(),command=sys.argv))
        self.jobs=[]
        self.audits={}

    def status(self,stage,**kwargs):
        write(OUT/'queue_status.json',dict(stage=stage,queue_pid=os.getpid(),updated=time.time(),
            audited_jobs=len(self.audits),scheduled_jobs=len(self.jobs),**kwargs),replace=True)

    def child(self,script,args,log,cpu=False):
        log=Path(log)
        log.parent.mkdir(parents=True,exist_ok=True)
        envs=os.environ.copy()
        envs['PYTHONDONTWRITEBYTECODE']='1'
        envs['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0'
        command=[str(PYTHON),'-B',str(HERE/script)]+[str(x) for x in args]
        with open(str(log)+'.stdout.log','xb') as stdout, open(str(log)+'.stderr.log','xb') as stderr:
            p=subprocess.Popen(command,cwd=str(ROOT),env=envs,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.status(script,active_pid=p.pid,active_command=command,log_prefix=str(log))
            code=p.wait()
        if code:
            raise RuntimeError('%s exited %d; preserve logs: %s' % (script,code,log))

    def job(self,arm,phase,s,temp,h,ratio=None):
        ident='%s_%s_t%g_h%d_s%d%s' % (phase,arm,temp,h,s,'' if ratio is None else '_p%02d'%round(ratio*100))
        job=dict(id=ident,task='walker2d-morph',arm=arm,phase=phase,seed=s,temp=float(temp),steps=h,
            out=str(OUT/'runs'/ident))
        if ratio is not None: job['ratio']=ratio
        check_job(job)
        file=OUT/'jobs'/(ident+'.json')
        write(file,job)
        self.jobs.append(job)
        write(OUT/'training_plan.json',dict(scope='task1 stages1-3 only',jobs=self.jobs,
            C_grid=False,formal=False,task2=False,stop_after='U/G screening'),replace=True)
        self.child('train.py',['--job',file],OUT/'logs'/ident)
        self.child('audit.py',['--out',job['out']],OUT/'logs'/(ident+'_audit'),cpu=True)
        a=read(Path(job['out'])/'audit.json')
        assert a['passed']
        self.audits[ident]=a
        self.status('job_audited',last_audited=ident)
        return a

    def protect(self,name):
        records=read(OUT/'protected_manifest_final.json')['sha256']
        changed=[path for path,expected in records.items() if not os.path.exists(winpath(path)) or sha(path)!=expected]
        write(OUT/('protected_'+name+'.json'),dict(passed=not changed,count=len(records),changed=changed))
        assert not changed,'Protected old files changed'

    def main(self):
        self.protect('before_gpu')
        smoke_t=self.job('T','smoke',7001,3.,5000)
        write(OUT/'T_smoke_audited.json',dict(passed=True,audit=smoke_t,
            warning='smoke return excluded from scientific selection'))
        candidates=[]
        # Three 500k temperatures finish before any 1M jobs; same full grid, no adaptive omission.
        for h in (500000,1000000):
            for temp in (1.,3.,10.):
                runs=[self.job('T','tune',s,temp,h) for s in (101,102,103)]
                mean=sum(a['Last5'] for a in runs)/3
                candidates.append(dict(temp=temp,steps=h,mean=mean,runs=runs))
        best_mean=max(c['mean'] for c in candidates)
        tied=[c for c in candidates if abs(c['mean']-best_mean)<=1e-8]
        best=min(tied,key=lambda c:(c['steps'],{3.:0,1.:1,10.:2}[c['temp']]))
        write(OUT/'T_selected.json',dict(temp=best['temp'],steps=best['steps'],mean=best['mean'],
            runs=best['runs'],candidates=candidates,criterion='three-seed Last-5 mean',
            horizon_trained_independently=True,selection_not_formal_evidence=True))
        self.protect('T_tuned')
        self.status('T_tuning_complete',selected_temp=best['temp'],selected_steps=best['steps'])
        temp,h=best['temp'],best['steps']
        ref=self.job('T','reference',1001,temp,h)
        done=read(Path(ref['job']['out'])/'completed.json')
        write(OUT/'reference_ready.json',dict(model=done['final_model'],model_sha256=done['final_model_sha256'],
            seed=1001,configuration_from_T_only=True,frozen_before_G_C_R_U=True))
        self.child('selection.py',['--action','diagnosis'],OUT/'logs'/'diagnosis',cpu=True)
        self.child('selection.py',['--action','score'],OUT/'logs'/'score',cpu=True)
        score=read(OUT/'scores/completed.json')
        assert score['passed'] and read(OUT/'scores/component_audit.json')['zero_additional_training']
        # All source arms share the same .20 smoke ratio; the real G screen still tests all three.
        rest=[]
        for arm in ('G','C','R','U','D'):
            rest.append(self.job(arm,'smoke',7001,temp,5000,.2 if arm in ('G','C','R') else None))
        write(OUT/'smokes_audited.json',dict(passed=True,T=smoke_t,others=rest,
            score_audit_sha256=sha(OUT/'scores/component_audit.json'),scientific_selection=False))
        self.protect('all_smokes')
        self.status('reference_diagnosis_scoring_smokes_complete')
        u=[self.job('U','screen',s,temp,h) for s in (101,102,103)]
        gs=[]
        for p in (.05,.2,.5):
            runs=[self.job('G','screen',s,temp,h,p) for s in (101,102,103)]
            gs.append(dict(ratio=p,mean=sum(a['Last5'] for a in runs)/3,runs=runs))
        gm=max(x['mean'] for x in gs)
        winner=max([x for x in gs if abs(x['mean']-gm)<=1e-8],key=lambda x:x['ratio'])
        # Verify full target sample stream AND initialization, not just first batch.
        for a in u+[v for group in gs for v in group['runs']]:
            t=next(r for r in best['runs'] if r['job']['seed']==a['job']['seed'])
            assert a['sampling']['target_stream_sha256']==t['sampling']['target_stream_sha256']
            def init(z):return read(Path(z['job']['out'])/'initialization.json')['sha256']
            assert init(a)==init(t)
        self.protect('stop')
        um=sum(a['Last5'] for a in u)/3
        passed=winner['mean']>best['mean'] and winner['mean']>um
        result=dict(passed_validation_stop=passed,status='AWAITING_USER_DECISION',
            T_mean=best['mean'],U_mean=um,G_best_mean=winner['mean'],G_ratio=winner['ratio'],
            G_candidates=gs,U_runs=u,T_selected_path=str(OUT/'T_selected.json'),
            conclusion='验证阶段见信号' if passed else '验证阶段未见信号',
            scientific_conclusion=False,formal_gate_applied=False,
            C_grid_started=False,formal_started=False,task2_started=False,
            audited_long_runs=31,audited_smokes=6,all_target_streams_and_initializations_paired=True,
            score_audit_path=str(OUT/'scores/SCORE_AUDIT.md'))
        # Independent paired summaries, explicitly exploratory validation only.
        import numpy as np
        def paired(lhs,rhs):
            lhs=sorted(lhs,key=lambda a:a['job']['seed']);rhs=sorted(rhs,key=lambda a:a['job']['seed'])
            d=np.array([a['Last5']-b['Last5'] for a,b in zip(lhs,rhs)])
            return dict(differences=d.tolist(),mean=float(d.mean()),sample_SD=float(d.std(ddof=1)),
                SE=float(d.std(ddof=1)/np.sqrt(len(d))),positive=int((d>0).sum()),n=len(d),
                descriptive_validation_only=True)
        result['G_minus_T']=paired(winner['runs'],best['runs'])
        result['G_minus_U']=paired(winner['runs'],u)
        write(OUT/'STOP_RESULT.json',result)
        lines=['# 第一任务 U/G 验证止损报告 R09','',
            '**%s。队列已停止，等待用户决定；未运行C网格、正式阶段或第二任务。**'%result['conclusion'],'',
            'T配置：temp=%g，H=%d；三个验证种子101–103，独立于未来正式种子。'%(temp,h),'',
            '|组别|三seed Last-5均值|','|---|---:|','|T|%.4f|'%best['mean'],'|U|%.4f|'%um]
        lines += ['|G %g%%|%.4f|'%(100*x['ratio'],x['mean']) for x in gs]
        lines += ['', '最佳G比例为%g%%。止损仅要求均值严格同时超过T/U，不是正式2SE/4正/5分判据。'%(100*winner['ratio']),
            '', '原始TensorBoard、模型有限性、完整target抽样流、初始化和旧文件保护哈希均逐项核验。',
            '', 'G分量主导情况见scores/SCORE_AUDIT.md；若触发90%必须以单分量选择描述。',
            'C尚未比例验证；固定V0缺口外推不可靠，不能据后续G/C比较宣称原则优劣。',
            '本结果不构成收益上界，不证明source无效/有害，不是严格95%置信结论。']
        (OUT/'STOP_REPORT_R09.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(OUT/'queue_completed.json',dict(stopped_as_authorized=True,report=str(OUT/'STOP_REPORT_R09.md'),
            time=time.time(),awaiting_separate_user_decision=True))
        self.status('AWAITING_USER_DECISION',report=str(OUT/'STOP_REPORT_R09.md'))

if __name__=='__main__':
    q=None
    try:
        q=Queue()
        q.main()
    except BaseException:
        error=dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc(),
            action='stop and retain all evidence; no automatic rerun')
        destination=OUT/('queue_failure_'+str(int(time.time()))+'.json')
        write(destination,error)
        if q is not None:q.status('FAILED',failure=str(destination))
        raise
