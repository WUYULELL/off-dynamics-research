"""Exactly twelve fresh U/G validation jobs, single GPU, then stop in every outcome."""
from runtime import *
from decision import decide
import subprocess
import traceback

class Queue:
    def __init__(self):
        check_frozen()
        assert read(OUT/'cpu_acceptance.json')['passed']
        assert not (OUT/'queue_started.json').exists(),'Do not restart an existing attempt'
        write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv))
        self.jobs=[];self.results=[]
        self.status('STARTING')

    def status(self,stage,**kw):
        write(OUT/'queue_status.json',dict(stage=stage,queue_pid=os.getpid(),updated=time.time(),
            dispatched_jobs=len(self.jobs),audited_jobs=len(self.results),total_jobs=12,
            U_validation_authorized=True,G_validation_authorized=True,
            C_grid_authorized=False,formal_authorized=False,task2_authorized=False,**kw),replace=True)

    def protect(self,label):
        records=read(OUT/'protected_manifest.json')['sha256']
        changed=[p for p,h in records.items() if not os.path.exists(winpath(p)) or sha(p)!=h]
        write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(records),changed=changed))
        assert not changed,'Protected old files changed'

    def child(self,script,args,name,cpu=False):
        command=[str(PYTHON),'-B',str(HERE/script)]+[str(a) for a in args]
        envs=os.environ.copy();envs['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';envs['PYTHONDONTWRITEBYTECODE']='1'
        with open(OUT/'logs'/(name+'.stdout.log'),'xb') as out,open(OUT/'logs'/(name+'.stderr.log'),'xb') as err:
            p=subprocess.Popen(command,cwd=str(ROOT),env=envs,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.status(script,active_pid=p.pid,active_command=command,log_prefix=str(OUT/'logs'/name))
            rc=p.wait()
        if rc:raise RuntimeError('%s exited %d; no automatic rerun; logs=%s'%(script,rc,name))

    def run_job(self,job):
        check_job(job)
        file=OUT/'jobs'/(job['id']+'.json')
        assert read(file)==job
        self.jobs.append(job)
        write(OUT/'dispatch_progress.json',dict(jobs=self.jobs),replace=True)
        self.child('train.py',['--job',file],job['id'])
        self.child('audit.py',['--out',job['out']],job['id']+'_audit',cpu=True)
        result=read(Path(job['out'])/'audit.json');assert result['passed']
        baseline=next(a for a in read(OUT/'T_selected.json')['runs'] if a['job']['seed']==job['seed'])
        assert result['sampling']['target_stream_sha256']==baseline['sampling']['target_stream_sha256']
        assert read(Path(job['out'])/'initialization.json')['sha256']==read(Path(baseline['job']['out'])/'initialization.json')['sha256']
        self.results.append(result);self.status('job_audited',last_audited=job['id'])

    def main(self):
        self.protect('before_gpu')
        for job in read(OUT/'training_plan.json')['jobs']:self.run_job(job)
        assert len(self.results)==12
        selected=read(OUT/'T_selected.json')
        u=[a for a in self.results if a['job']['arm']=='U']
        groups=[]
        for p in (.05,.2,.5):
            runs=[a for a in self.results if a['job']['arm']=='G' and a['job']['ratio']==p]
            groups.append(dict(ratio=p,mean=sum(a['Last5'] for a in runs)/3,runs=runs))
        result=decide(selected['runs'],u,groups)
        result.update(U_runs=u,T_runs=selected['runs'],audited_new_long_jobs=12,
            all_target_streams_and_initializations_paired=True,
            R11_score_audit_path=str(PRIOR/'scores/SCORE_AUDIT.md'),
            protocol_sha256=sha(OUT/'protocol.json'),new_authorization_required_for_any_next_stage=True)
        write(OUT/'STOP_RESULT.json',result)
        self.protect('stop')
        lines=['# R12 第一任务 U/G 验证止损报告','',
            '**%s。已停止，等待用户决定。**'%result['conclusion'],'',
            'T温度10，H500k；验证种子101–103；Last-5为460k–500k。','',
            '|组别|三种子Last-5均值|','|---|---:|','|T|%.5f|'%result['T_mean'],'|U|%.5f|'%result['U_mean']]
        lines += ['|G %g%%|%.5f|'%(100*g['ratio'],g['mean']) for g in groups]
        lines += ['', '选定G比例%g%%；比例按同一验证种子选出，存在选择偏差，不是独立正式证据。'%(100*result['G_ratio']),
            '止损仅要求均值严格同时超过T/U；没有应用正式四项门槛，也不触发任何开发或后续派发。',
            '', '## 启动前固定的解释限制',result['prospective_interpretation']['restriction'],
            result['T_boundary_limitation']['text'],
            '5%未触发90%主导也不构成组合效应证明；20%/50%按R11审计必须称访问相关性主导。',
            '', '## 配对描述统计（仅验证）']
        for name in ('G_minus_T','G_minus_U','U_minus_T'):
            s=result[name];lines.append('- %s：差%s，均值%.5f，样本SD%.5f，SE%.5f，正号%d/3。'%(name,s['differences'],s['mean'],s['sample_SD'],s['SE'],s['positive']))
        lines += ['', 'T64与混合64+64，总batch不同。未来正式G−T在[3,7]时原T128描述性对照要求仍保留，本轮不运行。',
            '本轮没有C网格、正式训练、第二任务/Hopper缓存、参考重训、诊断重采样、额外T调参、估计器开发或桌面更新。',
            '验证止损失败只记为验证阶段未见信号；不是收益上界，不证明source无效/有害，也不证明方案等效。',
            '原始TB、模型有限性、shifted评估与归一化、抽样/初始化配对及保护哈希逐项核验。交付前还需独立复算。']
        report=OUT/'STOP_REPORT_R12.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(OUT/'STAGE_COMPLETED.json',dict(status='AWAITING_USER_DECISION',report=str(report),
            U_completed=3,G_completed=9,C_grid_jobs=0,formal_jobs=0,task2_jobs=0,
            passed_validation_stop=result['passed_validation_stop'],time=time.time(),
            interpretation_restrictions_precommitted=True,no_automatic_next_stage=True))
        self.status('AWAITING_USER_DECISION',report=str(report))

if __name__=='__main__':
    q=None
    try:
        q=Queue();q.main()
    except BaseException:
        path=OUT/('queue_failure_%d.json'%int(time.time()))
        write(path,dict(time=time.time(),pid=os.getpid(),traceback=traceback.format_exc(),
            action='stop; preserve evidence; CPU diagnosis only; no automatic retraining'))
        if q:q.status('FAILED',failure=str(path))
        raise
