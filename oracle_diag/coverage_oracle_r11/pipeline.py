"""R11 single-GPU queue: no screen/formal dispatch path exists."""
from runtime import *
from decision import choose_temperature,choose_horizon
import subprocess
import traceback

class Queue:
    def __init__(self):
        check_frozen()
        assert read(OUT/'cpu_acceptance.json')['passed']
        write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv))
        self.jobs=[]
        self.results={}
        self.reuse=read(OUT/'reuse_receipts.json')
        self.plan()

    def status(self,stage,**kw):
        write(OUT/'queue_status.json',dict(stage=stage,queue_pid=os.getpid(),updated=time.time(),
            new_jobs=len(self.jobs),audited_new_jobs=len(self.results),reused_jobs=2,
            U_validation_authorized=False,G_validation_authorized=False,**kw),replace=True)

    def plan(self):
        write(OUT/'training_plan.json',dict(scope='T12 + reference1 + diagnosis5000 + score audit + five remaining smokes',
            logical_T_jobs=12,reused_T_jobs=1,new_T_jobs=11,reference_T_jobs=1,
            reused_T_smoke=1,new_smoke_jobs=5,new_GPU_job_total=17,
            logical_long_total=13,logical_smoke_total=6,
            U_validation_jobs=0,G_validation_jobs=0,C_grid_jobs=0,formal_jobs=0,task2_jobs=0,
            grid500k=[dict(temp=t,seed=s,steps=500000) for t in (1.,3.,10.) for s in (101,102,103)],
            grid1M=dict(temp='locked_500k_winner',seeds=[101,102,103],steps=1000000,additional_temp_search=False),
            dispatched_new_jobs=self.jobs,reuse=self.reuse,stop_after='five remaining5k smokes'),replace=True)

    def protect(self,label):
        records=read(OUT/'protected_manifest.json')['sha256']
        changed=[p for p,h in records.items() if not os.path.exists(winpath(p)) or sha(p)!=h]
        write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(records),changed=changed))
        assert not changed,'Old files changed'

    def child(self,script,args,name,cpu=False):
        logs=OUT/'logs'
        logs.mkdir(exist_ok=True)
        command=[str(PYTHON),'-B',str(HERE/script)]+[str(a) for a in args]
        envs=os.environ.copy()
        envs['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0'
        envs['PYTHONDONTWRITEBYTECODE']='1'
        with open(logs/(name+'.stdout.log'),'xb') as stdout,open(logs/(name+'.stderr.log'),'xb') as stderr:
            p=subprocess.Popen(command,cwd=str(ROOT),env=envs,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.status(script,active_pid=p.pid,active_command=command,log_prefix=str(logs/name))
            result=p.wait()
        if result:raise RuntimeError('%s exited %d; logs=%s'%(script,result,name))

    def job(self,arm,phase,s,temp,h,ratio=None):
        ident='%s_%s_t%g_h%d_s%d%s'%(phase,arm,temp,h,s,'' if ratio is None else '_p%02d'%round(100*ratio))
        job=dict(id=ident,task='walker2d-morph',arm=arm,phase=phase,seed=s,temp=float(temp),steps=h,
            out=str(OUT/'runs'/ident))
        if ratio is not None:job['ratio']=ratio
        check_job(job)
        file=OUT/'jobs'/(ident+'.json')
        write(file,job)
        self.jobs.append(job)
        self.plan()
        self.child('train.py',['--job',file],ident)
        self.child('audit.py',['--out',job['out']],ident+'_audit',cpu=True)
        result=read(Path(job['out'])/'audit.json')
        assert result['passed']
        self.results[ident]=result
        self.status('job_audited',last_audited=ident)
        return result

    def T500(self,t,s):
        if t==1. and s==101:
            reuse=self.reuse['T500_temp1_seed101']
            assert sha(reuse['audit'])==reuse['audit_sha256']
            return read(reuse['audit'])
        return self.job('T','tune',s,t,500000)

    def main(self):
        self.status('pre_gpu_protection')
        self.protect('before_new_gpu')
        candidates=[]
        for temp in (1.,3.,10.):
            runs=[self.T500(temp,s) for s in (101,102,103)]
            candidates.append(dict(temp=temp,mean=sum(a['Last5'] for a in runs)/3,runs=runs))
        best=choose_temperature(candidates)
        write(OUT/'TEMP_500K_SELECTED.json',dict(temp=best['temp'],mean=best['mean'],runs=best['runs'],
            candidates=candidates,choice_based_on_500k_only=True,no_additional_temperature_search=True))
        long=[self.job('T','tune',s,best['temp'],1000000) for s in (101,102,103)]
        choice=choose_horizon(best['runs'],long,best['temp'])
        write(OUT/'T_selected.json',choice)
        self.protect('T_locked')
        lines=['# R11 第一任务 T 配置锁定','',
            '选定temp=%g，H=%d。'%(choice['temp'],choice['steps']),'',
            '1M只训练500k胜出temp的seed101–103，没有追加温度搜索。','',
            '|seed|500k Last-5|1M Last-5|1M−500k|','|---|---:|---:|---:|']
        for a,b,d in zip(sorted(best['runs'],key=lambda x:x['job']['seed']),
                         sorted(long,key=lambda x:x['job']['seed']),choice['paired_differences']):
            lines.append('|%d|%.5f|%.5f|%.5f|'%(a['job']['seed'],a['Last5'],b['Last5'],d))
        lines += ['', '配对均值=%.5f，样本SD=%.5f，SE=%.5f，2SE=%.5f。'%
            (choice['paired_mean'],choice['sample_SD'],choice['SE'],choice['threshold']),
            '严格超过2SE才采用1M，否则500k；该规则有意偏向较短H。',
            '', '**窗口限定：500k为460k–500k，1M为960k–1M，比较不同训练后期阶段；不是相同训练阶段或相同绝对步数窗口。cosine周期也不同。**',
            '这不证明两档等效，不是严格95%置信结论。',
            '', '参考T、5000步诊断、分数审计和五组冒烟继续执行本轮授权。U3/G9验证尚未授权，程序不能派发。']
        (OUT/'T_LOCKED_R11.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(OUT/'T_LOCKED_READY.json',dict(report=str(OUT/'T_LOCKED_R11.md'),config_sha256=sha(OUT/'T_selected.json'),
            U_validation_authorized=False,G_validation_authorized=False))
        self.status('T_locked',temp=choice['temp'],steps=choice['steps'])
        ref=self.job('T','reference',1001,choice['temp'],choice['steps'])
        done=read(Path(ref['job']['out'])/'completed.json')
        write(OUT/'reference_ready.json',dict(model=done['final_model'],model_sha256=done['final_model_sha256'],
            seed=1001,configuration_from_T_only=True,frozen_before_G_C_R_U=True))
        self.child('selection.py',['--action','diagnosis'],'diagnosis',cpu=True)
        self.child('selection.py',['--action','score'],'score',cpu=True)
        assert read(OUT/'scores/completed.json')['passed']
        smokes=[]
        for arm in ('G','C','R','U','D'):
            smokes.append(self.job(arm,'smoke',7001,choice['temp'],5000,.2 if arm in ('G','C','R') else None))
        write(OUT/'smokes_audited.json',dict(passed=True,reused_T=self.reuse['T_smoke'],five_new=smokes,
            scientific_selection=False,validation_training=False))
        self.protect('stage_end')
        score=read(OUT/'scores/component_audit.json')
        lines=['# R11 T调参、参考诊断与冒烟交付','',
            '**授权范围已完成并停止，等待U/G验证的单独授权。**','',
            'T temp=%g，H=%d。完整T调参12次含复用合格的seed101/temp1/500k；参考T1次。'%
                (choice['temp'],choice['steps']),
            '本轮新增12次长训练＋5次5k冒烟；T冒烟另复用旧合格结果，没有重跑。','',
            'H比较为460k–500k与960k–1M两个不同训练后期窗口；不是同阶段比较。规则有意偏短H，不证明等效。','',
            '固定参考seed1001在真实G/C/R/U回报出现前封存；CPU诊断5000步已记录，未计入任何G/C/R/U训练数据。',
            'source→V+Q、actor只用target；静态G/C掩码封存前完成零新增训练分量审计，公式和权重未搜索。','',
            '## G分量审计摘要']
        for x in score['results']:
            lines.append('- 保留%g%%审计%d行：%s'%(100*x['retention_ratio'],x['audited_n'],x['interpretation']))
        lines += ['', '超过90%主导时，必须写本次G实质单一分量选择，不声称组合效果；未触发也不构成组合效果的因果证明。',
            'C的V0仅5000 target，缺口外推最不可靠；未来G胜C仅可比较本协议中信号可靠性，不推为原则优劣。',
            '', '## 验收与边界',
            '原始TensorBoard、模型有限性、实际shifted对象和归一化、64/64抽样及source/oracle对齐逐作业审计；旧文件保护哈希通过。',
            '**U只有一次5k冒烟，G只有一次5k冒烟；U3验证、G9比例验证均为0次。C网格、正式、第二任务及Hopper缓存均未启动。**',
            '冒烟回报不用于筛选原理或科研成功判断。',
            '', '详细T判定见T_LOCKED_R11.md，分数审计见scores/SCORE_AUDIT.md，逐作业见audit.json。']
        report=OUT/'STAGE_REPORT_R11.md'
        report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(OUT/'STAGE_COMPLETED.json',dict(status='AWAITING_SEPARATE_U_G_AUTHORIZATION',
            report=str(report),T_selected=str(OUT/'T_selected.json'),
            completed_T_jobs=12,reference_jobs=1,reused_jobs=2,new_GPU_jobs=17,
            U_validation_jobs=0,G_validation_jobs=0,all_five_smokes_passed=True,time=time.time()))
        self.status('AWAITING_SEPARATE_U_G_AUTHORIZATION',report=str(report))

if __name__=='__main__':
    q=None
    try:
        q=Queue()
        q.main()
    except BaseException:
        path=OUT/('queue_failure_%d.json'%int(time.time()))
        write(path,dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc(),
            action='stop, preserve evidence, no automatic retraining'))
        if q is not None:q.status('FAILED',failure=str(path))
        raise
