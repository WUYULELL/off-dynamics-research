"""Exactly T/U/G5/R5 x seeds201-205, no next-stage dispatch."""
from runtime import *
from decision import decide
import subprocess,traceback

class Queue:
    def __init__(self):
        check_frozen();assert read(OUT/'cpu_acceptance.json')['passed']
        write(OUT/'queue_started.json',dict(pid=os.getpid(),time=time.time(),command=sys.argv))
        self.jobs=[];self.results=[];self.status('STARTING')
    def status(self,stage,**kw):
        write(OUT/'queue_status.json',dict(stage=stage,queue_pid=os.getpid(),updated=time.time(),
            dispatched_jobs=len(self.jobs),audited_jobs=len(self.results),total_jobs=20,
            C_grid_authorized=False,original_formal_authorized=False,task2_authorized=False,
            estimator_development_authorized=False,**kw),replace=True)
    def protect(self,label):
        records=read(OUT/'protected_manifest.json')['sha256']
        bad=[p for p,h in records.items() if not os.path.exists(winpath(p)) or sha(p)!=h]
        write(OUT/('protected_'+label+'.json'),dict(passed=not bad,count=len(records),changed=bad));assert not bad
    def child(self,script,args,name,cpu=False):
        command=[str(PYTHON),'-B',str(HERE/script)]+[str(a) for a in args]
        envs=os.environ.copy();envs['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';envs['PYTHONDONTWRITEBYTECODE']='1'
        with open(OUT/'logs'/(name+'.stdout.log'),'xb') as stdout,open(OUT/'logs'/(name+'.stderr.log'),'xb') as stderr:
            p=subprocess.Popen(command,cwd=str(ROOT),env=envs,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.status(script,active_pid=p.pid,active_command=command,log_prefix=str(OUT/'logs'/name));rc=p.wait()
        if rc:raise RuntimeError('%s exited %d; no automatic retraining; %s'%(script,rc,name))
    def main(self):
        self.protect('before_gpu')
        for job in read(OUT/'training_plan.json')['jobs']:
            check_job(job);self.jobs.append(job)
            write(OUT/'dispatch_progress.json',dict(jobs=self.jobs),replace=True)
            self.child('train.py',['--job',OUT/'jobs'/(job['id']+'.json')],job['id'])
            self.child('audit.py',['--out',job['out']],job['id']+'_audit',cpu=True)
            a=read(Path(job['out'])/'audit.json');assert a['passed']
            if job['arm']!='T':
                t=next(r for r in self.results if r['job']['arm']=='T' and r['job']['seed']==job['seed'])
                assert a['sampling']['target_stream_sha256']==t['sampling']['target_stream_sha256']
                assert read(Path(job['out'])/'initialization.json')==read(Path(t['job']['out'])/'initialization.json')
            self.results.append(a);self.status('job_audited',last_audited=job['id'])
        groups={arm:[a for a in self.results if a['job']['arm']==arm] for arm in ('T','U','G','R')}
        result=decide(groups);result['group_means']={arm:sum(x['Last5'] for x in rs)/5 for arm,rs in groups.items()}
        result['runs']=groups;result['protocol_sha256']=sha(OUT/'protocol.json')
        write(OUT/'CONFIRMATION_RESULT.json',result);self.protect('stage_end')
        lines=['# R13 独立种子确认报告','', '**'+result['conclusion']+'。已停止。**','',
            '本方案在看到R12验证结果之后修订，在任何本阶段结果出现前固定。',
            '本轮不含C，原四项开发条件不可能满足；任何结果都不触发估计器开发，原5分门槛未调低。','',
            '|组别|五种子Last-5均值|','|---|---:|']
        lines+=['|%s|%.5f|'%(arm,result['group_means'][arm]) for arm in ('T','U','G','R')]
        lines+=['','|配对比较|均值|样本SD|SE|正号|>2SE且≥4/5|','|---|---:|---:|---:|---:|---|']
        for name,s in result['paired_Last5'].items():
            lines.append('|%s|%.5f|%.5f|%.5f|%d/5|%s|'%(name,s['mean'],s['sample_SD'],s['SE'],s['positive'],s['passed']))
        lines+=['','三项全过才记小幅稳定挑选效应；任一未过结束方向二、转方向一，不追加训练。']+result['notes']
        lines+=['','无论结果如何不追加训练；未来重启方向二须新假设、新数据，不得根据本轮结果重解释门槛。',
            'T温度在网格上界且三档单调上升；G−T若在[3,7]，列为主要保留意见。T128/D未授权，本轮不追加。',
            'oracle和固定参考访问分布仅为诊断，不能声称可部署方法优于论文；小样本工程判据非严格95%。',
            '原始TB/模型/抽样/初始化/行对齐逐项审计；交付前须独立复算Last-5、Final、六项配对与规则。']
        report=OUT/'CONFIRMATION_REPORT_R13.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(OUT/'STAGE_COMPLETED.json',dict(status='AWAITING_USER_DECISION',report=str(report),
            completed_jobs=20,all_three_passed=result['all_three_passed'],estimator_development_triggered=False,
            no_additional_training=True,time=time.time()))
        self.status('AWAITING_USER_DECISION',report=str(report))

if __name__=='__main__':
    q=None
    try:q=Queue();q.main()
    except BaseException:
        path=OUT/('queue_failure_%d.json'%int(time.time()))
        write(path,dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc(),action='CPU diagnosis and report; no automatic retraining'))
        if q:q.status('FAILED',failure=str(path))
        raise
