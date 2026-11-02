"""Create new, isolated V9 stage3 files. Never edit an existing experiment file."""
from pathlib import Path
import ast
import textwrap

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'oracle_diag/v9_hopper_label_headroom/stage3'
DEST.mkdir(exist_ok=False)
old=(ROOT/'oracle_diag/coverage_raw_r14/runtime.py').read_text(encoding='utf-8-sig')
tree=ast.parse(old)
def function(name):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(old.splitlines()[node.lineno-1:node.end_lineno])+'\n'
def put(name,source):
    with (DEST/name).open('x',encoding='utf-8') as f:f.write(textwrap.dedent(source).lstrip())

runtime='''
import os,sys,json,time,uuid,hashlib,math
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261031/stage3'
STAGE2=ROOT/'outputs/v9_hopper_label_headroom_20261030/stage2'
PYTHON=Path('D:/ANACONDA/envs/offdynamics/python.exe')
for p in (ROOT,ROOT/'algo'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='1'
os.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
mu=Path.home()/'.mujoco210/mujoco210'
os.environ['MUJOCO_PY_MUJOCO_PATH']=str(mu)
os.environ['PATH']=str(mu/'bin')+os.pathsep+os.environ.get('PATH','')
'''
runtime+='\n'.join(function(n) for n in ('winpath','shortpath','sha','read','write','fresh','configure_cpu','env','target_data'))
runtime+='''
def protocol():return read(OUT/'protocol.json')
def task_spec(task):return next(v for v in protocol()['tasks'] if v['task']==task)
def seed(purpose,run_seed,task):
    return int.from_bytes(hashlib.sha256(('V9-stage3-20261031|%s|%d|%s'%(task,run_seed,purpose)).encode()).digest()[:4],'little')
def code_hashes():
    return {str(p):sha(p) for p in sorted(list(HERE.glob('*.py'))+[ROOT/'algo/offline_offline/iql.py',ROOT/'algo/offline_offline/iql_dyn.py'])}
def check_frozen():
    a=read(OUT/'authorization.json');assert a['new_jobs']==45 and a['order']==['hopper-morph','walker2d-morph']
    for p,h in read(OUT/'code_frozen.json')['sha256'].items():assert sha(p)==h,p
    for p,h in read(OUT/'input_frozen.json')['sha256'].items():assert sha(p)==h,p
    assert read(OUT/'cpu_acceptance.json')['passed']
    assert sha(OUT/'cpu_acceptance.json')==read(OUT/'CPU_ACCEPTANCE_COMMITMENT.json')['sha256']
def ids(job):return read(OUT/'selections.json')[job['task']]['variants'][job['variant']]['rows']
def sample_hash(job,steps=200000):
    import numpy as np
    rows=np.asarray(ids(job),dtype=np.int64);rng=np.random.RandomState(seed('target_sample',job['seed'],job['task']));h=hashlib.sha256()
    remaining=steps*64
    while remaining:
        n=min(remaining,262144);h.update(rows[rng.randint(len(rows),size=n)].astype('<i8').tobytes());remaining-=n
    return h.hexdigest()
def validate(job):
    check_frozen();assert job in read(OUT/'training_plan.json')['jobs']
    assert job['steps']==200000 and job['beta']==0 and job['batch']==64
def protect(label):
    m=read(OUT/'protected_manifest.json')['sha256'];changed=[p for p,h in m.items() if not Path(winpath(p)).exists() or sha(p)!=h]
    write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(m),changed=changed,time=time.time()))
    assert not changed,changed[:10]
'''
put('runtime.py',runtime)
put('learner.py','''
from runtime import *
import random
import numpy as np
import torch
from offline_offline.iql import Policy
class Actor:
    def __init__(self,job,device):
        self.job=job;self.device=device;self.total_it=0
        s=seed('initialization',job['seed'],job['task']);random.seed(s);np.random.seed(s);torch.manual_seed(s)
        if device!='cpu':torch.cuda.manual_seed_all(s)
        spec=task_spec(job['task'])
        self.policy=Policy(spec['state_dim'],spec['action_dim'],1.,256).to(device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=3e-4)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,200000)
    def step(self,state,action):
        self.total_it+=1
        loss=self.policy.bc_loss(state,action).mean()
        self.optimizer.zero_grad();loss.backward();self.optimizer.step();self.scheduler.step()
        return loss.detach()
    def deterministic_action(self,state):
        with torch.no_grad():
            x=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(1,-1)
            return torch.tanh(self.policy.network(x).chunk(2,dim=1)[0]).squeeze(0).cpu().numpy()
    def digest(self):
        return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in self.policy.state_dict().values())).hexdigest()
    def finite(self):return all(bool(torch.isfinite(p).all()) for p in self.policy.parameters())
    def checkpoint(self):return dict(policy=self.policy.state_dict(),actor_optimizer=self.optimizer.state_dict(),actor_scheduler=self.scheduler.state_dict(),step=self.total_it,job=self.job)
class Sampler:
    def __init__(self,arrays,job,device):
        self.arrays=arrays;self.job=job;self.device=device;self.ids=np.asarray(ids(job),dtype=np.int64)
        self.rng=np.random.RandomState(seed('target_sample',job['seed'],job['task']));self.hash=hashlib.sha256();self.draws=0
    def sample(self):
        ix=self.ids[self.rng.randint(len(self.ids),size=64)];self.hash.update(ix.astype('<i8').tobytes());self.draws+=64
        return tuple(torch.as_tensor(np.asarray(a[ix]),dtype=torch.float32,device=self.device) for a in self.arrays[:2])
''')
train='''
from runtime import *
import argparse,random,traceback
import numpy as np
import torch
from learner import Actor,Sampler
from tensorboardX import SummaryWriter
'''
# The environment rollout and normalization are the same as R14, with stage3's seed offset.
src=(ROOT/'oracle_diag/coverage_raw_r14/train.py').read_text(encoding='utf-8-sig');t=ast.parse(src)
n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
evaluation='\n'.join(src.splitlines()[n.lineno-1:n.end_lineno]).replace("job['seed']-201","job['seed']-301")
train+=evaluation+'\n'+'''
def run(job_path):
    job=read(job_path);validate(job);configure_cpu();assert torch.cuda.is_available()
    assert read(STAGE2/'STAGE_COMPLETED.json')['passed'] and read(STAGE2/'DELIVERY_VERIFIED_STAGE2.json')['passed']
    out=Path(job['out']);fresh(out);write(out/'started.json',dict(pid=os.getpid(),time=time.time(),job_path=str(job_path)))
    write(out/'config.json',dict(job=job,policy_class='offline_offline.iql.Policy',actor_lr=3e-4,hidden=256,batch=64,beta=0,advantage_weight=1,
        policy_loss='mean(IQL.Policy.bc_loss(state,action))',Q_or_V_constructed=False,Q_or_V_optimized=False,source_rows=0,
        cosine_T_max=200000,observation_scaling='raw',reward_scaling=1,protocol_sha256=sha(OUT/'protocol.json'),code_sha256=code_hashes()))
    try:
        arrays=target_data(job['task']);model=Actor(job,'cuda:0');sample=Sampler(arrays,job,'cuda:0')
        write(out/'initialization.json',dict(sha256=model.digest()));e=env(job['task']);writer=SummaryWriter(str(Path(shortpath(out))/'tb'))
        start=time.time();events=[];envsteps=0
        for step in range(1,200001):
            loss=model.step(*sample.sample())
            if step%1000==0:
                assert model.finite() and bool(torch.isfinite(loss))
                write(out/'progress.json',dict(step=step,budget=200000,pid=os.getpid(),elapsed=time.time()-start,finite=True,draws=sample.draws,sampling_sha256=sample.hash.hexdigest(),evaluation_steps=envsteps),replace=True)
                writer.add_scalar('loss/actor',float(loss),step)
            if step%5000==0:
                v=evaluate(model,e,job);v['step']=step;events.append(v);envsteps+=v['steps']
                writer.add_scalar('evaluation/target_score',v['mean'],step);writer.flush();write(out/'evaluations.json',dict(events=events),replace=True)
            if step%50000==0:
                path=out/('model_%d.pt'%step);torch.save(model.checkpoint(),winpath(path))
        writer.close();e.close();path=out/'model_200000.pt'
        write(out/'sampling.json',dict(draws=sample.draws,sampling_sha256=sample.hash.hexdigest(),selected_rows=len(sample.ids)))
        write(out/'completed.json',dict(job=job,steps=200000,finite=model.finite(),final_model=str(path),final_model_sha256=sha(path),model_digest=model.digest(),elapsed=time.time()-start,evaluation_steps=envsteps,ended=time.time()))
    except BaseException:
        write(out/'failed.json',dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);run(p.parse_args().job)
'''
put('train.py',train)
put('metrics.py','''
import math,statistics
def group(values,seeds):
    a=list(values);assert len(a)==len(seeds) and len(a)>=2 and all(math.isfinite(v) for v in a)
    b=sorted(a);f=(len(a)-1)*.25;lo=int(f);hi=math.ceil(f)
    return dict(n=len(a),seeds=seeds,values=a,mean=statistics.mean(a),sample_SD=statistics.stdev(a),worst=min(a),worst_seeds=[s for s,v in zip(seeds,a) if v==min(a)],Q1=b[lo]+(b[hi]-b[lo])*(f-lo))
def welch(a,b):
    return dict(mean_difference=a['mean']-b['mean'],SE=math.sqrt(a['sample_SD']**2/a['n']+b['sample_SD']**2/b['n']),comparison='unpaired Welch standard error')
def gate(H,SE):
    if H>=10 and H>2*SE:return '估分员是瓶颈，存在提升空间'
    if H<5:return '该结构下提升空间不足'
    if 5<=H<10:return '不确定'
    return 'V9未覆盖：H>=10但未超过2SE，停止解释并请用户决定'
''')
put('audit.py','''
from runtime import *
import argparse
import numpy as np
import torch
from learner import Actor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def finite_tree(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
    if isinstance(x,(tuple,list)):return all(finite_tree(v) for v in x)
    return not isinstance(x,float) or math.isfinite(x)
def raw_scores(path,job,every=5000,last=200000):
    events=read(path/'evaluations.json')['events'];spec=task_spec(job['task']);expected=list(range(every,last+1,every))
    tb=EventAccumulator(str(Path(shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
    assert [e['step'] for e in events]==[e.step for e in tb]==expected
    points=[]
    for e,s in zip(events,tb):
        start=500000+10*(job['seed']-(301 if every==5000 else 201))
        assert e['env_seeds']==list(range(start,start+10)) and e['xml_sha256']==spec['xml_sha256'] and e['timestep']==.002
        assert e['actual_shifted_object'] and e['preserve_training_RNG']
        v=(np.asarray(e['raw_returns'],dtype=np.float64)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100
        assert len(v)==10 and np.isfinite(v).all() and abs(v.mean()-s.value)<1e-4
        assert abs(v.mean()-e['mean'])<1e-10;points.append(float(v.mean()))
    return dict(Last5=float(np.mean(points[-5:])),Final=points[-1])
def audit(path):
    configure_cpu();path=Path(path);done=read(path/'completed.json');job=done['job'];validate(job)
    assert done['finite'] and done['steps']==200000 and sha(done['final_model'])==done['final_model_sha256']
    model=torch.load(winpath(done['final_model']),map_location='cpu')
    assert set(model)=={'policy','actor_optimizer','actor_scheduler','step','job'} and finite_tree(model)
    assert model['step']==model['actor_scheduler']['last_epoch']==200000 and model['actor_scheduler']['T_max']==200000
    states=model['actor_optimizer']['state'];assert states and all(float(v['step'])==200000 for v in states.values())
    assert read(path/'initialization.json')['sha256']==Actor(job,'cpu').digest()
    sampling=read(path/'sampling.json');assert sampling['draws']==12800000 and sampling['sampling_sha256']==sample_hash(job)
    config=read(path/'config.json');assert not config['Q_or_V_constructed'] and not config['Q_or_V_optimized'] and config['advantage_weight']==1
    result=dict(passed=True,CPU_only=True,job=job,actor_optimizer_scheduler_finite=True,initialization_and_full_sampling_checked=True,
        final_model_sha256=done['final_model_sha256'],raw_TB_and_episode_returns_checked=True,time=time.time(),**raw_scores(path,job))
    write(path/'audit.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
''')
put('analyze.py','''
from runtime import *
import argparse
from audit import raw_scores
from metrics import group,welch,gate
def analyze(task):
    check_frozen();jobs=[j for j in read(OUT/'training_plan.json')['jobs'] if j['task']==task]
    refs=[v for v in read(OUT/'T_reuse.json')['records'] if v['task']==task]
    assert len(jobs)==(30 if task=='hopper-morph' else 15) and all((Path(j['out'])/'audit.json').exists() for j in jobs)
    records=[];inputs={}
    for j in jobs:
        p=Path(j['out']);a=read(p/'audit.json');assert a['passed'];v=raw_scores(p,j)
        assert abs(a['Last5']-v['Last5'])<1e-8 and abs(a['Final']-v['Final'])<1e-8
        records.append(dict(variant=j['variant'],seed=j['seed'],out=j['out'],**v))
        for f in [p/'evaluations.json',p/'audit.json',p/'completed.json',p/'initialization.json',p/'sampling.json']+list((p/'tb').iterdir()):inputs[str(f)]=sha(f)
    for j in refs:
        p=Path(j['out']);a=read(p/'audit.json');assert a['passed'];v=raw_scores(p,j,10000,500000)
        assert abs(a['Last5']-v['Last5'])<1e-8
        records.append(dict(variant='T',seed=j['seed'],out=j['out'],**v))
        for f in [p/'evaluations.json',p/'audit.json',p/'completed.json']+list((p/'tb').iterdir()):inputs[str(f)]=sha(f)
    order=['BC_all','BC50','BC25'];stats={}
    for metric in ('Last5','Final'):
        groups={}
        for variant in order+['T']:
            v=sorted([r for r in records if r['variant']==variant],key=lambda r:r['seed']);groups[variant]=group([r[metric] for r in v],[r['seed'] for r in v])
        stats[metric]=dict(groups=groups,comparisons_to_T={k:welch(groups[k],groups['T']) for k in order})
    groups=stats['Last5']['groups'];best=max(order,key=lambda k:groups[k]['mean']);ties=[k for k in order if groups[k]['mean']==groups[best]['mean']]
    effect=welch(groups[best],groups['T']);bc=welch(groups['BC_all'],groups['T'])
    verdict=dict(C=best,tied_best=ties,H=effect['mean_difference'],SE_H=effect['SE'],headroom=gate(effect['mean_difference'],effect['SE']),
        T_not_strong=bc['mean_difference']>2*bc['SE'],BC_all_minus_T=bc,selection_bias_disclosed=True,no_new_T_tuning=True)
    result=dict(passed=True,task=task,records=records,statistics=stats,decision=verdict,input_sha256=inputs,
        raw_TB_and_episode_recomputed=True,audit_only_crosscheck=True,time=time.time())
    prefix='HOPPER' if task=='hopper-morph' else 'WALKER'
    write(OUT/(prefix+'_RESULT.json'),result)
    lines=['# V9阶段3 '+task+' BC/%BC结果','',
        '200k actor-only IQL BC，优势权重1，cosine200k；每5k十episode，Last5=180/185/190/195/200k。T复用R14/R13的500k结果，training seed不同，以Welch非配对SE比较。','',
        '|组别|Last5均值|样本SD|最差/seed|Q1|Final均值|Final SD|','|---|---:|---:|---|---:|---:|---:|']
    for k in ['T']+order:
        a=groups[k];b=stats['Final']['groups'][k]
        lines.append('|%s|%.5f|%.5f|%.5f/%s|%.5f|%.5f|%.5f|'%(k,a['mean'],a['sample_SD'],a['worst'],a['worst_seeds'],a['Q1'],b['mean'],b['sample_SD']))
    lines+=['','C=%s；H=%+.5f；Welch SE_H=%.5f。'%(best,verdict['H'],verdict['SE_H']),
        '按V9判定：**%s**。'%verdict['headroom'],'T不是强基线：**%s**。后续须加入BC-all基线：%s。'%('是' if verdict['T_not_strong'] else '否','是' if verdict['T_not_strong'] else '未触发此规则'),
        'T调参未授权，未执行；C为同批种子中选出的最高均值，存在选择偏差，只作V9门槛，不当作精确可达上限或一般因果证明。','']
    if task=='hopper-morph':
        stage2=read(STAGE2/'LABEL_ATTRIBUTION_RESULT.json');o=stage2['statistics']['Last5']['groups']['oracle_1.0']
        od=welch(dict(n=o['n'],mean=o['mean'],sample_SD=o['sample_SD']),groups['BC_all'])
        result['O1_minus_BC_all_descriptive']=od
        lines+=['O1 Last5=%.5f；O1−BC-all=%+.5f，非配对SE=%.5f（描述性比较，不新增判据）。'%(o['mean'],od['mean_difference'],od['SE']),
            'O1的标签修正对照效应保留，是否超越简单模仿以及强基线意义需结合本表解释。','']
        # Separate receipt carries this descriptive comparison without replacing the immutable result.
        write(OUT/'HOPPER_O1_VS_BC.json',od)
    lines+=['逐seed、配对BC变体差、全部原始输入hash和配置见RESULT.json及各run。BC vs T为非配对，不能报告同seed正号数。',
        'n=8（7完整+1EOF尾段）；BC-all保留5000行。BC50/25排名的尾段未入选，n=7或8选出的数据完全相同。无BC10。',
        '无source、无Q/V训练、无T重跑、无额外调参。下一方法阶段4仍须用户确认。','']
    # Report paired differences only between new BC variants sharing the same seeds.
    lines+=['|BC变体对比|配对均值差|差的SE|正号数|','|---|---:|---:|---:|']
    import statistics,math
    for v in ['BC50','BC25']:
        d=[a-b for a,b in zip(groups[v]['values'],groups['BC_all']['values'])]
        lines.append('|%s−BC_all|%+.5f|%.5f|%d/%d|'%(v,statistics.mean(d),statistics.stdev(d)/math.sqrt(len(d)),sum(a>0 for a in d),len(d)))
    report=OUT/(prefix+'_REPORT_ZH.md')
    with open(winpath(report),'x',encoding='utf-8') as f:f.write('\n'.join(lines))
    write(OUT/(prefix+'_ANALYSIS_COMPLETED.json'),dict(passed=True,report=str(report),report_sha256=sha(report),result_sha256=sha(OUT/(prefix+'_RESULT.json')),CPU_only=True,time=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);analyze(p.parse_args().task)
''')
put('pipeline.py','''
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
                self.child('analyze.py',['--task','hopper-morph'],'HOPPER_analysis',True);protect('Hopper_end')
                write(OUT/'HOPPER_PRIORITY_COMPLETED.json',dict(passed=True,jobs=30,Walker_follows=True,time=time.time()))
        self.child('analyze.py',['--task','walker2d-morph'],'WALKER_analysis',True);protect('stage_end')
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
''')
print(DEST)
