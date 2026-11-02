"""One-time CPU preparation; no GPU launch and no old file writes."""
import json, hashlib, shutil, sys, time, copy
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'oracle_diag/coverage_oracle_r11_resume2'
PRIOR=ROOT/'oracle_diag/results/coverage_oracle_20261016_r11_resume2'
OUT=ROOT/'oracle_diag/results/coverage_oracle_20261018_r12'
DOC=ROOT/'outputs/research_idea_protocol_20261018_r12'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open('\\\\?\\'+str(Path(p).resolve()),'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert not OUT.exists() and not DOC.exists()
    delivery=read(PRIOR/'STAGE_DELIVERY_RECEIPT_20261017.json')
    assert sha(delivery['report'])==delivery['report_sha256']
    assert sha(delivery['verification'])==delivery['verification_sha256']
    assert read(delivery['verification'])['passed']
    assert read(PRIOR/'STAGE_COMPLETED.json')['status']=='AWAITING_SEPARATE_U_G_AUTHORIZATION'
    t=read(PRIOR/'T_selected.json')
    assert t['temp']==10 and t['steps']==500000
    OUT.mkdir();DOC.mkdir()
    for name in ('jobs','logs','scores'):(OUT/name).mkdir()
    copies={}
    for name in ('learner.py','train.py','audit.py'):
        shutil.copyfile(OLD/name,HERE/name)
        assert sha(OLD/name)==sha(HERE/name)
        copies[name]=dict(source=str(OLD/name),before_sha256=sha(OLD/name),after_sha256=sha(HERE/name),byte_identical=True)
    runtime=(OLD/'runtime.py').read_text(encoding='utf-8')
    runtime=runtime.replace('"""Isolated R11 runtime. Old experiments are read-only dependencies."""','"""R12 U/G screen only. Frozen R11 science and pre-result interpretation rules."""')
    runtime=runtime.replace("OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261016_r11_resume2'", "OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261018_r12'\nPRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261016_r11_resume2'")
    runtime=runtime.replace("DOC = ROOT / 'outputs/research_idea_protocol_20261017_r11_resume2'", "DOC = ROOT / 'outputs/research_idea_protocol_20261018_r12'")
    start=runtime.index('def check_frozen():');end=runtime.index('def configure_cpu():')
    guards='''def check_frozen():
    a=read(OUT/'stage_authorization.json')
    assert a['scope']=='task1_U3_G9_validation_stop_only'
    assert a['protocol_sha256']==sha(OUT/'protocol.json')
    assert a['U_validation_authorized'] and a['G_validation_authorized']
    assert all(a[k] is False for k in ('C_grid_authorized','formal_authorized','task2_authorized',
        'Hopper_cache_authorized','estimator_development_authorized','extra_T_search_authorized'))
    assert a['interpretation_rules_committed_before_dispatch']
    for path,expected in read(OUT/'code_frozen.json')['sha256'].items():
        assert sha(path)==expected,'Code changed after freeze: '+path
    for path,expected in read(OUT/'dependency_hashes.json')['sha256'].items():
        assert sha(path)==expected,'Frozen R11 dependency changed: '+path

def validate_spec(job):
    assert job['task']=='walker2d-morph' and job['phase']=='screen'
    assert job['arm'] in ('U','G') and job['seed'] in (101,102,103)
    assert job['steps']==500000 and job['temp']==10.
    if job['arm']=='G':assert job['ratio'] in (.05,.2,.5)
    else:assert 'ratio' not in job
    assert OUT in Path(job['out']).resolve().parents
    assert read(OUT/'T_selected.json')['temp']==10 and read(OUT/'T_selected.json')['steps']==500000

def check_job(job,for_launch=True):
    check_frozen();validate_spec(job)
    assert read(OUT/'cpu_acceptance.json')['passed']
    assert read(OUT/'T_smoke_audited.json')['passed']
    if for_launch:
        assert not (OUT/'STAGE_COMPLETED.json').exists(),'Stage ended; new user authorization required'

'''
    runtime=runtime[:start]+guards+runtime[end:]
    with open(HERE/'runtime.py','x',encoding='utf-8',newline='\n') as f:f.write(runtime)
    for name in ('T_selected.json','T_smoke_audited.json'):
        shutil.copyfile(PRIOR/name,OUT/name)
    for name in ('completed.json','component_audit.json','SCORE_AUDIT.md','G_05.npy','G_20.npy','G_50.npy','C_05.npy','C_20.npy','C_50.npy'):
        shutil.copyfile(PRIOR/'scores'/name,OUT/'scores'/name)
        assert sha(PRIOR/'scores'/name)==sha(OUT/'scores'/name)
    p=read(PRIOR/'protocol.json')
    p['version']='R12';p['date']='2026-10-18';p['status']='AUTHORIZED_TASK1_U3_G9_ONLY'
    p.pop('recovery',None);p.pop('engineering_attempt2',None)
    p['execution_authorized']='2026-10-18 explicit user: first task U3/G9 validation; stop after screening; C/formal/task2 not authorized'
    p['stage_scope']=dict(allowed=['U validation seeds101-103','G5/20/50percent validation each seeds101-103'],
        new_long_runs=12,new_smoke_runs=0,reference_retraining=False,diagnosis_resampling=False,
        prohibited=['C grid','formal','task2','Hopper cache','additional T search','T128 control','estimator development','desktop archive'],
        mandatory_stop='after all12 audited and fixed validation stop evaluated, pass or fail')
    p['recommended_defaults']['new_code_path_proposal']=str(HERE)
    p['recommended_defaults']['new_result_path_proposal']=str(OUT)
    p['recommended_defaults']['runtime']['long_training_requires_separate_user_authorization']='U3/G9 authorized now; all later stages require new authorization'
    p['recommended_defaults']['runtime']['report_and_pause_between_stages']='mandatory stop after this U/G screen, regardless of result'
    p['locked_T']=dict(temp=10.,H=500000,seeds=[101,102,103],baseline_receipt=str(PRIOR/'T_selected.json'),
        baseline_sha256=sha(PRIOR/'T_selected.json'),mean=t['mean'],extra_temperature_search=False)
    p['pre_result_interpretation_R12']=dict(committed_before_any_UG_long_result=True,
        formula_weights_grid_unchanged=True,
        p_G_5percent='Original four future formal comparisons remain required in at least2tasks; no automatic development authorization',
        p_G_20_or_50percent='Even if all future four thresholds pass: do not trigger coverage-gap estimator development; conclude only 访问相关性选择在本协议下有效; first reassess novelty against OTDF/DVDF similarity filtering; user separately decides',
        validation_stop_unchanged=True,
        novelty_review_not_executed_now=True,
        T_boundary_limitation='T温度在网格上边界10选出，500k三档均值单调上升，T可能未达该算法族最强；未来G−T在[3,7]为主要保留意见；不追加温度搜索',
        monotonic_T500k_means={'1':37.97317555226203,'3':39.37682605431254,'10':41.634296760004794},
        future_formal_G_minus_T_interval_closed=[3.,7.],
        existing_future_T128_descriptive_rule_preserved=True,
        no_T128_run_authorized_now=True)
    write(OUT/'protocol.json',p);write(DOC/'PROTOCOL_R12.json',p)
    auth=dict(scope='task1_U3_G9_validation_stop_only',protocol_sha256=sha(OUT/'protocol.json'),
        U_validation_authorized=True,G_validation_authorized=True,C_grid_authorized=False,formal_authorized=False,
        task2_authorized=False,Hopper_cache_authorized=False,estimator_development_authorized=False,
        extra_T_search_authorized=False,interpretation_rules_committed_before_dispatch=True,
        user_evidence='2026-10-18：确认R11交付，授权第一任务U3/G9验证，执行止损后停下；启动前写入pG解释触发限制与T网格边界局限，不改公式权重网格。',
        committed_at_unix=time.time())
    write(OUT/'stage_authorization.json',auth)
    jobs=[]
    for arm,ratios in (('U',[None]),('G',[.05,.2,.5])):
        for ratio in ratios:
            for seed in (101,102,103):
                ident='screen_%s_t10_h500000_s%d%s'%(arm,seed,'' if ratio is None else '_p%02d'%round(100*ratio))
                job=dict(id=ident,task='walker2d-morph',arm=arm,phase='screen',seed=seed,temp=10.,steps=500000,
                    out=str(OUT/'runs'/ident))
                if ratio is not None:job['ratio']=ratio
                jobs.append(job);write(OUT/'jobs'/(ident+'.json'),job)
    write(OUT/'training_plan.json',dict(scope=auth['scope'],U_jobs=3,G_jobs=9,total_GPU_jobs=12,
        gradient_updates=6000000,steps_each=500000,temp=10.,seeds=[101,102,103],G_ratios=[.05,.2,.5],
        jobs=jobs,order='U101/102/103 then G5%,20%,50%, each101/102/103',
        stop_rule='selected G mean strictly greater than T and U mean; equality fails',
        ratio_tie_tolerance=1e-8,ratio_tie_rule='larger ratio first',
        C_grid_jobs=0,formal_jobs=0,task2_jobs=0,extra_T_jobs=0,extra_smoke_jobs=0,
        stop_after='U/G validation screening regardless of pass or fail'))
    deps={}
    files=[PRIOR/'T_selected.json',PRIOR/'reference_ready.json',PRIOR/'diagnosis/completed.json',
        PRIOR/'diagnosis/trajectories.npz',PRIOR/'scores/completed.json',PRIOR/'scores/component_audit.json',
        PRIOR/'STAGE_DELIVERY_RECEIPT_20261017.json',Path(delivery['verification']),Path(delivery['report']),
        OUT/'T_selected.json',OUT/'T_smoke_audited.json']
    for r in t['runs']:files += [Path(r['job']['out'])/name for name in ('audit.json','initialization.json','sampling.json')]
    files += list((OUT/'scores').iterdir())
    for f in files:deps[str(f)]=sha(f)
    write(OUT/'dependency_hashes.json',dict(sha256=deps,reference_and_scores_reused_without_retraining=True))
    protected=read(PRIOR/'protected_manifest.json')['sha256']
    for path,expected in protected.items():assert sha(path)==expected,'Historical protection mismatch: '+path
    for oldroot in (PRIOR,OLD,ROOT/'outputs/research_idea_protocol_20261017_r11_resume2'):
        for path in oldroot.rglob('*'):
            if path.is_file() and '__pycache__' not in str(path):protected[str(path)]=sha(path)
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,no_old_files_written=True))
    write(OUT/'engineering_delta.json',dict(science_copies=copies,
        runtime_before_sha256=sha(OLD/'runtime.py'),runtime_after_sha256=sha(HERE/'runtime.py'),
        allowed_changes='runtime output root and phase/authorization guards; dispatcher and validation-only decision/report; no science changes',
        old_paths_read_only=True,new_trainings_started=0))
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),queue_script=str(HERE/'pipeline.py'),
        audit_script=str(HERE/'audit.py'),queue_status=str(OUT/'queue_status.json'),training_plan=str(OUT/'training_plan.json'),
        prior_stage=str(PRIOR),old_queues_must_not_restart=True))
    contract='''# R12 第一任务 U/G 验证执行协议

用户于2026-10-18确认R11交付，并授权本轮12次长验证；本协议在任何U/G长训练结果出现之前写入。当前只有第一任务walker2d-morph。

## 执行与止损

- T固定温度10、H=500k，复用R11种子101–103的对应T基线；不追加调参、不重训参考、不重采诊断、不重跑冒烟。
- U：100% source，seeds101–103，共3次；G：静态保留全部source的5%/20%/50%，各seeds101–103，共9次。
- 每项500k，共600万次更新。单GPU串行隐藏窗口；source标签全部oracle，source进入V+Q，actor只用原target。T64，混合64+64，source系数1.0。公式、k20、1/2权重、标准化、网格、种子与损失保持不变。
- Last-5为460k、470k、480k、490k、500k；Final另报。G按三验证种子均值选比例，1e-8内平局优先较大比例。
- 止损通过仅指选定G均值严格同时超过同种子T与U均值，等号失败。通过/失败均完成本轮后停止。仅为止损，不是正式四项门槛或科学成功结论。

## 启动前新增解释与触发限制（不改变上述筛查）

1. R11审计中20%、50%为rel主导。若p_G=5%，未来仍按原四项规则（G−T、G−C、G−R、G−U，至少两个任务）执行；5%未触发90%也不证明组合效果。若p_G=20%或50%，即使未来四项门槛全部通过，也不触发覆盖缺口估计器开发；仅可写“访问相关性选择在本协议下有效”，先对照OTDF、DVDF相似度筛选重新评估新颖性，再由用户另行决定。本轮筛查通过本身不支持该有效性结论。
2. T的温度在网格上边界10选出，500k三档均值37.97318、39.37683、41.63430单调上升，T可能未达该算法族最强。未来G−T落在[3,7]时须作为主要保留意见报告。不追加温度搜索；原未来正式T128描述性对照规则保留，但本轮不执行该对照。

## 解释与边界

验证种子与比例选择共用，需披露选择偏差；描述性配对SD用ddof1、SE=SD/sqrt3。不将3种子验证当作正式5种子，不把2SE称严格95%。T仅5000数据约8轨迹段，V为学得函数，oracle依赖历史恢复隐藏状态；参考诊断仅一阶固定访问分布。C的V0外推局限保留，C训练不在本轮。

完成后从原始TensorBoard独立复算逐seed/Last-5/Final及止损；核对模型有限性、实际shifted评估与归一化、初始化、完整target抽样流配对、source/oracle行对齐、静态掩码和旧文件保护hash。修复仅在新目录，保留前后hash及失败证据，不自动重训；审计故障只修审计不重训。

本轮禁止C网格、正式阶段、第二任务/Hopper缓存、估计器开发、额外T搜索、T128、D/R长训练、桌面更新。没有任何从止损结果自动派发下一阶段的路径。完成交付后删除跟进。
'''
    with open(DOC/'STAGE_CONTRACT_R12.md','x',encoding='utf-8') as f:f.write(contract)
    write(OUT/'PRE_RESULT_COMMITMENT.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        contract=str(DOC/'STAGE_CONTRACT_R12.md'),contract_sha256=sha(DOC/'STAGE_CONTRACT_R12.md'),
        authorization_sha256=sha(OUT/'stage_authorization.json'),
        any_UG_long_result_exists=False,committed_before_first_dispatch=True,time=time.time()))
    write(OUT/'code_frozen.json',dict(sha256={str(f):sha(f) for f in sorted(HERE.glob('*.py'))},
        science_code_matches_R11=True,pre_result_interpretation_committed=True))
    print('R12 prepared before any GPU job; protected files=%d'%len(protected),flush=True)

if __name__=='__main__':main()
