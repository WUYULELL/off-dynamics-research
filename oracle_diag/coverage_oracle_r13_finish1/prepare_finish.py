# -*- coding: utf-8 -*-
"""Audit-only repair accepted; launch only the original two unstarted R jobs."""
import ast,hashlib,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
BASE=ROOT/'oracle_diag/coverage_oracle_r13_resume3'
PRIOR=ROOT/'oracle_diag/results/coverage_oracle_20261023_r13_resume3'
ORIGINAL=ROOT/'oracle_diag/results/coverage_oracle_20261021_r13'
OUT=ROOT/'oracle_diag/results/coverage_oracle_20261024_r13_finish1'
DOC=ROOT/'outputs/research_idea_protocol_20261024_r13_finish1'
INCIDENT=ROOT/'outputs/research_idea_protocol_20261023_r13_resume3/audit_incident_20261024_022849'
def wp(p):
    s=str(Path(p).resolve()); return s if os.name!='nt' or s.startswith('\\\\?\\') else '\\\\?\\'+s
def read(p):
    with open(wp(p),encoding='utf-8-sig') as f:return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with open(wp(p),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(wp(p),'x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def source(n,t):
    ast.parse(t)
    with open(HERE/n,'x',encoding='utf-8',newline='\n') as f:f.write(t)

def main():
    assert not OUT.exists() and not DOC.exists()
    repair=read(INCIDENT/'AUDIT_REPAIR_RECEIPT.json'); assert repair['passed'] and repair['training_updates']==0
    assert sha(repair['repaired_audit'])==repair['repaired_audit_sha256']
    for n in ('learner.py','audit.py','train.py','decision.py'):shutil.copyfile(wp(BASE/n),wp(HERE/n))
    rt=(BASE/'runtime.py').read_text(encoding='utf-8')
    rt=rt.replace("OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume3'","OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261024_r13_finish1'")
    rt=rt.replace("DOC = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume3'","DOC = ROOT / 'outputs/research_idea_protocol_20261024_r13_finish1'")
    rt=rt.replace('R13_resume_R203_100k_then_R204_205_only','R13_finish_R204_R205_only_after_CPU_audit_repair')
    rt=rt.replace("a['new_GPU_jobs'] == 3","a['new_GPU_jobs'] == 2")
    rt=rt.replace("job['seed'] in (203, 204, 205)","job['seed'] in (204, 205)")
    source('runtime.py',rt)
    pp=(BASE/'pipeline.py').read_text(encoding='utf-8')
    pp=pp.replace('== 17','== 18').replace('reused_completed_jobs=17','reused_completed_jobs=18').replace('total_new_GPU_jobs=3','total_new_GPU_jobs=2')
    pp=pp.replace('reused17=True','reused18=True')
    pp=pp.replace("script = 'resume_train.py' if job['seed'] == 203 else 'train.py'","script = 'train.py'")
    pp=pp.replace('原17项完整结果复用；','原18项完整结果复用；')
    pp=pp.replace("'R13是在看到R12验证后修订、确认结果前固定。", "'R203原100k TensorBoard标量未落盘；暂停前封存10个episode回报重建这一辅助审计点，原始文件不改。49个原TB评估点与50个逐episode记录均核验；主指标460k–500k与Final原TB齐全，结果不受影响。',\n            'R13是在看到R12验证后修订、确认结果前固定。")
    source('pipeline.py',pp)
    OUT.mkdir(); DOC.mkdir()
    for n in ('jobs','logs','scores','random_masks'):(OUT/n).mkdir()
    for n in ('protocol.json','PRE_RESULT_COMMITMENT.json','R_mask_receipts.json'):shutil.copyfile(wp(ORIGINAL/n),wp(OUT/n))
    for folder in ('scores','random_masks'):
        for f in (ORIGINAL/folder).iterdir():
            if f.is_file():shutil.copyfile(wp(f),wp(OUT/folder/f.name))
    reused=list(read(PRIOR/'reuse_receipts.json')['runs'])
    a=read(repair['repaired_audit']); done=read(Path(a['job']['out'])/'completed.json')
    reused.append(dict(job=a['job'],audit=repair['repaired_audit'],audit_sha256=sha(repair['repaired_audit']),
        completed_sha256=sha(Path(a['job']['out'])/'completed.json'),final_model_sha256=sha(done['final_model']),
        audit_repair_receipt=str(INCIDENT/'AUDIT_REPAIR_RECEIPT.json'),no_original_TB_scalar_at100k=True))
    assert len(reused)==18
    for r in reused:
        a=read(r['audit']); p=Path(r['job']['out']); done=read(p/'completed.json')
        assert a['passed'] and sha(r['audit'])==r['audit_sha256']
        assert sha(p/'completed.json')==a['completed_sha256']
        assert sha(p/'evaluations.json')==a['evaluations_sha256']
        assert sha(done['final_model'])==a['final_model_sha256']
    jobs=[]
    for s in (204,205):
        j=dict(next(j for j in read(ORIGINAL/'training_plan.json')['jobs'] if j['arm']=='R' and j['seed']==s))
        old=next(j for j in read(PRIOR/'training_plan.json')['jobs'] if j['arm']=='R' and j['seed']==s)
        assert not Path(old['out']).exists()
        j['out']=str(OUT/'runs'/j['id']); jobs.append(j); write(OUT/'jobs'/(j['id']+'.json'),j)
    write(OUT/'reuse_receipts.json',dict(runs=reused,complete_audited=18,no_retraining=True))
    write(OUT/'training_plan.json',dict(jobs=jobs,new_GPU_jobs=2,original_logical_jobs=20,new_updates=1000000,no_R203_retraining=True))
    write(OUT/'stage_authorization.json',dict(scope='R13_finish_R204_R205_only_after_CPU_audit_repair',protocol_sha256=sha(OUT/'protocol.json'),
        explicit_user_resume_20261023=True,user_instruction='继续实验',audit_failure_only_repair_no_retraining=True,new_GPU_jobs=2,logical_total=20,
        C_authorized=False,D_authorized=False,task2_authorized=False,original_formal_authorized=False,estimator_development_authorized=False,
        extra_T_search_authorized=False,T128_authorized=False,no_extra_training_after_stage=True,committed=time.time()))
    write(OUT/'recovery.json',dict(completed_R203_reused=True,old_GPU_loss_overlap_available=False,
        expected_GPU_uuid='GPU-4893eae9-0a98-dde8-c0ef-bad66a87d788',scientific_protocol_changed=False,
        prior_resume3=str(PRIOR),R203_GPU_state_restored_receipt=str(Path(a['job']['out'])/'GPU_STATE_RESTORED.json'),
        audit_repair_receipt=str(INCIDENT/'AUDIT_REPAIR_RECEIPT.json'),audit_repair_receipt_sha256=sha(INCIDENT/'AUDIT_REPAIR_RECEIPT.json'),
        R203_original100k_TB_scalar_missing=True,R203_missing_point_from_pre_pause_raw_episode_receipt=True,
        Last5_Final_original_TB_points_complete=True))
    protected=dict(read(PRIOR/'protected_manifest.json')['sha256'])
    for root in (PRIOR,BASE,ROOT/'outputs/research_idea_protocol_20261023_r13_resume3'):
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in str(f):protected[str(f)]=sha(f)
    bad=[p for p,h in protected.items() if sha(p)!=h]; assert not bad,bad
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,includes_prior4541=True))
    for n in ('learner.py','audit.py','train.py','decision.py'):
        assert (HERE/n).read_bytes()==(ROOT/'oracle_diag/coverage_oracle_r13'/n).read_bytes()
    tree=ast.parse((ROOT/'oracle_diag/coverage_oracle_r13/runtime.py').read_text())
    newtree=ast.parse(rt)
    for name in ('seed','configure_cpu','env','target_data','source_data','model_config'):
        get=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(get(tree))==ast.dump(get(newtree))
    for s,r in read(OUT/'R_mask_receipts.json')['masks'].items():
        assert sha(r['path'])==r['sha256']==sha(OUT/'random_masks'/('R_%s.npy'%s))
        assert r['draws_of_candidate_masks']==1 and not r['selected_using_returns']
    write(OUT/'engineering_repair.json',dict(reason='Original100k TB scalar absent but sealed raw episodes complete; narrow CPU audit repaired in new evidence path',
        original_files_unchanged=True,failed_audit_preserved=True,old_files_protected=len(protected),training_loss_or_formula_changes=False,
        resumed_completed_run_retrained=False,only_R204_R205_new_GPU_jobs=True))
    write(OUT/'RESTORE_ACCEPTANCE_EXPECTATIONS.json',dict(no_more_checkpoint_resume_required=True,R203_GPU_loss_overlap_available=False,
        R203_audit_repair_sha256=repair['repaired_audit_sha256']))
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),queue_script=str(HERE/'pipeline.py'),
        queue_status=str(OUT/'queue_status.json'),training_plan=str(OUT/'training_plan.json'),all_prior_queues_historical=True))
    with open(DOC/'RECOVERY_SCOPE.md','x',encoding='utf-8') as f:
        f.write('# R13审计修复后仅完成剩余R204/R205\n\n用户“继续实验”授权不变；不是新阶段。原18项完成结果复用，只派发原R204/R205各500k，共100万更新。R203不重训。\n\n原100k TensorBoard标量未落盘，但暂停前封存10个episode原始回报完整，CPU重新归一化仅为该点新增派生审计凭据，不伪称原TB测量、不修改原日志。其余49个TB点和50个逐episode事件核验，Last5(460/470/480/490/500k)及Final原TB全部齐全，模型有限、完整两条32M抽样流和同seed T初始化/target流匹配。保留原失败审计和全部证据。\n\n新代码coverage_oracle_r13_finish1；旧R13/resume1/resume2/resume3及R09/R11/R12队列均不得重启。科学learner/audit/fresh-train/decision逐字节原R13，runtime科学函数AST一致，protocol逐字节原R13；seed、mask、oracle标签、64+64通道、source进V+Q/actor仅target、temp10/500k/cosine、独立评估和所有判定不变。R203有精确GPU状态恢复证明，无旧GPU损失重叠，不得冒用G205的24项证明。\n\n原完整三分支与六项配对均保留，SD ddof1、SE=SD/sqrt5，G-T/G-U/G-R各严格mean>2SE且>=4/5正号，任一未过方向二结束并仅保留方向一描述性证据，任何结果停下。原5分门槛不变，无C、任何结果不触发估计器。保留R12结果后修订来源、T温度上界局限及G-T在[3,7]的主要保留意见。未来方向二新假设/新数据，不改门槛。\n\n单GPU串行隐藏，审计CPU-only，不终止其他应用。故障CPU诊断留证，不自动重训；审计故障只修审计。全部完成后新目录CPU独立复算全部20项原始TB/逐episode回报、Last5/Final、六配对/三判定/分支，明确R203缺失辅助TB点的派生来源，核对恢复状态、模型、XML/timestep.002/归一化、初始化/完整target-source流、oracle对齐、封存G/R masks和新旧保护hash。交付中文报告及Canvas结果记录、不冻结新贡献，删除跟进。\n\n不称严格95%、普遍source有害/无效/等效/收益上界，不直接与OTDF/TABB排名。C网格、原六组正式、D、第二任务/Hopper、额外T搜索、T128、估计器、桌面归档与任何追加训练均未授权。\n')
    write(OUT/'code_frozen.json',dict(sha256={str(f):sha(f) for f in sorted(HERE.glob('*.py'))},GPU_not_started=True))
    files=[OUT/n for n in ('protocol.json','stage_authorization.json','training_plan.json','recovery.json','reuse_receipts.json','R_mask_receipts.json','PRE_RESULT_COMMITMENT.json','RESTORE_ACCEPTANCE_EXPECTATIONS.json')]
    files+=list((OUT/'jobs').iterdir())+list((OUT/'scores').iterdir())+list((OUT/'random_masks').iterdir())
    write(OUT/'dependency_hashes.json',dict(sha256={str(f):sha(f) for f in files}))
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        CPU_split_resume_bitwise_identical=read(PRIOR/'cpu_acceptance.json')['CPU_split_resume_bitwise_identical'],
        prior_R203_CPU_restore_acceptance_reused=True,R203_full_stream_and_model_audit_repaired_passed=True,
        old18_audit_model_and_episode_hashes_verified=True,scientific_code_and_runtime_functions_unchanged=True,
        original_sealed_R_masks_verified=True,prior_old_file_protection_passed=True,protected_old_files=len(protected),
        GPU_gradient_probe_updates=0,new_training_updates=0,environment_steps=0,time=time.time()))
    write(OUT/'CPU_ACCEPTANCE_COMMITMENT.json',dict(cpu_acceptance_sha256=sha(OUT/'cpu_acceptance.json'),
        restore_expectations_sha256=sha(OUT/'RESTORE_ACCEPTANCE_EXPECTATIONS.json'),code_frozen_sha256=sha(OUT/'code_frozen.json'),time=time.time()))
    print('Audit-only repair and CPU acceptance passed:18 reused, only R204/R205;protected=%d'%len(protected),flush=True)

if __name__=='__main__':main()
