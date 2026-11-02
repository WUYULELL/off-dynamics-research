# -*- coding: utf-8 -*-
"""New engineering attempt after the second explicit user pause. No old writes."""
import ast, hashlib, json, os, shutil, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'oracle_diag/coverage_oracle_r13_resume2'
ORIGINAL_CODE = ROOT / 'oracle_diag/coverage_oracle_r13'
ORIGINAL = ROOT / 'oracle_diag/results/coverage_oracle_20261021_r13'
PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume2'
OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume3'
DOC = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume3'
PAUSE = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume2/pause_20261023_163222'

def wp(p):
    s = str(Path(p).resolve())
    return s if os.name != 'nt' or s.startswith('\\\\?\\') else '\\\\?\\' + s
def read(p):
    with open(wp(p), encoding='utf-8-sig') as f: return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with open(wp(p),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(wp(p),'x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def source(name,text):
    ast.parse(text)
    with open(HERE/name,'x',encoding='utf-8',newline='\n') as f: f.write(text)
def change(text,old,new):
    assert old in text, old
    return text.replace(old,new)

def main():
    assert not OUT.exists() and not DOC.exists()
    proof=read(PAUSE/'PAUSE_VERIFICATION.json')
    assert proof['overall_saved_state_verified'] and proof['checkpoint_step']==100000
    assert sha(proof['checkpoint'])==proof['checkpoint_sha256']
    for manifest in ('PARTIAL_RUN_MANIFEST.json','COMPLETED_RUN_MANIFEST.json'):
        for p,h in read(PAUSE/manifest)['sha256'].items(): assert sha(p)==h,p
    for n in ('learner.py','audit.py','train.py','decision.py','resume_state.py'):
        shutil.copyfile(wp(BASE/n),wp(HERE/n))
    runtime=(BASE/'runtime.py').read_text(encoding='utf-8')
    runtime=change(runtime,"OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume2'","OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261023_r13_resume3'")
    runtime=change(runtime,"DOC = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume2'","DOC = ROOT / 'outputs/research_idea_protocol_20261023_r13_resume3'")
    runtime=change(runtime,"'R13_resume_G205_100k_then_R201_205_only'","'R13_resume_R203_100k_then_R204_205_only'")
    runtime=change(runtime,"a['new_GPU_jobs'] == 6","a['new_GPU_jobs'] == 3")
    runtime=change(runtime,"assert (job['arm'] == 'G' and job['seed'] == 205) or (job['arm'] == 'R' and job['seed'] in (201, 202, 203, 204, 205))","assert job['arm'] == 'R' and job['seed'] in (203, 204, 205)")
    source('runtime.py',runtime)

    acceptance=(BASE/'acceptance.py').read_text(encoding='utf-8')
    acceptance=change(acceptance,"for change in (dict(arm='T'), dict(arm='U'), dict(arm='C'), dict(seed=101),","for change in (dict(arm='G'), dict(arm='T'), dict(arm='U'), dict(arm='C'), dict(seed=201), dict(seed=101),")
    acceptance=acceptance.replace('205','203')
    acceptance=change(acceptance,"mask = np.load(OUT / 'scores/G_05.npy')","mask = np.load(read(OUT / 'R_mask_receipts.json')['masks']['203']['path'])")
    acceptance=change(acceptance,"        advance_sampler(s, 9000)\n        last = s.hash.hexdigest()\n        assert first == rec['expected_streams'][purpose]['100000']['sha256']\n        assert last == rec['expected_streams'][purpose]['109000']['sha256']\n        streams[purpose] = {'100000': first, '109000': last}","        assert first == rec['expected_streams'][purpose]['100000']['sha256']\n        streams[purpose] = {'100000': first}")
    begin=acceptance.index('    ea = EventAccumulator(')
    end=acceptance.index("    for item in read(OUT / 'reuse_receipts.json')",begin)
    acceptance=acceptance[:begin]+'''    ea = EventAccumulator(str(Path(shortpath(original)) / 'tb'), size_guidance={'scalars': 0}).Reload()
    assert all(not [v for v in ea.Scalars(tag) if v.step > 100000]
               for tag in ('loss/V', 'loss/Q', 'loss/actor'))
    write(OUT / 'RESTORE_ACCEPTANCE_EXPECTATIONS.json', dict(
        checkpoint_sha256=rec['checkpoint_sha256'], checkpoint_step=100000,
        GPU_loss_overlap_available=False,
        reason='User-requested stop immediately after first checkpoint; no old post-checkpoint GPU losses exist',
        GPU_exact_full_state_roundtrip_required=True,
        no_new_GPU_gradient_probe_updates=True))
'''+acceptance[end:]
    acceptance=acceptance.replace('original_109k_target_and_source_streams_match=True','original_100k_target_and_source_streams_match=True')
    acceptance=acceptance.replace('reused14_audit_model_and_evaluation_hashes_verified=True','reused17_audit_model_and_evaluation_hashes_verified=True')
    acceptance=acceptance.replace('GPU_overlap_pending=True, exact_GPU_overlap_comparisons=24,','GPU_loss_overlap_available=False, GPU_exact_full_state_roundtrip_pending=True,')
    acceptance=acceptance.replace("overlap_sha256=sha(OUT / 'OVERLAP_LOSSES.json')","restore_expectations_sha256=sha(OUT / 'RESTORE_ACCEPTANCE_EXPECTATIONS.json')")
    acceptance=acceptance.replace('GPU overlap pending','GPU exact full-state roundtrip pending; no old GPU loss overlap available')
    source('acceptance.py',acceptance)

    pipeline=(BASE/'pipeline.py').read_text(encoding='utf-8')
    pipeline=pipeline.replace("assert sha(OUT / 'OVERLAP_LOSSES.json') == c['overlap_sha256']","assert sha(OUT / 'RESTORE_ACCEPTANCE_EXPECTATIONS.json') == c['restore_expectations_sha256']")
    pipeline=pipeline.replace('== 14','== 17').replace('reused_completed_jobs=14','reused_completed_jobs=17').replace('total_new_GPU_jobs=6','total_new_GPU_jobs=3')
    pipeline=pipeline.replace("script = 'resume_train.py' if job['arm'] == 'G' else 'train.py'","script = 'resume_train.py' if job['seed'] == 203 else 'train.py'")
    pipeline=pipeline.replace("if job['arm'] == 'G':\n                assert read(Path(job['out']) / 'OVERLAP_VERIFIED.json')['passed']","if job['seed'] == 203:\n                assert read(Path(job['out']) / 'GPU_STATE_RESTORED.json')['passed']")
    pipeline=pipeline.replace('reused14=True, resumed_G205_from100k=True','reused17=True, resumed_R203_from100k=True, R203_GPU_loss_overlap_available=False')
    pipeline=pipeline.replace('原14项完整结果复用；G205从100k完整状态续跑、24项旧GPU损失严格相等，R五项从零训练且使用原封存掩码。','原17项完整结果复用；R203从100k完整状态续跑，CPU连续/恢复逐位一致且GPU完整状态精确核对；R203无旧GPU损失重叠窗口，不宣称该项已通过GPU损失重叠验证。R204/205使用原封存掩码从零训练；此前G205的24项严格重叠证明保留。')
    source('pipeline.py',pipeline)
    OUT.mkdir(); DOC.mkdir()
    for n in ('jobs','logs','scores','random_masks'): (OUT/n).mkdir()
    for n in ('protocol.json','PRE_RESULT_COMMITMENT.json','R_mask_receipts.json'):
        shutil.copyfile(wp(ORIGINAL/n),wp(OUT/n))
    for folder in ('scores','random_masks'):
        for f in (ORIGINAL/folder).iterdir():
            if f.is_file(): shutil.copyfile(wp(f),wp(OUT/folder/f.name))
    original_plan=read(ORIGINAL/'training_plan.json')['jobs']
    prior_plan=read(PRIOR/'training_plan.json')['jobs']
    candidates={j['id']:j for j in original_plan}
    candidates.update({j['id']:j for j in prior_plan if (Path(j['out'])/'completed.json').exists()})
    reused=[]
    for name in proof['completed_audited_jobs']:
        j=candidates[name]; p=Path(j['out']); a=read(p/'audit.json'); done=read(p/'completed.json')
        assert a['passed']
        reused.append(dict(job=j,audit=str(p/'audit.json'),audit_sha256=sha(p/'audit.json'),
            completed_sha256=sha(p/'completed.json'),final_model_sha256=sha(done['final_model'])))
    assert len(reused)==17
    jobs=[]
    for s in (203,204,205):
        j=dict(next(j for j in original_plan if j['arm']=='R' and j['seed']==s)); j['out']=str(OUT/'runs'/j['id'])
        jobs.append(j); write(OUT/'jobs'/(j['id']+'.json'),j)
    write(OUT/'training_plan.json',dict(jobs=jobs,new_GPU_jobs=3,original_logical_jobs=20,
        new_updates=1400000,resumed_R203_new_updates=400000,fresh_R204_205_updates=1000000,
        original_plan=str(ORIGINAL/'training_plan.json'),original_plan_sha256=sha(ORIGINAL/'training_plan.json')))
    write(OUT/'reuse_receipts.json',dict(runs=reused,complete_audited=17,no_retraining=True))
    write(OUT/'recovery.json',dict(checkpoint=proof['checkpoint'],checkpoint_sha256=proof['checkpoint_sha256'],
        checkpoint_step=100000,original_run=str(Path(proof['checkpoint']).parent),
        pause_proof=str(PAUSE/'PAUSE_VERIFICATION.json'),pause_proof_sha256=sha(PAUSE/'PAUSE_VERIFICATION.json'),
        prefix_through=100000,old_GPU_loss_overlap_available=False,expected_streams=proof['sampling_stream_reconstruction_verified'],
        expected_GPU='NVIDIA GeForce RTX 2060 SUPER',expected_GPU_uuid='GPU-4893eae9-0a98-dde8-c0ef-bad66a87d788',
        scientific_protocol_changed=False,G205_old24_overlap_evidence=str(PRIOR/'runs/confirm_G_t10_h500000_s205_p05/OVERLAP_VERIFIED.json')))
    write(OUT/'stage_authorization.json',dict(scope='R13_resume_R203_100k_then_R204_205_only',protocol_sha256=sha(OUT/'protocol.json'),
        explicit_user_resume_20261023=True,user_instruction='继续实验',supersedes_latest_user_pause=True,new_GPU_jobs=3,logical_total=20,
        C_authorized=False,D_authorized=False,task2_authorized=False,original_formal_authorized=False,
        estimator_development_authorized=False,extra_T_search_authorized=False,T128_authorized=False,
        no_extra_training_after_stage=True,committed=time.time()))
    protected=dict(read(PRIOR/'protected_manifest.json')['sha256'])
    for root in (PRIOR,BASE,ROOT/'outputs/research_idea_protocol_20261023_r13_resume2'):
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in str(f): protected[str(f)]=sha(f)
    assert all(sha(p)==h for p,h in protected.items())
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,includes_prior4398=True))
    delta={n:dict(original_sha256=sha(ORIGINAL_CODE/n),new_sha256=sha(HERE/n)) for n in ('learner.py','audit.py','train.py','decision.py')}
    assert all(x['original_sha256']==x['new_sha256'] for x in delta.values())
    write(OUT/'engineering_repair.json',dict(delta=delta,old_files_protected=len(protected),no_old_writes=True,
        changes='Paths and three-job scope; R203 original sealed-mask full-state recovery; explicitly unavailable old GPU overlap; exact GPU checkpoint roundtrip before any update',
        unchanged='Scientific loss/evaluation/data/formula/seeds/cosine/budget/decision code and protocol'))
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),queue_script=str(HERE/'pipeline.py'),
        queue_status=str(OUT/'queue_status.json'),training_plan=str(OUT/'training_plan.json'),original=str(ORIGINAL),prior_resume2=str(PRIOR),
        old_queues_must_not_restart=True,resume_step=100000))
    text='''# R13 第二次暂停后的恢复范围（resume3）

用户最新明确指示“继续实验”，撤销此前等待继续的暂停请求，仅恢复原R13剩余任务。原R13、resume1、resume2及R09/R11/R12旧队列绝不重启。

复用已审计完成17项；仅R203从100k完整状态续跑400k至原500k，再R204/R205各500k。共3项GPU作业、140万剩余更新，逻辑任务仍为原20项。五张R mask均只加载原封存掩码，不重新抽签。

完整恢复policy/V/Q/targetQ、三个Adam、原500k cosine及计数；双抽样流重建至100k并核对暂停前SHA。CPU实际检查点的连续/恢复测试逐位一致。R203在检查点后没有旧GPU损失窗口，不能宣称已通过G205的24项GPU重叠测试；本次在首个新更新前逐项精确核对GPU完整恢复状态，且不增加GPU梯度探针或测试训练。

只继承<=100k的原始TB和逐episode评估，旧后续未保存更新不计入指标。learner/audit/fresh-R train/decision逐字节原R13，runtime科学函数AST一致，protocol逐字节原R13。单GPU串行隐藏，审计CPU-only，不停止其他应用；故障先CPU诊断保存证据，不自动重训。

原科学设置及判定不变：T64target，混合64target+64oracle source，系数1，source进V+Q、actor仅target，temp10/H500k；独立评估500000+10*(seed-201)+episode0..9；Last5为460/470/480/490/500k，Final另报。

全部20项后按原三项严格mean>2SE且>=4/5正号（SD ddof1）与完整分支判定，报告六项配对。任一未过结束方向二、转方向一的描述性证据，不追加训练。无C，任何结果不触发估计器；原5分门槛不变。保留结果后修订来源、T温度边界局限，未来方向二须新假设/新数据；不称严格95%、普遍source有害/无效/等效/收益上界，也不与OTDF/TABB直接排名。

C网格、原六组正式、D、第二任务/Hopper、额外T搜索、T128、估计器、桌面归档及任何追加训练均未授权。完成后新目录CPU独立复算全部20项原始TB/逐episode回报、六项配对和分支，核对模型、初始化/完整抽样流、oracle对齐、封存mask、shifted XML/timestep.002/归一化和新旧保护hash，交付中文报告及Canvas结果记录后停用跟进。
'''
    with open(DOC/'RECOVERY_SCOPE.md','x',encoding='utf-8') as f: f.write(text)
    for f in HERE.glob('*.py'): ast.parse(f.read_text(encoding='utf-8'))
    write(OUT/'code_frozen.json',dict(sha256={str(f):sha(f) for f in sorted(HERE.glob('*.py'))},GPU_not_started=True))
    files=[OUT/n for n in ('protocol.json','stage_authorization.json','training_plan.json','recovery.json','reuse_receipts.json','R_mask_receipts.json','PRE_RESULT_COMMITMENT.json')]
    files+=list((OUT/'jobs').iterdir())+list((OUT/'scores').iterdir())+list((OUT/'random_masks').iterdir())
    write(OUT/'dependency_hashes.json',dict(sha256={str(f):sha(f) for f in files}))
    print('Prepared resume3: 17 reused, R203 remainder + R204/R205; protected=%d'%len(protected),flush=True)

if __name__=='__main__': main()
