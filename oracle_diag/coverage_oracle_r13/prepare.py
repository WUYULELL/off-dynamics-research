# -*- coding: utf-8 -*-
"""Prepare a post-R12, pre-confirmation revision, with no GPU execution."""
from pathlib import Path
import json,hashlib,shutil,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'oracle_diag/coverage_oracle_r12';PRIOR=ROOT/'oracle_diag/results/coverage_oracle_20261018_r12'
OUT=ROOT/'oracle_diag/results/coverage_oracle_20261021_r13'
DOC=ROOT/'outputs/research_idea_protocol_20261021_r13'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open('\\\\?\\'+str(Path(p).resolve()),'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def seed(purpose,s):return int.from_bytes(hashlib.sha256(('20261016|walker2d-morph|%d|%s'%(s,purpose)).encode()).digest()[:4],'little')
def main():
    assert not OUT.exists() and not DOC.exists()
    receipt=read(PRIOR/'DELIVERY_RECEIPT_20261019.json')
    assert all(sha(p)==v for p,v in receipt['sha256'].items())
    assert read(receipt['verification'])['passed']
    OUT.mkdir();DOC.mkdir()
    for n in ('jobs','logs','scores','random_masks'):(OUT/n).mkdir()
    copies={}
    for name in ('learner.py','audit.py'):
        shutil.copyfile(OLD/name,HERE/name)
        copies[name]=dict(source=str(OLD/name),before_sha256=sha(OLD/name),after_sha256=sha(HERE/name),byte_identical=True)
    text=(OLD/'train.py').read_text(encoding='utf-8')
    old="    base = 300000 if phase in ('reference', 'smoke') else 400000\n    index = 0 if phase in ('reference', 'smoke') else (101, 102, 103).index(run_seed)"
    new="    assert phase == 'confirmation'\n    base = 500000\n    index = (201, 202, 203, 204, 205).index(run_seed)"
    assert old in text;text=text.replace(old,new)
    old="                ids = np.sort(np.random.RandomState(seed('R_mask', job['seed'])).choice(len(raw_ids),\n                    size=int(np.floor(job['ratio'] * len(raw_ids))), replace=False))"
    new="                mask_record = read(OUT / 'R_mask_receipts.json')['masks'][str(job['seed'])]\n                assert sha(mask_record['path']) == mask_record['sha256']\n                ids = np.load(mask_record['path'])"
    assert old in text;text=text.replace(old,new)
    with open(HERE/'train.py','x',encoding='utf-8',newline='\n') as f:f.write(text)
    copies['train.py']=dict(source=str(OLD/'train.py'),before_sha256=sha(OLD/'train.py'),after_sha256=sha(HERE/'train.py'),
        changes=['confirmation evaluation seeds500000+10*(seed-201)+episode','load one precommitted uniform R mask instead of drawing the identical mask at launch'],
        training_update_loop_and_science_unchanged=True)
    rt=(OLD/'runtime.py').read_text(encoding='utf-8')
    rt=rt.replace('coverage_oracle_20261018_r12','coverage_oracle_20261021_r13').replace('research_idea_protocol_20261018_r12','research_idea_protocol_20261021_r13')
    rt=rt.replace("PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261016_r11_resume2'", "PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261018_r12'")
    start,end=rt.index('def check_frozen():'),rt.index('def configure_cpu():')
    guards='''def check_frozen():
    a=read(OUT/'stage_authorization.json')
    assert a['scope']=='task1_confirmation_T_U_G5_R5_seeds201_205_only'
    assert a['protocol_sha256']==sha(OUT/'protocol.json')
    assert all(a[k] is False for k in ('C_authorized','D_authorized','task2_authorized','original_formal_authorized',
        'estimator_development_authorized','extra_T_search_authorized','T128_authorized'))
    for file,h in read(OUT/'code_frozen.json')['sha256'].items():assert sha(file)==h,'Frozen code changed: '+file
    for file,h in read(OUT/'dependency_hashes.json')['sha256'].items():assert sha(file)==h,'Frozen dependency changed: '+file

def validate_spec(job):
    assert job['phase']=='confirmation' and job['task']=='walker2d-morph'
    assert job['arm'] in ('T','U','G','R') and job['seed'] in (201,202,203,204,205)
    assert job['temp']==10 and job['steps']==500000
    if job['arm'] in ('G','R'):assert job['ratio']==.05
    else:assert 'ratio' not in job
    assert OUT in Path(job['out']).resolve().parents

def check_job(job,for_launch=True):
    check_frozen();validate_spec(job)
    assert job in read(OUT/'training_plan.json')['jobs'],'Not one of the20authorized jobs'
    assert read(OUT/'cpu_acceptance.json')['passed']
    if for_launch:assert not (OUT/'STAGE_COMPLETED.json').exists(),'Stage complete; no further training'

'''
    rt=rt[:start]+guards+rt[end:]
    with open(HERE/'runtime.py','x',encoding='utf-8',newline='\n') as f:f.write(rt)
    for n in ('G_05.npy','completed.json','component_audit.json','SCORE_AUDIT.md'):
        shutil.copyfile(PRIOR/'scores'/n,OUT/'scores'/n)
    random_masks={}
    for s in range(201,206):
        ids=np.sort(np.random.RandomState(seed('R_mask',s)).choice(999322,size=49966,replace=False))
        file=OUT/'random_masks'/('R_%d.npy'%s);np.save(file,ids)
        random_masks[str(s)]=dict(path=str(file),sha256=sha(file),seed=seed('R_mask',s),n=len(ids),
            draws_of_candidate_masks=1,uniform_without_replacement=True,selected_using_returns=False)
    write(OUT/'R_mask_receipts.json',dict(root_seed=20261016,purpose='R_mask',masks=random_masks,
        generated_before_any_confirmation_results=True,one_mask_per_seed_no_retry_no_selection=True))
    p=read(PRIOR/'protocol.json');p['version']='R13-confirmation';p['date']='2026-10-21'
    p['status']='AUTHORIZED_CONFIRMATION_ONLY';p['revised_after_observing_R12']=True
    p['execution_authorized']='User2026-10-21: T/U/G5/R5 x seeds201-205,500k,temp10; after wall-clock estimate; any failed comparison ends direction2; no further training'
    p['stage_scope']=dict(arms=['T','U','G','R'],seeds=list(range(201,206)),steps=500000,temp=10,
        jobs=20,gradient_updates=10000000,G_ratio=.05,R_ratio=.05,
        old_T_or_UG_runs_reused_as_confirmation=False,
        paused=['Cgrid','original six-arm formal','D','task2','Hoppercache','T128','extraTsearch','estimator','desktoparchive'])
    p['confirmation_decision']=dict(primary='Last5 at460/470/480/490/500k',sample_SD_ddof=1,SE='sampleSD/sqrt5',
        contrasts=['G-T','G-U','G-R'],each='strict mean>2SE AND positive>=4/5',
        all_three_pass='存在小幅稳定的挑选效应; stop for user decision',
        any_fail='end direction2, switch to direction1 descriptive evidence; no additional training',
        G_R_fails='未建立按覆盖缺口挑选优于同数量随机子集的证据',
        G_R_pass_but_other_fails='覆盖缺口选择优于同数量随机子集，但未建立相对只用target或相对全部保留的稳定收益',
        G_T_fails_caveat='G-R优势可能来自随机小子集损失与高频重复抽样，不能解释为source选择带来正收益',
        G_U_fails_caveat='未建立挑选优于全部保留的证据，不称等效',
        descriptive_contrasts=['U-T','R-T','R-U'],report_Final_separately=True,
        development_trigger=False,C_absent_so_original_four_conditions_impossible=True,
        original_minimum5_point_development_threshold_unchanged=True,
        no_training_followup_in_any_outcome=True,
        future_direction2_restart='new hypothesis and new data; no threshold reinterpretation from this stage')
    p['revision_interpretation']=dict(post_R12_revision_not_original_preregistration=True,
        conservative_in_development_decision_and_claim_scope=True,
        not_claim_all_new_criteria_are_stricter='Small-effect confirmation is a new narrower question, not original>=5point method-development success',
        provided_probability_example='User simulation: true effect3,SD3,~7%; conditional illustration, not estimated posterior development probability')
    p['recommended_defaults']['new_code_path_proposal']=str(HERE);p['recommended_defaults']['new_result_path_proposal']=str(OUT)
    p['recommended_defaults']['runtime']['long_training_requires_separate_user_authorization']='current20only; no extra stage'
    p['recommended_defaults']['runtime']['report_and_pause_between_stages']='stop after20 pass or fail'
    write(OUT/'protocol.json',p);write(DOC/'PROTOCOL_R13.json',p)
    auth=dict(scope='task1_confirmation_T_U_G5_R5_seeds201_205_only',protocol_sha256=sha(OUT/'protocol.json'),
        C_authorized=False,D_authorized=False,task2_authorized=False,original_formal_authorized=False,
        estimator_development_authorized=False,extra_T_search_authorized=False,T128_authorized=False,
        confirmation20_authorized=True,wall_clock_estimate_communicated_before_start='45-60hours',
        user_confirmed_any_failed_contrast_ends_direction2=True,committed=time.time())
    write(OUT/'stage_authorization.json',auth)
    jobs=[]
    for arm in ('T','U','G','R'):
        for s in range(201,206):
            ident='confirm_%s_t10_h500000_s%d%s'%(arm,s,'_p05' if arm in ('G','R') else '')
            j=dict(id=ident,task='walker2d-morph',arm=arm,phase='confirmation',seed=s,temp=10.,steps=500000,out=str(OUT/'runs'/ident))
            if arm in ('G','R'):j['ratio']=.05
            jobs.append(j);write(OUT/'jobs'/(ident+'.json'),j)
    write(OUT/'training_plan.json',dict(jobs=jobs,total_GPU_jobs=20,total_updates=10000000,
        group_order=['T','U','G5','R5'],seeds=list(range(201,206)),eval_seeds='500000+10*(run_seed-201)+episode0..9',
        Last5=[460000,470000,480000,490000,500000],source_rows=999322,G_and_R_rows=49966,
        expected_source_draws_each=32000000,average_reuses_G_R=32000000/49966,average_reuses_U=32000000/999322,
        stop_after='all20 audited, evaluate all branches, then stop regardless of outcomes'))
    oldplan=read(PRIOR/'training_plan.json')
    mixed=[read(Path(j['out'])/'completed.json')['elapsed']/3600 for j in oldplan['jobs']]
    ts=read(PRIOR/'T_selected.json')['runs'];th=[read(Path(a['job']['out'])/'completed.json')['elapsed']/3600 for a in ts]
    budget=dict(measured_T_hours=th,measured_mixed_hours=mixed,
        central_T5_hours=5*np.mean(th),central_mixed15_hours=15*np.mean(mixed),
        central_training_hours=5*np.mean(th)+15*np.mean(mixed),
        quoted_wall_clock_hours=[45,60],includes='20training+CPUverification+report+speedvariation',
        no_guaranteed_completion_time=True,other_GPU_load_or_interruption_can_extend=True)
    write(DOC/'BUDGET_ESTIMATE.json',budget)
    protected=read(PRIOR/'protected_manifest.json')['sha256']
    assert all(sha(path)==h for path,h in protected.items())
    for root in (PRIOR,OLD,ROOT/'outputs/research_idea_protocol_20261018_r12'):
        for file in root.rglob('*'):
            if file.is_file() and '__pycache__' not in str(file):protected[str(file)]=sha(file)
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,no_old_writes=True))
    files=[OUT/'training_plan.json',OUT/'R_mask_receipts.json',PRIOR/'DELIVERY_RECEIPT_20261019.json',Path(receipt['verification'])]
    files+=list((OUT/'random_masks').iterdir())+list((OUT/'scores').iterdir())+list((OUT/'jobs').iterdir())
    write(OUT/'dependency_hashes.json',dict(sha256={str(f):sha(f) for f in files}))
    write(OUT/'engineering_delta.json',dict(science_copies=copies,runtime_before=sha(OLD/'runtime.py'),runtime_after=sha(HERE/'runtime.py'),
        unchanged='losses,optimizer,schedule,samplingseedderivation,oraclelabels,channels,Gformula',
        changed='fresh roots,authorization guard,independent eval namespace,presealed R masks,confirmationdecision',new_training_updates=0))
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),queue_script=str(HERE/'pipeline.py'),
        audit_script=str(HERE/'audit.py'),queue_status=str(OUT/'queue_status.json'),training_plan=str(OUT/'training_plan.json'),
        prior_stage=str(PRIOR),old_queues_must_not_restart=True))
    contract='''# R13 独立种子确认阶段：结果后修订、确认前固定

## 修订来源与时间

用户2026-10-21确认R12交付后提出本修订。此时已看到R12三比例择优、同验证种子选择与报告的G5%−T=+1.28008。故这是**看到验证结果之后的研究路径修订**，不是原方案的事前成功标准。修订在任何确认阶段长训练启动前写入。

保守性限定在收缩研究范围、开发决策和可声称的结论：原≥5分开发门槛不变，本轮不含C、原四项条件不可能满足，任何结果均不触发估计器开发。“小幅稳定的挑选效应”是新增且更窄的独立确认问题，不冒充原≥5分方法开发成功；不声称新规则的每个数值条件都比原规则严格。

## Task Contract（用户已提供并确认的执行设置）

- 仅walker2d-morph，T/U/G5%/R5%各5次，seeds201–205，每项500k、temp10，20次共1000万更新；全部从0训练，不复用R12验证种子结果。
- T每步64target；U/G/R为64target+64source，source系数1，source→V+Q，actor只用原target。oracle标签与R11/R12相同。算法、损失、学习率、周期、抽样seed派生不变。
- G固定原49966/999322条静态mask；R每个seed只生成一次相同大小的均匀无放回子集，不按回报挑选、不重新抽签。五张R mask在任何确认训练结果前封存，生成次数和hash入JSON。
- 原target5000行，固定参考seed1001、5000步诊断与G分数复用；不重训参考、不重新访问target诊断、不改分数/权重/网格。
- 独立确认种子201–205；按原正式评估命名空间500000+10*(seed−201)+episode(0–9)，与验证400000、诊断200000隔离。实际shifted XML/timestep.002和归一化不变。
- Last-5主指标固定460k–500k五点，先seed内平均再配对。Final另报，不反向改主判据。

## 启动前固定的完整结果树

G−T、G−U、G−R各自均须配对均值严格>2SE且≥4/5正号；SD用ddof1，SE=SD/sqrt5，等号失败。只在全部20项完成后评定，不按中途回报改规则或提前挑停。

1. 三项全部通过：记“存在小幅稳定的挑选效应”，停下由用户决定后续。
2. G−R未通过：记“未建立按覆盖缺口挑选优于同数量随机子集的证据”，方向二结束，转方向一，保留描述性证据，不追加训练。
3. G−R通过，但G−T或G−U任一未通过：记“覆盖缺口选择优于同数量随机子集，但未建立相对只用target（或相对全部保留）的稳定收益”，方向二结束，转方向一，不追加训练。
   - G−T未过：注明G相对R的优势可能来自随机小子集自身损失，例如相同更新预算下高频重复抽样，不能解释为source选择带来正收益；同时报告R−T、R−U描述性统计。
   - G−U未过：注明未建立“挑选优于全部保留”的证据，不写为等效。

所有分支均报告U−T、R−T、R−U五种子配对统计和逐seed值。R/G平均每行约640.44次source抽取，U约32.02次，这是曝光数量描述，不据此单独认定失败机制。

## 研究边界与停止

本阶段没有C，原四项开发门槛不可能满足；原5分门槛不变，不能事后调低。任何结果不触发估计器开发。日后重启方向二须作为新假设、使用新数据检验，不能依据本阶段结果重新解释门槛。转方向一只表示保留证据与研究叙事转向，不是新的训练授权。

暂停原C比例网格、第一任务六组正式、D和第二任务；没有Hopper缓存、T128、额外温度搜索、论文方法新复现、桌面归档或其他GPU作业。当前G−T若落[3,7]，T温度在上界且三档单调上升、可能非算法族最强须作主要保留意见；本轮不因此追加T128。

这是单任务、固定标签oracle+固定参考访问分布下的确认。即使三项通过也不证明覆盖机制或组合效果，不是严格95%结论，不直接与无oracle的OTDF/TABB排名。未通过是预设研究决策所需证据未建立，不证明普遍无效、等效或所有source有害。四组本轮独立训练种子并不等于新数据集。

## 预算、验收与运行规则

已向用户说明预计墙钟45–60小时：T5约7.5h，混合15约35h，余量用于CPU验收、审计、报告和速度波动；其他GPU负载或中断会延长。精确来源见BUDGET_ESTIMATE.json。用户约7%模拟仅作所假设真实效应3/SD3下的条件例子，不能当作本项目开发成功的后验概率。

启动前CPU检查科学代码差异、20项权限、结果树、独立评估seed映射及R mask。保留旧文件hash，后台隐藏窗口、单GPU串行、不终止其他应用。故障先CPU诊断并保存证据，不自动重训；审计失败只修审计，不重训成功作业。完成后从原始TB/逐episode回报独立复算Last-5、Final、六项配对/SD/SE/正号与分支，核对模型有限、XML/归一化、完整抽样流、初始化、oracle行对齐、mask和新旧保护hash。停下交付并删除跟进。
'''
    with open(DOC/'STAGE_CONTRACT_R13.md','x',encoding='utf-8') as f:f.write(contract)
    canvas='''# Working Idea Canvas：R13

- VERIFIED：R12 G5%−T=+1.28008，三比例择优，同种子选择与报告，不能作为独立确认。
- PROVIDED：收缩为T/U/G5/R5、201–205、500k/temp10的20项；三比较任一未过即结束方向二，不追加训练。
- PROVIDED：R−T、R−U及U−T描述性统计必须保留；任何结果不触发估计器，原≥5门槛不变。
- INFERRED：与随机同数量子集的独立比较可区分数量/曝光效应与本协议下的选择收益，但不单独证明覆盖机制。
- PROVISIONAL：G5%的小幅收益是否在独立种子上稳定，尚无确认结果。
- UNKNOWN：未来方向一的具体论文贡献与新增数据来源，本轮不冻结、不执行。
- Checkpoint记录：用户本轮文字直接确认Task Contract与完整失败分支；不重问已确认设置。假设/贡献仍受证据约束，本轮不得冻结为可部署新方法。
'''
    with open(DOC/'WORKING_IDEA_CANVAS.md','x',encoding='utf-8') as f:f.write(canvas)
    write(OUT/'PRE_RESULT_COMMITMENT.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        contract=str(DOC/'STAGE_CONTRACT_R13.md'),contract_sha256=sha(DOC/'STAGE_CONTRACT_R13.md'),
        authorization_sha256=sha(OUT/'stage_authorization.json'),R_masks_receipt_sha256=sha(OUT/'R_mask_receipts.json'),
        revised_after_R12_results=True,confirmation_results_exist=False,committed_before_dispatch=True,time=time.time()))
    files=list(HERE.glob('*.py'))+[ROOT/'algo/offline_offline'/name for name in ('iql.py','iql_dyn.py','iql_unc.py')]
    write(OUT/'code_frozen.json',dict(sha256={str(f):sha(f) for f in files},science_update_loop_unchanged=True))
    print('Prepared R13 before GPU; protected=%d; central training%.2fh'%(len(protected),budget['central_training_hours']),flush=True)

if __name__=='__main__':main()
