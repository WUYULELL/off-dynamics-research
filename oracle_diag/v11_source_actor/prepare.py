# -*- coding: utf-8 -*-
"""Create new manifests only; no training, optimizer updates, environment or CUDA."""
from common import *
import ast
import difflib
import shutil

INSTRUCTION=Path('C:/Users/TXin/Desktop/Claude命令/给Code的指令_v11_源数据动作进入策略的上限检验.md')

def text_file(path,value):
    with open(r.winpath(path),'x',encoding='utf-8') as f:f.write(value)

def main():
    assert not OUT.exists() and not STAGE1.exists(),'Never overwrite prepared/failed/completed artifacts'
    OUT.mkdir(parents=True);STAGE1.mkdir(parents=True);(OUT/'logs').mkdir();(OUT/'runs').mkdir();(STAGE1/'runs').mkdir()
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    # Import-only static check: resolve new audit/report modules, without initializing a learner.
    import audit,report,verify,stage1_report,stage1_verify,torch
    assert Path(audit.__file__).resolve()==HERE/'audit.py'
    assert not torch.cuda.is_initialized()
    shutil.copyfile(str(INSTRUCTION),str(OUT/'INSTRUCTION_V11_ORIGINAL.md'))
    snapshot=OUT/'O1_SCIENCE_SNAPSHOT';snapshot.mkdir()
    for name in ('runtime.py','learner.py','train.py'):
        shutil.copyfile(str(O1_CODE/name),str(snapshot/name))
    cfg=r.model_config(10,500000,201,TASK,1.)
    correction='用户更正：v11的−log π作废；沿用O1实际exp(10A).clamp(max=100)优势加权动作MSE；源动作同类MSE并入。'
    authorization=dict(scope='V11_Stage0_then_Hopper_Stage1_then_stop',
        source='本聊天用户明确要求：Walker BC后启动阶段0；最新指令阶段0完成不用停下，直接阶段1。',
        earlier_stop_after_stage0_superseded=True,stage0_authorized=True,stage1_authorized=True,
        stage2_authorized=False,V9_stage4_authorized=False,T_tuning_authorized=False,
        no_automatic_retraining=True,no_existing_file_edits=True,correction=correction,time=time.time())
    r.write(OUT/'authorization.json',authorization)
    spec=r.task_spec(TASK)
    cache=r.read(r.CACHE_RECEIPT)
    config=dict(scope=authorization['scope'],stage1_authorized=True,stage2_authorized=False,
        actor_loss='advantage_weighted_action_MSE',correction=correction,learner=cfg,
        target_actor_reduction='mean_over_action_dimensions_then_mean_over_64_target_samples',
        source_all_reduction='mean_over_action_dimensions_then_mean_over_64_source_samples',
        source_top_reduction='mean_over_action_dimensions_then_mean_over_selected_source_samples; none =>0; per v11 table',
        source_top_selection='top16_of64 by pre-V advantage, then A>0; ties: earlier original batch position',
        actor_beta=10.,actor_weight_max=100.,source_actor_w=1.,source_critic_w=1.,
        divide_VQ_or_actor_loss_by_1plusw=False,target_batch=64,source_batch=64,
        advantage='min(target_Q1,target_Q2)-V, both before this step V update',
        resample_source_for_actor=False,observations='raw',reward_scale=1.,
        sampling='original O1 independent uniform-with-replacement target/source RNG streams',
        random_seed_namespace='20261016|hopper-morph|trainingseed|initialization/target_sample/source_sample',
        stage0_seed=201,stage1_seeds=list(range(201,211)),cosine_T_max=500000,
        stage0_trials=[dict(id=i,variant=v,actor_w=w,steps=n) for i,v,w,n in TRIALS],
        stage0_original_O1_reference_updates=10000,stage0_new_updates=50000,
        evaluation_every=10000,episodes_per_evaluation=10,max_episode_steps=1000,timestep=.002,
        evaluation_seed='500000+10*(trainingseed-201)+episode[0..9]',preserve_training_RNG=True,
        Last5_steps=[460000,470000,480000,490000,500000],Final_auxiliary=True,
        source_labels='accepted V9 Stage1 Hopper oracle target-dynamics next_state/reward/not_done',
        task_spec=spec,cache_receipt=str(r.CACHE_RECEIPT),cache_receipt_sha256=r.sha(r.CACHE_RECEIPT),
        cache_arrays={k:v for k,v in cache['cache_arrays'].items()},
        diagnostic_recording='every update: six raw64-vector columns(target_A,source_A,target_MSE,source_MSE,target_weight,selected_source_weight), scalar JSONL; 1000-step CPU curves',
        diagnostic_disk_estimate_GB=[20,40],no_result_dependent_parameter_or_seed_selection=True,
        original_instruction_path=str(INSTRUCTION),original_instruction_sha256=r.sha(INSTRUCTION))
    r.write(OUT/'config.json',config)
    jobs=[dict(id=('SA_all' if v=='SA-all' else 'SA_top')+'_s'+str(s),variant=v,seed=s,steps=500000,
               out=str(STAGE1/'runs'/(('SA_all' if v=='SA-all' else 'SA_top')+'_s'+str(s))))
          for v in ('SA-all','SA-top') for s in range(201,211)]
    r.write(STAGE1/'training_plan.json',dict(jobs=jobs,new_jobs=20,total_updates=10000000,order='SA-all then SA-top',
        stage0_trial_outputs_not_reused=True,stop_after_Stage1=True))
    references=[]
    for item in r.read(r.OUT/'reuse_receipts.json')['runs']:
        if item['label']=='T' or (item['label']=='raw' and item['w']==1.):
            references.append(dict(label='T' if item['label']=='T' else 'raw1',seed=item['seed'],out=item['out'],
                audit=item['audit'],every=10000,last=500000,BC=False))
    for job in r.read(r.OUT/'training_plan.json')['jobs']:
        references.append(dict(label='O1' if job['source_w']==1. else 'O.25',seed=job['seed'],out=job['out'],
            audit=str(Path(job['out'])/'audit.json'),every=10000,last=500000,BC=False))
    hopper=r.read(ROOT/'outputs/v9_hopper_label_headroom_20261101/hopper_stage3_correction/HOPPER_RESULT.json')
    for item in hopper['records']:
        if item['variant'] in ('BC_all','BC50'):
            references.append(dict(label='BC-all' if item['variant']=='BC_all' else 'BC50',seed=item['seed'],out=item['out'],
                audit=str(Path(item['out'])/'audit.json'),every=5000,last=200000,BC=True))
    assert len(references)==60
    for item in references:
        assert r.read(item['audit'])['passed']
        item['audit_sha256']=r.sha(item['audit'])
    r.write(STAGE1/'reference_receipts.json',dict(records=references,
        references_are_readonly=True,BC50_selected_from_prior_three_BC_variants=True,BC_seed_pairing=False))
    rules=dict(primary='Last5',BC_reference='V9 Hopper BC50 seeds301-310 (54.049437037.. prior descriptive value)',
        BC_actual_values_recomputed_from_raw=True,
        condition1='M-BC50 >=5 AND M-BC50 >2*WelchSE',
        condition2='paired M-O1 mean >2*SE AND positive_count>=7/10',
        positive='condition1 AND condition2',negative='M-BC50 <2 OR paired M-O1 mean <=0',other='uncertain',
        comparisons_paired=['T','O1','raw1'],descriptive_only=['O.25','BC-all'],
        SD='ddof1',Q1='linear percentile25',all_seeds_retained=True,no_bootstrap=True,
        no_extra_experiments=True,stage2_unapproved=True)
    r.write(STAGE1/'protocol.json',rules)
    historic=r.read(r.OUT/'EXECUTION_SUMMARY.json')['individual_training_seconds']
    mean=sum(historic.values())/len(historic)
    r.write(OUT/'BUDGET_ESTIMATE.json',dict(historical_O1_s201_seconds=historic['hopp_O1_s201'],
        historical_20_O1_O025_mean_seconds=mean,historical_20_total_hours=sum(historic.values())/3600.,
        Stage0_minutes_estimate=[20,35],Stage1_pretrial_hours_estimate=[45,60],
        Stage1_20jobs_short_trial_measured_budget_will_be_written_before_launch=True,
        CPU_audit_overhead_and_load_uncertain=True,diagnostic_disk_estimate_GB=[20,40],
        estimate_reported_before_queue_launch=True,automatic_Stage1_explicitly_authorized=True))
    # A readable science delta, preserving the original source snapshot byte-for-byte.
    original=(O1_CODE/'learner.py').read_text(encoding='utf-8')
    changed=(HERE/'actor.py').read_text(encoding='utf-8')
    text_file(OUT/'O1_V11_ACTOR_DIFF.patch',''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
        fromfile='unchanged_O1_learner.py',tofile='V11_actor.py')))
    text_file(OUT/'RUNNING_HANDOFF.md','''# V11：Walker BC结束后，阶段0通过直接接续Hopper阶段1

用户最新指令优先：−log π作废，采用O1实际的exp(10A)截断100的优势加权动作MSE；目标批均值+w_a×源项。SA-top仍按v11表格对入选样本求均值，批次前16且A>0，其余权重0，无入选源项0。配置、原文快照、diff及最新authorization.json均已记录。

唯一入口oracle_diag/v11_source_actor/pipeline.py。新队列先CPU等待，不初始化CUDA/环境/真实数据优化；只有V9 Walker15项全部完成、DELIVERY_VERIFIED通过、原38944队列及其训练退出后才运行V11。每次核对launch1/真实命令行、创建时间与父子关系防PID复用，不重复启动，不终止其他应用，不改任何旧文件。

Walker原文档outputs/v9_hopper_label_headroom_20261101/walker_stage3；原队列维持原设置与15项范围。完成可单独通知Walker，不暂停已授权V11接续。通知去重写V11目录WALKER_DELIVERY_NOTIFIED.json，不改旧冻结目录。

阶段0：CPU8次合成learner更新（24Adam）、0真实数据优化/环境步、无CUDA；w_a0 seed201 10k新实现+同步原O1 10k逐步损失及完整模型/Adam/cosine比较；SA-all和SA-top各seed201 20k。总60k GPU learner更新（180kAdam），评估50episode。cosine始终500k，复用原采样/初始化、oracle缓存、XML/归一化。三审计与原始向量代数核验全部通过后独立verify.py，输出STAGE0_REPORT_ZH.md/DELIVERY_VERIFIED/STAGE0_COMPLETED；实测预算MEASURED_BUDGET先落盘，再直接阶段1，用户最新要求不在阶段0停下等确认。

阶段1：SA-all201–210后SA-top201–210，20个全新500k/temp10共1000万更新，单GPU串行隐藏；不把短试跑当正式实验。原64target+64source、source oracle仅替换critic标签，actor只学原s/a；优势pre-V，V/Q完全O1。每10k十episode；Last5固定460/470/480/490/500k，Final辅助；保留训练RNG。每步原始优势/MSE/权重和JSONL诊断保留，CPU逐项审计四网络/三Adam/cosine有限、初始化/双流、原始TB/episode和策略代数。预计诊断数据20–40GB；历史单项约2.1h，新增源actor与记录开销暂估20项45–60h，以阶段0单模型20k实测外推更新，不拿双模型parity时间当单模型速度。

全部20项后从新20+旧60=80原始TB/episode重算，不用audit分数作计算输入；参考T/raw1=R14，O1/O.25=V9阶段2，BC-all/BC50=V9阶段3Hopper。BC训练seed301–310与方法201–210不同用Welch；T/O1/raw1同seed配对。V11规则在stage1/protocol.json，严格>=5、>2SE、>=7正号；负判据<2或paired O1<=0，其余不确定，不能改阈值。逐组Last5/Final均值样本SD/最差与seed/线性Q1，描述均值差与诊断曲线完整报告；无新增bootstrap。BC50同批最高选择偏差披露，不宣称精确95%或完美critic。

最终stage1/STAGE1_RESULT.json、STAGE1_REPORT_ZH.md、ANALYSIS_COMPLETED、独立DELIVERY_VERIFIED、STAGE1_COMPLETED/AWAITING_USER_DECISION后退出。阶段2Walker上限、V9阶段4、T调参、估计器和额外训练均未授权。故障CPU诊断留证hash，不自动重训/恢复/覆盖失败或完成产物。

跟进沿用v9-walker-3，不新建重复跟进；健康等待/训练保持安静，Walker完成可通知，阶段0正常完成按用户要求不暂停不单独催确认；最终阶段1完成、故障、重大变化或需用户处理才通知。Walker通知与最终通知在本V11输出去重。断网不影响本地隐藏队列，重启/休眠会中断；不得因聊天未响应停队列。最终中文交付及凭据后删除跟进。
''')
    dependencies={str(p):r.sha(p) for p in HERE.glob('*.py')}
    dependencies.update(r.code_hashes())
    dependencies.update({str(p):r.sha(p) for p in (OUT/'config.json',OUT/'authorization.json',OUT/'INSTRUCTION_V11_ORIGINAL.md',
        OUT/'BUDGET_ESTIMATE.json',STAGE1/'training_plan.json',STAGE1/'reference_receipts.json',STAGE1/'protocol.json')})
    # Data, XML, normalization and accepted cache hashes frozen before any V11 result.
    for key in ('source_dataset','target_dataset','xml'):dependencies[spec[key]]=spec[key+'_sha256']
    dependencies[str(r.CACHE_RECEIPT)]=r.sha(r.CACHE_RECEIPT)
    for item in cache['cache_arrays'].values():dependencies[item['path']]=item['sha256']
    r.write(OUT/'FROZEN.json',dict(sha256=dependencies,time=time.time()))
    r.write(OUT/'PRE_RESULT_COMMITMENT.json',dict(config_sha256=r.sha(OUT/'config.json'),
        frozen_sha256=r.sha(OUT/'FROZEN.json'),authorization_sha256=r.sha(OUT/'authorization.json'),
        protocol_sha256=r.sha(STAGE1/'protocol.json'),stage1_20jobs=True,stage2_authorized=False,
        no_V11_optimization_or_environment_steps_yet=True,CUDA_initialized=False,time=time.time()))
    r.write(OUT/'STATIC_ACCEPTANCE.json',dict(passed=True,all_python_AST_parsed=True,
        new_module_resolution_verified=True,new_optimizer_updates=0,environment_steps=0,CUDA_initialized=False,
        files_created_only_in_new_V11_directories=True,time=time.time()))
    guard()
    print(json.dumps(dict(prepared=True,stage0=str(OUT),stage1=str(STAGE1),frozen_files=len(dependencies),CUDA_initialized=False)))

if __name__=='__main__':main()
