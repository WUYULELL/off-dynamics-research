"""Seal the explicitly authorized R14 scope before any new GPU results."""
from runtime import *
import numpy as np

def main():
    fresh(OUT);fresh(DOC)
    for name in ('jobs','logs'):fresh(OUT/name)
    old=ROOT/'oracle_diag/results/coverage_oracle_20261021_r13'
    original=read(old/'protocol.json');tasks=[]
    for entry in original['tasks']:
        keys=('task','target_rows','source_rows','xml','xml_sha256','target_dataset','target_dataset_sha256',
            'source_dataset','source_dataset_sha256','timestep','score_min','score_max')
        spec={k:entry[k] for k in keys}
        spec.update(state_dim=17 if entry['task']=='walker2d-morph' else 11,
            action_dim=6 if entry['task']=='walker2d-morph' else 3,source_labels='original_dataset',target_dataset_fixed=True)
        for path_key in ('xml','target_dataset','source_dataset'):assert sha(spec[path_key])==spec[path_key+'_sha256']
        tasks.append(spec)
    protocol_value=dict(version='R14-original-label-source-strength-precheck',date='2026-10-26',tasks=tasks,
        authorization='User explicitly authorized walker M(.25/1) seeds201-205 and Hopper T/M(.25/1) seeds201-210, 500k/temp10 after wall-clock quote',
        new_GPU_jobs=40,new_gradient_updates=20000000,walker_T5_reused=True,walker_R13_U5_oracle_descriptive_only=True,
        source_w=[0.,.25,1.],lambda_relation='w/(1+w)',lambda_values=[0.,.2,.5],
        losses='V_target_mean+w*V_source_mean; Q_target_mean+w*Q_source_mean; actor_target_only',
        no_division_by_one_plus_w=True,lambda_is_relative_weight_not_full_optimizer_equivalence=True,
        batch_target=64,batch_source=64,source_subset_selection=False,source_labels='HDF5 original next_observations/rewards/terminals',
        source_rows='R13 canonical valid rows: omit timeout rows and final adjacency-unavailable row',
        observation_normalization='none (raw), as R13',reward_scale=1.,gamma=.99,expectile=.7,
        learning_rates=dict(actor=.0003,V=.0003,Q=.0003),hidden_sizes=[256,256],tau=.005,
        advantage_clip=100,actor_advantage='pre-V-step target advantage',actor_schedule='cosine original H500k',
        target_Q_update='every gradient update',temperature=10,steps=500000,
        evaluation=dict(frequency=10000,episodes=10,deterministic=True,max_episode_steps=1000,
            seeds='500000+10*(run_seed-201)+episode0..9 per task; no online selection',
            preserve_training_RNG=True,Last5=[460000,470000,480000,490000,500000],Final='500000 auxiliary only'),
        statistics=dict(primary='seed-level Last5',sample_SD_ddof=1,lower_quartile='np.quantile(.25), linear interpolation',
            paired_mean='within task same training seed; report differences/mean/SD(ddof1)/SE/positive',
            SD_ratio='SD(group)/SD(T)',bootstrap='paired seed indices, conditional on >=2 unique indices, 20000 replicates, percentile [.025,.975]',
            denominator_undefined='SD(T)<=1e-12: report undefined, never silently replace/drop after seeing results',
            bootstrap_seed='uint32 little endian SHA256(R14-bootstrap|task|metric|group)',
            interval_interpretation='descriptive 95% percentile bootstrap; not exact coverage, no new formal gate',
            best_w='argmax mean Last5 over {0,.25,1}; within1e-8 choose smaller w; oracle U excluded',
            same_best_w='no task-dependent preference established on tested tasks/grid; not proof adaptation globally unnecessary',
            different_best_w='descriptive task dependence only; not estimator feasibility or development authorization',
            selection_bias='choose and report best w on same seeds; different n per task; no follow-up thresholds changed'),
        variance_object='training random seeds conditional on fixed dataset; not target dataset sampling variance',
        study_revision_after_R13_and_literature_results=True,all_new_results_unknown_at_commitment=True,
        estimator_development_authorized=False,other_experiments_authorized=False,
        stop='after all40 complete and audited, report then stop for user',
        hopper_temperature_not_separately_tuned=True,T_temperature_boundary_limitation_retained=True,
        wall_clock_quote_hours=[85,110],single_GPU_serial_hidden=True,no_automatic_retraining=True)
    write(OUT/'protocol.json',protocol_value)
    reuse=[]
    for job in read(old/'training_plan.json')['jobs']:
        if job['arm'] not in ('T','U'):continue
        path=Path(job['out']);audit=path/'audit.json'
        assert read(audit)['passed'] and read(path/'completed.json')['steps']==500000
        reuse.append(dict(task='walker2d-morph',seed=job['seed'],w=0. if job['arm']=='T' else 1.,
            label='raw' if job['arm']=='T' else 'R13_oracle',out=str(path),audit=str(audit),audit_sha256=sha(audit),
            files_sha256={str(path/f):sha(path/f) for f in ('completed.json','evaluations.json','sampling.json','initialization.json','config.json')}))
    assert len(reuse)==10;write(OUT/'reuse_receipts.json',dict(runs=reuse,no_retraining=True))
    jobs=[]
    for task,arms,last_seed in [('walker2d-morph',(.25,1.),205),('hopper-morph',(0.,.25,1.),210)]:
        for w in arms:
            for value in range(201,last_seed+1):
                ident='%s_w%s_s%d'%('walk' if task=='walker2d-morph' else 'hopp',{0.:'0',.25:'025',1.:'1'}[w],value)
                job=dict(id=ident,task=task,phase='precheck',arm='T' if w==0 else 'M',seed=value,temp=10.,steps=500000,
                    source_w=w,source_lambda=w/(1+w),source_labels='none' if w==0 else 'original_dataset',out=str(OUT/'runs'/ident))
                validate_spec(job);jobs.append(job);write(OUT/'jobs'/(ident+'.json'),job)
    assert len(jobs)==40;write(OUT/'training_plan.json',dict(jobs=jobs,new_GPU_jobs=40,new_updates=20000000,
        order='Walker w.25 five, w1 five; Hopper T ten, w.25 ten, w1 ten',no_next_stage=True))
    elapsed_T=[read(Path(r['out'])/'completed.json')['elapsed']/3600 for r in reuse if r['w']==0]
    elapsed_mix=[read(Path(r['out'])/'completed.json')['elapsed']/3600 for r in reuse if r['w']==1]
    finish=ROOT/'oracle_diag/results/coverage_oracle_20261024_r13_finish1/runs'
    elapsed_mix += [read(p)['elapsed']/3600 for p in finish.glob('*/completed.json')]
    budget=dict(measured_R13_T_hours=elapsed_T,measured_mixed_hours=elapsed_mix,
        estimate_training_hours=float(10*np.mean(elapsed_T)+30*np.mean(elapsed_mix)),
        quoted_wall_clock_hours=[85,110],new_jobs=40,mixed_jobs=30,T_jobs=10,hardware='RTX2060 SUPER 8GB',
        includes='CPU preparation, 40 sequential runs, audits, raw-data report, runtime variation',
        assumptions='continuous single GPU; no pauses or major competing GPU load; Hopper runtime unmeasured')
    write(DOC/'BUDGET_ESTIMATE.json',budget)
    contract='''# R14：方向一原始标签 source 强度前置验证

## 授权与结果前承诺

本方案由用户在阅读 LITERATURE_CHECK_DIR1 后明确授权；已看到R13的T/U历史结果，不能称从未看过任何相关结果的首次预注册。以下设置与分析在本阶段任何新长训练结果前固定。预计墙钟85–110小时（约3.5–4.5天），单卡持续运行；暂停和其他GPU负载会延长。精确实测来源见BUDGET_ESTIMATE.json。

## 唯一执行范围

- Walker2d-morph：原始标签source w=.25与1，各seeds201–205，10项；复用R13同seed T，不重跑；另列原R13 oracle-U(w=1)作描述对照。
- Hopper-morph：T、原始标签source w=.25与1，各seeds201–210，30项。从0训练；使用已验收OTDF hopper_morph_medium.hdf5与hopper_morph.xml，归一化min=-26.3360015397715、max=3152.75。
- 合计40项、2000万更新；IQL temp10、500k。T每次64target，混合64target+64source；全量source均匀有放回，无mask/筛选。source仅进V+Q，actor只用target；保留R13 pre-V target advantage、优化器/学习率/原500k cosine、targetQ更新及抽样seed派生。
- source从原HDF5读取next_observations/rewards/terminals，不用oracle。保持R13 canonical可抽样行：排除timeouts及末行。原HDF5显式next-state与相邻obs仅在terminal行不同，terminal mask为0；记录完整标签与行hash。
- w的含义为L_target+w L_source，V/Q同时使用；lambda=w/(1+w)，三点为0/.2/.5。为沿用R13更新，不额外除以1+w；相对权重换算不意味着固定学习率下完整优化轨迹等同于TMLR归一化目标。
- 两档w同seed初始化与完整target/source抽样流相同；Walker target流和初始化与复用T逐项核对。target数据固定5000行，learner用原始观测/reward，不新增观测归一化或奖励缩放。
- 评估按R13每10k十个episode，seed=500000+10*(training_seed-201)+episode0..9，各任务相同规则，保存/恢复训练RNG。实际shifted环境、timestep.002、TimeLimit1000。只为预定评估访问环境，不据回报选参数/种子，不新增诊断交互。
- Last5为460/470/480/490/500k五点评估均值，先seed内平均；Final另报。Hopper temp10沿用，未单独调参；T温度在旧网格上边界的限制保留，不因此追加搜索。

## 在新结果前固定的统计

1. 每组Last5均值、训练种子样本SD(ddof1)、最差分数及seed、Q1（25%分位，线性插值）。原始逐seed值和Final辅助统计保留。
2. SD(w)/SD(T)：按同seed联合重采样比较组与T，20000个bootstrap样本，百分位[2.5%,97.5%]描述性区间；仅排除所有索引相同的退化抽样并重抽，记录拒绝数。SD(T)<=1e-12或非退化抽样分母为0时报告不可识别，不事后改算法。分析随机seed由独立R14-bootstrap命名空间固定。
3. 两档w相对T同seed配对差，报告逐seed、均值、样本SD(ddof1)、SE=SD/sqrt(n)、严格正号数；Walker另列R13 oracle-U对照。
4. 每个任务在{0,.25,1}中按Last5均值选最佳w；1e-8内并列取较小w，报告并列集合；oracle-U不进入原始标签候选。此处为有限网格经验优选，不是Chen等Corollary2理论lambda*的估计器，不开发任何估计器。
5. 两任务最佳点相同：本次任务/网格未建立配比任务依赖的描述性证据，不证明自适应普遍无必要。不同：仅提供任务依赖线索，不证明估计器可行或授权开发。选择与报告同批seed，披露选择偏差、Walker5与Hopper10的不等精度。
6. 本轮测固定数据下的训练随机性，不是重抽target数据的统计估计方差；SD变小不自动证明价值误差/MSE降低或偏差—方差因果机制。均值、下尾、离散程度一起解释。

## 不变的边界

不新增任何正式成功/失败门槛，不修改原>=5分等后续门槛，不用本次最佳w改变后续标准；bootstrap不是精确95%保证。全部40项审计、原始TB/episode复算后停止交付，任何结果均不开发估计器、不派发其他实验。无C/G/R/D、T128、额外温度搜索、新oracle缓存或其他长训练。

## 工程验收与证据

新目录、单GPU串行隐藏窗口；不改旧文件，不终止其他应用，不恢复任何旧队列。CPU验收核对原始标签、XML/归一化、初始化/抽样、w=1与旧代数一致、评估seed/报告口径；无额外GPU探针。CPU合成数据的少量更新只作代数验收，不使用实验数据拟合或访问环境。故障保存stderr/完成失败证据并CPU诊断，不自动重训；审计问题只修审计。旧文件/数据/科学代码hash在启动前与完成后复核。
'''
    with open(DOC/'STAGE_CONTRACT_R14.md','x',encoding='utf-8') as file:file.write(contract)
    write(OUT/'stage_authorization.json',dict(scope='R14_raw_source_strength_precheck_40_only',protocol_sha256=sha(OUT/'protocol.json'),
        new_GPU_jobs=40,estimator_development_authorized=False,other_experiments_authorized=False,
        wall_clock_estimate_communicated_before_start=[85,110],committed=time.time()))
    prior_manifest=ROOT/'oracle_diag/results/coverage_oracle_20261024_r13_finish1/protected_manifest.json'
    protected=read(prior_manifest)['sha256']
    for directory in [p for p in (ROOT/'oracle_diag').glob('coverage_oracle_r13*') if p.is_dir()]+[p for p in (ROOT/'oracle_diag/results').glob('*r13*') if p.is_dir()]+[p for p in (ROOT/'outputs').glob('*r13*') if p.is_dir()]:
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in str(path):protected[str(path)]=sha(path)
    # Preserve the earlier hash for every previously protected file: detect, never refresh, a mismatch.
    previous=read(prior_manifest)['sha256']
    for path,h in previous.items():assert protected[path]==h and sha(path)==h,'Old file changed: '+path
    for spec in tasks:
        for key in ('xml','target_dataset','source_dataset'):protected[spec[key]]=spec[key+'_sha256']
    literature=ROOT/'outputs/research_idea_dir1_litcheck_20261026/LITERATURE_CHECK_DIR1.md';protected[str(literature)]=sha(literature)
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,no_old_writes=True))
    dependencies=[OUT/'protocol.json',OUT/'training_plan.json',OUT/'reuse_receipts.json',DOC/'STAGE_CONTRACT_R14.md',OUT/'stage_authorization.json']+list((OUT/'jobs').glob('*.json'))
    for spec in tasks:dependencies += [Path(spec[k]) for k in ('xml','target_dataset','source_dataset')]
    write(OUT/'dependency_hashes.json',dict(sha256={str(p):sha(p) for p in dependencies}))
    write(OUT/'PRE_RESULT_COMMITMENT.json',dict(time=time.time(),before_any_new_GPU_result=True,existing_R13_baselines_seen=True,
        protocol_sha256=sha(OUT/'protocol.json'),contract_sha256=sha(DOC/'STAGE_CONTRACT_R14.md'),
        training_plan_sha256=sha(OUT/'training_plan.json'),bootstrap_rule_fixed=True,new_training_results_exist=False))
    write(OUT/'code_frozen.json',dict(sha256=code_hashes(),approved_science_change='original source labels + scalar w; Hopper task plumbing only'))
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),docs=str(DOC),queue_script=str(HERE/'pipeline.py'),
        original_R13_queues_must_not_restart=True,new_GPU_jobs=40,old_T_reused=5,old_U_descriptive=5))
    print('Prepared R14;40 jobs;protected%d;estimated training%.2fh'%(len(protected),budget['estimate_training_hours']),flush=True)

if __name__=='__main__':main()
