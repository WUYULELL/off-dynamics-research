"""Create R09 documents and explicit stage-limited authorization, once."""
from runtime import *
import copy

def main():
    fresh(OUT)
    fresh(DOC)
    previous = ROOT / 'outputs/research_idea_protocol_20261016_r08'
    p = read(previous / 'PROTOCOL_DEFAULTS_R08.json')
    p['version'] = 'R09'
    p['status'] = 'EXECUTION_PARAMETERS_CONFIRMED_FOR_STAGES_1_3; research claims remain provisional'
    p['execution_authorized'] = 'stages 1-3 only, task 1 through validation stop'
    d = p['recommended_defaults']
    d['tuning_budget']['equal_ceiling_requires_confirmation'] = False
    d['diagnostics']['interaction_budget_interpretation_requires_confirmation'] = False
    d['ratio_selection']['task2_transfers_percentages_not_counts'] = True
    d['C']['interpretation_limit'] = 'V0仅由5000 target学得，在缺口状态外推最不可靠。G胜C仅说明本协议几何缺口信号更可靠，不说明覆盖原则优于一致性原则。'
    d['batch']['historical_control'] = dict(T128_minus_T64=-1.30, positive=2, n=5,
        failed_engineering_gate=True, does_not_prove_zero_effect=True)
    d['batch']['future_T128_descriptive_trigger'] = dict(user_wording='G-T paired mean near 5 points',
        operational_interval_provisional=[4.,6.], must_confirm_before_future_formal=True,
        enters_gate=False, authorized_now=False)
    d['G']['pre_mask_seal_audit'] = dict(top5='within each retained subset; also global top5',
        component_magnitude='mean absolute log-score pair difference shares',
        component_ranking='two-component Shapley allocation of signed pairwise rank agreement with full score',
        pairs=100000, seed='SHA256 score_contribution namespace',
        conservative_dominance='either contribution >90%; annotate effectively single-component, no combined-effect claim',
        extra='report component ranges, ties and full-pool deletion top-set overlap; no searches or rule changes')
    d['runtime']['long_training_requires_separate_user_authorization'] = 'already provided for stage 1-3; still required for anything else'
    d['runtime']['report_and_pause_between_stages'] = 'report stage receipts; continue within this explicitly authorized 1-3 bundle; mandatory pause after U/G stop result'
    d['new_code_path_proposal'] = str(HERE)
    d['new_result_path_proposal'] = str(OUT)
    p['stage_scope'] = dict(allowed=['CPU acceptance', 'six task1 5k smoke runs', 'full T grid18',
        'reference T seed1001', 'diagnosis5000', 'zero-additional-training score audit', 'U validation3', 'G validation9'],
        prohibited=['C validation grid', 'formal seeds', 'task2 execution', 'estimator development', 'old queues'],
        mandatory_stop='after U/G stop result irrespective of pass/fail',
        C_grid_order='only after stop passes AND separate authorization',
        T_search='full18 maintained; lean version cost comparison only')
    write(OUT / 'protocol.json', p)
    write(DOC / 'PROTOCOL_DEFAULTS_R09.json', p)
    write(OUT / 'stage_authorization.json', dict(authorized_stages=[1,2,3], task='walker2d-morph',
        protocol_sha256=sha(OUT/'protocol.json'), formal_authorized=False, C_grid_authorized=False,
        task2_authorized=False, old_queue_resume_authorized=False,
        evidence='2026-10-16用户：本轮只授权第1–3步（新管线CPU验收与冒烟、第一任务T调参、参考策略训练与诊断、第一任务U与G的比例验证）。拿到止损结果后单独决定是否授权正式阶段。',
        never_interpret_parameter_confirmation_as_broader_execution=True))
    budget = dict(current='full18', alternative='cost comparison only; not selected',
        T_full=dict(runs=18, updates=13500000, gpu_hours=[40.5,56.7]),
        T_lean_three_seeds=dict(runs=12, updates=7500000, gpu_hours=[22.5,31.5],
            schedule='3 temps at500k x3seeds, selectedtemp at1M x3seeds'),
        T_lean_literal_one_seed=dict(runs=10, updates=5500000, gpu_hours=[16.5,23.1],
            limitation='one-seed horizon check vs three-seed selection; unbalanced uncertainty'),
        savings_three_seed=dict(runs=6, updates=6000000, gpu_hours=[18.,25.2]),
        authorized_full_to_stop=dict(long_runs=31, smoke_runs=6, smoke_steps=5000,
            selected500k=dict(updates=20000000, gpu_hours_with20pctReserve=[84.96,117.36]),
            selected1M=dict(updates=26500000, gpu_hours_with20pctReserve=[121.32,166.68])),
        alternative_lean_to_stop=dict(long_runs=25, selected500k=dict(updates=14000000,
            gpu_hours_with20pctReserve=[63.36,87.12]), selected1M=dict(updates=20500000,
            gpu_hours_with20pctReserve=[99.72,136.44])),
        excluded='CPU scoring/audit, smoke overhead .5-1.5h, unavailable GPU time; no guaranteed completion date',
        full_two_task_count_if_later_authorized=dict(original119=119, lean107=107, savings_updates=12000000))
    write(DOC / 'COST_COMPARISON_R09.json', budget)
    text = '''# R09 修订与阶段执行合同

2026-10-16。承接 R08，以下条款优先；未修订的实验默认值按用户逐项确认保留。研究贡献仍属待验证假设，不把执行参数确认写成科学结论。

## 本轮变更

- P03：T=64，混合=64+64。历史T128−T64为−1.30、2/5正、未过门槛；这不证明batch效应为零。将来正式G−T接近5分时，增加同配置五种子T128描述性对照，不进入门槛。本轮不执行；“接近”的操作区间建议[4,6]分，须正式阶段前确认。
- P06：第二任务沿用百分比。Walker全池999322行，Hopper999998行，不能照搬绝对行数。
- P07：参考T及诊断固定后，任何G比例训练与掩码封存前进行零新增训练分数审计。5%、20%、50%各自选中子集的前5%分别占全池约0.25%、1%、2.5%；另报全池前5%，避免把同一份全池前5%重复三次。审计固定k20、两项相关性权重1/2，不再调参。
- P08：V0仅由5000条target学得，缺口外推最不可靠。G胜C只可解释为本协议下几何缺口信号比Bellman一致性信号更可靠，不能推为覆盖原则优于一致性原则。
- P14：先完成T→参考与诊断→分布审计/冒烟→U/G验证止损。只有止损通过并另行授权，才跑C的9次比例验证；不与G并行。

## 分量贡献的可复算口径

设a=log(rel)、b=log(gap)，总排名按a+b。每个审计集合固定随机抽10万对，计算a、b成对差绝对值的平均量，占两者合计的份额。这是排名分差幅度的描述，不能当因果贡献。

另以总分排序为参照，v(a)=mean(sign(Δa)sign(Δ(a+b)))，v(b)同理；总分差绝对值≤1e-12的并列对不计。双分量Shapley排名份额为 phi_a=(1+v(a)−v(b))/2，phi_b=1−phi_a。该约定将冗余信息均分；同时报告删去另一项后、在全池重新排序的top集合重合率、数值分布及并列率。

幅度份额或排名份额任一超过90%，报告必须标明本次G实质为单一分量选择，不得声称相关性×缺口的组合效果；未超过也不证明组合有因果贡献。审计只限制解释，不触发重新设计或搜索权重。全常量时份额不可识别，单列说明，不能伪填50%作为证据。

## T精简版成本比较（尚未替换完整网格）

|方案（第一任务）|长训练次数|更新步数|GPU占用估算，不含20%余量|
|---|---:|---:|---:|
|已确认：3temp×2步数×3seed|18|1350万|40.5–56.7小时|
|精简：3temp@500k×3seed，胜出temp@1M×3seed|12|750万|22.5–31.5小时|
|若1M字面只做1seed|10|550万|16.5–23.1小时|

建议精简版的“一次”指一个配置、仍用3个验证seed；节省6次训练、600万步、18–25.2小时。只做一个seed则1M和500k的不确定性不对等。历史500k/1M接近支持少搜索，但不保证所有温度下均接近；两种H的cosine周期不同，必须分别从零训练。

本轮依用户“补充成本对比”保留完整18次网格，精简版仅供决定，不能静默换方案。完整网格到止损为31次长训练＋6次5k冒烟。若选500k，约85–117小时；若选1M，约121–167小时，均含20%余量，CPU评分/审计与等待GPU另计。

## 执行边界

只执行第一任务CPU验收、T冒烟、T调参18次、参考T1次、5000步诊断、零新增训练分布审计、其余五组冒烟、U验证3次及G比例验证9次。source只进V+Q、actor只target，全部source训练标签oracle。参考在看到G/C/R/U短程或长期回报前固定。

止损：G三比例中最佳三seed Last-5均值严格同时超过已选T与U的均值才记“验证阶段见信号”；否则记“验证阶段未见信号”。这只是筛查，不使用正式2SE/4正/5分门槛，不作科学负结论。无论结果如何，到此停下并交付，等待用户另行决定。C网格、正式阶段、第二任务、旧队列、桌面归档均不在本轮授权范围。

所有新代码位于oracle_diag/coverage_oracle_r09，新结果位于oracle_diag/results/coverage_oracle_20261016_r09；旧代码/结果不改。单GPU串行、后台隐藏窗口、固定命名空间seed。异常停止保留目录，不覆盖、不自动重跑。
'''
    (DOC / 'REVISION_AND_STAGE_CONTRACT_R09.md').write_text(text, encoding='utf-8')
    canvas = '''# Working Idea Canvas R09

|项目|状态与内容|
|---|---|
|任务与默认参数|PROVIDED：R08默认值经用户逐项确认，四项修改见R09修订合同|
|核心待验证主张|PROVISIONAL：几何缺口信号能否带来超过强T、独立择优C、同数量R和全量U的增量收益|
|G/C解释边界|PROVIDED：固定V0外推局限；G胜C不等于两种原则的普遍优劣|
|预算公平|PROVIDED：相同验证上限；C网格在U/G止损通过且另行授权后运行|
|本轮执行授权|PROVIDED：仅第一任务阶段1–3，通过或失败均停在U/G止损结果|
|分布审计|IMPLEMENTATION DEFAULT：双口径成对分量贡献＋删除分量排名检查；不搜索权重/k|
|T搜索|CONFIRMED：完整18次；精简12次目前仅成本比较，不自动替换|
|正式/第二任务|UNAUTHORIZED：逐项参数确认不是执行授权|
|研究贡献|PROVISIONAL：基线选择的定量影响；预设oracle方案族内最佳观测证据。无数学上界、无严格95%置信结论|
'''
    (DOC / 'WORKING_IDEA_CANVAS_R09.md').write_text(canvas, encoding='utf-8')
    write(DOC / 'PROVENANCE.json', dict(previous={str(x):sha(x) for x in previous.iterdir() if x.is_file()},
        user_decision='P03/P06/P07/P08/P14 revised; rest confirmed; only stages1-3 authorized'))
    write(OUT / 'active_execution.json', dict(code=str(HERE), output=str(OUT),
        queue_script=str(HERE/'pipeline.py'), queue_status=str(OUT/'queue_status.json'),
        scope='only task1 through U/G validation stop; no C grid/formal/task2',
        protocol_sha256=sha(OUT/'protocol.json')))
    print(str(DOC), flush=True)

if __name__ == '__main__':
    main()
