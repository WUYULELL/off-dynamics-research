"""Consolidate audited stage evidence without overwriting the pipeline report."""
import json,re
from pathlib import Path
H=Path(__file__).resolve().parent; O=H/'results/verified_20261003'; D=O/'diagnostics'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
A=read(O/'final_audit_20261005.json');G=read(D/'stage1_stage2a.json')
envs=list(G); short={e:e.split('-')[0].replace('walker2d','Walker2d').replace('hopper','Hopper').replace('halfcheetah','HalfCheetah') for e in envs}
scores={(r['env'],r['method']):r['mean'] for r in A['jobs']}
auto=(H/'REPORT.md').read_text(encoding='utf-8')
def section(title):return auto.split('## '+title+'\n',1)[1].split('\n## ',1)[0].strip()
lines=[r'''# Bellman oracle 实验：全阶段结果与分析

核验日期：2026-10-05。范围：用户《给Code的指令_oracle诊断.md》规定的阶段 0、1、2a、2b、3；三个环境，训练 seed 0。本报告是人工核验后的统一汇总，优先于旧自动摘要 `REPORT.md` 中的描述。原始结果、旧报告和训练代码均保留。

## 一、结论先读

**全部阶段已完成，完成的是三环境 seed 0 的 oracle 诊断和训练对照；没有扩展到 5 seeds，也没有新增 OT 训练。**

1. **原 kNN 失配不是可靠的真实动力学失配排序器。** 固定 V 下，六组 batch 内相关范围约 −0.172 至 +0.170；Walker2d 使用方法自身的 V 时，相关为 −0.172，最差 20% 重合率仅 2.9%。
2. **差分修正改善了下一状态预测误差，但未得到稳定可靠的 Bellman 排序。** 三环境的平均状态误差分别降低约 40.6%、80.1%、3.2%，却仍都差于直接使用 source 下一状态的基线。Bellman 层面改善依赖环境和 V。
3. **把估计器换成模拟器 oracle，仍不能使当前指数加权公式获得稳定优势。** Walker2d 从常数权重的 58.42 降至 18.12；Hopper 基本持平；HalfCheetah 只有小幅点差，且与 target-only 几乎相同。
4. **直接修正 source Q 目标值得与“样本加权”分开看。** ORACLE-BACKUP 在 Walker2d 得到 66.37，高于常数权重 58.42；但 Hopper 下降，HalfCheetah 基本与 target-only 持平。因此只有局部后续验证线索，尚无跨任务优势证据。

我的判断：暂不扩大 OT＋原 Bellman 指数加权的训练。现有证据说明“估计器不准”与“即使给定 oracle，当前加权目标也未必有效”两层问题同时存在。不能据此宣布所有 Bellman 路线失败，也没有证据宣称能超过 TABB 或 OTDF；本实验未与两者作同协议对照。

## 二、所有阶段是否完成

| 阶段 | 计划与实际交付 | 核验结果 |
|---|---|---|
| 0：模拟器回放 | 每环境抽取 5000 条 target；反向回放完整 source | 完成；source 各 4999 条通过；Walker/Hopper target 各排除 1 条不可恢复样本 |
| 1：全量 source oracle | 三份 source→target 一步转移、哈希与索引对应 | 完成；各 4999 行，索引无重复、顺序对应 |
| 2a：动力学估计 | A/B/C、误差分布、失配排序、近邻距离相关 | 完成；source 不足 20000，使用全部 4999 个唯一点 |
| 2b：固定 V | 每环境 TARGET 与 MIX_FB 两个 V、相关/重合/ESS、散点图 | 完成；6 组 JSON/NPZ，6 张图各有 PNG/SVG/PDF |
| 3：训练验证 | 每环境 TARGET、FB、CONST、ORACLE、BACKUP，500k | 完成；14 个新训练＋Walker2d FB 1 个核验复用 |
| 历史对照 | IQL-mix seed 0 / 500k | 三环境历史分数纳入，明确标为历史参考 |
| 结果汇总 | 原始日志复算、完成状态与保存产物检查 | 完成；本报告及机器可读核验记录 |

本次检查直接读取 TensorBoard：15 个任务均含 10k–500k 每 10k 的评估点，Last-5 与原汇总一致。14 个新任务退出码均为 0，完成标记与进度均为 500000；98 个最终模型/优化器等状态文件及 70 个 V 快照可在 CPU 读取，张量均为有限值。三组状态误差、六组全局及 batch 相关已从 NPZ 独立复算；oracle 文件与固定 V 的哈希吻合，26 个受保护代码文件哈希未变。

队列运行约 **41.49 小时**，2026-10-05 12:38 完成。本次检查没有重跑训练或模拟器；回放有效性依据已保存的验收记录，检查不等于新的独立重复实验。

### 执行与原始方案的差异

- 20000 个 source 查询改为全部 4999 个唯一点，避免重复抽样虚增样本量。
- 新算法与注册适配器均在 `oracle_diag/`；没有修改原算法注册表。
- CONST 重新训练以保留一致的模型和日志；Walker2d FB 复用已核验模型；IQL-mix 使用历史分数。
- “oracle 上界”改称“理想转移对照”：非线性函数逼近、离线覆盖和优化使其不具备回报数学上界性质。
- 本轮验证的是 kNN/差分/oracle 以及 oracle 权重和 backup，不包含 OT reference 的直接精度比较。不能把 oracle 结果称为 OT 实测结果。

## 三、实验定义与公平比较范围

source 为 shifted ODRL medium，target 为正常 D4RL medium 的 seed 0 / 10% 子集。各环境 target 子集数量见后表。kNN 使用原代码的归一化 [s,a]、k=5、20000 点 target bank。训练 batch=128，gamma=0.99，eta=1，source_budget=0.20，clamp=[0.05,1]，IQL 与 actor/critic 流程沿用原实现。每 10k 评估 10 个 episode，Last-5 为 460k–500k 五次归一化评估的算术均值。

令 nd 表示非终止掩码，给定同一 V：

\[
y_s=r_s+\gamma nd_s V(s'_s),\qquad y^\circ=r^\circ+\gamma nd^\circ V(s'^\circ),\qquad b^\circ=|y_s-y^\circ|.
\]

原 kNN 使用近邻 Bellman 值的均值作为 reference；ORACLE 用同一 source 状态与动作在 target 模拟器的转移替换 reference。权重均沿用：

\[
z_i=b_i/(\operatorname{mean}_{batch}b+10^{-6}),\quad u_i=e^{-z_i},\quad
w_i=\operatorname{clip}\left(0.20u_i/(\operatorname{mean}_{batch}u+10^{-6}),0.05,1\right).
\]

**预算参数是 0.20，但最终平均权重不必精确为 0.20**，因为最后有截断。原实现的权重作用于 source 的 V、Q、actor 三项损失。ORACLE-BACKUP 则固定 source 权重 0.20，仅将 source Q 回归目标换为 y°；V/actor 的 source 状态动作仍保留。

名称 `MIX_FB` 指 IQL_BELLMAN_FB 的 V/训练，**不等同于历史 IQL-mix**。固定 V 诊断使用最终 500k 网络；ORACLE 训练使用当时在线更新的 V，因此两者不能按同一静态打分器理解。

## 四、阶段 0：回放验收
''',section('阶段 0：回放验收')]
lines+=['''
**分析：** 回放残差远低于既定验收阈值，支持把该 oracle 用于本数据上的诊断。Walker 的历史恢复是发现首轮失败后做的修复，保留了失败记录与未放宽的阈值；不能把恢复值称为无误差的原始隐藏状态。Hopper 的物理 shift 在本 source 支持集上约为浮点零，后续必须将数值噪声和跨轨迹边界分开。

## 五、阶段 1：全量 oracle 与实际 shift

下列状态量均为按 target 状态标准差缩放后的 L2 范数。物理 shift 用同一恢复状态在两个模拟器的下一状态比较；“加载器差异”用实际训练 buffer 的 next_state 与 oracle 比较。

| 环境 | source / target 子集 | 物理 shift 均值 / 中位 / p95 | 加载器差异均值 / 中位 / p95 | 奖励差均值 / 中位 / p95 | 边界行数 |
|---|---|---|---|---|---:|''']
def dist(d):return ' / '.join('{:.6g}'.format(d[k]) for k in ['mean','median','p95'])
for e,g in G.items():
    lines.append('| {} | 4999 / {} | {} | {} | {} | {} |'.format(short[e],g['target_subset_size'],dist(g['physical_shift_standardized']),dist(g['standardized_dynamics_mismatch']),dist(g['reward_difference']),len(g['boundary_rows'])))
lines+=['''
**更正旧摘要：** 边界行数为 Walker2d 4、Hopper 12、HalfCheetah 4，不是三个环境各 4。去除这些行后的样本数分别为 4995、4987、4995；相应轨迹片段约为 5、13、5（最后片段不保证完整，也不等于独立随机重复）。原训练 buffer 的跨边界 next_state 保留以对齐旧流程，敏感性诊断另外剔除。

Hopper 去边界后的加载器状态差异均值仅 3.216×10⁻⁷，物理 shift 均值 1.804×10⁻¹⁶。因此其全量 0.02293 不能当作实际动力学 shift 的证据。Walker 和 HalfCheetah 去边界后均值分别为 1.61995 与 0.48228，主要差异仍存在。跨环境标准差不同，上述单位不等于同一物理长度单位。

## 六、阶段 2a：下一状态估计质量

A：近邻 next_state 的平均；B：source 当前状态＋近邻平均位移；C：直接使用 source next_state。误差均相对于 oracle next_state。

| 环境 | 估计 | 误差均值 / 中位 / p95 | 去边界误差均值 | 估计失配 vs 真失配 Spearman | 邻居距离 vs 估计误差 Spearman |
|---|---|---|---:|---:|---:|''']
for e,g in G.items():
    for n,m in g['stage2a'].items():
        rho=m['estimated_vs_true_spearman'];lines.append('| {} | {} | {} | {:.6g} | {} | {:.4f} |'.format(short[e],n,dist(m['error']),m['clean_error']['mean'],'未定义' if rho is None else '{:.4f}'.format(rho),m['neighbor_distance_vs_error_spearman']))
lines+=['''
**分析：** B 相对 A 的平均误差下降约 40.6%、80.1%、3.2%，但 C 在三个环境都更接近 oracle。不能把“比 A 好”写成“动力学校正已经有效”。Walker A 的近邻距离/误差相关为 0.837，Hopper 为 0.913：邻居离查询点较远与误差大明显关联，支持存在邻域匹配误差；这些相关不能定量证明误差中多少比例由距离造成。

Walker 的估计失配排序仍为负相关；Hopper 的真 shift 几乎为零，相关主要涉及残差/边界；HalfCheetah B 的相关虽由负转正，仍很弱。C 预测的“source 与自身之差”恒为零，因此其失配排序相关未定义，不能填成 0。

## 七、阶段 2b：固定 V 的 Bellman 排序
''',section('阶段 2b：固定 V 下 Bellman 排序质量').replace('每环境只有约 5 条 source 轨迹','按边界划分，Walker/Hopper/HalfCheetah 分别约有 5/13/5 个 source 轨迹片段')]
lines+=['''
### 结果解释与敏感性

- **Walker2d：** kNN 的 batch 相关从 TARGET V 的 +0.170 变为 MIX_FB V 的 −0.172，说明判断依赖 V。MIX_FB 最差 20% 重合仅 2.9%，远低于随机排序约 20% 的参照。B 在该 V 下相关进一步降到 −0.203，不能称为修复成功。
- **Hopper：** 两种估计的相关均接近零。由于真实物理差异接近零，即使数值上计算出相关或 oracle ESS，也不宜将其解释为有意义的动力学识别能力。
- **HalfCheetah：** B 改善较明确，TARGET V 的 batch 相关从 −0.048 到 +0.230、重合率从 15.5% 到 38.7%；但 MIX_FB V 下相关只有 +0.068。尚不支持跨 V 的稳定方案。
- **ESS 是权重集中程度，不是准确率。** 接近 1 表示权重较均匀；低于 1 表示样本贡献不均匀，不直接说明选择正确。oracle 的自相关为 1 只是定义上的自比较，也不能证明 V 准确。

下面完整保留去边界全局相关与“逐邻居平移后评价 V 再平均”的额外敏感性。它与先平均 next_state 再评价 V 不等价，不能挑选较好的一个作为同一方法结果。

| 环境 | V | kNN 去边界相关 | B 去边界相关 | 逐邻居变体全局 / batch 相关 | 逐邻居 top20% | oracle 平均失配 | oracle 平均权重 |
|---|---|---:|---:|---|---:|---:|---:|''']
for e in envs:
    for k in ['TARGET','MIX_FB']:
        m=read(D/(e+'_'+k+'_bellman.json'))['metrics'];s=m['diff_per_neighbor_sensitivity']
        lines.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} / {:.4f} | {:.1%} | {:.6g} | {:.5f} |'.format(short[e],k,m['knn']['clean_global_spearman'],m['diff_mean_state']['clean_global_spearman'],s['global_spearman'],s['batch_spearman_mean'],s['top20_overlap'],m['oracle']['mismatch']['mean'],m['oracle']['source_weight_mean']['mean']))
lines+=['''
## 八、阶段 3：500k 训练结果

主指标为 Last-5 normalized score；不是最高 checkpoint，不是 5 个训练种子的均值。各行新训练除 Walker2d FB 为核验复用外；IQL-mix 为独立列示的历史结果。

| 方法 | Walker2d | Hopper | HalfCheetah |
|---|---:|---:|---:|''']
labels={'TARGET':'IQL-target','MIX_FB':'原 kNN BELLMAN_FB','CONST':'CONST 0.20','ORACLE':'ORACLE 权重','BACKUP':'ORACLE-BACKUP'}
for m,label in labels.items():lines.append('| {} | {} |'.format(label,' | '.join('{:.3f}'.format(scores[e,m]) for e in envs)))
lines+=['| IQL-mix（历史参考） | 60.077 | 46.070 | 42.244 |','', '| 对照点差 | Walker2d | Hopper | HalfCheetah |','|---|---:|---:|---:|']
for m,b in [('ORACLE','CONST'),('ORACLE','MIX_FB'),('BACKUP','CONST'),('BACKUP','MIX_FB'),('BACKUP','TARGET')]:
    lines.append('| {} − {} | {} |'.format(labels[m],labels[b],' | '.join('{:+.3f}'.format(scores[e,m]-scores[e,b]) for e in envs)))
lines+=['''
### 训练结果说明什么

**Walker2d：当前 oracle 权重方案出现严重退化。** 18.12 对 58.42 的差距远大于该运行末尾 checkpoint 的波动；它是此 seed 明确的负结果，但仍不等于跨 seed 统计结论。另一方面，BACKUP 的 66.37 高于常数权重约 7.94 分，提示“修正目标”可能比“按失配降权”更值得小规模验证。BACKUP 末五次分数在 54.48–80.75，波动较大。

**Hopper：oracle 权重与常数权重相近，BACKUP 反而下降。** 权重仅高 0.109 分，不能据此说有效；BACKUP 低 8.118 分。由于 source 支持集几乎无物理 shift，这不是“纠正真实动力学反而伤害训练”的干净证据。跨轨迹 next_state、终止掩码、浮点残差和非线性训练路径均可能参与差异，现有实验没有分离各自贡献。

**HalfCheetah：oracle 两种方式均有小幅点差，却没有超出 target-only 的明显收益。** ORACLE 44.724、BACKUP 44.722，target-only 44.705，差距不到 0.02；不能当作 source 迁移优势的证据。

### 训练末段的实际权重

下表取 460k–500k 区间内 9 个已记录训练时点的均值；不同于 Last-5 的五次评估，也不同于阶段 2b 的固定 V。

| 环境 | ORACLE 失配均值 | 实际 source 权重均值 | ESS ratio |
|---|---:|---:|---:|''']
for row in A['jobs']:
    if row['method']=='ORACLE':
        t=row['train_tail'];lines.append('| {} | {:.6g} | {:.5f} | {:.4f} |'.format(short[row['env']],t['train/bellman_mismatch_mean']['mean'],t['train/source_weight_mean']['mean'],t['train/source_weight_ess_ratio']['mean']))
lines+=['''
**机制线索（并非因果证明）：** Hopper 的失配约 10⁻⁵，经过除以 batch 均值＋10⁻⁶ 的归一化，仍足以形成不均匀权重。该公式主要反映相对大小，不能自动把低于有意义误差尺度的残差当作“无 shift”。此外，权重同时影响 V/Q/actor；较大失配样本不必是不值得学习的样本。Walker 退化可能涉及样本选择、状态覆盖与 V 的反馈，但本实验尚未定位具体原因，不能仅凭 ESS 下结论。

## 九、对原始三个问题的直接回答

1. **kNN 与真实失配一致吗？** 不稳定，通常很弱，部分情形方向相反，尤其是 Walker 的方法自身 V。原 kNN 权重即使偶尔提高回报，也不能据此声称它准确识别动力学差异。
2. **差分修正有效吗？** 状态估计误差部分改善；Bellman 排序只有局部改善，不能作为已验证的普适替代方案。
3. **oracle 权重胜过常数 0.20 吗？** 没有稳定优势：Walker 明显负结果，Hopper 基本持平，HalfCheetah 只有较小点差。当前结果不支持继续仅靠改善 reference 估计器来押注原指数加权公式。

## 十、下一步判断与局限

**建议顺序：** 先核实并单独处理原加载器的跨轨迹转移/终止语义，建立修复前后的对照版本；不要悄悄改动旧结果。随后若继续训练，优先限定 Walker2d 的 CONST 与 BACKUP 配对复核，再决定是否推广。Hopper 应先采用真正激活差异的状态支持集或另选 shift，不能继续把当前近零物理差异当作主要识别任务。若继续开发加权，应先验证绝对误差尺度、数值残差和 actor/V/Q 加权位置的影响；本次没有执行这些新实验。

关键限制：

- 只有 seed 0，不能给出种子间置信区间、显著性或跨任务稳定优势；末五 checkpoint 和每次 10 个 episode 都不能替代独立训练种子。
- source 只有少量轨迹片段；4999 个相关转移及 39 个随机 batch 不是独立实验重复。不能据此做伪独立的显著性检验。
- 固定 V 是学到的函数，未经真实回报校准；oracle 只消除了给定恢复状态下一步转移的估计问题，不保证 V 正确。
- 隐藏速度截断、float32 数据和 Walker 历史恢复带来近似性。Hopper 的残差尤其容易主导相对归一化。
- 训练保留原加载器边界行为以保证方法比较口径；因此本轮训练同时带有该数据处理局限。去边界诊断不是去边界重训练。
- V 的最终 checkpoint 被用于事后诊断，未证明训练全过程排序一致；固定 V 结果不能直接等同于在线 oracle 权重轨迹。
- 历史 IQL-mix 是参考，未在本轮重训；模型复用有单独验证记录。相同 seed 也不能自动保证所有变体共享完全相同的随机数消耗路径。
- 没有 OT、TABB、OTDF 的本轮直接对照，不据此推断它们的实际分数或优劣。

## 十一、Last-5 原始数值（便于核对）

| 环境 | 方法 | 460k | 470k | 480k | 490k | 500k | 均值 |
|---|---|---:|---:|---:|---:|---:|---:|''']
for r in A['jobs']:lines.append('| {} | {} | {} | {:.3f} |'.format(short[r['env']],labels[r['method']],' | '.join('{:.3f}'.format(v) for v in r['last5']),r['mean']))
lines+=['','## 十二、图表与证据索引','',section('图表'),'''

上述散点图属于固定 V 描述分析，线性坐标下可能受极端值压缩；应结合相关与重合率表阅读，不用图面密度代表统计显著性。本次核验了 18 个图文件存在且非空，没有重新生成图。

| 证据 | 用途 |
|---|---|
| results/verified_20261003/final_audit_20261005.json | 本次日志复算、模型读取、哈希核对的记录，含每个任务 TensorBoard 原始路径 |
| results/verified_20261003/queue_status.json、pipeline_completed.json | 任务退出码、完成时间、复用来源 |
| results/verified_20261003/final_acceptance.json | 最终 source 回放验收、Walker 修复行和哈希 |
| results/verified_20261003/acceptance.json、warmstart_acceptance.json | 初始失败与 warmstart 排查，不能把旧失败状态当最终状态 |
| results/verified_20261003/diagnostics/stage1_stage2a.json | 全量与去边界状态误差、奖励误差、相关 |
| results/verified_20261003/diagnostics/*_source_oracle.npz | 与 source buffer 逐行对应的 oracle 转移 |
| results/verified_20261003/diagnostics/*_bellman.json / .npz | 六个固定 V 的指标与逐点数据 |
| results/verified_20261003/gpu_smoke.json、checkpoint_reuse_verification.json | 新方法 smoke、公式/索引验证及复用模型验证 |
| results/training_20261003/<环境>_<方法>/ | effective_config、data_verified、completed、TensorBoard、最终模型和 V 快照 |
| ../analysis_results/const_w020_vs_bellman/score_by_seed.csv | IQL-mix 历史 seed 0 / 500k 分数 |

核心实现：`oracle_methods.py` 的 `compute_source_weight` 对应 oracle 失配与权重；`IQL_ORACLE_BACKUP.q_loss_per_sample` 对应 source Q 目标替换；`bellman_diagnostics.py` 对应固定 V 比较；`prepare_diagnostics.py` 对应状态估计。复核脚本 `audit_final_results.py` 仅在 CPU 读取产物，不触发训练；`build_final_report_cn.py` 从这些证据生成本文件。

**最终状态：阶段交付完整，可以用于本轮 seed 0 的研究决策；尚不足以支持普适方法优势或论文中的统计优越性声明。**
''']
text='\n'.join(lines)
# Use absolute links for all existing local linked artifacts.
def link(m):
    label,target=m.group(1),m.group(2)
    return '[{}](<{}>)'.format(label,(H/target).resolve().as_posix())
text=re.sub(r'\[([^\]]+)\]\((results/[^)]+)\)',link,text)
dest=H/'FINAL_REPORT_CN.md';dest.write_text(text,encoding='utf-8')
assert all(x in text for x in ['阶段 0','阶段 1','阶段 2a','阶段 2b','阶段 3'])
print(str(dest));print('characters:',len(text))
