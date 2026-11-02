# Oracle v3：标准方向实验最终报告

**阶段 A 与已授权的阶段 B 已完成并核验。3/3 组 GPU smoke、15/15 项正式训练全部完成；阶段 C 未执行。**

交付日期：2026-10-10（Asia/Shanghai）。主任务为 **walker2d-morph：正常 D4RL medium source → 5000 条 OTDF shifted medium target，在 shifted 环境评估**。诊断及最终核验使用 CPU，训练使用 GPU；结果标记为 `explicit_next_v3`。

## 核心结论

1. **ORACLE 没有超过 TARGET 的稳定收益证据。** Last-5 五种子均值为 35.39 对 39.30，配对差 **−3.91 分，SE=4.05，3/5 个正号**，未通过预设门槛。
2. **ORACLE 没有优于直接混数据的稳定增量证据。** 相对 CONST（即 MIX-NAIVE）的配对差为 **−0.0247 分，SE=3.29，2/5 个正号**。平均分接近并不证明两种方法等效，种子间正负变化很大。
3. **本轮建议：结束当前协议下的扩大训练，不启动阶段 C。** 结果不支持现在继续投入大规模 OT/动力学修正器训练；也不能据此证明 source 永远无用、Bellman 路线不可能有效，或 TARGET 是数学上界。

以上分数为 OTDF 归一化分数的点差，不是相对百分比。判据始终为“配对均值 > 2×SE 且至少 4/5 个种子为正”，没有根据结果调整预算、eta、种子或筛选标准。


## 1. 方案审查后的修正

1. v2 的负均值不证明 source 本身没有价值，也不证明 target-only 是任何方法的数学上限。v3 据此检验当前协议下模拟器修正是否产生可重复收益。
2. 核对旧实现后，v2 各组总采样量都是 256：TARGET 为 256 target，混合组为 128 source＋128 target。v3 统一总量 128：TARGET 128 target；混合组 64＋64。
3. 文档中的 MIX-NAIVE 与 MIX-CONST 完全相同：全量 source、系数 0.20、记录的转移标签。因此保留一个实验控制，MIX-NAIVE 作为别名，避免重复五个训练。
4. shifted target 的 5000 条原域回放，只能用于验收该模拟器，不能直接拿来给约 100 万条正常 source 提供 oracle 标签。本轮另建完整 source→shifted 缓存并验证行索引。
5. `infos/qvel` 虽以 float64 保存，实际仍在 ±10 截断。首轮将它视为完整状态的假设已被验收否定，原始失败记录保留；随后沿用 v2 的历史恢复流程，没有改样本、阈值或剔除失败行。

## 2. 阶段 A：接入与回放验收

四个任务均通过独立 HDF5→target buffer **全部行**的观测、动作、显式 next_observations、奖励和终止标记对齐检查。严格拒绝未支持的任务名。shifted 环境使用 gym v3 对应类加载 OTDF XML，外包 TimeLimit(1000)。

### 2.1 shifted target：复用 v2 验收，不重复回放

复用前重新核对了原数据和 XML 的 SHA256，并从保存的物理下一状态、奖励重算误差。旧结果没有保存逐行 factual done 数组，所以 done=0 差异的证据来自原验收记录；新 NPZ 明确保存的是 recorded_done，未伪称重新观测。

| 任务 | 通过/总量 | 状态最大误差 | 奖励最大误差 | 速度截断比例 | 历史修复行数 |
|---|---:|---:|---:|---:|---:|
| walker2d-morph | 5000/5000 | 3.91805e-05 | 7.51471e-07 | 37.80% | 0 |
| walker2d-gravity | 5000/5000 | 0.000933576 | 3.13257e-05 | 40.04% | 7 |
| hopper-morph | 5000/5000 | 1.74607e-06 | 3.20311e-07 | 5.80% | 0 |
| hopper-gravity | 5000/5000 | 1.63065e-06 | 4.68084e-07 | 4.76% | 0 |

验收阈值保持：状态 L∞≤0.001，奖励绝对误差≤0.0001，终止标记一致。Walker-gravity 最大误差并非 1e-6 量级；不能把文档中“全部 1e-6”作为事实复述。

### 2.2 正常 source：同一批 5000 点抽样验收

| 环境 | 首轮直接恢复失败 | 历史恢复后通过 | 状态最大误差 | 奖励最大误差 | done 差异 |
|---|---:|---:|---:|---:|---:|
| walker2d | 1560 | 5000/5000 | 3.17692e-05 | 1.08499e-06 | 0 |
| hopper | 861 | 5000/5000 | 1.90716e-06 | 5.46932e-07 | 0 |

抽样 seed=0；5000 条从保留的 source buffer 行中无放回抽取，两次验收使用相同原始行。历史恢复只沿同轨迹已验证前驱；额外短历史候选长度为 1、2、4、8、16、32、64，观测一致性阈值 1e-4。通过的不同历史还需给出一致的 shifted 转移。

### 2.3 主任务完整 source oracle 缓存

- **999322 / 999322 条保留 source 行全部通过正常域事实回放检查**；未删样本。
- 状态最大误差 `7.6900723e-05`；奖励最大误差 `6.2626989e-06`；done 不一致 `0`；短历史修复 `40` 行。
- oracle 缓存包含全部 source 行对应的 shifted `next_state/reward/not_done`。正常终止 513 条，shifted oracle 终止 506 条，其中终止状态变化 17 条。
- 保存的 D4RL source 行映射、全部数组有限性、文件哈希均通过；另以 seed=20261008 抽取 256 行重算 shifted 转移，float32 缓存逐位一致。
- source 继续使用原 D4RL qlearning_dataset 口径，target 使用文件显式 next_observations。Walker source 有 513 条 terminal 行的 next_observation 不同，但这些行的原 source bootstrap mask 为 0；非 terminal 行差异为 0。

**验收的范围：** 两个原域分别复现记录值，支持环境实现和状态恢复的可信度；同一模拟器生成后再重算的缓存一致性检查也不构成对“source 在 shifted 环境中的真实一步转移”的独立外部验证。观测截断意味着 oracle 还利用了轨迹历史中的额外信息。

## 3. 规模与重用次数

| 任务 | source 条数/轨迹段 | target 条数/轨迹段 | source:target | source 混合重用 | target 混合重用 | target-only 重用 |
|---|---:|---:|---:|---:|---:|---:|
| walker2d-morph | 999322/1191 | 5000/8 | 199.864:1 | 32.022 | 6400 | 12800 |
| walker2d-gravity | 999322/1191 | 5000/7 | 199.864:1 | 32.022 | 6400 | 12800 |
| hopper-morph | 999998/2187 | 5000/8 | 200.000:1 | 32.000 | 6400 | 12800 |
| hopper-gravity | 999998/2187 | 5000/8 | 200.000:1 | 32.000 | 6400 | 12800 |

重用次数=500000×每步该域采样条数/N，是有放回采样下的期望值。source 段数按 terminals/timeouts 划分，target 段数还使用记录的 next_obs 与下一行 obs 不连续边界。**5000 target 行只来自约 7–8 个轨迹段；五个训练 seed 不能代替独立重采数据。**


## 4. 阶段 B：实验协议与完成情况

| 组 | 每步 source / target | source 损失系数 | source Q 标签 | 完成情况 |
|---|---:|---:|---|---|
| TARGET | 0 / 128 | 无 source | 无 | seeds 0–4，各 500k，5/5 |
| MIX-CONST（MIX-NAIVE） | 64 / 64 | 0.20 | 原记录 | seeds 0–4，各 500k，5/5 |
| MIX-ORACLE-BACKUP | 64 / 64 | 0.20 | shifted oracle | seeds 0–4，各 500k，5/5 |

三组 IQL 网络结构、优化器及其参数相同；同一 seed 的初始网络哈希相同。总 batch=128，gamma=0.99，eta=1，source 系数=0.20，target_fraction=1，未交换 source/target。两混合组的 source V、actor 更新公式和系数一致，唯一有意改动是 source Q 的回归标签：

`y_s = r° + 0.99 × not_done° × V(next_state°)`。

CONST 使用 source 原记录的 `(r_s, next_state_s, not_done_s)`；ORACLE 使用按同一 source 行索引取得的 shifted 模拟器缓存。混合损失为 `mean(target_loss) + 0.20 × mean(source_loss)`；0.20 是损失系数，不是归一化后 source 严格占总损失的 20%。V 在训练中更新，**本实验不是固定同一个 V 的 kNN/OT reference 对比**，也没有运行 OT coupling。

TARGET 每项实际抽取 6400 万条 target；每项混合训练实际抽取 source、target 各 3200 万条。有放回采样的重复使用不等于获得新轨迹。每步域计数均有运行时断言，完成时总计数符合协议；同 seed 的混合组首批索引也逐项相同。未保存全程索引，不能把这些检查表述为逐步完整采样序列的事后逐位比对。

### 4.1 主指标及统计口径

- 每 10k 步评估一次，每次 10 个 episode；正式分数使用 460k、470k、480k、490k、500k 五次评估的均值（Last-5）。没有按最佳 checkpoint 或最好 seed 筛选。
- 每个 seed 先得到一个 Last-5 均值，再对五个 seed 统计。样本 SD 使用分母 `n−1=4`，`SE=SD/√5`；配对差先在同 seed 内相减。
- Last-5 的五个 checkpoint 相关，不能当成五次独立训练；也不能把五 seed × 五 checkpoint × 十 episode 当成 250 个独立训练样本。
- 主比较为 ORACLE−TARGET；ORACLE−CONST 判断相同混合流程下 source Q 修正的增量价值。ORACLE−TARGET 同时包含数据构成、source actor/V 和每步 target 采样数的差别。
- 正向工程门槛为 `mean > 2×SE` 且至少 4 个正号。`2×SE` 不是五种子下严格的 95% 置信界；门槛未通过不等于证明普遍不存在收益。

### 4.2 五种子汇总

下表 SD/SE 是各方法五个 seed 的 Last-5 均值之间的波动。原始 return 均值来自同一批 25 次 checkpoint 评估，供分数单位复核。

| 方法 | Last-5 均值 | 样本 SD | SE | 原始 return 均值 |
|---|---:|---:|---:|---:|
| TARGET | 39.2960 | 3.7528 | 1.6783 | 1734.53 |
| MIX-CONST | 35.4155 | 2.0761 | 0.9285 | 1564.24 |
| MIX-ORACLE-BACKUP | 35.3908 | 7.9920 | 3.5741 | 1563.15 |

### 4.3 每个 seed 的分数与配对差

| seed | TARGET | CONST | ORACLE | ORACLE−TARGET | CONST−TARGET | ORACLE−CONST |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 37.8097 | 34.0564 | 39.5621 | +1.7524 | -3.7533 | +5.5057 |
| 1 | 42.9067 | 36.4699 | 45.3595 | +2.4528 | -6.4368 | +8.8896 |
| 2 | 42.2916 | 32.6466 | 24.0872 | -18.2045 | -9.6450 | -8.5595 |
| 3 | 39.8190 | 37.9248 | 32.1130 | -7.7060 | -1.8942 | -5.8118 |
| 4 | 33.6532 | 35.9801 | 35.8323 | +2.1791 | +2.3268 | -0.1477 |

### 4.4 配对统计与预设判据

| 比较 | 配对均值 | 样本 SD | SE | 2×SE | 正号数 | 工程门槛 |
|---|---:|---:|---:|---:|---:|---|
| ORACLE−TARGET | -3.9052 | 9.0604 | 4.0520 | 8.1039 | 3/5 | 未通过：观察均值非正 |
| CONST−TARGET | -3.8805 | 4.5348 | 2.0280 | 4.0560 | 1/5 | 未通过：观察均值非正 |
| ORACLE−CONST | -0.0247 | 7.3552 | 3.2894 | 6.5787 | 2/5 | 未通过：观察均值非正 |

**解释：** ORACLE 对 TARGET 在 seeds 0、1、4 有约 1.75–2.45 分收益，但 seeds 2、3 分别为 −18.20、−7.71 分；小幅正收益未能稳定抵消负向结果。ORACLE 对 CONST 在 seeds 0、1 为 +5.51、+8.89，在 seeds 2、3 为 −8.56、−5.81，seed4 接近零。不能只选两个正向 seed 作为有效证据，也不能把 seed2 的较低回报自行当作故障剔除：该项的完成标记、有限性、数据、评估及日志检查均通过。

ORACLE 的跨 seed SD 为 7.99，当前样本中比 CONST 的 2.08 和 TARGET 的 3.75 更大；这是描述性观察，尚不足以证明该方法在总体上必然更不稳定。CONST 相对 TARGET 为 −3.88 分（1/5 正号），当前混合流程本身也未显示收益。

### 4.5 Last-5 原始评估分数

以下每格均为该 checkpoint 的 10-episode 平均 normalized return。完整 10k–500k 曲线保留在原始 TensorBoard 与正式审计 JSON 中。

| 方法 | seed | 460k | 470k | 480k | 490k | 500k | Last-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TARGET | 0 | 32.4735 | 38.2069 | 39.4456 | 39.3173 | 39.6051 | 37.8097 |
| CONST | 0 | 29.4042 | 27.7112 | 29.0752 | 45.0171 | 39.0742 | 34.0564 |
| ORACLE | 0 | 35.2749 | 39.0205 | 36.9173 | 44.2778 | 42.3198 | 39.5621 |
| TARGET | 1 | 48.9457 | 37.6404 | 38.3730 | 45.8846 | 43.6895 | 42.9067 |
| CONST | 1 | 31.4709 | 34.0604 | 31.1038 | 42.2460 | 43.4683 | 36.4699 |
| ORACLE | 1 | 43.5795 | 48.9049 | 39.9777 | 45.3213 | 49.0139 | 45.3595 |
| TARGET | 2 | 41.7105 | 46.0208 | 43.6905 | 43.4688 | 36.5675 | 42.2916 |
| CONST | 2 | 40.4327 | 33.8747 | 30.2882 | 27.7155 | 30.9219 | 32.6466 |
| ORACLE | 2 | 29.7908 | 25.1354 | 32.4620 | 19.1145 | 13.9330 | 24.0872 |
| TARGET | 3 | 41.2873 | 57.5341 | 38.4290 | 27.6295 | 34.2151 | 39.8190 |
| CONST | 3 | 34.2360 | 37.6485 | 45.7064 | 39.1767 | 32.8561 | 37.9248 |
| ORACLE | 3 | 33.6838 | 29.8922 | 39.1129 | 32.9281 | 24.9480 | 32.1130 |
| TARGET | 4 | 34.3930 | 33.9473 | 40.0521 | 26.9845 | 32.8893 | 33.6532 |
| CONST | 4 | 41.5460 | 25.2355 | 38.3402 | 42.8350 | 31.9437 | 35.9801 |
| ORACLE | 4 | 27.0117 | 40.9790 | 29.6269 | 38.7719 | 42.7722 | 35.8323 |

## 5. 独立核验与实际运行成本

最终核验没有启动任何新训练。先重新调用实际审计脚本的审计函数，复现 3 项 smoke 和 15 项正式审计；再独立读取原始 TensorBoard 的精确标签，用 NumPy float64 计算均值、样本 SD、SE、配对差和正号数，与正式审计逐项核对一致。

| 核验对象 | 结果与证据范围 |
|---|---|
| 训练完成性 | 3/3 smoke、15/15 正式；正式均到 500k，750 万步全部完成 |
| 评估完整性 | 750 次正式 checkpoint 评估 + 3 次 smoke 评估，步数齐全，无重复或缺失 |
| 实际评估对象 | 运行时从原 train.py 调用帧确认 `tar_eval_return` 来自创建的 shifted 对象；753 条记录的 Walker2d v3 类、OTDF XML 哈希及 lookup key 均复核 |
| OTDF 归一化 | `(return−10.079455055289959)/(4398.43−10.079455055289959)×100`；原始 return 转分数的最大浮点差约 4.46e−6，低于 2e−5 容差 |
| 参数与配对 | seed/target_only 之外的正式训练配置相同；同 seed 初始化哈希相同；首批混合采样一致且独立 RNG 重算一致；source 平均权重 .20、ESS 比例 1 |
| source/oracle 对齐 | 当前 HDF5 重建 999322 条保留行及五个字段哈希；raw_indices、三个 oracle 数组及恢复证据哈希一致，数组有限；终止变化仍为 17 条 |
| target 对齐 | 5000 条，显式 next_observations；当前文件哈希与每项训练的全行对齐记录一致；轨迹段仍为 8 |
| 模型与日志有限性 | 18 项共 126 个最终模型/优化器文件、78 个 V 快照均有限；日志中的全部 simple-value 标量也无 NaN/Inf |
| 受保护文件 | 1521 个旧文件 SHA256 均未变；阶段 A 冻结的 12 个 v3 脚本也未变 |
| 恢复证据 | attempt3 只变更实际清单/审计入口；清单只替换 ORACLE seed3 的输出路径；旧中断文件及 recovery1 原错误日志哈希未变 |

实际 TensorBoard 标签为 `test/target_return` 和 `test/target_normalized_score`。源码字符串中的空格会被 TensorBoardX 转成下划线；`effective_config.json` 保留源码标签字符串，最终核验以实际事件标签为准。

主环境 XML 是 `reproductions/OTDF/envs/assets/walker_morph.xml`，使用 `Walker2dEnv(xml_file=...)` 与 TimeLimit(1000)，没有回退至默认正常环境。对实际训练对象的核验依据是训练时断言和保存的逐次记录，并非训练结束后重新评估策略。

三个 smoke 均为 seed0、5000 步，进度计时分别为 TARGET 66.08 秒、CONST 104.88 秒、ORACLE 105.19 秒；只用于运行验收，不计入效果统计。

15 项正式训练进度计时合计 **38.29 小时**；从首次正式启动到最终训练完成的墙钟时间约 **53.53 小时**，后者包含中断和等待。进度计时含初始化及过程中评估、不含末次评估尾部，不能称为纯 GPU 计算时间。额外中断的40k和smoke不包含在正式750万步内。

训练时串行使用一张 RTX 2060 SUPER（8GB），每项 2 个 CPU 计算线程；oracle 数组在 CPU 内存映射，取 batch 后才转到 GPU。监控快照显示 GPU 总占用约 1.6GB，这不是全过程峰值。进度中的 PyTorch 张量分配峰值约 6–7MB 也不包含 CUDA 上下文等开销，不能把它当成整张卡的显存需求。

## 6. 研究判断、限制与后续停点

**问题一：oracle 相对 target-only 有多大提升？** 本轮观察是 −3.91 分，不是正提升；预设正向门槛未通过。原方案称其为“天花板不存在”，更准确的表述是“当前单任务、数据与 IQL 混合协议下，未观察到 oracle 修正带来稳定正收益”。它不否定其他预算、数据、算法或全部 Bellman 方法的可能性。

**问题二：oracle 是否优于直接混数据？** 没有稳定增量证据：ORACLE−CONST 为 −0.0247 分、SE=3.29、2/5 正号。MIX-NAIVE 与 CONST 完全同义，本轮未浪费预算另跑重复组。接近零的点估计不是等效性证明，也不是对某个正收益幅度的排除证明。

**问题三：无模拟器估计器能否接近 oracle？** 尚未检验。阶段 C 未执行；本轮也没有重跑 kNN/OT reference 方法，因此不能从这个训练结果宣布 OT 比 kNN 更好或更差。

需要保留的解释边界：

- **target 只有约 8 个轨迹段。** 五个优化 seed 复用了同一份数据，不能估计独立收集 target 数据的波动；Last-5 也不能解决这一限制。
- **V 是学得函数。** 替换模拟器的一步转移不意味着得到真实最优 Bellman 值；critic、V 与 actor 的后续学习会互相影响。
- **source actor/V 仍按原流程更新。** 它们可能造成负迁移；oracle只改 source Q，因此这个组不是所有修正方法的数学上界。
- **oracle 使用特权状态及历史恢复。** source 的速度存在截断，历史信息有助于恢复；这不等价于部署时只根据单个观测 `(s,a)` 可获得的条件转移。
- **两边事实回放与缓存复算有明确范围。** 它们支持环境与状态恢复的一致性，不能独立证明正常 source 在 shifted 环境下的反事实转移是真实外部真值。
- **与 OTDF/TABB 不是完整同协议对照。** 对齐 OTDF 归一化只统一了分数单位；本轮是 IQL、.20 系数、64+64 batch、5000 target、显式 loader 及模拟器/history oracle。不能凭本轮数字声称胜过或不如论文方法。

**决策建议（判断）：** 当前证据不支持直接扩大 OT 训练或进入估计器开发。先将这轮作为已完成的负向诊断保留。如果未来仍投入，优先选择能够区分“source actor/V 影响”“V 学习误差”“target 数据覆盖”的小规模归因实验，并使用独立 target 轨迹；这些是候选解释，不是本轮已证明的机制。本次没有启动这些新实验，也没有改变 .20 系数、eta 或种子来追逐正结果。

## 7. 中断、修复与产物选择

1. **Stage A 初次状态恢复失败保留。** 首先误将截断的 infos/qvel 视为完整物理状态；同一批样本用历史恢复后通过，阈值和样本未改变（见第2节）。
2. **recovery1 仅修复 TensorBoard 标签读取。** 3 项 smoke 本身成功，首版审计误期待带空格标签而失败；新审计统一标签后通过，没有重跑训练。
3. **recovery2 恢复训练队列。** 发现原队列与 ORACLE seed3 进程已退出，原任务停在40k；未发现同期系统重启，日志无 traceback，原因仍不明。保留11项已完成训练及原 `b1/o3`，将相同 seed3 在 `b2/o3` 从0重跑500k，再完成seed4三项。没有把40k片段拼接进正式曲线，也没有按得分决定重跑。
4. **最终统计只读取实际清单。** ORACLE seed3 使用 `b2/o3`，其余正式任务在 `b1/`；旧 `b1/o3` 仅作中断证据。
5. **交付复核脚本一次形状修复。** `final_review1` 中的新核验将 `(5000,)` 边界与 `(5000,1)` 原始终止数组广播，导致段数断言失败。原训练 loader 本已展平；新 `review_stage_b_v3_attempt2.py` 显式展平后重算仍为8段，`final_review2/review.json` 全部通过。失败记录及修复前后哈希保留；训练代码和数据未改。

## 8. 可复核入口

- [最终独立核验及全部精确数字](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/final_review2/review.json>)
- [原始 TensorBoard 正式审计](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/formal_audited.json>)
- [GPU smoke 审计](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/smoke_audited.json>)
- [实际训练清单（含替代seed3）](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/recovery2/training_plan.json>)
- [冻结协议](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/protocol.json>)
- [GPU 授权](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/gpu_authorization.json>)
- [Stage A 最终审计](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/stage_a_completed.json>)
- [source 历史恢复抽样验收](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/stage_a2/sample_completed.json>)
- [全量 source oracle](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/source_oracle1/completed.json>)
- [数据规模与轨迹段](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/stage_a1/scale.json>)
- [CPU 前向公式及采样检查](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/preflight1/completed.json>)
- [TensorBoard 标签修复记录](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/recovery1/repair.json>)
- [队列恢复记录](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/recovery2/repair.json>)
- [交付复核形状修复记录](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/final_review1/failure.json>)
- [实际运行审计脚本](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v3/audit_runs_v3_attempt3.py>)
- [交付独立复核脚本](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v3/review_stage_b_v3_attempt2.py>)
- [报告生成脚本](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v3/report_stage_b_v3.py>)
- [训练入口](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v3/train_entry_v3.py>)
- [source Q 接口](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v3/methods_v3.py>)
- [环境/数据适配器](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v3/task_env.py>)

原始 TensorBoard 文件的绝对路径与 SHA256、每个 checkpoint 的完整分数均在正式审计 JSON 中；原报告的交付前快照保存在 final_review2/REPORT_v3.md.before。旧实验、失败产物与历史恢复证据均保留。

### 只读复算命令

以下命令重读原始日志和模型，在控制台打印已指定审计的结果，不会启动训练或覆盖原审计文件。

```powershell
Set-Location -LiteralPath 'C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main'
& 'D:\ANACONDA\envs\offdynamics\python.exe' -B -c "import os,sys,json; os.environ['CUDA_VISIBLE_DEVICES']='-1'; sys.path.insert(0,'oracle_diag/v3'); from audit_runs_v3_attempt3 import audit; print(json.dumps(audit('formal')['comparisons'],indent=2))"
```

交付复核脚本也可使用新的空目录重跑；它会拒绝覆盖已有核验目录。阶段 B 到此结束，阶段 C 留待单独决定。
