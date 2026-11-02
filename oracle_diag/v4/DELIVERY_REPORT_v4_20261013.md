# Oracle v4 归因实验：最终交付与独立核验

交付日期：2026-10-13。**20/20 项正式训练完成，每项 500000 步，独立核验通过。** 5 组 GPU smoke 已通过；TARGET 的五个正式基线按冻结方案复用 v3。

## 1. 结论先读

**oracle 标签对纯源域训练有帮助，但这种改善尚未转化为超过 TARGET 的表现。提高混合训练的 source 权重到 1.0，也没有解决本轮的性能差距。**

- ORACLE-ONLY 相比 SOURCE-ONLY 提高 **6.5464 分**，五个 seed 全为正，达到预设工程门槛。这支持“在相同纯源域训练流程中，替换为 oracle 标签有益”。
- ORACLE-ONLY 相比 TARGET 低 **20.5412 分**，五个 seed 全为负。其纯源域训练性能仍明显落后于本轮 TARGET 基线；标签改善本身不足以弥补差距。
- ORACLE-W1 相比 TARGET 低 **12.2320 分**，五个 seed 全为负。当前这套权重为 1.0 的混合方案没有带来收益。
- TARGET-64 的平均分最高，为 **40.5911**；TARGET（batch 128）为 **39.2960**。两者配对差的波动较大，不能据此认定 TARGET-64 稳定更好，也不能认定二者等效。

这些结果支持的是本任务、数据和训练流程下的具体判断。它们不能推出“所有源域数据都有害”，也不能唯一证明瓶颈就是状态分布、权重、V 函数或某一条损失路径。

## 2. 完成范围与指标

任务为 walker2d-morph 标准方向：正常 D4RL medium source → shifted OTDF medium target；评分使用实际 shifted Walker 环境。每组 seeds 0–4、500k 步，eta=1.0，其余参数遵守冻结协议。

| 方法 | 每步实际抽样 | 本轮执行 | Last-5 均值 | 样本 SD | SE |
| --- | --- | --- | --- | --- | --- |
| TARGET | 128 target | 复用 v3 五项 | 39.2960 | 3.7528 | 1.6783 |
| ORACLE-ONLY | 128 source，oracle 标签 | 五项完成 | 18.7549 | 0.7883 | 0.3525 |
| SOURCE-ONLY | 128 source，原始标签 | 五项完成 | 12.2085 | 6.1733 | 2.7608 |
| TARGET-64 | 64 target | 五项完成 | 40.5911 | 3.0984 | 1.3856 |
| ORACLE-W1 | 64 source + 64 target，source 系数 1.0 | 五项完成 | 27.0640 | 7.7425 | 3.4626 |

前两组训练不读取 shifted target 数据或其经验统计。ORACLE-W1 的 Q、V、actor 损失均按 target 均值 + 1.0 × source 均值组合，oracle 仅替换 source Q backup 标签。

**分数单位是 OTDF 归一化分，不是百分比增幅。** 每个 seed 的 Last-5 是 460k、470k、480k、490k、500k 五个 checkpoint 的归一化评估均值；每次评估 10 个 episode。先计算每个 seed 的 Last-5，再在五个 seed 间计算均值、样本 SD（分母 4）和 SE=SD/√5。不能把五次 checkpoint 或评估 episode 当成额外的独立训练 seed。

归一化使用 `(原始回报 − 10.079455055289959) / (4398.43 − 10.079455055289959) × 100`。原报告附录 A 的“原始分数”指直接读取 TensorBoard 的**归一化分数**，不是未经归一化的环境回报。

## 3. 四个预设主比较

差值方向保持冻结协议定义；“正号数”是左侧方法高于右侧方法的 seed 数。

| 比较 | 配对平均差 | 配对样本 SD | 配对 SE | 正号数 | 预设正向门槛 |
| --- | --- | --- | --- | --- | --- |
| Q1：ORACLE-ONLY − TARGET | -20.5412 | 3.7038 | 1.6564 | 0/5 | 未通过 |
| Q2：ORACLE-ONLY − SOURCE-ONLY | +6.5464 | 5.8666 | 2.6236 | 5/5 | 通过 |
| Q3：TARGET − TARGET-64 | -1.2951 | 5.3340 | 2.3854 | 2/5 | 未通过 |
| Q4：ORACLE-W1 − TARGET | -12.2320 | 8.5345 | 3.8167 | 0/5 | 未通过 |

**Q1/Q2：标签修正有效，但仍有明显性能缺口。** Q2 在相同 source 状态、动作和训练流程中比较标签替换，提供本轮最直接的正向证据。Q1 的五个差值均为负，表明该完整 source-only 方案仍低于 TARGET；Q1 同时改变数据量、状态及动作分布，不能把全部差距归给某个单一因素。

**Q3：未发现降低 target batch 必然导致退化的证据。** TARGET-64 的均值略高，五个 seed 中三正两负（这是将 Q3 反向阅读）。当前数据不支持“混合训练只是因为 target batch 从 128 降到 64 才变差”的简单解释；Q3 属于 target-only 对照，不能定量分解混合训练中的交互效应。

**Q4：权重 1.0 的混合方案没有超过 TARGET。** 相比 v3 权重 0.20 的 MIX-ORACLE-BACKUP，W1 的描述性平均差为 **-8.3268 分，SE 4.7207，正号 1/5**；这个描述性结果也不支持直接提高权重的收益预期。W1 同时改变多项损失的 source 权重和总体尺度，尚未分离各条路径的作用。

预设门槛为配对均值 > 2×SE 且至少 4/5 正号。**这是工程门槛，不是严格 95% 置信结论，也不是经多重比较校正的显著性检验。** 未通过门槛不等于证明零效应；本轮不改判据，不根据得分挑选重跑。

## 4. 独立核验结果

2026-10-13 的交付检查使用单独的 EventAccumulator 读取原始事件，以 `math.fsum` 和显式样本方差公式复算；统计部分不调用生产报告的计算函数。

| 核验对象 | 结果 |
| --- | --- |
| 20 项 v4 正式训练及 15 项 v3 历史对照 | 35 份原始事件文件核验通过 |
| 全程评估与 Last-5 | 1750 个归一化评分 checkpoint、175 个 Last-5 点核验通过，无重复步骤或非有限标量 |
| 四个主比较及三个描述性比较 | 逐 seed 配对差、样本 SD、SE、正号数与报告一致 |
| 归一化 | 从原始回报复算，最大绝对差 0.0000044581，小于冻结容差 0.00002 |
| 新训练模型、优化器、调度器及 V 快照 | 240 个产物完成 CPU 有限性和哈希复查 |
| 实际评估对象 | shifted 环境对象、XML 哈希、TimeLimit 1000、归一化查询键均与协议一致 |
| 数据与训练 | 逐项复核初始化、首批索引、累计抽样量、source/oracle 对齐凭据与缓存哈希 |
| 数据对齐基础证据 | CPU 验收覆盖全部 999322 行 source；行映射精确，缓存数组哈希一致；not_done 标签变化 17 行 |
| 冻结文件 | 训练代码、协议与计划一致；2111 项受保护旧文件哈希全部未变 |
| 报告 | 主表、逐 seed 表、完整配对表、Last-5 附录和标签诊断表均与来源一致，来源链接有效 |

核验全程使用 CPU，没有重新启动训练。正式队列已正常结束；中期报告及交付凭据保留。此前的失败、恢复和历史结果均保留，本次解读另存为新文件，原始最终报告保持不变。

## 5. 科学结论的范围

本轮最明确的研究发现是：**在该离线跨动力学任务中，oracle 一步标签能改善纯源域学习，但这一改善尚不足以超过仅使用 target 数据的策略；单独扩大 source 权重也没有消除差距。** 这提供了归因实验的证据，不构成一个已超过现有论文方法的新算法结果。

数据只有单个任务、五个优化 seed；target 5000 条转移仅约 8 条轨迹段，未覆盖独立采集数据的变化。V 是学得函数；oracle 依赖历史恢复的隐藏物理状态，属于特权信息。缓存事实回放支持工程可信度，但不提供理论性能上界，也不保证长期价值估计准确。

当前可把 TARGET/TARGET-64 作为本任务的强基线。要写“优于某论文方法”，仍需要该方法在统一数据、预算、评估对象和评分口径下的直接对照；本轮没有提供这项证据，不能把论文表格分数与这里的数值直接拼接排名。

## 6. 可追溯交付文件

- [完整原始最终报告：逐 seed、配对统计及 Last-5 明细](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v4/REPORT_v4.md>)
- [交付独立核验凭据](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/final_delivery1/verified.json>)
- [独立核验脚本](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/final_delivery1/verify_final.py>)
- [正式训练审计：原始事件与模型文件哈希](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/formal_audited.json>)
- [冻结协议](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/protocol.json>)
- [训练清单](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/training_plan.json>)
- [已交付的中期报告](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v4/INTERIM_v4.md>)

原始最终报告 SHA256：`71ea803afd21a58eb9ade9bf5fc91b57a7ee56a1ca99565969e002452a91d49f`。

独立核验凭据 SHA256：`42444e5aa994433367fa419ee9566325aa3ab76dcbd9950773f723daaa674394`。
