# Oracle v5：CPU 准备与零训练诊断交付（待 GPU 确认）

## 状态

A、B 实现和 CPU 检查完成；**v5 尚未运行 GPU 冒烟或正式训练，不能据此回答 QA1–QA4、QB1–QB4。**
已准备 30 项正式训练（A 15 项、B 15 项）、8 项单独 GPU 冒烟及 2 项并行冒烟。已有基线仅重读并核验日志，跨入口复用还须通过 GPU 检查。

## 1. 实现与证据

- A：CRITIC-ORACLE / CRITIC-SOURCE 仅在 actor 中排除 source；Q-ORACLE 还在 V 中排除 source。每步仍抽样 64+64，source 系数 1.0。
- A：固定网络和 target batch、更换 source batch 后，前两组 actor 逐位一致；Q-ORACLE 的 actor、V 逐位一致；应受影响的网络确实改变。独立梯度检查也确认被排除的 source 项无梯度。
- A：v5 的 ORACLE-W1 复用入口与 v4 在 CPU 的 10 步参数、损失轨迹和抽样计数完全一致；GPU 复用检查单独执行。
- B：TargetIQL 只覆盖 train；V、Q、actor、目标 Q 更新、VAE 和学习率调度均沿用官方实现。IQL-TARGET 与 +REG 仅用官方 noreg 开关区分，source 训练抽样为 0，target 为 128。
- B：OTDF-NOREG 使用官方类，CPU 检查确认抽样 128+128、过滤后 source 为 102，三个损失的有效 batch 均为 230。Q 使用 OT 权重，V 和 actor 不使用该权重。
- B：保留官方 actor 中优势 [B,1] 与 BC 损失 [B] 的广播行为，未在消融中悄悄修复；因此结果解释限定于该官方实现。
- 数据：A 使用 5000 条 explicit_next_v3 target；B 保留官方 4999 条 next-row target；B 原始数据、OT 代价、导出行次序和代码通过历史哈希核验。
- 旧文件保护：本次完整核验 3037 项哈希，全部一致。

证据：[completed.json](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/cpu_a1/completed.json>), [completed.json](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_v5/results_20261013/cpu_b3/completed.json>), [existing_baselines.json](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/existing_baselines.json>), [protocol.json](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/protocol.json>)。

## 2. A5：固定 5000 条 target 状态的动作诊断

MSE 在全部状态与 6 个动作维度上平均。source 参考动作来自 target 均值/标准差（总体 SD + 1e-6）标准化空间的精确 5 近邻，再平均近邻动作；12 个预先固定查询点通过独立穷举距离核对。表中为五种子均值 ± 样本 SD。

| 策略 | 与 target 动作的 MSE ↓ | 与近邻 source 均值动作的 MSE ↓ |
|---|---:|---:|
| TARGET-64 | 0.00013234 ± 0.00000460 | 0.42354135 ± 0.00003816 |
| ORACLE-W1 | 0.00282482 ± 0.00081592 | 0.41956532 ± 0.00082456 |
| ORACLE-ONLY | 0.54386661 ± 0.01694803 | 0.26457329 ± 0.00825183 |

两个含 source 的策略都更远离 target 动作、更接近近邻 source 动作。ORACLE-W1 向 source 靠近的绝对幅度较小（MSE 约 0.42354→0.41957）；不能把相对倍数或 source-only 的大幅差异外推为混合策略的主要因果机制。
这是描述性支持，不能代替路径切断后的回报对照；近邻 source 动作是代理参照，并非 shifted 环境的最优动作。
原始逐 seed 记录、模型路径及哈希：[completed.json](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/a5_2/completed.json>)。

## 3. 分组与逐 seed 结果

**A 与 B 的分数不得直接排名比较。** A 为 500k 步、seeds 0–4；B 为 1M 步、seeds 1–5，且 loader、网络更新实现和 batch 不同。B 有正则时另预训练 VAE 10k 步；无正则组不执行 VAE 预训练。

以下仅为已存在基线的重新核验值，不是 v5 新训练结果。

### 实验 A

| 方法 | seed | Last-5 | Final |
|---|---:|---:|---:|
| TARGET-64 | 0 | 40.457115 | 34.781967 |
| TARGET-64 | 1 | 40.028674 | 35.885654 |
| TARGET-64 | 2 | 43.571122 | 37.263664 |
| TARGET-64 | 3 | 35.803947 | 34.569355 |
| TARGET-64 | 4 | 43.094704 | 46.446632 |
| ORACLE-W1 | 0 | 15.755073 | 13.461314 |
| ORACLE-W1 | 1 | 31.525633 | 34.001793 |
| ORACLE-W1 | 2 | 22.571896 | 22.237652 |
| ORACLE-W1 | 3 | 34.672418 | 28.741562 |
| ORACLE-W1 | 4 | 30.795204 | 36.658112 |
| TARGET | 0 | 37.809705 | 39.605148 |
| TARGET | 1 | 42.906663 | 43.689526 |
| TARGET | 2 | 42.291613 | 36.567467 |
| TARGET | 3 | 39.818993 | 34.215076 |
| TARGET | 4 | 33.653234 | 32.889317 |

### 实验 B

| 方法 | seed | Last-5 | Final |
|---|---:|---:|---:|
| OTDF | 1 | 46.864701 | 36.919565 |
| OTDF | 2 | 38.587349 | 39.773131 |
| OTDF | 3 | 40.366880 | 37.207272 |
| OTDF | 4 | 46.282511 | 55.171332 |
| OTDF | 5 | 46.542937 | 52.596900 |

## 4. 预设比较

通过条件：五种子配对差均值 > 2×SE 且至少 4 个正号，SD 使用 ddof=1，SE=SD/√5。这是工程门槛，不是严格 95% 置信结论。

| 编号 | 比较（左 − 右） | 配对差 seed 顺序 | 均值 | SD | SE | 正号数 | 门槛 |
|---|---|---|---:|---:|---:|---:|---|
| QA1 | CRITIC-ORACLE − TARGET-64 | 待训练 | — | — | — | — | 待定 |
| QA2 | CRITIC-ORACLE − ORACLE-W1 | 待训练 | — | — | — | — | 待定 |
| QA3 | CRITIC-ORACLE − CRITIC-SOURCE | 待训练 | — | — | — | — | 待定 |
| QA4 | Q-ORACLE − CRITIC-ORACLE | 待训练 | — | — | — | — | 待定 |
| QB1 | OTDF − IQL-TARGET | 待训练 | — | — | — | — | 待定 |
| QB2 | OTDF − IQL-TARGET+REG | 待训练 | — | — | — | — | 待定 |
| QB3 | IQL-TARGET+REG − IQL-TARGET | 待训练 | — | — | — | — | 待定 |
| QB4 | OTDF − OTDF-NOREG | 待训练 | — | — | — | — | 待定 |

## 5. 结论的解释范围

- QA1 检验当前 critic-only source 使用方案相对 TARGET-64 的回报；QA2 检验移除 source 直接 actor 损失的效果，不单独证明 actor 是唯一或主要伤害来源。source 仍可通过学得的 Q、V 间接影响后续 actor。
- QA3 衡量这一特权标签修正干预的实际增益，不是动力学修正的数学上限；QA4 检验再移除 source V 损失的影响。
- QB1 是同一官方代码中的 OTDF 与 target-only 比较；QB2 衡量过滤加权 source 及 batch 从 128 增到 230 的组合干预，不是严格固定 batch 下的纯数据效应。
- QB3、QB4 检验正则流程的效果；VAE 预训练会消耗随机数，同 seed 不意味着之后抽样序列相同。通过也不能宣称该正则在任何方法中都必要。
- 不通过正向门槛只能报告未观察到稳定正向差异，不能推断等效、所有源数据无价值或所有源数据有害，也不能证明优于论文方法。
- 局限：单任务；target 仅约 8 条轨迹段；V 是学得函数；oracle 利用历史恢复的隐藏物理状态；历史复用与新实验运行时间不同；Windows 兼容复现的软件环境与论文有差异。保留阶段 A 失败和历史恢复证据。

## 6. 下一阶段与确认点

先执行两条管线各自的复用冒烟和三个新组冒烟，再执行 A/B 并行吞吐冒烟。每个进程限制 2 个 CPU 线程。测得吞吐后给出两条正式队列的预计耗时；如 B 超过 72 小时，将列出可选缩减方案，由用户决定。当前完整 30 项计划保持不变。
GPU 冒烟及正式训练均未启动。用户文件要求 GPU 前确认，并要求冒烟汇报后再次确认正式训练。
若正式并行时发现吞吐低于单独运行的一半，脚本将阻止 B 启动下一项，优先完成 A 前两组；正在运行的 B 任务保留到安全作业边界，避免丢失不可续训的优化器状态。该操作方式会在冒烟汇报中明确列出。
准备阶段修复：A5 使用旧 SciPy 支持的精确单线程近邻查询；B 模块改用独立名称以防误导入 A；旧 v3 评估凭据没有 time_limit 字段，历史审计按其已有实际环境标识核验，新运行仍强制检查 1000 步限制。所有原始训练、完成标记和失败训练产物均未改变。

## 附录：逐 checkpoint 与来源

| 实验 | 方法 | seed | Last-5 五个原始归一化分数 | TensorBoard 来源 |
|---|---|---:|---|---|
| A | TARGET-64 | 0 | 48.220737, 40.319080, 39.240665, 39.723125, 34.781967 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/h0/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r0/tb/events.out.tfevents.1791752584.DESKTOP-B3JMOOO>) |
| A | TARGET-64 | 1 | 43.575279, 39.521606, 34.332016, 46.828812, 35.885654 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/h1/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r1/tb/events.out.tfevents.1791758617.DESKTOP-B3JMOOO>) |
| A | TARGET-64 | 2 | 46.051147, 55.205074, 40.527481, 38.808243, 37.263664 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/h2/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r2/tb/events.out.tfevents.1791764661.DESKTOP-B3JMOOO>) |
| A | TARGET-64 | 3 | 34.807980, 29.522648, 40.501373, 39.618382, 34.569355 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/h3/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r3/tb/events.out.tfevents.1791770676.DESKTOP-B3JMOOO>) |
| A | TARGET-64 | 4 | 45.319263, 39.499420, 42.437340, 41.770863, 46.446632 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/h4/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r4/tb/events.out.tfevents.1791776965.DESKTOP-B3JMOOO>) |
| A | ORACLE-W1 | 0 | 19.998363, 10.567570, 17.412724, 17.335392, 13.461314 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/w0/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r0/tb/events.out.tfevents.1791783842.DESKTOP-B3JMOOO>) |
| A | ORACLE-W1 | 1 | 26.693624, 31.838863, 31.554821, 33.539062, 34.001793 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/w1/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r1/tb/events.out.tfevents.1791795510.DESKTOP-B3JMOOO>) |
| A | ORACLE-W1 | 2 | 26.025995, 16.861601, 18.495335, 29.238895, 22.237652 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/w2/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r2/tb/events.out.tfevents.1791807146.DESKTOP-B3JMOOO>) |
| A | ORACLE-W1 | 3 | 35.527145, 34.598740, 35.957741, 38.536903, 28.741562 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/w3/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r3/tb/events.out.tfevents.1791819326.DESKTOP-B3JMOOO>) |
| A | ORACLE-W1 | 4 | 31.678026, 32.061821, 27.665968, 25.912094, 36.658112 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v4_20261011/b1/w4/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r4/tb/events.out.tfevents.1791831312.DESKTOP-B3JMOOO>) |
| A | TARGET | 0 | 32.473530, 38.206921, 39.445621, 39.317307, 39.605148 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/b1/t0/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r0/tb/events.out.tfevents.1791447495.DESKTOP-B3JMOOO>) |
| A | TARGET | 1 | 48.945667, 37.640450, 38.373035, 45.884636, 43.689526 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/b1/t1/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r1/tb/events.out.tfevents.1791476399.DESKTOP-B3JMOOO>) |
| A | TARGET | 2 | 41.710495, 46.020767, 43.690525, 43.468811, 36.567467 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/b1/t2/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r2/tb/events.out.tfevents.1791502847.DESKTOP-B3JMOOO>) |
| A | TARGET | 3 | 41.287323, 57.534115, 38.428982, 27.629471, 34.215076 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/b1/t3/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r3/tb/events.out.tfevents.1791530493.DESKTOP-B3JMOOO>) |
| A | TARGET | 4 | 34.392998, 33.947254, 40.052090, 26.984512, 32.889317 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v3_20261008/b1/t4/IQL/walker2d-morph-srcdatatype-medium-tardatatype-medium-medium/r4/tb/events.out.tfevents.1791613207.DESKTOP-B3JMOOO>) |
| B | OTDF | 1 | 48.534214, 54.985304, 42.869213, 51.015209, 36.919565 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_setup/runs/train_1M_restart_20260921_1052_seed1/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r1/tb/events.out.tfevents.1789959341.DESKTOP-B3JMOOO>) |
| B | OTDF | 2 | 35.205557, 33.503866, 46.093776, 38.360416, 39.773131 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_setup/runs/train_5seeds/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r2/tb/events.out.tfevents.1789982321.DESKTOP-B3JMOOO>) |
| B | OTDF | 3 | 27.452063, 44.145778, 45.449253, 47.580033, 37.207272 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_setup/runs/train_5seeds/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r3/tb/events.out.tfevents.1790004368.DESKTOP-B3JMOOO>) |
| B | OTDF | 4 | 47.137483, 44.890220, 41.799709, 42.413812, 55.171332 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_setup/runs/train_5seeds/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r4/tb/events.out.tfevents.1790026576.DESKTOP-B3JMOOO>) |
| B | OTDF | 5 | 52.151051, 43.157452, 40.674657, 44.134624, 52.596900 | [原始日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_setup/runs/train_5seeds/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r5/tb/events.out.tfevents.1790048722.DESKTOP-B3JMOOO>) |
