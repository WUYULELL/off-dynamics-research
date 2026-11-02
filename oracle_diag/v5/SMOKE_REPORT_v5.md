# Oracle v5：GPU 冒烟、复用与并行吞吐交付报告

交付日期：2026-10-13（北京时间）。范围：A、B 冒烟阶段；正式训练等待确认。

## 结论

**10 项 GPU 冒烟全部完成，A/B 历史复用通过，3037 项旧文件哈希保持一致。当前可以提交完整 30 项正式训练供确认。**
本轮完成 8 项单独冒烟、2 项同时运行的冒烟，每项 5000 策略更新步；有正则的 B 组另完成 10000 步 VAE 预训练。正式输出目录尚未创建，未启动正式训练。
并行测试未触发“吞吐低于单独一半”的门槛。预计 A 约 43–47 小时、B 约 66–70 小时；同时启动的整体排期约 3 天，建议预留 3–4 天。这是短测试外推，具体计算与限制见第 5 节。
**冒烟通过说明实现和运行链路可用，不能据此判断 source 有益、有害，或回答 QA1–QA4、QB1–QB4。**

核验依据：[独立交付核验（含原始路径、哈希及未舍入数值）](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/smoke_delivery1/verified.json>)；[原自动审计](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/smoke_audited.json>)。

## 1. 实现与 CPU 检查

| 实验 | 干预及固定条件 | 检查结果 |
|---|---|---|
| A：CRITIC-ORACLE | source 用于 V、Q；actor 仅用 target；source Q 使用 oracle backup | 固定初始网络与 target batch，更换 source 后直接 actor 更新不变 |
| A：CRITIC-SOURCE | 同上，source Q 使用原始 source 转移 | 直接 actor 更新不变；V、Q 对 source 干预有响应 |
| A：Q-ORACLE | source 仅用于 Q；V、actor 仅用 target | source 干预下直接 V、actor 更新均不变 |
| B：IQL-TARGET / +REG | 继承官方 OTDF 更新函数，仅 target 128；区别是 VAE 与 actor 正则开关 | 两种开关均与独立单步更新一致；source 抽样为零 |
| B：OTDF-NOREG | 官方过滤、Q 权重保留；关闭 VAE 与 actor 正则 | 抽样 128+128，source 保留 102，有效 batch 230 |

A 固定抽样 64 source + 64 target，source 系数 1.0。单步不变性与独立梯度检查确认的是被移除的直接损失路径；source 仍可能通过后续学得的 Q、V 间接影响 actor。
A 复用入口与 v4 在 CPU 的前 10 步参数、损失和抽样计数完全一致。B 沿用官方 V、Q、actor、目标 Q、学习率调度以及 actor 的广播行为；未在消融中额外修复官方算法。
证据：[A CPU](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/cpu_a1/completed.json>)；[B CPU](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_v5/results_20261013/cpu_b3/completed.json>)；[独立 CPU 验证](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/cpu_validation1/completed.json>)。

## 2. GPU 复用、数据及模型核验

- A ORACLE-W1 seed 0 对照 v4：初始参数哈希、首批样本、有效配置一致；前 10 步损失最大绝对差为 0；5000 步 actor、critic、V 参数最大绝对差均为 0；归一化评估分数差为 0。
- B OTDF seed 1 对照历史运行：初始 source 回报 53.1623987225、target 回报 47.8972691693；首个 VAE 损失 0.4300073087。上述差值均为 0，有效配置一致，达到预设复用条件。B 的复用判据是这三项，不能扩写为与历史完整训练逐参数一致。
- 5 个 B 冒烟任务在 VAE/策略更新前记录相同 seed 1 初始参数哈希。A/B 的并行与对应单独冒烟 actor、critic 最终参数分别完全一致，评估分数也一致。
- 10 项均对实际 shifted Walker 对象评估，核对 walker_morph.xml 哈希、TimeLimit=1000、每次 10 episodes。归一化为 100×(原始回报−10.079455055289959)/(4398.43−10.079455055289959)。
- A 使用 5000 条 explicit_next target；B 保留官方 4999 条 next-row target。source 999322 行，oracle/source 行对齐与 OT 代价行序核验通过。B target-only 可以加载 source，但训练抽样计数为零。
- 两种 TensorBoard 读取实现交叉核对 10 项原始 target_return 和归一化分数；CPU 加载全部模型及 V/VAE 保存参数，均为有限值。冻结的 18 个原实现脚本哈希一致，3037 项历史保护文件完整通过。
- 未改变协议、预算、种子、损失、正则开关或容差；未重跑已完成冒烟。此次核验与报告生成均使用 CPU。

## 3. 冒烟结果（工程检查）

下表速度 = 5000 / 累计 train 调用时间，不含初始化、VAE 预训练和评估；不是完整作业吞吐。5000 步分数仅用于日志与复用核验，不是 Last-5，也没有五种子统计。A、B 使用不同协议，分表展示，不能横向排名。

### A：单独及并行冒烟

| 方法 | seed | 运行方式 | 策略步数 | 训练速度（步/秒） | 5000 步归一化分数 | 累计 source / target 抽样 |
|---|---:|---|---:|---:|---:|---:|
| ORACLE-W1 | 0 | 单独 | 5000 | 52.16 | 9.504565 | 320000 / 320000 |
| CRITIC-ORACLE | 0 | 单独 | 5000 | 54.12 | 20.557388 | 320000 / 320000 |
| CRITIC-SOURCE | 0 | 单独 | 5000 | 59.72 | 8.702707 | 320000 / 320000 |
| Q-ORACLE | 0 | 单独 | 5000 | 52.17 | 10.680468 | 320000 / 320000 |
| CRITIC-ORACLE | 0 | 并行 | 5000 | 51.00 | 20.557388 | 320000 / 320000 |

### B：单独及并行冒烟

| 方法 | seed | 运行方式 | 策略步数 | 训练速度（步/秒） | 5000 步归一化分数 | 累计 source / target 抽样 |
|---|---:|---|---:|---:|---:|---:|
| OTDF | 1 | 单独 | 5000 | 42.61 | 11.843082 | 640000 / 1920000 |
| IQL-TARGET | 1 | 单独 | 5000 | 73.10 | 19.886543 | 0 / 640000 |
| IQL-TARGET+REG | 1 | 单独 | 5000 | 52.98 | 42.399817 | 0 / 1920000 |
| OTDF-NOREG | 1 | 单独 | 5000 | 79.25 | 15.454724 | 640000 / 640000 |
| IQL-TARGET | 1 | 并行 | 5000 | 77.23 | 19.886543 | 0 / 640000 |

B 有正则两组的 target 抽样包含 VAE 的 1280000 次，再加策略更新 640000 次；无正则组没有 VAE 抽样。所有运行都保留完成标记、原始 TensorBoard、配置、抽样计数与模型。

## 4. A5：已有策略的零训练动作诊断

全部 5000 个 target 状态，15 个已完成 actor。按 target 均值和总体 SD+1e-6 标准化状态，使用精确 k=5 source 近邻的平均动作；12 个固定查询点独立穷举核对。表中为五种子均值 ± 样本 SD，MSE 在状态及动作维度上平均。

| 策略 | 与 target 动作的 MSE ↓ | 与近邻 source 均值动作的 MSE ↓ |
|---|---:|---:|
| TARGET-64 | 0.00013234 ± 0.00000460 | 0.42354135 ± 0.00003816 |
| ORACLE-W1 | 0.00282482 ± 0.00081592 | 0.41956532 ± 0.00082456 |
| ORACLE-ONLY | 0.54386661 ± 0.01694803 | 0.26457329 ± 0.00825183 |

两个含 source 的策略均更远离 target 动作、更接近近邻 source 动作。ORACLE-W1 接近 source 的绝对幅度较小；近邻动作不是 shifted 环境的最优动作。这是描述性证据，不能证明 source actor 项是主要伤害通道，也不能代替正式回报对照。
[A5 逐 seed 原始结果与模型哈希](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/a5_2/completed.json>)

## 5. 并行吞吐与正式排期

记录显示两边同时进行策略更新约 64.32 秒。A 的末段在 B 更新结束后继续运行，直接用整段平均会高估真正并行速度。因此按 progress 文件时间锚定 throughput.jsonl，使用落在共同训练区间内的完整片段，并以 TensorBoard 训练标量时间校验锚点。

| 并行方法 | 对比步段 | 相同步段单独速度 | 并行速度 | 保留比例 |
|---|---|---:|---:|---:|
| A：CRITIC-ORACLE | 1000–3000 | 54.49 | 49.19 | 90.3% |
| B：IQL-TARGET | 1000–5000 | 71.86 | 77.11 | 107.3% |

两边均高于预设 50% 门槛。B 的 107.3% 来自单次短测试，不能解释为可靠加速；耗时外推将其截为 100%。仅测量了 CRITIC-ORACLE 与 IQL-TARGET 的并行组合，尚无 +REG 并行实测。

### 完整五种子清单的耗时估计

主估计逐组计算：一次初始化 + 一次 VAE（如有）+ 正式步数/实测训练速度 + 周期评估/保存开销。A 有 50 次、B 有 100 次周期评估；以冒烟末次训练结束至完成标记的时长作为每次评估/保存代理。训练及评估代理再除以并行保留比例（A=0.90277，B=1）。

| 队列 / 方法 | 每 seed 小时 | 五 seeds 小时 |
|---|---:|---:|
| A / CRITIC-ORACLE | 2.99 | 14.95 |
| A / CRITIC-SOURCE | 2.66 | 13.29 |
| A / Q-ORACLE | 3.04 | 15.21 |
| B / IQL-TARGET | 4.03 | 20.14 |
| B / IQL-TARGET+REG | 5.61 | 28.03 |
| B / OTDF-NOREG | 3.67 | 18.33 |

- 主估计：A **43.45 小时**，B **66.49 小时**。
- 另一种口径将冒烟 wall-time 中的启动开销随步数一并外推，扣除 VAE 后再单独加回每次 VAE，得到 A **46.59 小时**、B **70.26 小时**。它重复放大启动开销，且未显式加入周期评估，因此只作排期交叉参考。
两种估算形成约 A 43–47 小时、B 66–70 小时的规划范围；不是统计置信区间或严格上限。未来策略的 episode 长度、不同 seed、正则并行竞争、温度/后台负载及审计开销均可能延长时间。A 结束后 B 独占资源的变化也未单独建模。
当前两种 B 点估计均未超过 72 小时，因此保留原 15 项 B 清单，不自动缩减。接近 72 小时不构成保证；完整并行排期约 3 天，建议预留 3–4 天。
原自动审计中的 A=44.64 小时使用整段并行比例；本交付保留原文件，并以本节经重叠区间修正及补充评估开销的估计为准。没有修改训练或科学协议。

## 6. 待确认的正式实验与交付

| 队列 | 顺序 | seeds | 每项步数 | 新训练数 |
|---|---|---|---:|---:|
| A | CRITIC-ORACLE → CRITIC-SOURCE → Q-ORACLE | 0–4 | 500000 | 15 |
| B | IQL-TARGET → IQL-TARGET+REG → OTDF-NOREG | 1–5 | 1000000 | 15 |

A 复用 TARGET-64、ORACLE-W1 各五种子，TARGET-128 仅作描述比较；B 复用历史 OTDF 五种子。两队列按组依次推进，首组各五种子完成后交付 QA1、QA2、QB1 中期结果，再继续已授权剩余组。
正式主指标：A 的 460k–500k Last-5，B 的 960k–1M Last-5；B 同时报告 Final。逐 seed 配对，报告均值、样本 SD、SE=SD/√5、正号数。均值 >2SE 且至少 4 个正号仅为预设工程门槛，不是严格 95% 置信结论。
QA2 只识别移除 source 直接 actor 损失的干预效应；QA3 不构成数学上界。QB2 还包含有效 batch 128→230 的变化；正则预训练会改变后续随机序列。未通过门槛不能写为无效、等效或“所有源数据有害”，也不能宣称优于论文方法。
共同解释范围：单一 Walker 任务、target 约 8 条轨迹段、V 为学得函数、oracle 依赖历史恢复的隐藏物理状态、历史基线与新训练时间不同、Windows 软件环境与论文不同。A/B 分数不可横排。

正式训练须等待确认。依据用户提供的 [v5 指令](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/USER_SPEC_v5.md>)：“阶段 B：正式训练（必须等我确认）”。本次授权仅覆盖冒烟、复用及吞吐测试。

## 7. 证据与历史保留

- [冻结协议](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/protocol.json>)；[完整训练清单](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/training_plan.json>)；[GPU 冒烟授权](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/gpu_smoke_authorization.json>)
- [流水线完成记录](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/smoke_pipeline_status.json>)；[启动与审计日志](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/smoke_launch1>)
- [A 冒烟产物](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/s1>)；[B 冒烟产物](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_v5/results_20261013/s1>)；[A 并行产物](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/s_parallel1>)；[B 并行产物](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_v5/results_20261013/s_parallel1>)
- [独立核验脚本](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v5/deliver_smoke_v5.py>)；[本报告生成脚本](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/v5/write_smoke_report_v5.py>)；[历史基线原始重读](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/existing_baselines.json>)
- CPU 准备阶段旧 SciPy 不支持 workers、B 同名模块导入冲突的失败记录保留：[A5 原失败](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/oracle_diag/results/v5_20261013/a5_1>)、[B 原失败](<C:/Users/TXin/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main/reproductions/otdf_v5/results_20261013/cpu_b1>)。修复后的通过证据另存；此前 v2–v4 历史结果和恢复证据均由保护清单保留。
- 本轮 GPU 冒烟未发现训练失败。新增交付文件仅作核验与报告；未覆盖完成或失败的训练产物。
