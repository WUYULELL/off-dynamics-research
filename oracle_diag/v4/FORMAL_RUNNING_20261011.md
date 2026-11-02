# Oracle v4：20 项正式训练运行交接

## 授权与启动

用户明确授权：「启动全部 20 项正式训练」。2026-10-11 已启动全部正式任务的串行 GPU 队列；无需再次申请启动许可。5 组 GPU smoke 已通过，不能重复运行。

- 项目：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main`
- 代码：`oracle_diag/v4`
- 结果：`oracle_diag/results/v4_20261011`
- Python：`D:\ANACONDA\envs\offdynamics\python.exe`
- 队列：`run_queue_v4.py --kind formal`
- 初始队列 PID：26048；初始子进程 PID：30244。PID 仅供核对，必须验证实际命令行，不能仅据 PID 判断。
- 当前执行入口：结果目录 `active_execution.json`；队列状态：`queue_status.json`；清单：`training_plan.json`。
- 授权凭据：`gpu_formal_authorization.json`。
- 启动日志：`launch2/queue.stdout.log`、`launch2/queue.stderr.log`；原启动前状态保存在 `launch2/active_before_formal.json`。
- 启动验收：第一项 ORACLE-ONLY seed 0 的 `b1/o0/progress.json` 已到 10000 步，source 抽样 1280000 次、target 抽样 0 次；两个真实进程均存在，无运行异常。

## 冻结协议

walker2d-morph 标准方向：正常 D4RL medium source → shifted OTDF medium target，实际 shifted 环境评估。依次完成每组 seeds 0–4，每项 500000 步，共 20 项：

1. ORACLE-ONLY：`b1/o0` 至 `b1/o4`。128 source，oracle next-state/reward/not-done，无 target 数据，普通 batch 均值。
2. SOURCE-ONLY：`b1/s0` 至 `b1/s4`。128 source，原 source 标签，无 target 数据，普通 batch 均值。
3. TARGET-64：`b1/h0` 至 `b1/h4`。64 target，无 source。
4. ORACLE-W1：`b1/w0` 至 `b1/w4`。64 source + 64 target，source 系数 1.0；oracle 只替换 source Q backup 标签。V/Q/actor 的组间系数按冻结实现执行。

v3 的 source 系数 0.20 不变；v4 W1 的 1.0 是明确设定的实验组。不要把历史 v3 规则误套到 v4。eta、种子、阈值、训练步数、数据和评估对象均不能临时调整。

TARGET-128 的 5 项正式基线复用 v3：TARGET smoke 与原模型/优化器张量、初始化、抽样及评估已核验一致，不能额外重跑。凭据：`target_reuse_verified.json`、`smoke_audited.json`、`smoke_delivery_verified.json`。

- protocol SHA256：`721a98732b30bfa0c48b30a14255f5d84caf0fe42118091a40315c293ef0d708`
- training_plan SHA256：`c9b91d5d7b14b164ff587b8614f24e7487afdf9843939434bdbcdad53d0b6003`
- 冻结代码哈希在 `cpu_ready.json`，2111 项旧文件保护清单在 `protected_manifest.json`。
- oracle 缓存复用 v3 的已验收恢复结果；不重新生成，也不覆盖。

## 跟进与故障处理

先读 active execution、队列状态、计划和最新 progress，再核对真实进程、完成标记、错误日志。活跃队列不重复启动；不要终止其他应用。队列是本地隐藏独立进程，不依赖聊天保持连接；电脑关机、重启或睡眠会影响训练。断网时自动跟进不能工作，恢复连接后再核对实际状态。

若发生本任务故障，先诊断；保护已完成和失败目录，在新的 attempt 路径修复或恢复，并记录代码/计划修复前后哈希。科学协议不变；不得按分数挑选重跑结果。没有完整可恢复状态时，同一预设 seed 从第 0 步重跑并保留中断证据。报告或审计出错不能作为重跑已完成训练的理由。

## 中期与最终交付

前 10 项（ORACLE-ONLY 和 SOURCE-ONLY 各 5 seeds）完成后，队列自动审计并生成 `oracle_diag/v4/INTERIM_v4.md` 和结果目录 `interim_ready.json`。交付中期结果后继续其余 10 项；不要在中期自行停止。用户另有停止指令时再执行。

20 项完成后，检查 `audit_v4.py` 的原始 TensorBoard 读数和 `report_v4.py` 生成的 `oracle_diag/v4/REPORT_v4.md`。主指标是 460k、470k、480k、490k、500k 五次评估的均值。独立核对逐 seed 配对差、样本 SD、SE、正号数；确认 shifted 评估对象、归一化、实际抽样、初始化、source/oracle 行对齐、所有模型有限性和受保护文件哈希。

主比较：ORACLE-ONLY−TARGET、ORACLE-ONLY−SOURCE-ONLY、TARGET−TARGET-64、ORACLE-W1−TARGET。预设门槛为均值大于 2×SE 且至少 4/5 正号；这是工程判据，不能称严格 95% 置信结论。

解释时区分事实和归因：target 仅约 8 条轨迹段，V 为学得函数，oracle 使用历史恢复的隐藏状态；单组失败不能证明所有 source 数据有害，近零差不能证明等效。W1 同时影响多个损失的 source 系数和总体尺度；单纯 batch 对照不能独立分解所有混合组损失。不能据这些实验声称超过 TABB 或 OTDF 论文结果。

初始耗时粗估 44–51 小时，来自 smoke 和 v3 实际速度，非完成承诺；应按已完成正式任务更新。交付最终中文报告和核心结果后停用本次自动跟进。
