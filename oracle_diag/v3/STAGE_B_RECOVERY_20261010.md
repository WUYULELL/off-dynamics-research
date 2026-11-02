# Oracle v3 阶段 B：2026-10-10 恢复记录

## 当前状态与恢复依据

本次检查时，原队列 PID 24436 和 ORACLE seed3 子进程均已退出。11/15 项正式训练已有完成标记；原 `b1/o3` 仅更新到 40000 步，没有完成标记。队列 stderr 为空，子任务 stderr 仅有环境警告和数据加载记录；系统最近启动为 2026-10-06，未查到同期重启事件。因此进程退出的原因尚不明确，不能断言为重启或训练代码报错。

恢复前重新核对 1521 个受保护旧文件及阶段 A 冻结代码哈希，全部通过。旧状态、锁文件、报告和失败产物哈希保留在 `recovery2/`；原训练目录、数据、权重和日志均保留。

## 实际执行入口

以结果目录 `active_execution.json` 为准。当前：

- 队列：`oracle_diag/v3/run_queue_v3_attempt3.py`
- 审计：`oracle_diag/v3/audit_runs_v3_attempt3.py`
- 实际清单：`oracle_diag/results/v3_20261008/recovery2/training_plan.json`
- 实际队列日志：`recovery2/queue.stdout.log`、`recovery2/queue.stderr.log`
- 当前状态：结果根目录 `queue_status.json`
- 原始中断目录：`b1/o3`，40k，仅留作中断证据，不能计入正式结果。
- 替代目录：`b2/o3`，相同 ORACLE seed3 从第0步重跑500k；之后为原清单 `b1/t4`、`b1/c4`、`b1/o4`。

三组 smoke 与前11项完成训练由队列跳过，不重跑。原训练清单、attempt2脚本及 recovery1 的日志标签修复证据均保留。attempt3 的代码差异只有清单路径和审计入口；清单只改变 ORACLE seed3 输出路径及对应命令参数。训练入口未改，未改变 source=.20、eta=1、总batch128、mixed64+64、seeds0–4、500k或工程判据。

不使用40k权重续跑：该训练没有保存完整优化器、随机数和调度器恢复状态，不能声称能够精确续训。重复的40k作为中断开销记录，不拼接进正式500k曲线。

## 后续

不要重复启动队列，也不要再使用原 b1/o3 的陈旧 progress 判断当前停滞。按实际清单检查四项剩余任务。全部训练和正式审计完成后，继续独立复核 Last-5、配对 SD/SE/正号数、源/oracle 行对齐、实际 shifted 评估和保护哈希，更新 REPORT_v3.md 并交付；阶段 C 不执行。自动跟进 oracle-v3-b 已更新为上述入口。
