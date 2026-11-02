# Oracle v3：阶段 B 已启动

> **2026-10-10 更新：原队列已中断并恢复。当前入口为 attempt3，实际清单/日志在 recovery2，ORACLE seed3 替代目录为 b2/o3。详见 [恢复记录](STAGE_B_RECOVERY_20261010.md)。以下启动信息保留为历史记录；实时状态以 active_execution.json 为准。**

用户于 2026-10-08 明确指示“启动阶段 B”。阶段 A 报告保留其交付时状态，当前执行状态以本文及 active_execution.json 为准。

## 当前执行

- 标准方向 walker2d-morph，TARGET / MIX-CONST / MIX-ORACLE-BACKUP 各 seeds 0–4、500k，共 15 项正式训练。
- 三组 seed0、5000 步 GPU smoke 均已完成并通过审计。正式训练已进入 TARGET seed0。
- 总 batch=128；TARGET 128 target，混合组 64 source+64 target；source 系数0.20，eta=1；阶段 C 不执行。
- 不将 smoke 分数用于方法效果结论；正式结果按预定 Last-5 与五种子配对统计交付。

## GPU smoke 核验

原始 TensorBoard 中的 raw return、normalized score、评估步数、归一化与实际 shifted 评估对象均核对通过；三组初始网络一致、两个混合组首批采样一致、模型有限性和各域总采样量符合协议。

| 方法 | 5000 步进度计时（秒） |
|---|---:|
| TARGET | 66.08 |
| MIX-CONST | 104.88 |
| MIX-ORACLE-BACKUP | 105.19 |

计时包含数据接入与初始评估，尚未包含末次评估，不能视为纯梯度吞吐。简单外推 15 项正式训练约 38.35 小时；先按 35–45 小时安排，后续使用正式任务计时修正。单 GPU 串行运行，2 个 CPU 计算线程，oracle 为 CPU 内存映射数组。

## 已修复的审计兼容性问题

三个训练本身都正常完成。TensorBoardX 将源码日志标签里的空格转为下划线；首版审计期待空格，因此未读取到 score 事件。新审计统一标签后通过，没有重跑训练或修改数据、超参数、种子、判据。原脚本及错误日志均保留。实际事件标签是 `test/target_return` 和 `test/target_normalized_score`，应更正阶段 A 报告中把源码字符串当作实际落盘标签的表述。

## 恢复与检查入口

- 项目：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main`
- 实际队列：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v3\run_queue_v3_attempt2.py`
- 实际审计：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v3\audit_runs_v3_attempt2.py`
- 主状态：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\results\v3_20261008\active_execution.json`
- 当前子任务：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\results\v3_20261008\queue_status.json`
- 新队列日志：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\results\v3_20261008\recovery1\queue.stdout.log` / `C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\results\v3_20261008\recovery1\queue.stderr.log`
- 故障证据：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\results\v3_20261008\recovery1\repair.json`；旧 `queue.stderr.log` 不能单独用来判断当前任务失败。
- 自动跟进：`oracle-v3-b`，约每小时检查，正常运行不重复通知。

不要重复启动队列。检查实际进程与 progress.json 判断运行状态；进程结束后才诊断及选择新 attempt，保留失败目录。完成后从新审计生成的 formal_audited.json 整理并核验中文报告，停用跟进；不要启动阶段 C。
