# 本轮执行说明（2026-10-03）

## 当前任务

用户已授权完成 oracle 诊断全部阶段，范围为三个环境的 seed 0；不自动扩展到 5 seeds。

- 阶段 0：已完成三环境 CPU 回放验收及 Walker2d 的状态精度修复。
- 阶段 1：已生成三份与原 source buffer 对齐的 4999 行 oracle NPZ。
- 阶段 2a：已完成，使用原 seed 0 / 10% target 数据、20k bank、k=5。
- 阶段 2b：Walker2d 的已有 MIX_FB V 已分析，其余 V 由后台队列补齐。
- 阶段 3：实现、公式验证、GPU smoke 与真实入口保存检查均通过，正式对照已排入同一队列。

这些状态是启动记录。请以实时 `results/verified_20261003/queue_status.json` 和 `REPORT.md` 为准；不能把启动记录当作训练完成。

## 后台队列

启动时间：2026-10-03 19:07（Asia/Shanghai）。启动时 runner PID 为 **64516**，第一个训练进程 PID 为 **62864**。进程号仅作本次记录，结束后可能被系统复用。

共 15 个训练/复用任务：1 个核验通过的 Walker2d MIX_FB 500k 模型复用，14 次新 GPU 训练。每次 500k 步；全部保持 source_budget=0.20、eta=1、seed=0。每个新任务保存最终完整模型和 100k/200k/300k/400k/500k 的 V。

短程 GPU 测速推算纯训练合计 **37.16 小时**；评估、I/O、不同环境和机器负载会增加时间，按约 **40–48 小时**规划。不是完成时间保证。

队列会依次运行：

1. 三环境 TARGET 和 MIX_FB 的 V。
2. 每份 V 完成后计算 Bellman 排序、ESS 和散点图，并独立复算保存数组。
3. 三环境 CONST 0.20、ORACLE 权重、ORACLE-BACKUP。
4. 自动读取完整的 460k–500k 五次评估，更新 REPORT.md。

新任务调用原 train.py；通过内存适配器连接新类，没有改动原注册表或算法文件。Windows 下用已有目录的 NTFS 短路径处理原模型保存函数的路径兼容问题。

## 文件

- 总报告：`REPORT.md`。
- 队列计划、代码哈希及检查点复用证据：`results/verified_20261003/queue_plan.json`。
- 实时状态：`results/verified_20261003/queue_status.json`。
- 控制程序日志：`results/pipeline_20261003.queue.log`、`.queue.err.log`。
- 每任务日志：`results/verified_20261003/<environment>_<method>.console.log`。
- 每任务模型、V、训练步数和实际配置：`results/training_20261003/<environment>_<method>/`。
- 模拟器恢复、oracle 数据、诊断与图表：`results/verified_20261003/`。
- 若失败：`results/verified_20261003/pipeline_error.json`，不会冒充完成。
- 全部成功标记：`results/verified_20261003/pipeline_completed.json`，还需 REPORT 中的日志验收完整。

## 查看状态与重新汇总

```powershell
$oracle = 'C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag'
Get-Content -LiteralPath "$oracle\results\verified_20261003\queue_status.json" -Raw
& D:\ANACONDA\envs\offdynamics\python.exe -B "$oracle\report_pipeline.py"
```

已运行的启动命令如下。**不要重复启动**；脚本发现已有计划会拒绝重复执行。

```powershell
& D:\ANACONDA\envs\offdynamics\python.exe -B -u "$oracle\run_all_stages.py"
```

计算和数据均在本机，训练与自动汇总不需要联网。保持电脑开机、不手动睡眠、不结束训练进程。启动时已核对交流/直流电源的自动睡眠超时均为 0（从不）。

## 解释边界

- 三份 source 数据各来自约 5 条轨迹；4999 行不是 4999 个独立实验重复。
- source 原加载器的 4 条跨轨迹 next_state 跳变保持不变，另做剔除边界的诊断。
- 恢复后的 oracle 在已声明容差内通过验收，仍是基于舍入数据的近似模拟器参考。
- Hopper 在本批 source 支持集上没有观察到物理一步动力学差异，不能将数值噪声的排序当作识别 shift。
- ORACLE-BACKUP 是理想转移消融，不构成理论回报上界；单 seed 结果也不能证明统计显著。
