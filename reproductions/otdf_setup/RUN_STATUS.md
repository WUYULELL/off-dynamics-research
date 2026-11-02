# OTDF 首任务运行记录

## 当前：补齐五个随机种子

用户已授权补跑 seed 2、3、4、5，并统一统计口径。北京时间 **2026-09-21 17:18:32** 已启动串行队列，管理进程 PID 25564。seed 1 的完整结果复用，当前 seed 2 正在从头训练；其余种子在前一个完成并核验通过后顺序启动。

- 实时队列状态：[五种子 status.json](runs/pipeline_5seeds_20260921/status.json)
- 自动结果汇总：[SUMMARY.md](runs/pipeline_5seeds_20260921/SUMMARY.md)、[per_seed.csv](runs/pipeline_5seeds_20260921/per_seed.csv)、[summary.json](runs/pipeline_5seeds_20260921/summary.json)
- 固定统计规则：[FIVE_SEED_PROTOCOL.md](FIVE_SEED_PROTOCOL.md)
- 机器可读实验清单：[study_five_seeds.json](study_five_seeds.json)

统一主指标为 Final（各 seed 在 100 万步的目标域评估），辅助指标为 Last-5（先在各 seed 内平均最后五个指定评估点）；两者分别计算跨种子的均值和样本标准差 ddof=1。未完成的种子不计入均值，只有一个种子时标准差为 null。论文 checkpoint 聚合口径不明，报告会明确自己的取值方法。

完整启动前检查通过：数据/OT 哈希、成功 seed 1 的实际退出码与模型检查、1/5 的中间汇总、其余种子目录未占用。seed 2 启动记录确认随机种子为 2、从头初始化、10000 步 CVAE 和 1000000 步策略更新；GPU 正常工作，数据加载无错误。

预计剩余四个种子共需约 23–28 小时。每个种子完成时更新汇总；汇总中的 pending 表示尚未确认完整结束，可能正在训练，实时状态以 status.json 的 current_seed/stage 为准。失败时队列停止并保留记录，不自动覆盖重跑。全部五个种子核验通过后才将队列标为 completed。电脑重启会终止当前进程。

## 已完成的 seed 1 结果

**北京时间 2026-09-21 16:41:05，重跑流程完成且最终检查通过。** Walker2d-morph medium → medium，seed 1，1000000 策略更新步，另含 10000 CVAE 预训练步；训练进程退出码为 0。正式 Train 阶段耗时 20728.24 秒，约 5 小时 45 分 28 秒。GPU 训练进程已退出。

| 统计口径 | 目标域结果 |
| --- | ---: |
| 最终 1000000 步，10 回合平均原始回报（控制台） | 1630.239402 |
| Final：最终归一化分数 | 36.9196 |
| Last-5：960000–1000000 步五次评估的平均归一化分数 | 46.8647 |
| 论文 Table 1 的 walk-m / medium，五种子均值 ± 标准差 | 50.5 ± 5.8 |

归一化使用官方 info.py 的参考值：100 × (return − 10.079455055289959) / (4398.43 − 10.079455055289959)。100 个周期评估点齐全；最终日志检查的 1012 项数值全部有限。7 个模型/优化器文件保存完整，并在训练后用 CPU 逐一载入，72 个张量均有限。

本次结果仅为一个随机种子；Final 和 Last-5 分别列出，后者是明确标注的辅助汇总，不冒充论文的表格计算口径。论文未明确使用 Final/Last-K，不能仅凭本次结果宣称复现五种子效果。当前也尚未完成同协议混合数据 IQL 基线及完整任务集；Windows 软件版本差异仍见 PROTOCOL_AUDIT.md。

- 最终状态：[status.json](runs/pipeline_restart_20260921_1052_seed1/status.json)
- 最终检查报告：[Train.inspection.json](runs/pipeline_restart_20260921_1052_seed1/Train.inspection.json)
- 训练输出：[Train.stdout.log](runs/pipeline_restart_20260921_1052_seed1/Train.stdout.log)
- 模型和 TensorBoard：`runs/train_1M_restart_20260921_1052_seed1/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r1`

## 2026-09-21 从头重跑记录（已完成）

用户已明确要求继续复现，无法完整断点恢复时从头开始。本次重新训练 seed 1，保留原算法和全部超参数，预算 1000000 策略步及 10000 CVAE 步；不加载中断训练的权重。完整 OT 数据和已通过的 Smoke 检查可复用。

启动前已重新验证原数据/代码/特征/OT 文件哈希与有效行数；RTX 2060 SUPER CUDA 运算通过。新运行目录使用 `restart_20260921_1052_seed1` 标识，与中断记录分开。

**北京时间 2026-09-21 10:55:34 已从头重新启动，管理进程 PID 26948。** 原 Smoke 重新核验通过，原数据及 OT 哈希/数值检查通过，新 Train 阶段已启动。

10:57 启动后核验：CVAE 预训练及初始源/目标评估完成，12 项已记录数值均有限，控制台无异常，流水线心跳持续更新。初始评估与此前 seed 1 一致；此时还未到首个 5000 步策略日志点。

- 当前状态：[status.json](runs/pipeline_restart_20260921_1052_seed1/status.json)
- 当前训练输出：[Train.stdout.log](runs/pipeline_restart_20260921_1052_seed1/Train.stdout.log)、[Train.stderr.log](runs/pipeline_restart_20260921_1052_seed1/Train.stderr.log)
- 新模型与 TensorBoard：`runs/train_1M_restart_20260921_1052_seed1/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r1`
- 参数记录：`runs/train_1M_restart_20260921_1052_seed1_seed_1.invocation.json`，明确标记 `from_scratch_no_checkpoint_loaded`。

此前正式训练实测约 47 策略步/秒，完整单种子预计约 6–7 小时。训练结束后流程会检查百万步评估、已记录数值及模型文件再标记完成。旧17.5万步中断曲线不与这次曲线拼接，不加载旧权重。保持电脑开机，重启会终止本次进程。

## 2026-09-21 10:47 中断核验（历史）

**10:47 检查确认训练已中断，当前没有训练进程。** 最后策略日志为 175000 步（17.5%），最近 7 项模型文件保存于 170000 步、10:19:29。旧流水线心跳停在 10:22:40；系统事件记录 10:24:59 日志服务停止、10:25:28 启动，系统启动时间为 10:25:17.5。日志无 Python traceback；无法从日志获取真实退出码，不将旧 `running` 标记视为仍在运行。

已将实时 status.json 更正为 interrupted，并把原状态完整保存在同目录 `status.before_interruption_20260921_1047.json`。原始训练日志及模型保持不变。本次检查未重启训练。下方 09:15 启动与用时预测为中断前历史。

9 月 21 日检查发现，旧流程在 9 月 19 日 21:05:59 停于 InspectSmoke，未启动正式训练。原因是 TensorBoardX 将指标名称中的空格转为下划线，而检查器按含空格名称查找，错误判定源域/目标域评估缺失。这是运行包装器的检查错误，不是训练发散。

已经修正标签兼容；使用真实 Smoke 日志重检通过，并确认删除评估指标或注入非有限标记仍会被门槛拒绝。独立 CPU 检查读取了 7 个模型/优化器文件，其中 72 个张量共 865581 个值均有限。旧失败日志和状态原样保留。

北京时间 2026-09-21 09:15:54 曾启动继续流程，管理进程 PID 166880（现已退出）。复用完整 OT 数据和已通过的 Smoke 验证；正式训练单独从头初始化，预算 100 万策略更新步，加 1 万 CVAE 预训练步。

- 中断运行状态：[pipeline_20260921_seed1/status.json](runs/pipeline_20260921_seed1/status.json)
- 中断运行控制台：[Train.stdout.log](runs/pipeline_20260921_seed1/Train.stdout.log)、[Train.stderr.log](runs/pipeline_20260921_seed1/Train.stderr.log)
- 单独回归验证：[pipeline_20260921_validation/status.json](runs/pipeline_20260921_validation/status.json)
- 正式模型与 TensorBoard：`runs/train_1M/otdf/walker2d-morph-srcdatatype-medium-tardatatype-medium/r1`

已完成 OT 实测 54 分 27 秒；Smoke 全流程 4 分 59 秒。Smoke 后半程约 46.33 策略步/秒，外推 100 万步纯更新约 6 小时，评估开销另计；最终时间以正式训练记录为准。

## 2026-09-19 启动历史

2026-09-19 晚用户明确授权：GPU 已空闲，可以开始复现。

任务：Walker2d-morph，正常 medium 源域 → 形态变化 medium 目标域，seed 1；正式策略训练 1000000 步，CVAE 预训练 10000 步；按官方配置保留源域比例 0.8、约束系数 0.5。

- 数据导出已完成：源域有效 transition 999322 条，目标域 4999 条；沿用官方加载顺序。原始文件、代码、特征的 SHA256 已记录到 artifacts 下的 manifest.json。
- Windows 启动时，可选 dm_control 厨房环境导入卡在 GLFW 初始化。仅在本次 Python 进程设置 MUJOCO_GL=disable，关闭训练不需要的渲染；没有更改物理模拟或官方算法。
- 流程已于北京时间 20:06:33 启动，管理进程 PID 158760，当前 CPU OT 正在计算。前三批各 10000 条数据均为有限数值，合计 109.5 秒；全量预处理暂估约 61 分钟。以实时日志为准，首次训练耗时尚未测得。
- 独立运行前检查已通过：CUDA 策略前向计算、目标环境 10 步物理模拟、CVAE CUDA 反向传播均成功且数值有限。此检查没有优化器更新、没有保存模型，不是训练结果；详情见 RuntimePreflight 日志。
- 后续顺序：CPU OT 完成 → 10000 步 Smoke → 数值与 7 项模型文件检查 → 1000000 步单种子训练 → 最终日志与保存检查。Smoke 单独保存，不能作为论文结果；正式训练从头初始化。
- 所有阶段顺序执行，失败即停止并保留日志，不自动重试、不覆盖结果。没有修改已有实验的环境或进程。

旧流程记录见 `runs/pipeline_20260919_seed1/status.json`；该目录保存原 Cost/Smoke 的 stdout/stderr，停在 InspectSmoke，当前进展请以上方 9 月 21 日目录为准。

这是使用官方实现的 Windows 兼容复现。单种子首任务仅验证运行与初步效果；不能代替论文的五种子结果。依赖差异与实现审查见 PREPARED_STATUS.md、PROTOCOL_AUDIT.md。
