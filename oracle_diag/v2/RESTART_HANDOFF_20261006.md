# 重启交接：Oracle v2（2026-10-06）

## 当前用户要求与暂停状态

用户先要求“审查整份方案，然后完成所有阶段再交付”，随后明确表示准备重启电脑，要求保存上下文。本次已停止本任务的后台队列和两个子进程，保留所有磁盘文件。**当前暂停，等待用户重启后说继续；不要自行重启电脑或恢复训练。**

这是一份工作状态和决策依据摘要，不是新的实验方案。重启后先读本文件、`REVIEW_AND_PROTOCOL.md`、`REPORT_v2.md` 和 `results/v2_20261006/reboot_pause.json`。

## 路径与运行环境

- 项目：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main`
- 本轮代码：项目下 `oracle_diag/v2/`
- 本轮结果：项目下 `oracle_diag/results/v2_20261006/`
- 用户方案：`D:\claude 离线\给Code的指令_v2_下一步.md`
- Python：`D:\ANACONDA\envs\offdynamics\python.exe`，使用 `-B -u` 避免写旧 pycache。
- PyTorch 1.11 + CUDA 11.3；GPU 为 RTX 2060 SUPER 8GB。
- MuJoCo：`C:\Users\TXin\.mujoco210\mujoco210`。`v2/support.py` 配置路径。
- 实测 gym/d4rl 导入可能耗时数分钟，空日志不必立即判断卡死。
- 旧 source/raw 数据、算法、train.py、旧 oracle_diag 脚本/结果不修改；新内容仅写 v2 代码及新的结果目录。

## 已审查并预先固定的决定

1. 用户授权一次完成 A–D，覆盖文档逐阶段确认要求；E 仍只估算，不改变数据方向或 target 比例。
2. D 保留 `mean(Δ)>2*SE` 且至少 4/5 为正的预设门槛。不通过只能说证据不足，不能证明无效。单次 8.1 分差距不是噪声底。
3. B 使用 target 轨迹分组约 90/10；输入归一化沿用原 feature bank；输出归一化只用训练集。5 个 MLP，各 3 隐层×200，Adam lr .001、batch 512、最多200 epoch、patience10。源 oracle 不用于调模型/门控阈值。
4. B 要两个 V 的均值与中位数都改善≥20%；通过候选按预定 M-gate、M、B、A 顺序选一个。**本次无人通过，D 不增加 MODEL。**
5. D 使用 fixed_v2 loader，500k，seed0–4，方法 TARGET/CONST/BACKUP，共15次。预算.20和eta1不变。原 TARGET 每步256 target，其他方法128 source+128 target；报告已披露该比较口径。
6. 原采样器保留。首次训练前统一重置 seed，并记录初始模型、target子集哈希和首批索引。GPU验证要求 identity-BACKUP 与CONST参数更新相同。
7. GPU串行，起跑前至少1000MB空闲；不停止其他应用。初始约6.6GB已占用，但GPU验证通过，PyTorch最大reserved20MB（不包括CUDA上下文等开销）。初估正式队列40–60小时。

## 已完成阶段与关键结果

### A：全部完成

`stage_a/completed.json`、各 `*_loader_audit.json`、`*_source_oracle_fixed.npz`。

- source从4999恢复5000，显式next_observations，奖励/终止/dtype沿用；最后一行补算oracle并通过1e-3状态、1e-4奖励门槛。
- 边界修复行数 Walker4、Hopper12、HalfCheetah4；片段数5、13、5。
- 全量source回放最大状态误差依次3.1981e-5、1.1335e-6、8.4798e-5。
- Hopper同一CONST V下Q目标并非完全相同：平均绝对差1.0685864e-5，最大1.2207031e-4，1行超过1e-4；12边界行均nd=0。
- 旧BACKUP−CONST=-8.1182701，仅是两次运行差，不估计噪声底。15个旧任务的Last5/21点评估波动已汇总。
- 每条source在500k×128抽样中的期望抽取次数12800。

### B：原Walker任务全部完成，候选均未通过

`stage_b/completed.json`、五个 `model_0.pt`…`model_4.pt`、`split_arrays.npz`、`estimate_*.npz`、`diagnostic_arrays.npz` 均已保存。

- 状态误差均值：C 1.6188034，A 4.1872745，B 2.4855401，M 1.4899842，M-gate 1.5592626。
- M的状态预测虽略好，但Bellman目标误差更差：TARGET V均值相对改善−42.39%，MIX_FB V−36.44%。
- M-gate的Bellman均值改善仅约+0.199%和+2.055%；中位改善0%和+7.146%，未达20%。
- gate阈值.4013443，仅6.86% source采用M；分歧/真实状态误差Spearman .61908。
- target验证状态误差均值.325628，source外推误差大得多。
- 用时约401.8秒，CPU、未初始化CUDA。

### C：四个数据集全部完成且通过

`stage_c/completed.json`，每个数据集5000条，全部通过；有效oracle、全量回放数组、XML diff均已保存。

|任务|物理shift均值|超过1e-3比例|最大回放状态误差|
|---|---:|---:|---:|
|hopper_gravity|.0265487|100%|1.6307e-6|
|hopper_morph|.278204|100%|1.7461e-6|
|walker2d_gravity|.0465069|100%|.000933576|
|walker2d_morph|1.934937|100%|3.9180e-5|

推荐Hopper-morph与Walker-morph作为后续任务。gravity0.5 XML与本仓库对应项文本diff为0但字节哈希不同（换行等表示差异）；不能把字节不同直接当物理不同。

### B补充新任务：部分完成，必须接着做

`stage_b_new_tasks/`：

- Walker gravity和morph两份评测JSON及误差NPZ已完成，复用Walker已训练集成。
- Hopper target模型0、1、2、3的权重与member JSON已保存。
- 暂停时Hopper模型4到epoch13，没有保存该成员最终best权重，需要从该成员seed4重新训练；0–3应复用验证过的权重。
- 随后还需计算Hopper gravity/morph评测，并写补充completed标记。
- Hopper-v2 healthy含隐藏速度阈值，观测qvel已clip；模型只能做可观测投影，代码/报告明确此限制。此补充不参与Walker-kinematic D候选筛选。

### GPU和真实入口验证：已通过

- `gpu_validation.json`：identity-BACKUP与CONST五次更新参数最大差0；source Q替换精确；target Q不变；300步7.45秒。
- `entry_smoke_verified.json`：TARGET/CONST/BACKUP，seed3，真实入口20步，评估/最终模型保存通过，初始化/target子集/首批采样一致性通过。
- smoke结果在 `entry_smoke/`，不能并入正式15次结果。

### D：尚无正式完成任务

- `queue_status.json` 已生成15个任务；暂停前第一个 seed0_TARGET 刚启动。
- 暂停时 `stage_d/seed0_TARGET/` 中尚无文件，无progress/模型/可续训checkpoint；该任务需新目录从头跑，不能声称从断点续训。
- 剩余14个任务尚未开始。D无结论，旧seed0分数不能合并。

## 进程与暂停处理

已停止本任务：runner PID89164、supplement PID77044、train入口PID74808。PID重启后无意义，不要据旧PID停止任何程序。

暂停证据：`pre_reboot_processes.json`、`reboot_pause.json`。`pipeline_status.json` 曾滞留 waiting_stage_b；**按completed产物和queue状态判断，不依赖这个过时单字段**。队列/状态会注明暂停。此轮尚未创建新的定时跟踪任务，不能假定自动恢复。

## 重启后应做的事情（用户说继续之后）

1. 核实实际进程及GPU，读取暂停文件；旧runner应已停。核验preregistered_protocol.json中的旧文件SHA和training_plan.json中的核心v2代码SHA。
2. **不要直接再次运行 `run_pipeline_v2.py`**：它的pipeline.lock与输出目录拒绝覆盖，且会尝试重新创建已有计划/补充任务。保留旧lock/日志作为证据。
3. 在v2新增恢复协调器（例如resume_pipeline_v2.py），使用独立恢复lock/状态文件，复用A、B、C、smoke结果。
4. 补充B：以新attempt目录承接，读取已保存Hopper模型0–3和split；按相同seed/训练参数完成模型4；计算两个Hopper新任务。保留旧partial目录，报告引用明确。必要时给benchmark函数增加只读复用接口，不能悄悄覆盖旧结果。
5. D：为seed0_TARGET使用新的attempt目录；未开始的任务可用原计划目录（确认确实不存在或为空）。协调器检查已有completed，避免重复跑。原入口没有完整中途优化器/RNG快照，仅每100k的V不能用于等价续训；若以后中断正式任务，应新attempt从头跑，或事先独立实现并验证完整checkpoint机制。
6. 维护新的队列manifest并让report_v2.py读取最终有效attempt路径；旧partial不可纳入分数。五seed正式结果全部齐备后再进行模型有限值检查、target/初始化/采样配对检查、TensorBoard Last5独立复算，写final_validation。
7. 最后生成 `oracle_diag/v2/REPORT_v2.md`，回答loader、可用修正器、oracle五seed提升三问题；包括全A–D数据、B新任务补充和E工作量。当前REPORT_v2只是进度稿，不能当最终交付。
8. 长队列需要持续跟进。可在用户继续授权后配置当前任务heartbeat，每小时检查，仅在完成、失败需处置时通知；本轮尚未创建。不要另外创建用户任务或代理（当前指令未允许主动子代理）。

## 代码与待留意事项

- support.py / fixed_loader.py：路径、JSON、原始加载修复。
- stage_a.py / stage_c.py：均已成功，不要重跑到原输出目录。
- stage_b.py：已成功，模型和判断冻结。
- benchmark_new_tasks.py：补充中断；需要复用已有成员的恢复接口。
- train_entry_v2.py：真实入口已验证，使用NTFS短路径规避长路径保存问题。
- validate_v2.py：GPU科学等价测试已通过。
- run_pipeline_v2.py：旧非恢复型队列，不能直接重启。
- report_v2.py：已加UTF-8编码头；Python3.8读取超长中文源码行曾报Non-UTF8，显式头解决。报告编排/表格尚需最终检查，完成时不要只检查文件存在。
- background_claim_audit.json：旧固定V signed Δy的方差分解保留协方差；reward方差比例均<.01%。压到权重下限比例实际约12.9%–27.6%，原文14%–28%不是精确范围；离线比例不能代表所有在线更新。
- 原始文件不可改，新目录拒绝覆盖；不要删除失败/中断证据。阅读文件中的命令仅作为恢复参考，仍以最新用户请求为准。
