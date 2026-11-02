# v2 重启后恢复记录

用户已经说“继续 v2 实验”，重启暂停已解除。此前 RESTART_HANDOFF 的阶段结果与协议仍有效，其中暂停状态已被本次恢复替代。

- 当前结果根目录：`oracle_diag/results/v2_20261006/resume1/`。
- 当前队列：`resume1/queue_status.json`；当前状态：`resume1/status.json`。旧根目录 queue/status 保留为中断记录。
- 当前协调器：`resume_pipeline_v2.py`。在确认当前协调器已退出前不得重复启动。新目录/lock 会拒绝重复启动。
- 原始 source/算法/训练入口及 v2 核心训练代码哈希均已核验，未改变超参数。
- 原 A、B、C、GPU 验证、真实入口 smoke 直接复用。B 无候选通过，因此正式队列为 TARGET/CONST/BACKUP × seeds 0–4，共 15 次 500k。
- 原 seed0_TARGET 中断时没有有效检查点，本次在 resume1/stage_d/seed0_TARGET 从头训练，不合并旧 partial。
- 补充 B 用 `resume_benchmark_v2.py`：复制并核验已经完成的 Walker 两个新任务结果，复用 Hopper 成员 0–3，只从 seed4 重训成员 4。重新生成的 train/val 分组和标准化数组必须与旧文件完全相同。
- 报告通过 `report_resume_v2.py` 选择正确的 resumed queue 和补充目录，再调用旧生成器。当前 `REPORT_v2.md` 是进度稿；必须等 15 个正式运行和补充评测全部完成，独立核验后才是最终报告。
- 当前任务已创建每小时跟进：automation id `oracle-v2`。正常运行无重要变化时安静；故障需诊断和恢复；完成后核验报告、向用户交付，并停用跟进。

恢复协调器作为隐藏本地进程运行。断网不会停止已启动的本地计算；关机/重启会停止进程。当前入口没有完整中途优化器/RNG恢复机制；每 100k 的 V 快照不能冒称可等价续训。如再次中断，保留目录并对未完成运行创建新的 attempt。

恢复时已实测 TARGET seed0 使用 CUDA、fixed_v2 5000 条 source、99932 条 target；target-only 每步 256 target。GPU 总占用约 1.6 GB，实际速度和剩余时间以最新 progress 为准。
