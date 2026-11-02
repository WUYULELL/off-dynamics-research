# R11 resume1 检查点续跑交接

2026-10-17，Asia/Shanghai。用户询问“可以从中断的位置继续跑吗”，本次按续跑请求恢复最近已保存的400k点，保持R11原分段边界。不是从407k恢复，也不是从零重训。

## 位置与范围

- 新代码oracle_diag/coverage_oracle_r11_resume2；新结果oracle_diag/results/coverage_oracle_20261016_r11_resume2。
- 说明outputs/research_idea_protocol_20261017_r11_resume2/RECOVERY_SCOPE.md。
- 原R11中断目录及系统事件证据保持只读；9项500k和T冒烟复用，不重跑。
- 最新实际入口active_execution.json，queue_status.json和training_plan.json。首次队列PID16768；始终核对真实命令行，禁止重复启动。
- 首项用resume_train.py，temp10、seed101、总H1M，恢复点400k，剩余600k。之后仅同temp的1M seeds102/103、参考T、5000步CPU诊断、零新增训练分数审计、五组各5k冒烟。新增GPU作业9项，包含此续跑。
- U3/G9验证、C比例网格、正式阶段、第二任务/Hopper缓存、旧队列及桌面更新均未授权。

## 已核验的恢复条件

cpu_acceptance.json检查：policy/V/Q/targetQ、三个Adam、原T_max=1M的cosine调度、step400k完整恢复；actorLR保持0.0001963525491562418，不改成600k的新调度。

没有在原检查点保存抽样器状态，但抽样器是固定seed的独立RandomState。按原次序重建400k后，向后复算407k和500k抽样摘要分别与原中断progress、已完成同seed运行sampling.json的SHA256完全一致。CPU连续80步与40步保存/恢复后继续40步，模型/优化器/调度逐位一致；该target-only更新不消耗Torch随机数。

以上是恢复条件验收，不将CPU结果夸成GPU跨重启逐位一致的保证。实际GPU在401k–406k逐点对照旧TensorBoard中V/Q/actor三种loss，共18个float32值，**严格相等**才通过。OVERLAP_VERIFIED.json是后续继续的关键标记；不匹配则写overlap_failed.json、停止并保留现场，不能放宽容差。

## 日志与预算

新attempt只继承原TensorBoard/评估记录中≤400k部分，丢弃旧401k–407k记录，再重新产生后续日志；不混拼重复步数。有效完整训练为旧400k＋新600k＝1M，原400k–407k之间7000步重新计算；资源消耗中保留该中断损耗。

配置/温度/seed/评估和归一化规则均未改。旧首次10步loss缓存因进程崩溃不可恢复，early_losses.json须如实说明，不能伪造。新恢复检查点额外保存随机状态。

## 后续交付

H判据仍三seed Last5(1M)−Last5(500k)配对mean>2SE才取1M，否则500k；两档窗口分别960k–1M和460k–500k，为不同训练阶段，不证明等效。T锁定后独立核验并交付，已授权参考/诊断/冒烟继续；五组冒烟完成后停下等待U/G单独授权。

恢复前科学代码及9项完成产物已有哈希验收。队列各阶段核对保护清单，单GPU串行、后台隐藏窗口，诊断仅CPU，不改旧文件、不终止其他应用。故障不得自动重训。

工程attempt1导入冲突在训练开始前退出，无训练更新；attempt2用明确路径导入冻结evaluate。Windows文本换行字节检查问题在新代码冻结前已纠正；同一恢复CPU证明保留，旧attempt均只读。
