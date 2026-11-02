# 覆盖缺口 R09：第一任务阶段 1–3

## 依据与范围

用户于2026-10-16确认R08其余默认值，修订P03/P06/P07/P08/P14，明确只授权第一任务到U/G比例验证止损。修订合同在：

`outputs/research_idea_protocol_20261016_r09/REVISION_AND_STAGE_CONTRACT_R09.md`

所有新代码/结果均在R09新目录，旧代码和旧结果只读。**不启动C比例网格、正式阶段、第二任务、估计器开发或旧队列；不更新桌面归档。** C有一次管线冒烟，不是C验证网格。

T调参执行已确认完整网格18次，精简12次方案仅作成本对比。本轮到止损共31项长训练及6次5k冒烟；不能从逐项确认推导出完整协议授权。

## 当前入口

- 代码：`oracle_diag/coverage_oracle_r09/pipeline.py`
- 输出：`oracle_diag/results/coverage_oracle_20261016_r09`
- 身份与实际入口：`active_execution.json`、`queue_started.json`
- 实时状态：`queue_status.json`、`training_plan.json`、当前作业`progress.json`
- 首次启动日志：`launch1.stdout.log`、`launch1.stderr.log`；作业日志在`logs/`
- 初始队列PID 56268，必须结合实际命令行判断，不能只看PID。

2026-10-16首次CPU验收已通过。T五步新旧参数差最大4.1211e-7（等价浮点求平均次序），混合五步完全一致；source改变V/Q而不直接改变首步actor。初始化/target抽样隔离、确定性动作不消耗训练RNG、独立20NN计算、source/cache逐行对齐与实际XML环境通过。见`cpu_acceptance.json`。

一次CPU测试夹具失败已保留在`cpu_test_attempt1.json`：旧T对照共享配置字典，把target_only泄漏至混合测试夹具；修复测试副本，未修改训练代数。随后所有CPU测试通过。

初始保护快照漏计34个超过Windows普通路径长度的文件；已使用扩展路径恢复核对，实际没有文件缺失。最终保护清单3140项，见`protected_manifest_final.json`及`prelaunch_review.json`；GPU前全部通过，`protected_before_gpu.json`无变化。TensorBoard使用现有目录短路径，避免长路径问题。

## 队列顺序与固定规则

1. T冒烟5000步seed7001 → CPU审计。
2. T temp1/3/10 ×500k/1M，各seed101–103，18次。500k三温度先完成，再1M；全部都跑，不按中途结果删配置。
3. 三seed Last-5选择T；均值1e-8内先短H，再温度3/1/10。
4. 同配置独立参考seed1001；封存参考模型，CPU实际shifted环境采5000步。
5. CPU计算G/C分数，先出`component_audit.json`/`SCORE_AUDIT.md`，后封存三个比例的静态掩码。审计不搜索参数。
6. G/C/R/U/D各一次5000步冒烟。全体通过后才进U/G验证。
7. U三个验证seed；G三比例各三个验证seed，单GPU串行。
8. 核对完整target抽样流、初始化、TensorBoard、模型与3140项保护哈希。最佳G均值须严格同时超过T和U才记验证见信号。
9. 输出`STOP_RESULT.json`、`STOP_REPORT_R09.md`，状态`AWAITING_USER_DECISION`。无论通过与否，都停止。

参考策略在任何真实G/C/R/U短程或长期回报出现前固定。source进入V＋Q，actor仅原target，source训练标签仅oracle。T64，混合64＋64，source系数1。主评估每10k十回合，Last-5为H−40k至H；验证种子不参与未来正式判定。

## 报告限制

- rel/gap任一贡献口径超过90%，G须描述为实质单分量选择；不触发调整k或权重。
- C的V0只由5000target学得，缺口外推不可靠；未来G>C仅能说明本协议下信号可靠性差异。
- 止损是验证筛查，非正式门槛或科学负结论；最佳验证配置存在选择偏差。
- 未来T128描述性控制“接近5分”的建议区间[4,6]仍需正式阶段前确认，当前不执行。
- 第二任务转移百分比，不是绝对行数；当前不执行第二任务。

## 监测与异常

同任务静默跟进`r09-1-3`。只读核对实际Python进程、最新progress、日志和完成/失败标记。不得重复启动队列。异常时队列保留原目录并停止，不自动重跑；先CPU诊断并交付事实，不能放宽标准或扩大授权。已完成训练若只是审计故障，仅重审，不重训。交付止损结果后停用跟进。
