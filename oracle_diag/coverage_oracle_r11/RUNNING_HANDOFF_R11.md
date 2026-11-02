# R11 当前分段授权交接

日期：2026-10-16（Asia/Shanghai）。以最新用户分段授权和R11为准，不能沿用R09/R10较大的执行范围。

## 范围

- T调参逻辑共12次：500k三温度1/3/10各seed101–103；只选500k胜出温度，再以同温度、同三seed从零训练1M，不追加温度搜索。
- 参考T(seed1001)1次、CPU诊断5000步、零新增训练分数分布审计、G/C/R/U/D各一次5k冒烟(seed7001)。
- 已完成且审计通过的R09 T冒烟和T(temp1/500k/seed101)复用、不重跑。
- 实际新增12长训练＋5短冒烟，共17个新GPU作业；包含复用后为T12＋参考1及六组冒烟。
- **U的3次验证、G的9次比例验证、C比例网格、正式阶段、第二任务、Hopper缓存、估计器开发、旧队列恢复与桌面归档均未授权。** U/G各一次5k冒烟不是验证。

## 当前事实

旧R09派发在R10预算等待期间已停止，期间未新增派发。旧清单只有两项是历史记录，不代表活队列；原seed101/temp1已完成50万步并通过CPU独立审计。R11预算授权后才开始新派发。

R11代码位于oracle_diag/coverage_oracle_r11；输出oracle_diag/results/coverage_oracle_20261016_r11。pipeline.py首次队列PID57312，首次子任务PID57468为T(temp1、500k、seed102)。必须核对实际命令行，不只按PID判断。

## 核心文件

- outputs/research_idea_protocol_20261016_r11/STAGE_CONTRACT_R11.md 与 PROTOCOL_R11.json
- 输出目录active_execution.json、stage_authorization.json、queue_started.json、queue_status.json、training_plan.json
- 输出目录reuse_receipts.json、science_copy_provenance.json、cpu_acceptance.json、code_frozen.json
- 输出目录protected_manifest.json（3209项）、各阶段protected_*.json
- 首次日志launch1.stdout.log/launch1.stderr.log，子任务日志logs/

## 已完成CPU补验

R09学习器、训练、审计和评分四文件逐字节复制，hash完全一致。只改独立输出路径/授权边界、两阶段调参调度和H决策。旧完整CPU代数验收承接，不无故重跑训练。

复用项的audit/completed/evaluations/model哈希匹配，参数字典、CPU初始化digest及target首批抽样与新管线精确相同。补验检查了温度平局、H通过/失败/零差、拒绝1M额外温度、拒绝U/G验证/正式/第二任务。补验完全CPU，未初始化CUDA。

## H与报告口径

对同seed计算Last5(1M)−Last5(500k)，三项差用样本SD(ddof1)/sqrt3作SE；配对均值严格>2SE才用1M，否则500k。规则有意偏向短H，不证明等效或严格95%结论。

**500k的Last-5是460k/470k/480k/490k/500k；1M的是960k/970k/980k/990k/1M。两者是不同训练后期阶段，不是相同阶段或相同绝对训练步数窗口。cosine周期也不同。**

TEMP_500K_SELECTED.json记录只看500k选温度；T_selected.json记录H判定。T_LOCKED_READY.json及T_LOCKED_R11.md生成后交付T配置，记录交付凭据防重复。参考/诊断/分布审计/五组冒烟可在已授权范围继续，T锁定不自动授权U/G验证。

## 完成与异常

参考在真实G/C/R/U回报前固定；5000步诊断为CPU确定性均值策略。scores/component_audit.json与SCORE_AUDIT.md先于掩码封存；任何rel/gap贡献超过90%，须写实质单一分量选择，不宣称组合效果，不改公式或搜索权重。C的V0只有5000target且缺口外推不可靠，G胜C不等于原则优劣。

五组冒烟全部审计和保护核验后输出STAGE_REPORT_R11.md、STAGE_COMPLETED.json，状态AWAITING_SEPARATE_U_G_AUTHORIZATION，然后停止。不可恢复旧队列或启动U/G验证。单GPU串行、隐藏窗口、诊断CPU、不终止其他应用。异常先CPU诊断，保留原始证据，不自动重训成功/失败作业；若只是审计故障，只修审计并重审。

跟进沿用id r09-1-3，名称已更新为“覆盖缺口 R11 T锁定与冒烟交付”；以其最新R11提示为准。完成交付后停用。
