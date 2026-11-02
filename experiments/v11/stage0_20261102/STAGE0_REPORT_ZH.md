# V11阶段0：源动作策略通道实现与一致性检查

用户更正：原文 −log π 作废。实际采用 O1 的优势加权动作 MSE，exp(10A) 截断上限100。
Lπ=目标加权MSE均值+w_a×源加权MSE均值；动作维先求均值。源/目标优势均取同一步V更新前的 min(target Q1,target Q2)−V。
SA-all对64条源样本求均值。SA-top先取源批次优势前16/64，再要求A>0，未入选权重为0；按用户最新澄清对入选样本求均值（分母=入选数），无入选时源项0。优势并列按批内原位置优先。
V/Q与O1相同，w=1，64target+64source；actor复用该source批次，无额外抽样，不构造额外oracle缓存。cosine周期始终500k，短试跑不缩短周期。

w_a=0：seed201新实现10k与原O1同步10k，逐步三损失及最终四网络、三Adam、cosine完整状态逐位一致，并核验历史O1最初10步损失。
SA-all/SA-top各seed201从0运行20k；这是诊断，不作正式Last5或H_act判定。每10k评估10episode，保存恢复训练RNG。
计算计数：新实现50k+原O1参考10k=60k GPU learner更新（180k Adam步骤）；另有8次CPU合成learner更新（24次Adam步骤），CPU真实数据更新0、CPU环境步0、CPU验收未初始化CUDA。GPU评估共50episode。

|变体|源A>0比例|目标权重和均值|源有效权重和均值|源入选数均值|目标损失均值|源损失均值|实测秒|
|---|---:|---:|---:|---:|---:|---:|---:|
|SA_all_s201|0.3681281|718.423|1069.498|64|0.1406347|1.484859|405.75|
|SA_top_s201|0.3681281|718.423|945.7995|15.331|0.3008884|5.410003|405.58|

全部逐步权重、MSE、筛选与损失在CPU从原始向量独立复核；权重和未除批大小，另存归一化权重和，二者不混称。损失范围/1000步诊断曲线在STAGE0_RESULT.json，原始逐步向量及日志在runs/。未按回报修改参数，也未以短试跑得分选择变体。

## 后续预算（仅估计，未授权启动）

- SA_all_s201：单次500k约2.82小时，10seed约28.18小时。
- SA_top_s201：单次500k约2.82小时，10seed约28.17小时。

20项直接外推合计约56.34小时，另需逐项CPU审计/最终核验；短前缀与后期episode长度可能不同，非精确承诺。
本次按v11修订版只执行阶段0，完成后队列退出并停下汇报。阶段1/阶段2的任务、变体和先后顺序均须用户之后明确确认。Walker后续BC参照确定为BC-all均值38.85（精确值和收据见config.json）；V9阶段4和T调参未授权。
原始指令快照及更正见config.json、authorization.json；命令/模型配置/代码数据XML哈希见launch1.json、FROZEN.json、runs/*/config.json。


## 无入选样本的边界情况

SA-top源项=sum(入选exp(10A)×动作MSE)/入选数，再乘w_a。无人入选时源项0，不除以0。
- SA_all_s201：无人入选批次0/20000，占比0.000000。
- SA_top_s201：无人入选批次0/20000，占比0.000000。

## 修复文件、行号与一致性证据

故障修复本身仅明确O1评估函数的导入路径。另列修订版要求的停止范围、配置/报告/诊断改动。
|用途|文件|本次行号|说明|
|---|---|---|---|
|import_repair_and_scope|C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v11_source_actor_recovery_20261102_stage0\trial.py|1-1,9-14,17-17,134-135|Original failure fixed by explicit O1 evaluate file import; CLI restricted to3 Stage0 trials; training loop AST unchanged|
|scope_and_protection|C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v11_source_actor_recovery_20261102_stage0\common.py|1-1,18-18,32-33,45-47,125-125|New output only, require stages1/2 unauthorized; protect old failed evidence; reject formal_jobs|
|scope_controller|C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v11_source_actor_recovery_20261102_stage0\pipeline.py|1-1,5-5,7-7,10-17,19-21,23-25,27-27,29-29,31-34,38-43,46-46,49-49|Only3 Stage0 trials; reuse accepted CPU receipt; audit/report/verify then exit; no later-stage launch branch|
|diagnostic_only|C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v11_source_actor_recovery_20261102_stage0\audit.py|32-32,63-63,75-76|Compute zero-selected-batch count/fraction from saved scalar diagnostics; no optimizer or sampling change|
|report_only|C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v11_source_actor_recovery_20261102_stage0\report.py|33-34,39-39,51-51,54-54,56-71|Revised instruction, selected-count denominator and empty-batch ratio; repair lines and evidence; stop afterStage0|
|verification_only|C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v11_source_actor_recovery_20261102_stage0\verify.py|1-1,22-24|Verify revised authorization, absent later-stage outputs and repair-evidence hashes|

训练逻辑：actor.py逐字节等同原CPU验收版本；短试跑训练循环AST、V/Q、三Adam和cosine更新顺序与候选相同。SA-top分母按用户澄清维持候选，非修复时暗改。
随机序列：初始化与Sampler源码未改；同seed初始化和完整target/source索引流与O1独立重建值匹配，actor不额外抽样。
评估协议：明确文件路径加载原冻结O1同一evaluate，函数AST/文件SHA核验；每10k十episode、固定evalseed、shifted XML、.002/1000、原归一化、保存恢复Python/NumPy/Torch/CUDA训练RNG均沿用。
本次额外CPU优化0，复用此前8次合成learner更新/24Adam。原启动失败发生在trial.main之前，原GPU更新/环境步0。旧文件和失败现场保留，证据哈希见REPAIR_CHANGELOG.json及DELIVERY_VERIFIED.json。
