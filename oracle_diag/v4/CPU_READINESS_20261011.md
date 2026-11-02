# Oracle v4：阶段A CPU准备交付

日期：2026-10-11。**实现与CPU核验完成；GPU冒烟、TARGET完整复用验收及20项正式训练尚未开始。本文不是实验效果报告。**

## 已完成

- 四组方法、独立runpy入口、实际抽样量与source权重断言。
- ORACLE-ONLY / SOURCE-ONLY复用原TargetIQL；训练数据接入口返回全部source，完全不读取shifted target数据。原IQL无经验状态归一化；评分使用固定OTDF常数。
- 核对v3实际SourceBackup使用的receipt与三个缓存数组SHA256；全部999322行的source字段、保留行映射与oracle buffer逐行一致。缓存未重新生成。
- 单域路径与原IQL在真实数组上的前10步损失、更新后参数逐位一致；五组初始化均与历史TARGET seed0哈希一致。CPU测试共90次更新调用，属于实现验收，没有环境rollout或性能结论。
- ORACLE-W1的source Q标签替换、target Q未替换、两次Q调用顺序、三项损失合成均通过独立公式检查；日志source_weight_mean=1.0。
- TensorBoard下划线标签读取通过；重复事件与NaN负例被拒绝。控制台中的NaN提示来自刻意构造的测试负例，不是训练失败。
- 重新读取15份v3正式TensorBoard基线，Last-5与历史审计一致。
- 2111个受保护旧文件哈希未变；新代码只在v4，新结果只在v4_20261011。

## 标签变化

| 量 | 均值 | 中位数 | P95 | 最大值 |
|---|---:|---:|---:|---:|
| next_state逐行L2差 | 4.116684 | 1.765667 | 15.359165 | 34.084612 |
| reward绝对差 | 0.042187 | 0.034360 | 0.105871 | 0.927668 |

not_done变化17行。以上是原域记录与shifted标签的差异，不是缓存重建误差；状态坐标单位不同，L2不能直接解释为回报影响。

## 等待执行

1. GPU冒烟5项：TARGET复用验证＋四个新组，各seed0/5000步。
2. 对历史TARGET smoke检查初始化、首批索引、有效配置、5000步全部模型/优化器及首次shifted评估。预定容差已写入protocol.json，失败即停。
3. v3没有历史早期损失记录。已新生成冻结v3代码的CPU前10步重放，将与新GPU早期损失比较，并明确标明其证据来源。
4. 冒烟通过后更新耗时估计，报告并确认正式队列。正式20项按组排序；前10项报告Q1/Q2，队列继续。

GPU冒烟通常预计数分钟至十余分钟，需实际测量；正式训练以v3经验暂估约51小时，source-only可能更快，尚不能精确承诺。

## 启动命令（仅在对应授权记录建立后可运行）

```powershell
& 'D:/ANACONDA/envs/offdynamics/python.exe' -B oracle_diag/v4/run_queue_v4.py --kind smoke
& 'D:/ANACONDA/envs/offdynamics/python.exe' -B oracle_diag/v4/run_queue_v4.py --kind formal
```

两个命令均有授权检查。现在执行会拒绝，旧v3的GPU授权不能解锁v4。正式命令还要求冒烟与TARGET复用核验通过。

## 科学结论边界

四组、Q1–Q4和判据保持原方案；结论只按支持程度报告。Q1通过不能独自证明状态覆盖有效，Q4不能唯一归因权重，Q3不能无条件分解混合流程的3.9分差距；未通过不代表所有动力学修正无效。详细审查见REVIEW_AND_PROTOCOL.md。

## 恢复与交付入口

- 当前状态：oracle_diag/results/v4_20261011/status.json。
- 冻结代码与CPU核验：cpu_ready.json、cpu1/completed.json、tooling1/completed.json。
- 队列：training_plan.json、run_queue_v4.py；防重复queue.lock；任务子目录非空即拒绝。
- 后续失败：保留输出，用新的attempt目录从同seed第0步开始；不能按回报挑选重跑。
- 中期报告：INTERIM_v4.md；最终报告：REPORT_v4.md，均在对应真实GPU运行完成并审计后生成。
- 全流程不自动追加未通过复用验收的五个TARGET、不启动额外任务、不与论文分数并排排名。
