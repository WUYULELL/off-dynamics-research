# j006b_vgbc_walkerm：价值函数引导的 BC-混合（AWM、AFM）筛查（单任务子任务）

种子 [3106, 3107, 3108, 3109, 3110]（新种子）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。

本任务是 j006 的一个子任务（j006a/b/c 各一个任务）。支持/否定的判定需要三个子任务的结果，在 analysis/ 中合并后按 research/PLAN.md 的登记规则做出；这里只给出本任务上每个候选是否“筛查有效”。

## 预设规则（提交前写进 research/PLAN.md）

- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004/j005 metric)
- strong_control：per task, the group among T, BC, BCMIX with the highest Last5 mean over seeds 3106-3110
- screen_effective：candidate minus strong control, paired by seed: mean > 0 and mean > 1 * paired SE
- supported：screen-effective on at least 2 of the 3 tasks (j006a/b/c combined) -> a confirmation job with 5 never-used seeds per task on those tasks
- rejected：paired mean difference <= 0 on at least 2 of the 3 tasks
- otherwise：uncertain; not advanced to confirmation
- descriptive：2 * SE criterion, candidate minus BCMIX / BC / T, AFM vs AWM, guidance statistics (source weight share, kept fraction, advantages), episode lengths
- note：Rules are fixed before results; not changed after results are seen. The verdict is made only after all three sub-jobs finish; if any sub-job fails, no verdict.

## 本任务上的筛查结果

- AWM：筛查有效 否；均值差 ≤ 0：是
- AFM：筛查有效 否；均值差 ≤ 0：是

## 价值函数引导的统计（描述性，5 个种子平均；“后20%”指最后 20% 的策略更新步）

- AWM：全程 {"adv_target_mean": -0.2082, "adv_source_mean": -3.8674, "source_adv_positive": 0.4236, "target_adv_positive": 0.3772, "source_weight_share": 0.9313, "weight_max": 5.0234, "clipped_fraction": 0.1902}；后20% {"adv_target_mean": -0.1927, "adv_source_mean": -4.2869, "source_adv_positive": 0.4232, "target_adv_positive": 0.376, "source_weight_share": 0.94, "weight_max": 5.0496, "clipped_fraction": 0.1895}
- AFM：全程 {"adv_target_mean": -0.2082, "adv_source_mean": -3.8674, "source_adv_positive": 0.4236, "target_adv_positive": 0.3772, "source_kept_fraction": 0.4236}；后20% {"adv_target_mean": -0.1927, "adv_source_mean": -4.2869, "source_adv_positive": 0.4232, "target_adv_positive": 0.376, "source_kept_fraction": 0.4232}

## walker2d-kinematic-thighjnt-medium（强对照：BC，69.00）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|68.33|5.50|60.15|885.8|0.00|68.6 / 60.2 / 68.4 / 75.7 / 68.9|
|BC（只用目标数据）|69.00|7.16|57.48|898.2|0.02|70.2 / 73.6 / 67.7 / 76.0 / 57.5|
|BC-混合（目标 + 源数据）|67.50|6.42|61.65|859.4|0.00|74.4 / 64.3 / 74.5 / 62.6 / 61.7|
|AWM（候选：目标价值函数优势加权 BC-混合）|41.33|23.65|15.73|510.2|0.01|15.7 / 18.4 / 48.0 / 70.5 / 54.1|
|AFM（候选：目标价值函数优势过滤 BC-混合）|48.13|12.11|33.12|595.2|0.05|33.1 / 64.1 / 40.0 / 54.3 / 49.2|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|AWM|-27.67|11.45|0|否|否|-26.17（SE 12.15）|-27.67（SE 11.45）|-27.00（SE 8.81）|
|AFM|-20.87|5.48|0|否|否|-19.37（SE 7.90）|-20.87（SE 5.48）|-20.21（SE 6.66）|

描述性：BC-混合 − BC -1.50（SE 4.08，3/5 正）；AFM − AWM 6.80（SE 11.21，2/5 正）。

## 运行情况

总耗时 10.6 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
