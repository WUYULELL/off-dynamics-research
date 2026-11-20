# j006c_vgbc_walkerh：价值函数引导的 BC-混合（AWM、AFM）筛查（单任务子任务）

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

- AWM：全程 {"adv_target_mean": -0.2287, "adv_source_mean": -0.7142, "source_adv_positive": 0.4662, "target_adv_positive": 0.3796, "source_weight_share": 0.9242, "weight_max": 4.7156, "clipped_fraction": 0.2117}；后20% {"adv_target_mean": -0.2191, "adv_source_mean": -2.7421, "source_adv_positive": 0.4166, "target_adv_positive": 0.3804, "source_weight_share": 0.9185, "weight_max": 5.364, "clipped_fraction": 0.1904}
- AFM：全程 {"adv_target_mean": -0.2287, "adv_source_mean": -0.7142, "source_adv_positive": 0.4662, "target_adv_positive": 0.3796, "source_kept_fraction": 0.4662}；后20% {"adv_target_mean": -0.2191, "adv_source_mean": -2.7421, "source_adv_positive": 0.4166, "target_adv_positive": 0.3804, "source_kept_fraction": 0.4166}

## walker2d-kinematic-thighjnt-hard（强对照：BCMIX，73.14）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|55.16|16.50|28.80|703.3|0.30|51.1 / 66.9 / 28.8 / 70.2 / 58.8|
|BC（只用目标数据）|52.86|15.67|30.05|675.8|0.32|30.1 / 59.9 / 44.9 / 70.6 / 58.9|
|BC-混合（目标 + 源数据）|73.14|2.60|70.59|889.0|0.00|70.9 / 74.5 / 76.8 / 73.0 / 70.6|
|AWM（候选：目标价值函数优势加权 BC-混合）|44.51|13.60|31.02|588.1|0.12|54.3 / 31.0 / 62.6 / 33.2 / 41.5|
|AFM（候选：目标价值函数优势过滤 BC-混合）|44.43|10.64|28.15|615.4|0.13|42.8 / 56.6 / 44.1 / 50.5 / 28.2|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|AWM|-28.63|5.90|0|否|否|-28.63（SE 5.90）|-8.34（SE 12.41）|-10.65（SE 13.30）|
|AFM|-28.71|4.26|0|否|否|-28.71（SE 4.26）|-8.43（SE 7.64）|-10.73（SE 7.62）|

描述性：BC-混合 − BC 20.29（SE 7.00，5/5 正）；AFM − AWM -0.09（SE 8.96，2/5 正）。

## 运行情况

总耗时 4.6 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
