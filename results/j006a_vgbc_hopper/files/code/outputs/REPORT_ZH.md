# j006a_vgbc_hopper：价值函数引导的 BC-混合（AWM、AFM）筛查（单任务子任务）

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

- AWM：全程 {"adv_target_mean": -0.2108, "adv_source_mean": -7.5572, "source_adv_positive": 0.3255, "target_adv_positive": 0.3784, "source_weight_share": 0.8975, "weight_max": 6.6856, "clipped_fraction": 0.1433}；后20% {"adv_target_mean": -0.1925, "adv_source_mean": -7.218, "source_adv_positive": 0.3324, "target_adv_positive": 0.3763, "source_weight_share": 0.9145, "weight_max": 6.5801, "clipped_fraction": 0.1455}
- AFM：全程 {"adv_target_mean": -0.2108, "adv_source_mean": -7.5572, "source_adv_positive": 0.3255, "target_adv_positive": 0.3784, "source_kept_fraction": 0.3255}；后20% {"adv_target_mean": -0.1925, "adv_source_mean": -7.218, "source_adv_positive": 0.3324, "target_adv_positive": 0.3763, "source_kept_fraction": 0.3324}

## hopper-kinematic-legjnt-medium（强对照：BCMIX，54.00）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|43.61|4.48|38.59|412.8|0.00|44.3 / 46.1 / 38.6 / 49.4 / 39.7|
|BC（只用目标数据）|37.75|0.40|37.37|352.2|0.00|38.3 / 37.4 / 37.6 / 38.0 / 37.4|
|BC-混合（目标 + 源数据）|54.00|1.08|52.19|544.8|0.00|54.9 / 53.9 / 52.2 / 54.2 / 54.7|
|AWM（候选：目标价值函数优势加权 BC-混合）|14.67|7.91|8.41|266.1|0.52|14.9 / 28.1 / 9.8 / 12.1 / 8.4|
|AFM（候选：目标价值函数优势过滤 BC-混合）|17.31|10.61|7.92|268.1|0.26|13.2 / 13.1 / 16.9 / 35.4 / 7.9|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|AWM|-39.33|3.53|0|否|否|-39.33（SE 3.53）|-23.08（SE 3.57）|-28.94（SE 3.11）|
|AFM|-36.69|4.84|0|否|否|-36.69（SE 4.84）|-20.44（SE 4.67）|-26.30（SE 3.69）|

描述性：BC-混合 − BC 16.25（SE 0.46，5/5 正）；AFM − AWM 2.64（SE 6.27，2/5 正）。

## 运行情况

总耗时 6.8 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
