# j009b_nfbc_walkerm：源数据组成 2×2 筛查（远源权重 × 近邻项；候选 BCMIXN）（单任务子任务）

种子 [3121, 3122, 3123, 3124, 3125]（新种子）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。

本任务是 j009 的一个子任务（j009a/b/c 各一个任务）。支持/否定的判定需要三个子任务的结果，在 analysis/ 中合并后按 research/PLAN.md 的登记规则做出；这里只给出本任务上每个候选是否“筛查有效”。

## 预设规则（提交前写进 research/PLAN.md）

- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004-j008 metric)
- strong_control：per task, the group among T, BC, BCMIX with the highest Last5 mean over seeds 3121-3125
- screen_effective：candidate (BCMIXN) minus strong control, paired by seed: mean > 0 and mean > 1 * paired SE
- supported：screen-effective on at least 2 of the 3 tasks (j009a/b/c combined) -> a confirmation job with 5 never-used seeds per task on those tasks
- rejected：paired mean difference <= 0 on at least 2 of the 3 tasks
- otherwise：uncertain; not advanced to confirmation
- factorial_registered_secondary：not counted by the loop: near term helps if near_at_far1 and near_at_far05 are both > 0 and > 1 SE on a task; far weight hurts on walker medium if BCMIX-W50 < 0 and BCMIXN-W50N < 0 there, each beyond 1 SE
- descriptive：2 * SE criterion, candidate minus BCMIX / BC / T, all 2x2 effects and the interaction, W50N vs strong control on every task, near-set size, episode lengths
- note：Rules are fixed before results; not changed after results are seen. The verdict is made only after all three sub-jobs finish; if any sub-job fails, no verdict.

## 本任务上的筛查结果

- BCMIXN：筛查有效 是；均值差 ≤ 0：否

## 近邻集的统计（描述性，各种子相同）

- W50N：阈值 tau=1.549；近邻集 10132 / 999323 行（1.01%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}
- BCMIXN：阈值 tau=1.549；近邻集 10132 / 999323 行（1.01%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}

## walker2d-kinematic-thighjnt-medium（强对照：T，67.06）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|67.06|5.27|60.98|864.5|0.00|61.0 / 71.7 / 62.4 / 67.6 / 72.5|
|BC（只用目标数据）|64.72|4.52|57.89|851.2|0.04|64.6 / 57.9 / 67.3 / 69.9 / 63.8|
|BC-混合（远源权重 1，无近邻项）|55.12|14.87|29.17|834.3|0.02|67.0 / 58.5 / 61.4 / 29.2 / 59.5|
|W50（远源权重 0.5，无近邻项）|63.40|4.97|57.75|802.6|0.00|60.6 / 57.8 / 61.4 / 67.8 / 69.4|
|W50N（远源权重 0.5 + 近邻项 0.5）|71.85|4.32|64.46|894.0|0.00|74.2 / 71.9 / 73.4 / 64.5 / 75.3|
|BCMIXN（候选：远源权重 1 + 近邻项 0.5）|70.68|3.23|67.03|884.7|0.00|75.2 / 72.5 / 68.8 / 67.0 / 69.8|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|BCMIXN|3.62|3.07|3|是|否|15.56（SE 5.69）|5.96（SE 3.13）|3.62（SE 3.07）|

描述性：BC-混合 − BC -9.60（SE 7.94，2/5 正）。

2×2 效应（描述性，同种子配对）：

|效应|均值|配对SE|正号种子|
|---|---|---|---|
|near_at_far1 (BCMIXN-BCMIX)|15.56|5.69|5|
|near_at_far05 (W50N-W50)|8.45|3.28|4|
|far_without_near (BCMIX-W50)|-8.28|8.02|2|
|far_with_near (BCMIXN-W50N)|-1.17|1.63|3|
|interaction|7.12|8.69|2|

## 运行情况

总耗时 4.2 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
