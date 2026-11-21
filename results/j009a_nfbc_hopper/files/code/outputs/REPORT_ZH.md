# j009a_nfbc_hopper：源数据组成 2×2 筛查（远源权重 × 近邻项；候选 BCMIXN）（单任务子任务）

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

- BCMIXN：筛查有效 否；均值差 ≤ 0：是

## 近邻集的统计（描述性，各种子相同）

- W50N：阈值 tau=0.539；近邻集 14666 / 999999 行（1.47%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}
- BCMIXN：阈值 tau=0.539；近邻集 14666 / 999999 行（1.47%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}

## hopper-kinematic-legjnt-medium（强对照：BCMIX，54.56）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|42.09|2.53|39.16|398.2|0.00|39.2 / 41.5 / 40.6 / 43.7 / 45.5|
|BC（只用目标数据）|40.89|3.17|37.04|383.3|0.00|43.0 / 37.8 / 43.6 / 43.0 / 37.0|
|BC-混合（远源权重 1，无近邻项）|54.56|1.88|51.81|550.4|0.00|53.4 / 51.8 / 56.1 / 56.0 / 55.6|
|W50（远源权重 0.5，无近邻项）|48.24|2.34|45.42|486.5|0.00|49.7 / 46.1 / 49.1 / 50.8 / 45.4|
|W50N（远源权重 0.5 + 近邻项 0.5）|52.12|4.37|45.16|525.3|0.00|57.2 / 45.2 / 53.2 / 52.2 / 52.8|
|BCMIXN（候选：远源权重 1 + 近邻项 0.5）|54.31|2.19|51.28|545.8|0.00|55.2 / 51.3 / 56.9 / 55.1 / 53.0|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|BCMIXN|-0.26|0.75|2|否|否|-0.26（SE 0.75）|13.42（SE 0.70）|12.21（SE 1.75）|

描述性：BC-混合 − BC 13.68（SE 1.35，5/5 正）。

2×2 效应（描述性，同种子配对）：

|效应|均值|配对SE|正号种子|
|---|---|---|---|
|near_at_far1 (BCMIXN-BCMIX)|-0.26|0.75|2|
|near_at_far05 (W50N-W50)|3.89|1.66|4|
|far_without_near (BCMIX-W50)|6.33|1.09|5|
|far_with_near (BCMIXN-W50N)|2.18|1.42|4|
|interaction|-4.14|1.76|1|

## 运行情况

总耗时 4.2 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
