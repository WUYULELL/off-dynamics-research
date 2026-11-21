# j009c_nfbc_walkerh：源数据组成 2×2 筛查（远源权重 × 近邻项；候选 BCMIXN）（单任务子任务）

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

- W50N：阈值 tau=1.506；近邻集 5124 / 999323 行（0.51%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}
- BCMIXN：阈值 tau=1.506；近邻集 5124 / 999323 行（0.51%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}

## walker2d-kinematic-thighjnt-hard（强对照：BCMIX，72.29）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|45.86|12.16|33.98|598.0|0.42|34.0 / 51.1 / 36.6 / 64.1 / 43.5|
|BC（只用目标数据）|51.27|6.57|40.61|663.8|0.30|40.6 / 51.2 / 54.5 / 58.2 / 51.8|
|BC-混合（远源权重 1，无近邻项）|72.29|3.42|67.24|877.1|0.00|72.4 / 71.1 / 76.1 / 74.5 / 67.2|
|W50（远源权重 0.5，无近邻项）|65.31|7.31|53.92|824.7|0.08|64.5 / 73.3 / 69.7 / 65.1 / 53.9|
|W50N（远源权重 0.5 + 近邻项 0.5）|73.61|1.85|71.87|921.9|0.00|72.5 / 71.9 / 74.8 / 76.3 / 72.6|
|BCMIXN（候选：远源权重 1 + 近邻项 0.5）|69.33|4.05|64.43|871.4|0.00|65.8 / 73.9 / 72.0 / 70.5 / 64.4|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|BCMIXN|-2.96|1.57|1|否|否|-2.96（SE 1.57）|18.06（SE 2.61）|23.47（SE 5.04）|

描述性：BC-混合 − BC 21.01（SE 2.92，5/5 正）。

2×2 效应（描述性，同种子配对）：

|效应|均值|配对SE|正号种子|
|---|---|---|---|
|near_at_far1 (BCMIXN-BCMIX)|-2.96|1.57|1|
|near_at_far05 (W50N-W50)|8.31|3.33|4|
|far_without_near (BCMIX-W50)|6.98|2.57|4|
|far_with_near (BCMIXN-W50N)|-4.28|1.81|1|
|interaction|-11.26|4.34|1|

## 运行情况

总耗时 4.4 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
