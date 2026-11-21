# j007c_pbbc_walkerh：近邻加采样 BC-混合（PB25、PB50）筛查（单任务子任务）

种子 [3111, 3112, 3113, 3114, 3115]（新种子）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。

本任务是 j007 的一个子任务（j007a/b/c 各一个任务）。支持/否定的判定需要三个子任务的结果，在 analysis/ 中合并后按 research/PLAN.md 的登记规则做出；这里只给出本任务上每个候选是否“筛查有效”。

## 预设规则（提交前写进 research/PLAN.md）

- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004/j005/j006 metric)
- strong_control：per task, the group among T, BC, BCMIX with the highest Last5 mean over seeds 3111-3115
- screen_effective：candidate minus strong control, paired by seed: mean > 0 and mean > 1 * paired SE
- supported：screen-effective on at least 2 of the 3 tasks (j007a/b/c combined) -> a confirmation job with 5 never-used seeds per task on those tasks
- rejected：paired mean difference <= 0 on at least 2 of the 3 tasks
- otherwise：uncertain; not advanced to confirmation
- descriptive：2 * SE criterion, candidate minus BCMIX / BC / T, PB50 vs PB25, near-set size, near-row BC loss, episode lengths
- note：Rules are fixed before results; not changed after results are seen. The verdict is made only after all three sub-jobs finish; if any sub-job fails, no verdict.

## 本任务上的筛查结果

- PB25：筛查有效 否；均值差 ≤ 0：是
- PB50：筛查有效 否；均值差 ≤ 0：是

## 近邻集的统计（描述性，各种子相同）

- PB25：阈值 tau=1.506；近邻集 5124 / 999323 行（0.51%），每批 16/64 个源行来自近邻集；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}
- PB50：阈值 tau=1.506；近邻集 5124 / 999323 行（0.51%），每批 32/64 个源行来自近邻集；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}

## walker2d-kinematic-thighjnt-hard（强对照：BCMIX，72.57）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|50.35|10.30|35.28|649.1|0.36|49.7 / 47.0 / 35.3 / 61.0 / 58.8|
|BC（只用目标数据）|53.13|12.35|38.05|685.8|0.29|52.0 / 38.1 / 55.9 / 71.8 / 48.0|
|BC-混合（目标 + 源数据）|72.57|0.75|71.53|875.7|0.00|73.4 / 72.1 / 71.5 / 72.8 / 72.9|
|PB25（候选：近邻加采样 BC-混合，源批中 16/64 来自近邻集）|71.18|1.23|70.04|885.9|0.00|73.0 / 70.2 / 70.7 / 70.0 / 71.9|
|PB50（候选：近邻加采样 BC-混合，源批中 32/64 来自近邻集）|71.99|2.27|69.19|893.8|0.00|71.5 / 71.4 / 69.2 / 72.4 / 75.4|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|PB25|-1.39|0.42|0|否|否|-1.39（SE 0.42）|18.04（SE 5.67）|20.83（SE 4.62）|
|PB50|-0.57|0.86|1|否|否|-0.57（SE 0.86）|18.86（SE 5.69）|21.64（SE 3.80）|

描述性：BC-混合 − BC 19.43（SE 5.48，5/5 正）；PB50 − PB25 0.82（SE 1.03，3/5 正）。

## 运行情况

总耗时 3.2 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
