# j007b_pbbc_walkerm：近邻加采样 BC-混合（PB25、PB50）筛查（单任务子任务）

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

- PB25：筛查有效 否；均值差 ≤ 0：否
- PB50：筛查有效 是；均值差 ≤ 0：否

## 近邻集的统计（描述性，各种子相同）

- PB25：阈值 tau=1.549；近邻集 10132 / 999323 行（1.01%），每批 16/64 个源行来自近邻集；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}
- PB50：阈值 tau=1.549；近邻集 10132 / 999323 行（1.01%），每批 32/64 个源行来自近邻集；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}

## walker2d-kinematic-thighjnt-medium（强对照：BC，69.54）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|66.35|4.37|61.42|862.7|0.00|65.6 / 65.1 / 66.2 / 61.4 / 73.4|
|BC（只用目标数据）|69.54|6.97|58.00|908.8|0.00|72.7 / 74.2 / 68.1 / 58.0 / 74.7|
|BC-混合（目标 + 源数据）|61.73|10.18|47.07|797.4|0.02|65.0 / 75.1 / 62.8 / 47.1 / 58.7|
|PB25（候选：近邻加采样 BC-混合，源批中 16/64 来自近邻集）|71.26|3.67|68.18|890.8|0.00|68.2 / 68.9 / 68.8 / 76.0 / 74.5|
|PB50（候选：近邻加采样 BC-混合，源批中 32/64 来自近邻集）|72.45|2.49|69.54|900.4|0.00|73.0 / 75.6 / 70.4 / 69.5 / 73.7|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|PB25|1.72|4.22|2|否|否|9.53（SE 5.98）|1.72（SE 4.22）|4.91（SE 2.44）|
|PB50|2.91|2.23|4|是|否|10.72（SE 3.72）|2.91（SE 2.23）|6.10（SE 1.78）|

描述性：BC-混合 − BC -7.81（SE 2.83，1/5 正）；PB50 − PB25 1.19（SE 2.30，3/5 正）。

## 运行情况

总耗时 4.2 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
