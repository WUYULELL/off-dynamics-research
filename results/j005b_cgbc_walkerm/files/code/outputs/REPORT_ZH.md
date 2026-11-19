# j005b_cgbc_walkerm：覆盖门控 BC-混合（CG50、CG90）筛查（单任务子任务）

种子 [3101, 3102, 3103, 3104, 3105]（新种子）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。

本任务是 j005 的一个子任务（j005a/b/c 各一个任务）。支持/否定的判定需要三个子任务的结果，在 analysis/ 中合并后按 research/PLAN.md 的登记规则做出；这里只给出本任务上每个候选是否“筛查有效”。

## 预设规则（提交前写进 research/PLAN.md）

- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004 metric)
- strong_control：per task, the group among T, BC, BCMIX with the highest Last5 mean over seeds 3101-3105
- screen_effective：candidate minus strong control, paired by seed: mean > 0 and mean > 1 * paired SE
- supported：screen-effective on at least 2 of the 3 tasks (j005a/b/c combined) -> a confirmation job with 5 never-used seeds per task on those tasks
- rejected：paired mean difference <= 0 on at least 2 of the 3 tasks
- otherwise：uncertain; not advanced to confirmation
- descriptive：2 * SE criterion, candidate minus BCMIX and minus BC, CG90 vs CG50, kept source fraction, episode lengths
- note：Rules are fixed before results; not changed after results are seen. The verdict is made only after all three sub-jobs finish; if any sub-job fails, no verdict.

## 本任务上的筛查结果

- CG50：筛查有效 否；均值差 ≤ 0：是
- CG90：筛查有效 否；均值差 ≤ 0：否

## 覆盖门控的统计（描述性，各种子相同）

- CG50：阈值 tau=0.389；源数据保留 997815 / 999323 行（99.8%）；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}
- CG90：阈值 tau=1.549；源数据保留 989191 / 999323 行（99.0%）；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}

## walker2d-kinematic-thighjnt-medium（强对照：BC，65.56）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|62.45|6.98|53.96|819.0|0.00|63.4 / 72.8 / 58.7 / 63.4 / 54.0|
|BC（只用目标数据）|65.56|3.59|61.72|859.7|0.05|64.4 / 63.7 / 71.0 / 61.7 / 67.0|
|BC-混合（目标 + 源数据）|60.43|14.13|35.26|752.5|0.00|67.2 / 68.4 / 66.0 / 35.3 / 65.2|
|CG50（候选：覆盖门控 BC-混合，阈值 50% 分位）|64.63|3.71|58.81|796.7|0.00|67.3 / 66.9 / 67.1 / 58.8 / 63.0|
|CG90（候选：覆盖门控 BC-混合，阈值 90% 分位）|67.08|7.22|58.08|829.9|0.00|68.4 / 61.5 / 71.8 / 58.1 / 75.6|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|CG50|-0.93|1.64|2|否|否|4.20（SE 4.87）|-0.93（SE 1.64）|2.18（SE 3.15）|
|CG90|1.52|2.21|3|否|否|6.64（SE 4.95）|1.52（SE 2.21）|4.63（SE 5.97）|

描述性：BC-混合 − BC -5.13（SE 5.60，2/5 正）；CG90 − CG50 2.45（SE 3.01，3/5 正）。

## 运行情况

总耗时 7.1 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
