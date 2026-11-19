# j005a_cgbc_hopper：覆盖门控 BC-混合（CG50、CG90）筛查（单任务子任务）

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
- CG90：筛查有效 否；均值差 ≤ 0：是

## 覆盖门控的统计（描述性，各种子相同）

- CG50：阈值 tau=0.167；源数据保留 997596 / 999999 行（99.8%）；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}
- CG90：阈值 tau=0.539；源数据保留 985333 / 999999 行（98.5%）；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}

## hopper-kinematic-legjnt-medium（强对照：BCMIX，51.66）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|40.03|2.65|37.96|378.6|0.00|44.7 / 39.1 / 39.5 / 38.9 / 38.0|
|BC（只用目标数据）|38.24|0.49|37.86|357.7|0.00|38.8 / 38.8 / 37.9 / 37.9 / 37.9|
|BC-混合（目标 + 源数据）|51.66|2.76|47.98|523.2|0.00|55.0 / 52.8 / 48.0 / 49.8 / 52.7|
|CG50（候选：覆盖门控 BC-混合，阈值 50% 分位）|45.57|1.47|44.27|454.6|0.00|47.8 / 44.9 / 44.6 / 46.3 / 44.3|
|CG90（候选：覆盖门控 BC-混合，阈值 90% 分位）|40.94|1.85|38.31|398.0|0.02|42.7 / 42.1 / 38.3 / 39.7 / 41.8|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|CG50|-6.09|1.08|0|否|否|-6.09（SE 1.08）|7.33（SE 0.58）|5.54（SE 0.70）|
|CG90|-10.72|0.45|0|否|否|-10.72（SE 0.45）|2.69（SE 0.67）|0.91（SE 1.13）|

描述性：BC-混合 − BC 13.41（SE 1.07，5/5 正）；CG90 − CG50 -4.64（SE 0.85，0/5 正）。

## 运行情况

总耗时 5.8 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
