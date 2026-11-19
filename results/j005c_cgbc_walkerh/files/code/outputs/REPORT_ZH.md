# j005c_cgbc_walkerh：覆盖门控 BC-混合（CG50、CG90）筛查（单任务子任务）

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

- CG50：阈值 tau=0.547；源数据保留 998127 / 999323 行（99.9%）；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}
- CG90：阈值 tau=1.506；源数据保留 994199 / 999323 行（99.5%）；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}

## walker2d-kinematic-thighjnt-hard（强对照：BCMIX，72.83）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|61.40|6.31|52.80|774.4|0.22|60.6 / 70.5 / 62.4 / 60.7 / 52.8|
|BC（只用目标数据）|56.51|8.36|44.63|721.9|0.25|54.2 / 57.3 / 44.6 / 58.6 / 67.8|
|BC-混合（目标 + 源数据）|72.83|3.79|66.76|879.8|0.00|74.3 / 71.7 / 76.2 / 66.8 / 75.1|
|CG50（候选：覆盖门控 BC-混合，阈值 50% 分位）|61.76|9.01|47.59|775.0|0.12|64.5 / 68.3 / 69.8 / 47.6 / 58.7|
|CG90（候选：覆盖门控 BC-混合，阈值 90% 分位）|64.75|4.42|58.46|813.2|0.06|58.5 / 68.7 / 69.2 / 63.0 / 64.3|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|CG50|-11.07|2.97|0|否|否|-11.07（SE 2.97）|5.25（SE 6.81）|0.36（SE 3.74）|
|CG90|-8.08|2.39|0|否|否|-8.08（SE 2.39）|8.24（SE 4.72）|3.35（SE 2.62）|

描述性：BC-混合 − BC 16.32（SE 4.47，5/5 正）；CG90 − CG50 2.99（SE 3.63，3/5 正）。

## 运行情况

总耗时 4.8 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
