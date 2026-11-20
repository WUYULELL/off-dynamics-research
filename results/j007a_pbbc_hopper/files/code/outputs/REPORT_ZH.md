# j007a_pbbc_hopper：近邻加采样 BC-混合（PB25、PB50）筛查（单任务子任务）

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

- PB25：筛查有效 是；均值差 ≤ 0：否
- PB50：筛查有效 否；均值差 ≤ 0：是

## 近邻集的统计（描述性，各种子相同）

- PB25：阈值 tau=0.539；近邻集 14666 / 999999 行（1.47%），每批 16/64 个源行来自近邻集；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}
- PB50：阈值 tau=0.539；近邻集 14666 / 999999 行（1.47%），每批 32/64 个源行来自近邻集；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}

## hopper-kinematic-legjnt-medium（强对照：BCMIX，53.12）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|42.41|3.47|38.79|401.6|0.00|46.6 / 41.0 / 40.1 / 45.6 / 38.8|
|BC（只用目标数据）|40.95|1.66|39.26|383.7|0.00|41.4 / 39.3 / 40.6 / 43.5 / 39.9|
|BC-混合（目标 + 源数据）|53.12|2.30|50.12|538.3|0.00|55.8 / 53.2 / 51.7 / 50.1 / 54.9|
|PB25（候选：近邻加采样 BC-混合，源批中 16/64 来自近邻集）|54.27|1.13|53.12|546.1|0.00|55.2 / 55.7 / 53.1 / 54.0 / 53.4|
|PB50（候选：近邻加采样 BC-混合，源批中 32/64 来自近邻集）|53.03|1.68|50.47|533.9|0.00|53.6 / 52.7 / 55.1 / 53.3 / 50.5|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|PB25|1.16|0.98|3|是|否|1.16（SE 0.98）|13.32（SE 0.97）|11.86（SE 1.41）|
|PB50|-0.09|1.52|2|否|否|-0.09（SE 1.52）|12.08（SE 0.88）|10.61（SE 1.47）|

描述性：BC-混合 − BC 12.17（SE 1.55，5/5 正）；PB50 − PB25 -1.25（SE 0.91，1/5 正）。

## 运行情况

总耗时 4.1 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
