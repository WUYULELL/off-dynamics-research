# j011_w50n_confirm：W50N 新种子确认（远源权重 0.5 + 近邻项 0.5）

种子 [3129, 3130, 3131, 3132, 3133]（从未用过）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。

W50N 是看过 j009 结果后选出的组（j009 中登记的一个组，但不是 j009 的候选）。判定按 research/PLAN.md 的 j011 登记；主要任务：walker2d-kinematic-thighjnt-medium / walker2d-kinematic-thighjnt-hard；hopper 只作描述。

## 预设规则（提交前写进 research/PLAN.md）

- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004-j009 metric)
- strong_control：per task, the group among T, BC, BCMIX with the highest Last5 mean over seeds 3129-3133
- confirmed_on_task：W50N Last5 mean over seeds 3129-3133 > strong-control Last5 mean (PLAN section 1 success condition: mean above the strongest control; no significance test)
- success：confirmed on BOTH primary tasks (walker2d-kinematic-thighjnt-medium and walker2d-kinematic-thighjnt-hard; the two tasks where W50N was above the strong control by > 1 paired SE in j009) -> PLAN section 1 condition 1 met; loop stops and the user is notified
- not_confirmed：otherwise; W50N is not a confirmed method
- hopper：hopper-kinematic-legjnt-medium is reported descriptively and does not count toward success (W50N was below BCMIX there in j009)
- descriptive：paired difference vs strong control / BCMIX / BC / T with SE and positive-seed count; 1 SE and 2 SE criteria; mean +- SD per group; BCMIX - BC; near-set size; episode lengths
- note：Rules fixed before results; not changed after results are seen. Any failed run -> no verdict (failure handling). All seeds kept.

## 按登记规则的结果

- W50N / hopper-kinematic-legjnt-medium（描述）：均值高于强对照 否
- W50N / walker2d-kinematic-thighjnt-medium（主要任务）：均值高于强对照 是
- W50N / walker2d-kinematic-thighjnt-hard（主要任务）：均值高于强对照 是
- W50N：两个主要任务都高于强对照（成功条件）是

## 近邻集的统计（描述性，各种子相同）

- hopper-kinematic-legjnt-medium：阈值 tau=0.539；近邻集 14666 / 999999 行（1.47%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.03674176978256924, "0.5": 0.16663886406286188, "0.9": 0.5386192867070883, "0.99": 1.4027663315052643}；源数据到目标最近距离分位数 {"0.01": 0.46732948902407384, "0.1": 1.4727535978636592, "0.5": 2.740342881800882, "0.9": 6.81167857583935}
- walker2d-kinematic-thighjnt-medium：阈值 tau=1.549；近邻集 10132 / 999323 行（1.01%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.13700898634124306, "0.5": 0.38922955338398657, "0.9": 1.5488928357718932, "0.99": 2.806880087067943}；源数据到目标最近距离分位数 {"0.01": 1.5437614604556777, "0.1": 2.2529495582983703, "0.5": 3.1434323053111717, "0.9": 6.8600708145539695}
- walker2d-kinematic-thighjnt-hard：阈值 tau=1.506；近邻集 5124 / 999323 行（0.51%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.20194940123591554, "0.5": 0.5465978674576242, "0.9": 1.5062802815072758, "0.99": 4.534110936636478}；源数据到目标最近距离分位数 {"0.01": 1.7965641825695626, "0.1": 3.269299087104668, "0.5": 6.331804486954131, "0.9": 12.241044435187273}

## hopper-kinematic-legjnt-medium（强对照：BCMIX，54.92）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|40.05|1.37|38.08|378.6|0.00|40.7 / 40.5 / 39.3 / 41.6 / 38.1|
|BC（只用目标数据）|41.31|3.00|38.06|387.2|0.00|40.5 / 41.7 / 38.1 / 40.1 / 46.1|
|BC-混合（远源权重 1，无近邻项）|54.92|1.49|53.34|553.4|0.00|55.5 / 53.3 / 55.1 / 53.6 / 57.0|
|W50N（确认对象：远源权重 0.5 + 近邻项 0.5）|50.56|1.68|48.75|509.3|0.00|51.2 / 48.8 / 52.8 / 49.0 / 51.0|

|候选|相对强对照|配对SE|正号种子|均值高于强对照|>1SE（描述）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|-4.36|0.59|0|否|否|否|-4.36（SE 0.59）|9.24（SE 1.67）|10.51（SE 1.22）|

描述性：BC-混合 − BC 13.60（SE 1.12，5/5 正）。

## walker2d-kinematic-thighjnt-medium（强对照：T，69.49）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|69.49|4.46|61.74|900.0|0.00|71.8 / 72.2 / 61.7 / 72.1 / 69.7|
|BC（只用目标数据）|64.82|8.30|50.27|853.0|0.06|66.4 / 67.4 / 70.6 / 69.5 / 50.3|
|BC-混合（远源权重 1，无近邻项）|63.49|3.33|59.76|819.8|0.03|68.9 / 62.9 / 59.8 / 63.5 / 62.5|
|W50N（确认对象：远源权重 0.5 + 近邻项 0.5）|71.47|0.52|70.91|892.2|0.01|71.4 / 71.6 / 71.2 / 70.9 / 72.3|

|候选|相对强对照|配对SE|正号种子|均值高于强对照|>1SE（描述）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|1.98|1.97|2|是|是|否|7.98（SE 1.51）|6.66（SE 3.92）|1.98（SE 1.97）|

描述性：BC-混合 − BC -1.32（SE 4.00，2/5 正）。

## walker2d-kinematic-thighjnt-hard（强对照：BCMIX，71.63）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|52.38|7.52|44.25|665.7|0.35|55.3 / 46.6 / 44.3 / 52.6 / 63.2|
|BC（只用目标数据）|49.80|7.22|39.76|651.3|0.35|46.9 / 39.8 / 58.5 / 49.1 / 54.7|
|BC-混合（远源权重 1，无近邻项）|71.63|4.45|65.10|872.0|0.01|74.2 / 65.1 / 72.4 / 69.8 / 76.7|
|W50N（确认对象：远源权重 0.5 + 近邻项 0.5）|72.04|1.81|69.43|899.8|0.00|73.0 / 73.1 / 70.9 / 69.4 / 73.8|

|候选|相对强对照|配对SE|正号种子|均值高于强对照|>1SE（描述）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|0.41|1.93|1|是|否|否|0.41（SE 1.93）|22.24（SE 3.51）|19.66（SE 3.09）|

描述性：BC-混合 − BC 21.83（SE 2.30，5/5 正）。

## 运行情况

总耗时 8.0 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
