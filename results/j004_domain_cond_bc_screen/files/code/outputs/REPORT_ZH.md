# j004：域条件 BC-混合（DCBC）与质量-域条件 BC-混合（QDCBC）筛查

种子 3001–3005（新种子）；T、BC、BC-混合在同一台机器、同一批种子下重跑。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。

## 预设规则（提交前写进 research/PLAN.md）

- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21 metric)
- strong_control：per task, the group among T, BC, BCMIX with the highest Last5 mean over seeds 3001-3005
- screen_effective：candidate minus strong control, paired by seed: mean > 0 and mean > 1 * paired SE
- supported：screen-effective on at least 2 of the 3 tasks -> a confirmation job with 5 never-used seeds per task on those tasks
- rejected：paired mean difference <= 0 on at least 2 of the 3 tasks
- otherwise：uncertain; not advanced to confirmation
- descriptive：2 * SE criterion, candidate minus BCMIX and minus BC, DCBC vs QDCBC, episode lengths
- note：Rules are fixed before results; not changed after results are seen.

## 按规则的结论

- DCBC：**否定**；筛查有效的任务：无；均值差 ≤ 0 的任务：hopper-kinematic-legjnt-medium、walker2d-kinematic-thighjnt-medium、walker2d-kinematic-thighjnt-hard
- QDCBC：**否定**；筛查有效的任务：无；均值差 ≤ 0 的任务：hopper-kinematic-legjnt-medium、walker2d-kinematic-thighjnt-medium、walker2d-kinematic-thighjnt-hard

## hopper-kinematic-legjnt-medium（强对照：BCMIX，56.38）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|40.39|2.20|38.36|382.4|0.00|40.2 / 40.6 / 44.0 / 38.8 / 38.4|
|BC（只用目标数据）|39.92|3.21|37.99|374.0|0.00|38.0 / 38.5 / 38.3 / 45.6 / 39.2|
|BC-混合（目标 + 源数据）|56.38|2.61|52.76|567.3|0.00|52.8 / 57.3 / 59.9 / 55.3 / 56.6|
|DCBC（候选：域条件 BC-混合）|40.07|3.17|37.83|374.3|0.00|38.0 / 38.4 / 37.8 / 40.8 / 45.3|
|QDCBC（候选：质量与域条件 BC-混合）|40.52|3.21|38.03|378.7|0.00|38.0 / 46.0 / 40.5 / 38.7 / 39.4|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|DCBC|-16.31|1.87|0|否|否|-16.31（SE 1.87）|0.14（SE 1.74）|-0.32（SE 2.23）|
|QDCBC|-15.86|1.36|0|否|否|-15.86（SE 1.36）|0.60（SE 2.32）|0.13（SE 1.53）|

描述性：BC-混合 − BC 16.46（SE 2.01，5/5 正）；QDCBC − DCBC 0.45（SE 2.29，3/5 正）。

## walker2d-kinematic-thighjnt-medium（强对照：BC，69.74）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|58.88|6.57|49.97|773.6|0.03|50.0 / 67.4 / 59.3 / 55.7 / 62.1|
|BC（只用目标数据）|69.74|5.39|63.93|907.8|0.02|68.1 / 63.9 / 66.0 / 76.3 / 74.4|
|BC-混合（目标 + 源数据）|64.57|9.00|49.42|842.1|0.03|68.8 / 72.5 / 49.4 / 63.9 / 68.2|
|DCBC（候选：域条件 BC-混合）|68.70|5.05|61.15|881.0|0.02|67.4 / 61.2 / 74.1 / 68.4 / 72.4|
|QDCBC（候选：质量与域条件 BC-混合）|69.60|3.72|64.30|896.9|0.01|68.8 / 73.3 / 64.3 / 73.1 / 68.5|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|DCBC|-1.03|2.61|1|否|否|4.13（SE 5.89）|-1.03（SE 2.61）|9.82（SE 4.18）|
|QDCBC|-0.14|2.60|2|否|否|5.02（SE 3.00）|-0.14（SE 2.60）|10.71（SE 3.05）|

描述性：BC-混合 − BC -5.16（SE 4.51，2/5 正）；QDCBC − DCBC 0.90（SE 3.73，3/5 正）。

## walker2d-kinematic-thighjnt-hard（强对照：BCMIX，70.70）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|58.84|6.85|48.15|744.5|0.27|48.2 / 61.0 / 57.2 / 66.6 / 61.1|
|BC（只用目标数据）|52.29|9.24|41.15|672.8|0.29|48.7 / 52.3 / 41.2 / 52.7 / 66.6|
|BC-混合（目标 + 源数据）|70.70|2.59|66.74|853.0|0.00|70.9 / 72.7 / 66.7 / 69.9 / 73.2|
|DCBC（候选：域条件 BC-混合）|62.63|9.87|48.56|794.3|0.21|59.6 / 60.9 / 48.6 / 70.9 / 73.2|
|QDCBC（候选：质量与域条件 BC-混合）|64.02|10.36|49.68|804.7|0.20|49.7 / 56.9 / 73.8 / 72.1 / 67.6|

|候选|相对强对照|配对SE|正号种子|筛查有效（>1SE）|>2SE（描述）|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|
|DCBC|-8.07|3.70|1|否|否|-8.07（SE 3.70）|10.34（SE 2.09）|3.79（SE 3.85）|
|QDCBC|-6.68|5.31|2|否|否|-6.68（SE 5.31）|11.73（SE 6.23）|5.18（SE 3.39）|

描述性：BC-混合 − BC 18.41（SE 3.25，5/5 正）；QDCBC − DCBC 1.39（SE 6.21，2/5 正）。

## 运行情况

总耗时 9.6 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
