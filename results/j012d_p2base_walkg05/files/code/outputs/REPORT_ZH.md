# j012d_p2base_walkg05：第二阶段基线（不含 OTDF；描述性，不判定）

种子 [3129, 3130, 3131, 3132, 3133]（与 j011 相同，使第二阶段 6 个任务用同一组种子）；T、BC、BC-混合、W50N 同机同种子。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。本实验是第二阶段基线的一部分，不判定、不计入自动循环计数；W50N 只作描述。OTDF 在服务器恢复后用同一组种子补跑。

## 预设说明（提交前写进 research/PLAN.md）

- type：descriptive phase-2 baseline; no candidate verdict, not counted by the loop (PLAN section 1 phase 2, j012)
- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004-j011 metric)
- reported：per group mean / SD / worst seed / episode length / fraction under 200 steps; strongest control among T, BC, BCMIX; W50N minus strongest control / BCMIX / BC / T with paired SE (descriptive); BCMIX minus BC
- use：these runs plus j011 (two original tasks) form the T / BC / BCMIX columns of the phase-2 baseline; OTDF is added later with seeds 3129-3133; candidates are then judged by the phase-2 rules in PLAN section 1
- note：Rules fixed before results. Any failed run -> no summary (failure handling). All seeds kept.

## 近邻集的统计（描述性，各种子相同）

- walker2d-gravity-0.5：阈值 tau=1.892；近邻集 303604 / 999323 行（30.38%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.028458901992368755, "0.5": 0.32309512349403496, "0.9": 1.8920095411624414, "0.99": 4.3503619187173}；源数据到目标最近距离分位数 {"0.01": 0.5927774304705903, "0.1": 1.323614889561859, "0.5": 2.277139995256028, "0.9": 3.8360746533591366}

## walker2d-gravity-0.5（强对照：T，34.51）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|34.51|2.42|32.54|596.4|0.06|38.4 / 34.9 / 32.5 / 32.6 / 34.2|
|BC（只用目标数据）|28.74|2.70|24.80|521.0|0.13|24.8 / 30.6 / 31.4 / 29.5 / 27.3|
|BC-混合（远源权重 1，无近邻项）|26.80|5.48|20.90|509.5|0.18|20.9 / 24.3 / 31.3 / 23.6 / 33.8|
|W50N（描述性：远源权重 0.5 + 近邻项 0.5）|24.35|13.82|7.47|488.6|0.33|42.3 / 14.0 / 7.5 / 31.0 / 26.9|

|组（描述）|相对强对照|配对SE|正号种子|均值高于强对照|>1SE|>2SE|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|-10.17|5.54|1|否|否|否|-2.45（SE 7.76）|-4.39（SE 7.28）|-10.17（SE 5.54）|

描述性：BC-混合 − BC -1.93（SE 2.38，1/5 正）。

## 运行情况

总耗时 2.7 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
