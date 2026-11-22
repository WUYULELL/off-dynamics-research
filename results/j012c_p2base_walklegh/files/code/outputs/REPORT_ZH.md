# j012c_p2base_walklegh：第二阶段基线（不含 OTDF；描述性，不判定）

种子 [3129, 3130, 3131, 3132, 3133]（与 j011 相同，使第二阶段 6 个任务用同一组种子）；T、BC、BC-混合、W50N 同机同种子。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。本实验是第二阶段基线的一部分，不判定、不计入自动循环计数；W50N 只作描述。OTDF 在服务器恢复后用同一组种子补跑。

## 预设说明（提交前写进 research/PLAN.md）

- type：descriptive phase-2 baseline; no candidate verdict, not counted by the loop (PLAN section 1 phase 2, j012)
- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004-j011 metric)
- reported：per group mean / SD / worst seed / episode length / fraction under 200 steps; strongest control among T, BC, BCMIX; W50N minus strongest control / BCMIX / BC / T with paired SE (descriptive); BCMIX minus BC
- use：these runs plus j011 (two original tasks) form the T / BC / BCMIX columns of the phase-2 baseline; OTDF is added later with seeds 3129-3133; candidates are then judged by the phase-2 rules in PLAN section 1
- note：Rules fixed before results. Any failed run -> no summary (failure handling). All seeds kept.

## 近邻集的统计（描述性，各种子相同）

- walker2d-morph-leg-hard：阈值 tau=1.534；近邻集 3631 / 999323 行（0.36%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.20733560655941727, "0.5": 0.5367710291138088, "0.9": 1.5341873915305604, "0.99": 4.355743662285993}；源数据到目标最近距离分位数 {"0.01": 2.508656324683796, "0.1": 5.485429407299861, "0.5": 31.7431798473256, "0.9": 70.95473893564561}

## walker2d-morph-leg-hard（强对照：BC，60.57）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|60.00|1.12|58.33|970.5|0.00|61.0 / 59.4 / 58.3 / 60.5 / 60.8|
|BC（只用目标数据）|60.57|0.89|59.27|975.7|0.00|61.5 / 60.4 / 60.4 / 59.3 / 61.3|
|BC-混合（远源权重 1，无近邻项）|18.99|7.71|9.93|371.5|0.19|29.8 / 9.9 / 14.3 / 22.9 / 18.0|
|W50N（描述性：远源权重 0.5 + 近邻项 0.5）|15.92|3.52|10.51|311.6|0.38|18.0 / 15.8 / 10.5 / 15.5 / 19.8|

|组（描述）|相对强对照|配对SE|正号种子|均值高于强对照|>1SE|>2SE|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|-44.64|1.42|0|否|否|否|-3.07（SE 3.16）|-44.64（SE 1.42）|-44.08（SE 1.15）|

描述性：BC-混合 − BC -41.57（SE 3.36，0/5 正）。

## 运行情况

总耗时 2.8 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
