# j012a_p2base_hopfootm：第二阶段基线（不含 OTDF；描述性，不判定）

种子 [3129, 3130, 3131, 3132, 3133]（与 j011 相同，使第二阶段 6 个任务用同一组种子）；T、BC、BC-混合、W50N 同机同种子。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。本实验是第二阶段基线的一部分，不判定、不计入自动循环计数；W50N 只作描述。OTDF 在服务器恢复后用同一组种子补跑。

## 预设说明（提交前写进 research/PLAN.md）

- type：descriptive phase-2 baseline; no candidate verdict, not counted by the loop (PLAN section 1 phase 2, j012)
- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004-j011 metric)
- reported：per group mean / SD / worst seed / episode length / fraction under 200 steps; strongest control among T, BC, BCMIX; W50N minus strongest control / BCMIX / BC / T with paired SE (descriptive); BCMIX minus BC
- use：these runs plus j011 (two original tasks) form the T / BC / BCMIX columns of the phase-2 baseline; OTDF is added later with seeds 3129-3133; candidates are then judged by the phase-2 rules in PLAN section 1
- note：Rules fixed before results. Any failed run -> no summary (failure handling). All seeds kept.

## 近邻集的统计（描述性，各种子相同）

- hopper-kinematic-footjnt-medium：阈值 tau=0.315；近邻集 6514 / 999999 行（0.65%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.04469878470067892, "0.5": 0.1231373172937709, "0.9": 0.31540370265425677, "0.99": 2.840376006428274}；源数据到目标最近距离分位数 {"0.01": 0.3966532977557309, "0.1": 1.98799876746625, "0.5": 3.4788342773160825, "0.9": 4.580094515451433}

## hopper-kinematic-footjnt-medium（强对照：BC，76.42）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|74.33|2.96|70.41|893.3|0.00|78.7 / 75.0 / 73.8 / 70.4 / 73.8|
|BC（只用目标数据）|76.42|1.04|75.56|921.7|0.00|76.3 / 75.6 / 78.1 / 76.6 / 75.6|
|BC-混合（远源权重 1，无近邻项）|15.70|1.59|14.71|216.9|0.00|18.5 / 14.8 / 15.3 / 15.2 / 14.7|
|W50N（描述性：远源权重 0.5 + 近邻项 0.5）|17.02|2.66|12.45|235.5|0.20|19.3 / 17.2 / 12.4 / 17.8 / 18.3|

|组（描述）|相对强对照|配对SE|正号种子|均值高于强对照|>1SE|>2SE|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|-59.41|1.60|0|否|否|否|1.32（SE 1.13）|-59.41（SE 1.60）|-57.31（SE 1.53）|

描述性：BC-混合 − BC -60.72（SE 0.83，0/5 正）。

## 运行情况

总耗时 2.8 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
