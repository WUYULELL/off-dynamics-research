# j012b_p2base_walkg20：第二阶段基线（不含 OTDF；描述性，不判定）

种子 [3129, 3130, 3131, 3132, 3133]（与 j011 相同，使第二阶段 6 个任务用同一组种子）；T、BC、BC-混合、W50N 同机同种子。主指标 Last5（最后五个评估点平均，每点 10 回合），分数按 ODRL 参考回报归一化。本实验是第二阶段基线的一部分，不判定、不计入自动循环计数；W50N 只作描述。OTDF 在服务器恢复后用同一组种子补跑。

## 预设说明（提交前写进 research/PLAN.md）

- type：descriptive phase-2 baseline; no candidate verdict, not counted by the loop (PLAN section 1 phase 2, j012)
- metric：Last5 = mean of the last five evaluation points, 10 episodes each (V21/j004-j011 metric)
- reported：per group mean / SD / worst seed / episode length / fraction under 200 steps; strongest control among T, BC, BCMIX; W50N minus strongest control / BCMIX / BC / T with paired SE (descriptive); BCMIX minus BC
- use：these runs plus j011 (two original tasks) form the T / BC / BCMIX columns of the phase-2 baseline; OTDF is added later with seeds 3129-3133; candidates are then judged by the phase-2 rules in PLAN section 1
- note：Rules fixed before results. Any failed run -> no summary (failure handling). All seeds kept.

## 近邻集的统计（描述性，各种子相同）

- walker2d-gravity-2.0：阈值 tau=0.683；近邻集 3025 / 999323 行（0.30%），近邻项每批单独抽 32 行（权重 0.5）；目标留一轨迹最近距离分位数 {"0.1": 0.03624555608541186, "0.5": 0.22688767163261223, "0.9": 0.6825305881509504, "0.99": 2.3481922621270654}；源数据到目标最近距离分位数 {"0.01": 1.2726201728559914, "0.1": 2.2235255238103684, "0.5": 3.080591852810191, "0.9": 7.077963738349796}

## walker2d-gravity-2.0（强对照：BC，42.83）

|组|Last5均值|SD|最差种子|回合长度|<200步比例|各种子|
|---|---|---|---|---|---|---|
|T（只用目标数据的 IQL）|41.60|2.17|38.90|614.3|0.00|43.0 / 44.4 / 40.2 / 38.9 / 41.6|
|BC（只用目标数据）|42.83|3.17|40.26|638.7|0.00|41.5 / 47.8 / 44.2 / 40.4 / 40.3|
|BC-混合（远源权重 1，无近邻项）|15.96|7.84|8.28|280.2|0.62|19.4 / 9.7 / 27.6 / 14.9 / 8.3|
|W50N（描述性：远源权重 0.5 + 近邻项 0.5）|9.51|3.03|6.98|172.9|0.92|7.5 / 14.0 / 11.4 / 7.7 / 7.0|

|组（描述）|相对强对照|配对SE|正号种子|均值高于强对照|>1SE|>2SE|相对BC-混合|相对BC|相对T|
|---|---|---|---|---|---|---|---|---|---|
|W50N|-33.31|0.25|0|否|否|否|-6.44（SE 3.65）|-33.31（SE 0.25）|-32.08（SE 1.27）|

描述性：BC-混合 − BC -26.87（SE 3.75，0/5 正）。

## 运行情况

总耗时 3.1 小时。冒烟运行的回报不读取、不报告。所有种子都保留。
