# V12 阶段 0.3 单卡吞吐测量报告

测速已停止，等待用户决定后续实验。
所有结果均为丢弃的测速短跑，不计入正式实验，不按回报选择参数或并发数。

|算法|并发|整组更新每秒|各进程速率之和|平均GPU利用率|显存峰值 MiB|本任务CPU百分比 100代表一核|
|---|---:|---:|---:|---:|---:|---:|
|IQL|1|99.032|99.067|10.667|2035|99.742|
|IQL|2|197.645|198.111|94.625|4066|199.091|
|IQL|4|391.489|393.256|97.845|8129|397.030|
|IQL|6|542.542|547.043|98.875|12191|595.158|
|BC|1|339.363|339.864|7.571|2031|99.054|

## IQL_n1

|种子|前1万步 秒|后1万步 秒|训练总计 秒|一次10回合评估 秒|长度均值|长度中位数|最短|小于200比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|901|100.113|101.771|201.884|0.440|112.600|97.500|93|1.0000|

资源压力标记：[]

## IQL_n2

|种子|前1万步 秒|后1万步 秒|训练总计 秒|一次10回合评估 秒|长度均值|长度中位数|最短|小于200比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|901|102.410|99.932|202.342|0.676|112.600|97.500|93|1.0000|
|902|100.799|100.674|201.473|0.892|169.400|165.500|158|1.0000|

资源压力标记：[]

## IQL_n4

|种子|前1万步 秒|后1万步 秒|训练总计 秒|一次10回合评估 秒|长度均值|长度中位数|最短|小于200比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|901|100.898|102.500|203.398|0.837|112.600|97.500|93|1.0000|
|902|101.760|102.530|204.290|1.128|169.400|165.500|158|1.0000|
|903|101.907|101.948|203.855|1.088|157.500|170.500|97|1.0000|
|904|102.336|99.853|202.189|1.023|144.000|149.500|89|1.0000|

资源压力标记：[]

## IQL_n6

|种子|前1万步 秒|后1万步 秒|训练总计 秒|一次10回合评估 秒|长度均值|长度中位数|最短|小于200比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|901|109.407|109.281|218.688|1.242|112.600|97.500|93|1.0000|
|902|112.398|108.428|220.826|1.690|169.400|165.500|158|1.0000|
|903|110.998|109.350|220.348|1.620|157.500|170.500|97|1.0000|
|904|108.378|108.109|216.487|1.527|144.000|149.500|89|1.0000|
|905|111.125|110.033|221.158|1.728|180.000|173.500|127|0.7000|
|906|109.104|109.627|218.730|1.565|149.400|116.000|108|0.7000|

资源压力标记：[]

## BC_n1

|种子|前1万步 秒|后1万步 秒|训练总计 秒|一次10回合评估 秒|长度均值|长度中位数|最短|小于200比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|901|29.134|29.713|58.847|0.909|236.600|252.000|111|0.4000|

资源压力标记：[]

## 未测档位

[]

## 建议和完整训练耗时外推

{
  "recommended_measured_concurrency": 6,
  "selection": "Highest group updates/s among measured IQL tiers without resource-pressure flags",
  "future_concurrency_requires_user_confirmation": true,
  "not_a_formal_experiment_authorization": true,
  "estimates": [
    {
      "algorithm": "IQL",
      "concurrency": 1,
      "full_training_updates": 500000,
      "training_only_seconds": 5047.100303182378,
      "estimated_total_seconds": 5069.0828673075885,
      "evaluation_points": 50,
      "uses_early_policy_episode_length": true,
      "not_a_runtime_guarantee": true
    },
    {
      "algorithm": "IQL",
      "concurrency": 2,
      "full_training_updates": 500000,
      "training_only_seconds": 5047.688449732959,
      "estimated_total_seconds": 5086.889704363421,
      "evaluation_points": 50,
      "uses_early_policy_episode_length": true,
      "not_a_runtime_guarantee": true
    },
    {
      "algorithm": "IQL",
      "concurrency": 4,
      "full_training_updates": 500000,
      "training_only_seconds": 5085.822588088922,
      "estimated_total_seconds": 5136.769323376939,
      "evaluation_points": 50,
      "uses_early_policy_episode_length": true,
      "not_a_runtime_guarantee": true
    },
    {
      "algorithm": "IQL",
      "concurrency": 6,
      "full_training_updates": 500000,
      "training_only_seconds": 5484.327070578001,
      "estimated_total_seconds": 5562.432031768064,
      "evaluation_points": 50,
      "uses_early_policy_episode_length": true,
      "not_a_runtime_guarantee": true
    },
    {
      "algorithm": "BC",
      "concurrency": 1,
      "full_training_updates": 200000,
      "training_only_seconds": 588.4700798615813,
      "estimated_total_seconds": 624.8167575150728,
      "evaluation_points": 40,
      "uses_early_policy_episode_length": true,
      "not_a_runtime_guarantee": true
    }
  ],
  "limitation": "One short measurement per tier under shared background load; early short episodes can underestimate late evaluation time"
}

推荐仅针对本次已测且资源压力可接受的IQL档位；未来正式并发仍待用户确认。
整组吞吐按总更新数/同步放行到最后训练结束的时间计算；另一列是各进程自身更新速率之和，两者不混为一谈。
训练计时不包含初始化、环境创建、屏障等待、评估和CPU审计；每进程训练耗时包含相同的逐步损失记录开销。
IQL500k估计采用25倍20k训练时间加50次评估；BC200k采用10倍训练时间加40次评估。早期短回合可能低估后期评估耗时。
本次只有末尾一次评估，Last5不适用。没有重新训练正式种子或启动阶段1A/1B/ROOT。

## 停止位置

AWAITING_USER_STAGE1_DECISION
原迁移项目、原Conda、已完成0.2与原代码保持只读，详见保护凭据。
