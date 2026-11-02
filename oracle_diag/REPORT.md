# IQL Bellman oracle 诊断

**全部阶段已完成；以下为 seed 0 结果。**

更新时间：2026-10-05T12:38:02+08:00

## 执行范围与前提修正

- source=shifted ODRL medium；target=正常 D4RL medium 的 seed 0 / 10% 子集。原数据加载、target 标准化和 kNN 函数均复用。
- 每个 source 原文件 5000 行，原加载器实际输出 4999 行；本诊断全部使用，不重复抽样伪装成 20000 个独立点。
- 原算法、train.py、注册表、旧脚本和旧结果均未修改。新增算法与内存注册适配器位于 oracle_diag；原训练入口的 NTFS 短路径解决 Windows 保存兼容问题。
- source_budget=.20、eta=1、clamp=[.05,1]、batch=128。权重沿用原流程，作用于 source V/Q/actor 三项损失；clamp 后均值可能略高于 .20。
- ORACLE-BACKUP 使用常数 .20，只把 source Q 的回归目标换成 oracle 奖励、下一状态与终止掩码。它是理想转移对照，不是策略回报的数学上界；失败也不能证明所有 Bellman 方法无效。

## 阶段 0：回放验收

使用 CPU。每环境随机检查 5000 条 target、完整 4999 条 source；qvel 截断样本沿同一轨迹恢复。下表是最终 source 回放误差（逐样本最大状态分量误差）：

| 环境 | source n | 状态误差均值 | 中位数 | p95 | 最大 | 奖励误差均值 | 奖励中位数 | 奖励 p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| walker2d-kinematic-footjnt | 4999 | 8.8668e-07 | 2.987e-07 | 3.5738e-06 | 3.1981e-05 | 7.8712e-08 | 6.7266e-08 | 1.882e-07 |
| hopper-kinematic-legjnt | 4999 | 2.2065e-07 | 1.6972e-07 | 5.7851e-07 | 1.1335e-06 | 8.8283e-08 | 7.2131e-08 | 2.1967e-07 |
| halfcheetah-kinematic-footjnt | 4999 | 6.709e-07 | 4.4996e-07 | 1.3552e-06 | 8.4798e-05 | 1.8825e-07 | 1.5714e-07 | 4.4656e-07 |

| 环境 | target 可恢复样本 | 状态误差均值 / 中位数 / p95 | 奖励误差均值 / 中位数 / p95 | target / source 截断率 |
|---|---:|---|---|---|
| walker2d-kinematic-footjnt | 4999 / 5000 | 1.0491e-06 / 5.1154e-07 / 3.4499e-06 | 9.9717e-08 / 7.9426e-08 / 2.5664e-07 | 30.78% / 13.60% |
| hopper-kinematic-legjnt | 4999 / 5000 | 2.3148e-07 / 1.8442e-07 / 5.6034e-07 | 7.9744e-08 / 6.4642e-08 / 2.0158e-07 | 17.62% / 9.32% |
| halfcheetah-kinematic-footjnt | 5000 / 5000 | 1.0957e-06 / 5.8187e-07 / 2.8126e-06 | 1.5281e-07 / 1.2812e-07 / 3.6744e-07 | 0.00% / 0.00% |

- shift 由 XML 关节活动范围改变实现，未对动作额外裁剪。Hopper joint range 确实改变，但在本 source 支持集上，两套模拟器的一步转移相同；不能只凭任务名假设存在可识别动力学差异。
- Walker2d 首轮有 20 条 source 回放失败。求解器 warmstart 不是原因；改用 1–64 步候选历史恢复后全部达到原阈值（状态 1e-3，奖励 1e-4）。不同通过验收的历史产生的 target 结果也必须在同样阈值内一致。此前失败结果保留，未放宽阈值。
- Walker2d、Hopper 各 1 条 target 样本在轨迹开头已有速度截断，缺少可恢复历史；明确排除这两条，未将其计为通过。source 则全量通过。
- 这是经过回放验证的近似模拟器 oracle，仍受 float32 观测和有限状态恢复精度限制。诊断没有宣称取得原始未舍入的精确物理状态。
- 证据：results/verified_20261003/{acceptance,warmstart_acceptance,final_acceptance}.json 与每环境 NPZ。

## 阶段 1、2a：真实差异与下一状态估计

误差为 target 特征库状态标准差单位下的 L2；A=邻居下一状态均值，B=当前 source 状态加邻居平均位移，C=原 source 下一状态。

| 环境 | 物理 shift 均值 | A 误差均值 | B 误差均值 | C 误差均值 | A 差异相关 | B 差异相关 |
|---|---:|---:|---:|---:|---:|---:|
| walker2d-kinematic-footjnt | 1.6188 | 4.1875 | 2.4857 | 1.6228 | -0.43278 | -0.24171 |
| hopper-kinematic-legjnt | 1.804e-16 | 1.8119 | 0.36132 | 0.02293 | 0.11631 | 0.076693 |
| halfcheetah-kinematic-footjnt | 0.48233 | 3.6115 | 3.4962 | 0.48778 | -0.11478 | 0.075562 |

完整分布、近邻距离相关、奖励差异及剔除边界的敏感性见 `results/verified_20261003/diagnostics/stage1_stage2a.json`。
每环境 oracle NPZ 与原 source buffer 的行 i 一一对应，文件 SHA256 与索引说明均已记录。每个 source 加载器有 4 条跨轨迹 next_state 跳变；保留原训练数据，同时另报去除边界的诊断。

## 阶段 2b：固定 V 下 Bellman 排序质量

| 环境 | V | 估计器 | 全局 Spearman | batch 内均值 | top20% 重合 | ESS ratio |
|---|---|---|---:|---:|---:|---:|
| walker2d-kinematic-footjnt | TARGET | knn | 0.17078 | 0.17027 | 0.231 | 0.76583 |
| walker2d-kinematic-footjnt | TARGET | diff_mean_state | 0.14203 | 0.14242 | 0.357 | 0.80622 |
| walker2d-kinematic-footjnt | TARGET | oracle | 1 | 1 | 1 | 0.75421 |
| walker2d-kinematic-footjnt | MIX_FB | knn | -0.17255 | -0.17171 | 0.029 | 0.79348 |
| walker2d-kinematic-footjnt | MIX_FB | diff_mean_state | -0.20746 | -0.20316 | 0.166 | 0.79237 |
| walker2d-kinematic-footjnt | MIX_FB | oracle | 1 | 1 | 1 | 0.73722 |
| hopper-kinematic-legjnt | TARGET | knn | 0.050426 | 0.051476 | 0.183 | 0.81229 |
| hopper-kinematic-legjnt | TARGET | diff_mean_state | 0.066713 | 0.066832 | 0.214 | 0.8203 |
| hopper-kinematic-legjnt | TARGET | oracle | 1 | 1 | 1 | 0.75669 |
| hopper-kinematic-legjnt | MIX_FB | knn | -0.021754 | -0.020152 | 0.173 | 0.82055 |
| hopper-kinematic-legjnt | MIX_FB | diff_mean_state | 0.043065 | 0.044876 | 0.209 | 0.81209 |
| hopper-kinematic-legjnt | MIX_FB | oracle | 1 | 1 | 1 | 0.77936 |
| halfcheetah-kinematic-footjnt | TARGET | knn | -0.04818 | -0.047666 | 0.155 | 0.7803 |
| halfcheetah-kinematic-footjnt | TARGET | diff_mean_state | 0.22996 | 0.23019 | 0.387 | 0.82545 |
| halfcheetah-kinematic-footjnt | TARGET | oracle | 1 | 1 | 1 | 0.8255 |
| halfcheetah-kinematic-footjnt | MIX_FB | knn | -0.021432 | -0.021434 | 0.162 | 0.96287 |
| halfcheetah-kinematic-footjnt | MIX_FB | diff_mean_state | 0.067598 | 0.067823 | 0.241 | 0.80228 |
| halfcheetah-kinematic-footjnt | MIX_FB | oracle | 1 | 1 | 1 | 0.78192 |

batch 指 seed 0 打乱后的 39 个完整 128 样本组，余下 7 条只进入全局指标；这些组不是独立训练 seeds。top20% 取 1000 条，平分用原样本索引确定；独立随机排序的期望重合率约 20%。
差分主口径在 B 的均值下一状态上评价 V，奖励及 not_done 取邻居均值；另报逐邻居平移后计算 V 再平均的敏感性，不将两者混为同一估计。
kNN 失配直接调用原 compute_source_weight，并核验权重公式。V 是学到的网络；oracle 的“真实”只指给定 V 下、经回放验证的转移，不代表 V 自身准确。每环境只有约 5 条 source 轨迹，4999 条转移不能当作 4999 个独立样本；相关与重合率为该数据集上的描述指标。

## 阶段 3：seed 0 回报训练

进度：15/15 个训练/模型复用任务完成。新训练均为 500k 步，保存最终模型及每 100k 的 V。

| 环境 | 方法 | Last-5 normalized score | 来源 |
|---|---|---:|---|
| walker2d-kinematic-footjnt | IQL-target | 60.199 | 本轮新训练/经核验复用 |
| walker2d-kinematic-footjnt | IQL_BELLMAN_FB-b0.20 | 61.687 | 本轮新训练/经核验复用 |
| walker2d-kinematic-footjnt | IQL_CONST-w0.20 | 58.42 | 本轮新训练/经核验复用 |
| walker2d-kinematic-footjnt | IQL_BELLMAN_ORACLE-b0.20 | 18.123 | 本轮新训练/经核验复用 |
| walker2d-kinematic-footjnt | ORACLE-BACKUP | 66.365 | 本轮新训练/经核验复用 |
| walker2d-kinematic-footjnt | IQL-mix | 60.077 | 既有500k seed0，历史对照 |
| hopper-kinematic-legjnt | IQL-target | 49.72 | 本轮新训练/经核验复用 |
| hopper-kinematic-legjnt | IQL_BELLMAN_FB-b0.20 | 57.538 | 本轮新训练/经核验复用 |
| hopper-kinematic-legjnt | IQL_CONST-w0.20 | 59.657 | 本轮新训练/经核验复用 |
| hopper-kinematic-legjnt | IQL_BELLMAN_ORACLE-b0.20 | 59.766 | 本轮新训练/经核验复用 |
| hopper-kinematic-legjnt | ORACLE-BACKUP | 51.539 | 本轮新训练/经核验复用 |
| hopper-kinematic-legjnt | IQL-mix | 46.07 | 既有500k seed0，历史对照 |
| halfcheetah-kinematic-footjnt | IQL-target | 44.705 | 本轮新训练/经核验复用 |
| halfcheetah-kinematic-footjnt | IQL_BELLMAN_FB-b0.20 | 44.052 | 本轮新训练/经核验复用 |
| halfcheetah-kinematic-footjnt | IQL_CONST-w0.20 | 44.254 | 本轮新训练/经核验复用 |
| halfcheetah-kinematic-footjnt | IQL_BELLMAN_ORACLE-b0.20 | 44.724 | 本轮新训练/经核验复用 |
| halfcheetah-kinematic-footjnt | ORACLE-BACKUP | 44.722 | 本轮新训练/经核验复用 |
| halfcheetah-kinematic-footjnt | IQL-mix | 42.244 | 既有500k seed0，历史对照 |

指标统一取 460k、470k、480k、490k、500k 的 `test/target_normalized_score` 均值。仅日志齐备、有限且任务成功时纳入，不选 best checkpoint。归一化保持原项目口径。

## 当前结论

- walker2d-kinematic-footjnt / TARGET：kNN 的 batch 内相关 0.17027；差分修正 0.14242。这是该固定 V、该 source 支持集上的结果。
- walker2d-kinematic-footjnt / MIX_FB：kNN 的 batch 内相关 -0.17171；差分修正 -0.20316。这是该固定 V、该 source 支持集上的结果。
- hopper-kinematic-legjnt / TARGET：kNN 的 batch 内相关 0.051476；差分修正 0.066832。这是该固定 V、该 source 支持集上的结果。
- hopper-kinematic-legjnt / MIX_FB：kNN 的 batch 内相关 -0.020152；差分修正 0.044876。这是该固定 V、该 source 支持集上的结果。
- halfcheetah-kinematic-footjnt / TARGET：kNN 的 batch 内相关 -0.047666；差分修正 0.23019。这是该固定 V、该 source 支持集上的结果。
- halfcheetah-kinematic-footjnt / MIX_FB：kNN 的 batch 内相关 -0.021434；差分修正 0.067823。这是该固定 V、该 source 支持集上的结果。
- walker2d-kinematic-footjnt：ORACLE−CONST 的 Last-5 差值 -40.297。单个 seed 的点差不能宣称统计显著或证明总体优势。
- hopper-kinematic-legjnt：ORACLE−CONST 的 Last-5 差值 0.10885。单个 seed 的点差不能宣称统计显著或证明总体优势。
- halfcheetah-kinematic-footjnt：ORACLE−CONST 的 Last-5 差值 0.47019。单个 seed 的点差不能宣称统计显著或证明总体优势。

## 图表

- [walker2d-kinematic-footjnt / TARGET 散点图](results/verified_20261003/figures/walker2d-kinematic-footjnt_TARGET_bellman.png)；另存同名 SVG/PDF。
- [walker2d-kinematic-footjnt / MIX_FB 散点图](results/verified_20261003/figures/walker2d-kinematic-footjnt_MIX_FB_bellman.png)；另存同名 SVG/PDF。
- [hopper-kinematic-legjnt / TARGET 散点图](results/verified_20261003/figures/hopper-kinematic-legjnt_TARGET_bellman.png)；另存同名 SVG/PDF。
- [hopper-kinematic-legjnt / MIX_FB 散点图](results/verified_20261003/figures/hopper-kinematic-legjnt_MIX_FB_bellman.png)；另存同名 SVG/PDF。
- [halfcheetah-kinematic-footjnt / TARGET 散点图](results/verified_20261003/figures/halfcheetah-kinematic-footjnt_TARGET_bellman.png)；另存同名 SVG/PDF。
- [halfcheetah-kinematic-footjnt / MIX_FB 散点图](results/verified_20261003/figures/halfcheetah-kinematic-footjnt_MIX_FB_bellman.png)；另存同名 SVG/PDF。

## 复现与文件

- verify_replay_v2.py / verify_warmstart.py / investigate_walker_precision.py / finalize_replay.py：独立回放检查和恢复证据。
- prepare_diagnostics.py：阶段1与2a；bellman_diagnostics.py：阶段2b。
- oracle_methods.py：索引buffer、oracle权重和oracle Q backup；train_entry.py：不改原文件的入口适配。
- run_all_stages.py：单进程 GPU 队列及自动诊断/报告；queue_status.json 记录成功、失败与当前任务。
- 新结果全部位于 oracle_diag/results，既有结果不覆盖；失败 smoke 的目录也保留。
- 当次汇总的机器可读证据见 results/verified_20261003/report_data.json。
