# OTDF 复现协议静态审查

审查日期：2026-09-19。官方仓库：<https://github.com/dmksjfl/OTDF>。

锁定提交：`41e2e626844bd6b65e2316a418ce4d6df4bbe775`。以下代码行号均按本地该提交核查，不使用网页抽取器的行号。

**当前仅准备环境、源码和脚本。用户正在运行其他实验，明确要求等待通知后再启动；本次审查没有运行 OT 预计算、CVAE 训练、策略训练、环境 rollout 或性能测试。本文中的命令仅供后续使用。官方算法文件未修改。**

## 1. 建议首个任务与比较边界

首个任务采用 `walker2d-morph`，源域 `medium`、目标域 `medium`。这对应论文表 1 的 `walk-m / medium` 行：IQL* 为 23.0 ± 4.7，OTDF 为 50.5 ± 5.8。它是一个检查点，单个任务成功不能代表整篇论文已复现。

方向为**正常动力学 D4RL 源域 → 形态变化目标域**。用户此前 ODRL 实验的“变化源域 → 正常目标域”方向、目标域数据预算和训练步数不同，不能直接把旧实验分数与本表比较。

论文来源：[ICLR 2025 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4ce6738e84876aa79f13c8bc8b7c5eb-Paper-Conference.pdf)，重点核对正文表 1、附录 C、Algorithm 2、D.3 和表 5。

## 2. 数据、环境与预计算文件的对应

| 项目 | 首任务所需内容 | 代码证据 |
|---|---|---|
| 源域环境/数据 | `walker2d-medium-v2`；由 D4RL `qlearning_dataset` 加载 | `train_otdf.py:76–77,134` |
| 目标域原始文件 | `datasets/walker2d_morph_medium.hdf5`，已随官方仓库下载；本地文件大小 785756 字节 | `algo/utils.py:45–62` |
| 目标域配置 | `config/otdf/walker2d_morph.yaml` 中 `tar_env_config.env_name: Walker-morph` | 配置文件第 36–37 行 |
| 目标域实例 | Gym `walker2d_v3.Walker2dEnv` 加载 `envs/assets/walker_morph.xml`，外包 `TimeLimit(...,1000)` | `envs/common.py:12,69–76` |
| OT 输出/输入 | `costlogs/walker2d-morph-srcdatatype-medium-tardatatype-medium.hdf5`，键名 `cost` | `make_otdf_cost.py:112,175–178`；`algo/utils.py:27–43` |

不要自行换成 Gymnasium 新版环境、ODRL 另一套 XML 或反方向数据。即使环境简称相同，动力学、终止条件或奖励实现不同也会改变复现条件。官方 XML 在 `envs/assets/walker_morph.xml:17–38` 定义了两侧不同的腿部几何，应使用该文件。

泛化到其他任务时：CLI `--env walker2d` 代表 kinematic，目标文件为 `walker2d_kinematic_<quality>.hdf5`；`--env walker2d-gravity` 对应 `walker2d_gravity_0.5_<quality>.hdf5`；`--env walker2d-morph` 对应 `walker2d_morph_<quality>.hdf5`。质量名 `medium-expert` 在目标文件名中会转换为 `medium_expert`。见 `algo/utils.py:46–62`。

数据量不能根据文件名写成恰好 5000 条：论文采用约 5 条轨迹；官方加载器循环 `range(N-1)`，因此实际训练条数为原始奖励行数减 1，另需记录各数据文件哈希和处理后的条数。该加载器直接把 `observations[i+1]` 当下一状态，并不读取可能存在的显式 `next_observations`；timeouts 只用于计数/重置步数，不剔除跨轨迹边界样本。证据为 `algo/utils.py:75–122`。首次官方源码复现应记录并保持这一行为；若以后修正，另列实现版本。

## 3. 配对命令——收到启动通知后才运行

先激活已准备好的独立环境，再把当前目录切换至 `reproductions/OTDF`。下面两个命令的 `env/srctype/tartype` 必须完全一致：

```text
python make_otdf_cost.py --env walker2d-morph --srctype medium --tartype medium --metric cosine --seed 0

python train_otdf.py --env walker2d-morph --policy OTDF --srctype medium --tartype medium --weight --proportion 0.8 --reg_weight 0.5 --seed 1 --steps 1000000 --save-model
```

后续完整重复使用 seeds `1,2,3,4,5`，每个 seed 独立日志；同一数据组合的 OT 文件可以复用。这里没有提供自动启动循环，也没有执行这些命令。

衔接注意：

- README 第 22 行预计算示例用 `halfcheetah`，第 32 行训练示例用 `halfcheetah-morph`，两者不是配对示例；不能直接一前一后照抄。
- 预计算默认输出到当前工作目录下 `./costlogs`；训练从官方仓库绝对路径下 `costlogs` 读取。仅修改训练 `--dir` 不会改变 OT 文件路径。统一从仓库根目录执行可避免路径不一致。
- 使用连字符命名 `walker2d-morph`。预计算在规范化环境名之前就拼接输出文件名（`make_otdf_cost.py:112,117–118`）；传下划线可能导致它写出的文件名与训练读取名不一致。
- 原始 OT 分数是负匹配成本：数值越大越接近目标数据，训练使用 `topk` 保留大值。不要在外部加载时反号或重新排序。`make_otdf_cost.py:54–56`；`algo/OTDF.py:358–370`。
- 不直接执行官方 `run.sh`。它在同一 GPU 上用 `&` 同时启动多个任务和种子（第 3–13 行），与目前只准备、不运行的要求不符。

## 4. 参数与评估记录

首任务参数：每域采样 128，源域保留比例 0.8，软权重启用，数据集约束系数 0.5，CVAE 预训练 10000 步，策略更新 1000000 步。Actor/Q/V 隐藏层宽度 256，Actor/Critic 学习率 0.0003，gamma 0.99，目标网络系数 0.005，IQL expectile 0.7、inverse temperature 3.0。证据：`config/otdf/walker2d_morph.yaml:2–30`；`algo/OTDF.py:278–291`。

YAML 的 `proportion: 0.25` 是旧值；训练入口默认 CLI `0.8` 会覆盖它（`train_otdf.py:54,109–117`）。`state_dim/action_dim/max_action` 也会由实际源环境覆盖 YAML 占位值（第 99–113 行），不能据 YAML 的 `action_dim: 3` 断言 Walker2d 网络实际只输出 3 维。`--steps` 只改变训练循环，Actor 的 cosine 学习率调度仍由 YAML `max_step` 驱动（`algo/OTDF.py:275`）；短跑不能当作论文完整日程。

每 10000 个策略梯度步评估一次，源域与目标域各 10 个 episode；Actor 评估执行均值动作（`train_otdf.py:24–40,156–160`；`algo/OTDF.py:281–285`）。初始评估仅打印，不写入该标量序列。训练日志 `test/target return` 存的是原始回报，并未自动归一化。

目标域归一化使用仓库 `info.py`，不能套用普通 D4RL Walker2d 的参考上界：

| 目标环境 | 参考最小回报 | 参考最大回报 |
|---|---:|---:|
| `walker2d`（kinematic） | 10.079455055289959 | 3257.51 |
| `walker2d-morph` | 10.079455055289959 | 4398.43 |
| `walker2d-gravity` | 10.079455055289959 | 5194.713 |

`normalized_score = 100 * (raw_return - ref_min) / (ref_max - ref_min)`。证据：`info.py:10–12,25–27,33–36`。

论文说明训练 1M 梯度步、5 个随机种子，报告目标域归一化分数均值和标准差；本次未找到表格究竟使用 Final、Last-K 还是 Best checkpoint，以及标准差 ddof 的明确说明。不能把自己的汇总规则声称为作者已明确的规则。建议未来保留完整日志，并分别给出：

1. Final：每个 seed 的 step=1000000 评估均值，再跨 seeds 汇总。
2. Last-5：每个 seed 的 960000、970000、980000、990000、1000000 五次评估先平均，再跨 seeds 汇总。
3. 报告采用的标准差定义，例如样本标准差 ddof=1；需要时同时保留 ddof=0 的值供核对。

Best 不作为默认主结果，避免事后挑峰值。当前未生成任何实验得分，也未声称已经验证表 1 的效果。正式归因还需要在相同协议下复现混合数据 IQL 基线；仅运行 OTDF 无法验证提升来自 OTDF 本身。

## 5. 论文与官方源码的差异和静态风险

以下均为静态代码审查，未通过实际训练验证其性能影响。

| 项目 | 已核到的证据及含义 | 复现处理 |
|---|---|---|
| Actor 优势加权的广播 | Q/V 输出为 `(B,1)`（`algo/OTDF.py:216–227`），`adv` 因此为 `(B,1)`（第 318 行）；`bc_loss` 沿动作维求和返回 `(B,)`（第 210 行）；第 341 行相乘会广播为 `(B,B)`。这不等价于逐样本优势加权。 | 保留官方原样作为源码基线；若以后修正为逐样本乘法，单列版本与结果。 |
| 筛选影响的网络范围 | 论文 Algorithm 2 先更新 V 后筛源数据，附录 D.2 称筛选仅发生于 Q 优化；源码先筛选（第 361–370 行），再拼 batch（第 372–376 行），以该批次依次更新 V、Q、Actor（第 387、392、403 行）。指数 OT 软权重直接进入 Q 损失（第 335 行），但硬筛选影响三个网络收到的数据。 | 不应继续把官方代码解释为“筛选完全不影响 V/Actor”。源码复现与论文公式复现分开。 |
| 策略损失形式 | 论文 Algorithm 2 的 AWR 项使用动作 log-probability，源码使用均值动作的平方误差 BC（第 205–210、338–341 行）。 | 记录为实现差异，不能静默改成另一种 IQL Actor。 |
| 系数规则不一致 | 论文 D.3 给出：HalfCheetah/Walker2d 的 gravity 用 0.1，其他用 0.5；`run.sh:9,12` 却给 HalfCheetah gravity 0.5、kinematic 0.1。 | 首任务 Walker2d-morph 使用双方无冲突的 0.5；扩展任务时明确声明按论文还是按脚本。 |
| OT 分块空批次 | `make_otdf_cost.py:71–75` 用 `src_num // 10000 + 1`，源样本数恰好整除 10000 时最后会送入空批次。 | 启动前依据实际源数据处理后的条数检查；不得将未运行的条件风险描述为已发生报错。 |
| OT metric 参数未传递 | `filter_dataset(...,cost_type)` 接收参数，但第 69、75 行 JIT 调用 `solve_ot` 时只传两个数组，仍使用默认 cosine。 | 首轮使用 cosine，与论文一致；其他 metric 不能仅改 CLI 后声称已做消融。 |
| OT 小尾块量纲 | 每个分块独立求均匀边缘分布的 OT（第 44–54 行），尾块样本数可能与 10000 不同；输出没有再除以源行质量。 | 保留官方计算；未来比较替代分块或调整块大小时，应视为算法变更并重新核查分数尺度。 |
| 常数 cost 的归一化 | `algo/OTDF.py:358` 分母没有 epsilon；若批次 cost 完全相等，将产生除零。 | 保留风险记录，未来运行时检查有限值；当前不进行性能测试。 |
| 保存与重载 | `train_otdf.py:163–164` 每次覆盖同一模型名前缀；`algo/OTDF.py:428` 重载 V 时变量拼成未定义的 `file_name`；保存列表未包括 CVAE、目标 Q 或全部随机状态。 | `--save-model` 有助保留最终权重，但不是完整可恢复训练快照。未来不能承诺原代码可无缝 resume。 |

## 6. 两条复现路径必须区分

**路径 A：官方源码复现。** 锁定上述提交、官方数据和 XML，保留已发现的算法行为，只在独立环境/包装脚本解决必要的运行兼容问题，并逐项记录。先看发布代码能得到什么结果；当前只准备此路径，不启动。

**路径 B：论文公式复现。** 若后续希望修复逐样本 AWR、按 Algorithm 2 调整 V 更新批次或修改 Actor 目标，应在独立分支/目录进行，并显式标注变更。即使分数更好，也不能归为“未修改官方源码复现”。

官方 README 第 13 行列出的依赖为 PyTorch 1.8、Gym 0.23.1、JAX/JAXlib 0.4.9、ott-jax 0.4.5、jaxopt 0.8.3；这只是作者的版本陈述，不等于已经验证它们在当前 Windows/Python/CUDA 组合可安装兼容。源码还导入 D4RL、MuJoCo 相关依赖、POT (`ot`)、h5py、PyYAML、tensorboardX 等。实际安装方案与兼容修改应写入环境记录；本文件不声称依赖已全部安装或 import 测试已通过。

## 7. 可回溯的固定版本入口

- [训练入口与评估](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/train_otdf.py)
- [OT 预计算](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/make_otdf_cost.py)
- [算法实现](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/algo/OTDF.py)
- [数据与 cost 加载](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/algo/utils.py)
- [目标环境](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/envs/common.py)
- [目标域评分参考](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/info.py)
- [原始批量脚本](https://github.com/dmksjfl/OTDF/blob/41e2e626844bd6b65e2316a418ce4d6df4bbe775/run.sh)

本审查仅写入本 Markdown 文件；没有修改官方仓库，没有启动预计算、训练或评估。
