# V12 阶段 1A 1B 与 2 集群运行说明

用户于 2026 年 11 月 7 日确认阶段 0 通过，授权 1A、1B 和阶段 2。一个 tmux 控制器依次执行 1A、1B、2。每项审计通过后立即生成单独的报告和完成凭据，随后继续下一项已授权任务。阶段 2 结束于 `AWAITING_STAGE3_CONFIRMATION`。入口仅包含这三项任务。

## 训练范围和事先固定的解读

- 1A：Hopper 源域 D4RL medium-v2 canonical 999,998 行。以原始状态和动作为输入，匹配已验收的 V12 源数据范围和 SA-all 对照，目标域 Hopper-morph 评估。401–405，5 次各 200,000 actor 更新。Last5 组均值严格低于 20 且 Last5 所有种子、所有回合合并后长度严格小于 200 的比例超过 50%，采用 V12 的支持性解读。其他情况采用 V12 的另有原因解读。旧 SA-all 与 BC-all 只作描述。
- 1B：从 V9 证据取两任务各七条完整轨迹。对每任务独立用 `RandomState(777)` 排列编号 1–7，顺序为 `[3,1,6,7,5,2,4]`，嵌套前 2、4、7 条，排除不完整尾段。每组 401–405，合计 30 次。主指标 Last5。7−4 严格大于 2×Welch SE 判未饱和。差值小于 0 或严格小于 1×SE 判饱和。其他情况不确定。报告 4−2 和 7−4 的均值差与 Welch SE。跨硬件参考与轨迹组分变化只作描述。
- BC 沿用 V9 Actor、Sampler、Policy、评估定义。64 批大小，Adam 0.0003，β=0，动作均方误差，cosine 200,000。每 5,000 步评估 10 回合，保留逐回合原始回报、长度和种子。评估种子沿用 V9 公式 `500000 + 10*(seed−301) + episode`。Last5 为 180k、185k、190k、195k、200k。所有训练种子保留。阶段 1A 和 1B 合计 700 万次 actor 更新，1,400 次评估，14,000 个评估回合。
- 2：ROOT Walker2d-morph medium→medium，seed 901。先逐数组核对两任务目标数据与 XML。再做一次完整鲁棒 OT 权重试算和一次 20,000 步主循环。保留原 10,000 次 VAE 预训练，主循环 cosine 周期维持 1,000,000。主循环每步 3 次 Adam 更新。只在 10k、20k 评估，短跑的 Last5 不适用。

## ROOT 来源和适配

上游 `https://github.com/cristianhtvc/ROOT` 固定 commit `0ae9c05d70ed369c16732c71abda59b54b559da9`。包内保留此 commit 中使用的原文件和两份目标 HDF5，以 `ROOT_PROVENANCE.json` 记录来源和文件哈希。完整默认 YAML 和作者入口均可只读查阅。

1. ROOT 原 `algo/ROOT.py` 不改，直接调用原网络、更新、VAE 和采样。读取原 `algo/utils.py` 中的 replay buffer 与权重加载定义，避免原命令入口注册环境、下载数据和自动恢复。
2. 新入口以已验证的本地 HDF5 路由数据。源域维持 D4RL v2 canonical 行顺序并读取显式下一状态。目标域用全部 5,000 行及 `next_observations`。上游默认目标 loader 用相邻行并保留 4,999 行，两者会产生不同的转移和 OT 特征。报告逐项记录差异。
3. 原求解器的三个函数从冻结上游文件直接载入，保留 float16 特征、标准化、Euclidean cost、`max_cost`、epsilon 0.01、lambda_src 0.05、lambda_tar 0.5、原 10,000 源行分批、Sinkhorn 设置。独立 JAX GPU 进程只算一次并保存 `robust_weights.hdf5` 的 `cost`。其退出后同卡运行 PyTorch，校验 HDF5 和 canonical 源行哈希后按原全局均值归一化并在 learner 内截断到 5。分进程本身不改变求解器、权重归一化或 PyTorch 学习公式。输入与评估的本机协议变化见第 2、4 点。
4. V12 复用已验收的 v3 目标环境构造、1000 步限制、精确归一化常数、固定回合种子及训练 RNG 保存恢复。上游每 10k 评估源域和目标域各 10 回合，报告目标域当前评估点的归一化平均回报。试跑记录 V12 目标分数及相同回合按作者常数计算的分数。V12 的目标评估种子沿用 R14 公式，偏移 201。实际原配置仍采用温度 3、128+128 等，保留 YAML 默认 `proportion`，CLI 的默认 `metric=euclidean` 覆盖 YAML 的 `cosine`。
5. `$HOME/conda-envs/v12_root_20261107` 用 `conda --copy --clone` 建立副本。仅在副本安装固定版本 SciPy 1.10.1、JAX 0.4.9、jaxlib 0.4.9 CUDA 11/cuDNN 8.6、ott-jax 0.4.5、jaxopt 0.8.3 及其依赖。JAX CUDA 库只注入权重进程的路径。记录原环境和新环境在安装前后的完整 Conda/Pip 清单。需要服务器能访问 PyPI 和 JAX 官方旧版轮子仓库，并预留 12 GiB 存储空间。

## 资源和停止规则

同一个控制器锁定单一已分配的 GPU UUID。BC 首批最多两项，记录每项纯训练更新率、纯训练率之和及包含初始化和评估的整批更新率。后续按可用 CPU、内存和实测显存分批，最多 6 个训练进程。ROOT 权重和训练各单进程，且在所有 BC 结束后启动。每个 PyTorch/BLAS 进程使用 1 线程。资源采样和 CPU 审计也由同一控制器管理。每阶段保存 `nproc --all`、`nproc`、cgroup 配额和有效核数，以及基于 `/proc/stat` 起止差值的整机 CPU 占用。每批保存 GPU/CPU/RSS 采样及持续压力记录。

启动前核验阶段 0.3 完成凭据和哈希、迁移包和原 Conda/MuJoCo 保护。每批 GPU 准入沿用阶段 0 的已验收条件。错误或持续资源压力触发停止并保留新目录、日志、已经生成的权重或训练记录，控制器只清理自己启动的进程。不存在自动重跑或恢复。不要用重复执行入口覆盖现场。

## 启动与查阅结果

先把 ZIP 上传到服务器个人主目录，按聊天提供的校验和解包到 `$HOME/v12_linux/stage1_2_20261107`。进入一个 tmux 会话，确认 0 号卡仍分配给自己，执行：

```bash
bash "$HOME/v12_linux/stage1_2_20261107/run_in_tmux.sh" 0
```

各任务的输出目录和完成凭据：

|任务|`$HOME/v12_linux/outputs/` 下的目录|结果|
|---|---|---|
|1A|`stage1A_20261107`|`COMPLETED.json`、`REPORT_ZH.md`、`RESULT.json`|
|1B|`stage1B_20261107`|`COMPLETED.json`、`REPORT_ZH.md`、`RESULT.json`|
|2|`stage2_root_20261107`|`COMPLETED.json`、`REPORT_ZH.md`、`RESULT.json`|

首批实际并发的测量在相应任务目录 `FIRST_BC_CONCURRENCY.json`。若任务停止，请保留 `FAILED.json` 和 `logs/` 并回传报错。每个阶段结束的终端提示会给出相应报告路径。
