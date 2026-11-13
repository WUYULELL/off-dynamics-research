# V12 阶段 2 ROOT 迁移试跑报告

上游 commit `0ae9c05d70ed369c16732c71abda59b54b559da9`。会话 `v12_stage2_continue`，入口 `bash /mnt/cs_dsns_public/<USER>/v12_linux/stage2_continue_20261108/run_in_tmux.sh --registered`。
输出 `/mnt/cs_dsns_public/<USER>/v12_linux/outputs/stage2_root_20261108_continue1`。独立环境 `/mnt/cs_dsns_public/<USER>/conda-envs/v12_root_20261107`，环境清单见 ENVIRONMENT_BEFORE/AFTER。
Walker2d-morph medium→medium，seed 901。原 VAE 预训练 10,000 步及 ROOT 主循环 20,000 步全部完成。
主训练 60,000 次 Adam 更新，VAE 另有 10,000 次 Adam 更新。两次目标评估共 20 回合。

`nproc --all` 为 128，`nproc` 为 128，cgroup CPU 配额为 60.00 核，结合亲和性后的有效可用 CPU 为 60.00 核。
本阶段主机整体 CPU 占用 27.82%，进程可用 CPU 集合占用 27.81%。两项为整个阶段起止的 CPU tick 差，包含同期其他工作负载。

## 数据与评估核对

|任务|HDF5 逐数组|XML 字节|原 ROOT 目标行|V12 显式 next 行|相邻行替代差异／非 terminal 差异|ROOT 归一化|V12 归一化|
|---|---|---|---:|---:|---|---|---|
|walker2d-morph|相同|相同|4999|5000|7 / 3|[10.08, 4398.43]|[10.079455055289959, 4398.43]|
|hopper-morph|相同|相同|4999|5000|7 / 0|[-26.336, 3152.75]|[-26.3360015397715, 3152.75]|

两端构造 v3 目标环境，XML timestep 0.002，TimeLimit 1,000。ROOT 上游默认每 10,000 步对源域和目标域各评估 10 回合，输出当前点目标域 10 回合平均原始回报的 0–100 归一化分数。V12 入口在目标域按固定逐回合种子评估并保存长度，评估前后恢复训练 RNG。同一批回报额外计算上游常数下的分数。
V12 使用迁移包显式 next_observations。源域保留 D4RL v2 的 canonical 行顺序，目标域保留 5,000 行。这会改变上游相邻行构造的 OT 特征和训练转移，属于 V12 本机协议适配。

## 鲁棒 OT 权重

在 gpu:0 上调用原求解器一次，处理 999,322 源行、100 批。计算及写出用时 76.00 秒。该子进程退出后 PyTorch 子进程读取同一 HDF5，保留原归一化与截断。
权重阶段采样显存峰值 573 MiB，子进程峰值 RSS 1360.0 MiB。
归一化后超过 5 并实际截断的比例 0.0226%，原始零权重比例 0.0000%。

|权重|均值|最小|P01|P25|P50|P75|P95|P99|最大|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|raw|8.7844e-05|2.11526e-06|2.60412e-05|6.8375e-05|8.60199e-05|0.000100764|0.000141108|0.000202811|0.000865815|
|normalized_float32|1|0.0240798|0.296448|0.778369|0.979235|1.14708|1.60635|2.30877|9.85629|
|clipped|0.999799|0.0240798|0.296448|0.778369|0.979235|1.14708|1.60635|2.30877|5|

## 短跑评估与耗时

|步数|V12 目标分数|ROOT 常数分数|回合长度均值|中位数|最短|<200 步比例|
|---:|---:|---:|---:|---:|---:|---:|
|10000|14.4609|14.4609|248.8|247.0|227|0.0%|
|20000|14.3542|14.3542|246.0|228.0|141|10.0%|

VAE 65.24 秒，主循环 387.09 秒，两次评估合计 3.29 秒。单次 100 万步外推 5.44 小时，含一次 VAE 和估计的 100 次目标评估。若另计算一次权重，合计 5.46 小时。后期回合可能更长。
两处评估点为迁移试跑记录。Last5 不适用，本阶段完成后进入 AWAITING_STAGE3_CONFIRMATION。

## ROOT 配置

README 的版本要求为 JAX 0.4.9、jaxlib 0.4.9、ott-jax 0.4.5、jaxopt 0.8.3。完整上游 YAML、实际 CLI 默认值、维度覆盖、正则和求解器常量均保存在 config.json。上游 YAML 的 metric 为 cosine，run_root.sh 未传 metric，train_root.py 的默认 euclidean 会覆盖它。原 learner 的截断、VAE、采样及更新顺序直接沿用固定 commit。上游 Q 权重张量为 (256, 1, 1)，与 (256, 1) 的残差平方按原 PyTorch 广播相乘，试跑保留这一实现。

```json
{
  "stage": "2",
  "task": "walker2d-morph",
  "commit": "0ae9c05d70ed369c16732c71abda59b54b559da9",
  "seed": 901,
  "original_yaml": {
    "alpha": 0.2,
    "batch_size": 128,
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "gamma": 0.99,
    "max_epochs_since_update_decay_interval": 150000.0,
    "state_dim": 17,
    "action_dim": 3,
    "hidden_sizes": 256,
    "max_action": 1,
    "temperature_opt": false,
    "tau": 0.005,
    "update_interval": 2,
    "expl_noise": 0.2,
    "eval_episode": 10,
    "eval_freq": 10000,
    "start_steps": 5000,
    "max_step": 1000000,
    "tar_env_interact_freq": 10,
    "device": "cuda",
    "save_freq": 50000,
    "lam": 0.7,
    "temp": 3.0,
    "proportion": 0.25,
    "metric": "cosine",
    "env": "Walker-morph",
    "tar_env_config": {
      "env_name": "Walker-morph"
    }
  },
  "CLI_defaults_effective_except_seed_and_short_loop": {
    "policy": "ROOT",
    "env": "walker2d-morph",
    "srctype": "medium",
    "tartype": "medium",
    "seed": 901,
    "steps": 20000,
    "weight": false,
    "noreg": false,
    "reg_weight": 0.5,
    "epsilon": 0.01,
    "metric": "euclidean",
    "lambda_src": 0.05,
    "lambda_tar": 0.5,
    "filter_threshold": 0.0
  },
  "learner": {
    "alpha": 0.2,
    "batch_size": 128,
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "gamma": 0.99,
    "max_epochs_since_update_decay_interval": 150000.0,
    "state_dim": 17,
    "action_dim": 6,
    "hidden_sizes": 256,
    "max_action": 1.0,
    "temperature_opt": false,
    "tau": 0.005,
    "update_interval": 2,
    "expl_noise": 0.2,
    "eval_episode": 10,
    "eval_freq": 10000,
    "start_steps": 5000,
    "max_step": 1000000,
    "tar_env_interact_freq": 10,
    "device": "cuda",
    "save_freq": 50000,
    "lam": 0.7,
    "temp": 3.0,
    "proportion": 0.25,
    "metric": "euclidean",
    "env": "Walker-morph",
    "tar_env_config": {
      "env_name": "Walker-morph"
    },
    "weight": false,
    "noreg": false,
    "reg_weight": 0.5,
    "filter_threshold": 0.0
  },
  "source_rows": 999322,
  "target_rows": 5000,
  "batch_sizes": {
    "source": 128,
    "target": 128
  },
  "main_updates": 20000,
  "VAE_pretraining_updates": 10000,
  "VAE_lr": 0.001,
  "VAE_hidden": 750,
  "VAE_latent_dim": 12,
  "VAE_reg_beta": 0.4,
  "VAE_reg_samples": 10,
  "solver": {
    "epsilon": 0.01,
    "lambda_src": 0.05,
    "lambda_tar": 0.5,
    "cost": "Euclidean",
    "scale_cost": "max_cost",
    "source_chunk_size": 10000,
    "source_chunks": 100,
    "Sinkhorn_threshold": 1e-05,
    "Sinkhorn_max_iterations": 1000,
    "feature_dtype": "float16",
    "features": "concatenated s, a, explicit next state standardized by combined domain moments",
    "training_source_clip": 5.0,
    "training_normalization": "raw weight / (raw mean + 1e-12)"
  },
  "evaluation": {
    "points": [
      10000,
      20000
    ],
    "episodes": 10,
    "target_only": true,
    "seed_origin": 201,
    "metric": "mean normalized target score per point",
    "Last5_applicable": false
  },
  "threads": 1,
  "gpu_uuid": "GPU-965aff83-8d85-646b-eb0e-042db35f2e88",
  "separate_process_for_JAX_and_Torch": true,
  "runtime": {
    "deterministic_algorithms": true,
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "TF32": true,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
  },
  "adaptation": "local verified HDF5 and explicit next observations, V12 evaluation and resource logging"
}
```
