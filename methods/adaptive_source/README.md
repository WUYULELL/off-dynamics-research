# 自适应源数据批次（j013–j016）公共代码

登记见 `research/PLAN.md`（j013–j016 条目）。`src/` 是所有任务共用的代码，`build_jobs.py` 把它逐字节复制进各任务的 `code/`，再写入各自的 `config.json`、`job.json` 和 `FROZEN.json`（含 `shared_files_digest`，各任务相同）。

## src/ 的内容

| 文件 | 作用 |
|---|---|
| `otdf_official/` | OTDF 官方文件，提交 41e2e626844bd6b65e2316a418ce4d6df4bbe775 的原始字节（LF）。仓库 `reproductions/OTDF/` 里是 Windows 检出的 CRLF 版本，去掉 CR 后逐字节相同；两种哈希都写在 `asrc_common.py`，运行时校验。不修改。 |
| `d4rl/` | 实验室电脑 OTDF 复现所用的 d4rl 1.1（本仓库 `src/d4rl/d4rl/`，去掉 CR），只含 `__init__.py`、`infos.py`、`offline_env.py`；服务器 offdynamics38 环境没有安装 d4rl。 |
| `asrc_data.py` | 官方读法：源数据 hdf5 → dict → `d4rl.qlearning_dataset(None, dataset=dict)`；目标数据照抄官方 `call_tar_dataset` 循环（只改路径）。 |
| `asrc_support.py` | τ、τ_state、D_ret、D_near、随机分层。 |
| `otdf_tiered.py` | `OTDFTiered(OTDF)`：只覆盖 `train()`（加一个新方法 `configure_tiers()`）。 |
| `otdf_worker.py` | 一次 OTDF 训练（官方超参数；只评估目标环境；评估前后保存/恢复随机数状态；保存 Last5 策略）。 |
| `bc_worker.py` | j011 的 `worker.py` 只加保存模型（与 j011 的 diff 只有保存部分）。 |
| `ot_cost.py` | 用官方 `solve_ot`（AST 提取）算 OT 代价，在能导入 jax/ott-jax 的 Python 里运行。 |
| `dyn_worker.py` / `select_worker.py` / `simcheck_worker.py` | j015a 动力学集成与 H*、j015b 离线选择、j016 仿真器核对。 |
| `controller.py` + `ctl_*.py` | 各类任务的控制器（按 `config.json` 的 `kind` 分派）。 |
| `preflight_tests.py` | j013 单元测试 a–d 与层权重检查。 |

## 生成任务目录

```
python methods/adaptive_source/build_jobs.py --hopper-morph-hdf5 <官方 hopper_morph_medium.hdf5>
```

默认每个（种子、任务）一个 j014 任务；若 j013 实测单个任务超过 22 小时，用 `--j014-layout split` 重新生成（两半都含 ρ80），并同步重新生成 j015b / j016（它们的 `j014_layout` 会随之改变）。

j013 需要重跑时（例如服务器上补装了 OT 环境），用 `--only j013 --preflight-name j013b_adapt_preflight` 生成同一份代码的新任务名；下游任务按 j013 → j013b → j013c 的顺序寻找 PREFLIGHT.json 全部通过的预检。

服务器上若没有能导入 jax/ott-jax 的 Python，j013 会报告 OT 受阻。可在 offdynamics38 旁边建一个名为 `otdf_ot` 的 conda 环境（与 `reproductions/otdf_setup/requirements_ot_windows.txt` 同版本：jax/jaxlib 0.4.20、ott-jax 0.4.5、jaxopt 0.8.3、numpy 1.26.4、scipy 1.11.4、h5py 等，Python 3.9–3.12），j013 会自动找到它。

## 本地测试（CPU、合成数据、假环境；服务器上不会设置 ASRC_LOCAL_TEST）

```
python methods/adaptive_source/tests/local_e2e.py <工作目录> <Python 3.8 解释器> <带 jax/ott-jax 的解释器>
python methods/adaptive_source/tests/fake_results.py <假仓库目录> .   # 再对假结果运行 analysis_tools/j014_matrix.py
```
