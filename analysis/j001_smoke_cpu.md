# j001_smoke_cpu 分析（2026-11-18，Claude 定时会话）

## 做了什么
只用 CPU 的冒烟测试：检查 conda 环境中常用模块能否导入，并只读盘点服务器上自动任务要引用的资源。不训练。

## 结果
- 状态 `completed`，退出码 0，运行约 3 秒（23:11:30–23:11:33，北京时间）。
- Python 3.8.13（`offdynamics38`）；numpy 1.23.5、scipy 1.4.1、torch 1.11.0+cu113、gym 0.18.3、h5py 3.11.0 均可导入；CPU 矩阵乘正常。
- `mujoco_py` 导入失败：在 `$HOME/.mujoco/mujoco210` 找不到 MuJoCo。原因是本任务没有设置 MuJoCo 路径；V21 的入口脚本使用 `MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco210/mujoco210` 并把它的 `bin` 加入 `LD_LIBRARY_PATH`，还使用 mujoco_py 私有副本。`d4rl` 未安装（V21 也不依赖它，设置了 `D4RL_SUPPRESS_IMPORT_ERROR=1`）。
- 资源盘点：迁移项目 `cluster_zip_reupload_20261104/off-dynamics-transfer-20261104/`（含 MANIFEST.json、cluster/paths.example.json、source_snapshot/algo/offline_offline/ 下的 iql_dyn.py、iql_unc.py 等）、`odrl_target_20261115.zip`（3,800,530 字节）、`v13_action_fix/stage0/` 都存在，与 CONTEXT.md 第 6 节一致，无需更正。
- 其他：服务器 `nproc`=4（自动任务可见的 CPU 核数）；盘点时 4 张 A6000 利用率 96–99%，所以 j000（GPU 冒烟）一直停在 `waiting_gpu`。
- 注意：`paths.example.json` 里的 `cluster_root` 写的是 `${HOME}/off-dynamics-transfer-20261104`，实际目录在 `cluster_zip_reupload_20261104/` 之下；V21 代码用的是实际目录，任务数据路径按相对于实际目录的方式拼接。

## 按规则的结论
PLAN.md 登记的判定规则：状态 `completed`（numpy、torch、gym 都能导入），且迁移项目、ODRL 目标数据压缩包、v13 代码路径都存在，即为通过。三项都满足，**j001 通过**。

## 局限与后续
- 通过只说明链路和资源路径没问题；mujoco_py 在自动任务环境里还没有验证，训练和评估都依赖它。
- 后续：提交 j003_pipeline_smoke_cpu，按 V21 的方式设置 MuJoCo 并在 CPU 上走一遍完整训练链路，通过后再提交研究实验。
