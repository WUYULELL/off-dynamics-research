# j000_smoke 分析（2026-11-19，Claude 定时会话）

## 做了什么
GPU 冒烟测试：检查 conda 环境、GPU/CUDA、MuJoCo/ODRL 相关模块能否导入。不训练。

## 结果
- 自 2026-11-18 22:10（北京时间）起一直 `waiting_gpu`（4 张卡都被占满），2026-11-19 06:11 在 3 号卡上开始运行，约 7 秒结束，退出码 0，状态 `completed`。
- 主机 ecs-instance-525d908e；Python 3.8.13；numpy 1.23.5、torch 1.11.0+cu113、gym 0.18.3。
- `cuda_available: true`，GPU 为 NVIDIA RTX A6000，GPU 矩阵乘正常（`matmul_ok: true`）。
- `mujoco_py` 导入失败（在 `$HOME/.mujoco/mujoco210` 找不到 MuJoCo）；`mujoco`、`d4rl` 未安装。

## 按规则的结论
PLAN.md 登记的规则：`completed`，且 torch 能用 CUDA、gym/mujoco 相关模块能导入，即为通过。
- torch + CUDA：满足；gym：满足；mujoco 相关模块：**不满足**。
- 因此按字面规则，**j000 未完全通过**：GPU 链路（relay 分卡、`CUDA_VISIBLE_DEVICES`、CUDA 计算）通过，MuJoCo 一项不通过。

## 解释与处理
- MuJoCo 失败的原因与 j001 相同：j000 的脚本没有设置 MuJoCo 路径。这个问题已由 j003_pipeline_smoke_cpu 解决（按 V21 的方式设置 `MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco210/mujoco210`、`LD_LIBRARY_PATH` 和 mujoco_py 私有副本，8 项检查全部通过，见 analysis/j003_pipeline_smoke_cpu.md）。
- j004 沿用 j003 的环境设置，已于 2026-11-19 08:11 在 3 号卡上开始运行。GPU 端的 MuJoCo 环境在 j004 中实际验证；不再为 j000 单独重跑。

## 局限
- 本任务只验证了 GPU 可见与可算，未在 GPU 任务里验证 MuJoCo；这要看 j004 的冒烟阶段（种子 2999）是否通过。
