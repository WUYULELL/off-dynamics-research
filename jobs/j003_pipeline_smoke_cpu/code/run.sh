#!/usr/bin/env bash
# j003：只用 CPU 的整条训练链路冒烟测试（MuJoCo 私有副本、ODRL 目标数据、源数据、三个 V21 任务的环境、少量 IQL/BC 更新、一回合评估）。
# 不写任何已有目录；只写本任务的 code/ 目录（private_runtime/、data/、outputs/）。
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/outputs"
: "${CONDA_PREFIX:?conda 环境没有激活}"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=-1 MUJOCO_PY_FORCE_CPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 D4RL_SUPPRESS_IMPORT_ERROR=1
set +e
python -u -B "$here/pipeline_smoke.py" 2>&1 | tee "$here/outputs/stdout.txt"
rc=${PIPESTATUS[0]}
set -e
echo "[j003] exit code $rc"
exit "$rc"
