#!/usr/bin/env bash
# j008_wsel_screen：任务层面选择源数据权重（BC-混合 w ∈ {0,0.25,0.5,1} × 3 个任务 × 种子 3116–3120，共 60 项策略训练；另训练 10 个源动力学模型、15 次模型选权重；先跑冒烟）。
# 只使用 relay 设置的 CUDA_VISIBLE_DEVICES 里的卡；只写本任务的 code/ 目录。失败时以非 0 退出码结束。
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/outputs"
: "${CONDA_PREFIX:?conda 环境没有激活}"
echo "[j008_wsel_screen] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} host=$(hostname)"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_PY_FORCE_CPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=20261120 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset J008_LOCAL_TEST
set +e
python -u -B "$here/controller.py" 2>&1 | tee "$here/outputs/stdout.txt"
rc=${PIPESTATUS[0]}
set -e
echo "[j008_wsel_screen] exit code $rc"
exit "$rc"
