#!/usr/bin/env bash
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TMUX:-}" && "${1:-}" != '--scan' ]]; then
  echo '请先执行 tmux new -A -s v13_stage1_pid，再在会话内启动。' >&2
  exit 1
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/offdynamics38"
set -u
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=-1 MUJOCO_PY_FORCE_CPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset ODRL_PAPER_STYLE_SOURCE_TARGET
exec python -u -B "$here/continuation.py" "$@"
