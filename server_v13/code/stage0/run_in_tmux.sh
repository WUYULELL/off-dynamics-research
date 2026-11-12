#!/usr/bin/env bash
set -eo pipefail
if [[ -z "${TMUX:-}" ]]; then
  echo '先执行 tmux new -s v13_stage0，再在会话内运行本脚本。' >&2
  exit 1
fi
if [[ $# -ne 0 && ( $# -ne 1 || "$1" != '--after-reboot' ) ]]; then
  echo '本入口仅运行阶段 0。唯一可选参数为 --after-reboot。' >&2
  exit 2
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/offdynamics38"
set -u
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 MUJOCO_PY_FORCE_CPU=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset ODRL_PAPER_STYLE_SOURCE_TARGET
stage0_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python -u -B "$stage0_here/run_stage0.py" "$@"
