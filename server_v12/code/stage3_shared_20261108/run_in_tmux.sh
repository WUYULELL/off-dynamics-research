#!/usr/bin/env bash
set -eo pipefail
if [ "$#" -eq 1 ] && [ "$1" = "--scan" ]; then
  :
elif [ -n "${TMUX:-}" ] && [ "$#" -ge 2 ] && [ "$1" = "--registered" ]; then
  :
else
  echo 'Run --scan to list cards, then register your selected card and run --registered GPU_INDEX inside tmux.'
  exit 1
fi
code_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
test "$code_dir" = "$HOME/v12_linux/stage3_shared_20261108"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/offdynamics38"
set -u
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset ODRL_PAPER_STYLE_SOURCE_TARGET PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
exec python -u -B "$code_dir/stage3_controller.py" "$@"
