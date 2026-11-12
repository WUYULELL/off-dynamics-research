#!/usr/bin/env bash
set -eo pipefail
if [ -z "${TMUX:-}" ] || [ "$#" -ne 1 ]; then
  echo 'Enter tmux, then pass the allocated GPU index or UUID.'
  exit 1
fi
code_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
test "$code_dir" = "$HOME/v12_linux/stage1_2_20261107"
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
unset ODRL_PAPER_STYLE_SOURCE_TARGET PYTHONPATH PYTHONHOME
exec python -u -B "$code_dir/controller.py" --gpu "$1"
