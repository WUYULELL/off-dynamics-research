#!/usr/bin/env bash
set -eo pipefail
if [ -z "${TMUX:-}" ]; then
  echo 'Stop: enter tmux first: tmux new -s v12_stage0_2'
  exit 1
fi
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash run_in_tmux.sh ALLOCATED_GPU_INDEX_OR_UUID'
  echo 'Passing the device means you have confirmed its allocation under cluster rules.'
  exit 1
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/offdynamics38"
set -u
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export MUJOCO_PY_FORCE_CPU=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset ODRL_PAPER_STYLE_SOURCE_TARGET
ODRL_V12_GPU="$1"
ODRL_V12_PROJECT="$HOME/cluster_zip_reupload_20261104/off-dynamics-transfer-20261104"
ODRL_V12_PRIOR="$HOME/v12_linux/outputs/stage0_1_20261106"
ODRL_V12_OUTPUT="$HOME/v12_linux/outputs/stage0_2_20261106"
if [ -e "$ODRL_V12_OUTPUT" ]; then
  echo "Stop: output already exists; preserve it and report: $ODRL_V12_OUTPUT"
  exit 1
fi
python -u -B "$HOME/v12_linux/stage0_2/pipeline.py" \
  --project "$ODRL_V12_PROJECT" \
  --prior "$ODRL_V12_PRIOR" \
  --output "$ODRL_V12_OUTPUT" \
  --gpu "$ODRL_V12_GPU" \
  --allocation-confirmed
echo 'V12 0.2 stopped. No throughput or later experiment was started.'
