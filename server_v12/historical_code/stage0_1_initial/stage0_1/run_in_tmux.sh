#!/usr/bin/env bash
set -eo pipefail
if [ -z "${TMUX:-}" ]; then
  echo 'Stop: enter tmux first: tmux new -s v12_stage0_1'
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
ODRL_V12_PROJECT="$HOME/cluster_zip_reupload_20261104/off-dynamics-transfer-20261104"
ODRL_V12_OUTPUT="$HOME/v12_linux/outputs/stage0_1_20261106"
if [ -e "$ODRL_V12_OUTPUT" ]; then
  echo "Stop: output already exists; preserve it and report: $ODRL_V12_OUTPUT"
  exit 1
fi
python -u -B "$HOME/v12_linux/stage0_1/run_env_acceptance.py" \
  --project "$ODRL_V12_PROJECT" \
  --output "$ODRL_V12_OUTPUT"
echo 'V12 0.1 stopped. No GPU learner or later stage was started.'
