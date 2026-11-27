#!/usr/bin/env bash
# Adaptive-source batch (j013-j016): single entry point, executed by the relay in ~/auto_jobs/<job>/code/.
# Uses only the GPUs in CUDA_VISIBLE_DEVICES set by the relay; writes only inside this job's code/ folder.
# Exits with a non-zero code on failure. No host / user names are printed.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/outputs"
: "${CONDA_PREFIX:?conda environment not activated}"
echo "[asrc] job=${AUTO_JOB_ID:-local} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco210/mujoco210"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_PY_FORCE_CPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR="$here/private_runtime/d4rl_datasets"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=20261127 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset ASRC_LOCAL_TEST
# scrub the home directory, user name and host name from everything this job prints (outputs/stdout.txt is retrieved)
user="${USER:-$(id -un 2>/dev/null || true)}"
host="$(hostname 2>/dev/null || true)"
filt=(-e "s#${HOME}#~#g")
[ "${#user}" -ge 3 ] && filt+=(-e "s#${user}#<USER>#g")
[ "${#host}" -ge 3 ] && filt+=(-e "s#${host}#<HOST>#g")
set +e
python -u -B "$here/controller.py" 2>&1 | sed -u "${filt[@]}" | tee "$here/outputs/stdout.txt"
rc=${PIPESTATUS[0]}
set -e
echo "[asrc] exit code $rc"
exit "$rc"
