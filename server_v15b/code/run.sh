#!/usr/bin/env bash
# Read-only, takes seconds; no tmux needed.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/offdynamics38"
set -u
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset V15B_TEST_SHIM
exec python -u -B "$here/diagnose.py" "$@"
