#!/usr/bin/env bash
set -eo pipefail
dest="$1"
out="$2"
origin="$3"
code_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
test "$dest" = "$HOME/conda-envs/v12_root_20261107"
test "$origin" = "$HOME/conda-envs/offdynamics38"
test "$dest" != "$origin"
test ! -e "$dest"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONNOUSERSITE=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda list --prefix "$origin" --json > "$out/CONDA_ORIGINAL.json"
"$origin/bin/python" -B -m pip freeze --all > "$out/PIP_ORIGINAL.txt"
conda create --prefix "$dest" --offline --copy --clone "$origin" --yes
conda list --prefix "$dest" --json > "$out/CONDA_ENVIRONMENT_BEFORE.json"
"$dest/bin/python" -B -m pip freeze --all > "$out/PIP_ENVIRONMENT_BEFORE.txt"
"$dest/bin/python" -B -m pip install --disable-pip-version-check --no-cache-dir \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  -r "$code_dir/root_requirements.txt"
"$dest/bin/python" -B -m pip check
conda list --prefix "$dest" --json > "$out/CONDA_ENVIRONMENT_AFTER.json"
conda list --prefix "$dest" --explicit > "$out/CONDA_ENVIRONMENT_AFTER_EXPLICIT.txt"
"$dest/bin/python" -B -m pip freeze --all > "$out/PIP_ENVIRONMENT_AFTER.txt"
"$dest/bin/python" -B - "$dest" "$out" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

prefix, output = (Path(x).resolve() for x in sys.argv[1:])
assert Path(sys.prefix).resolve() == prefix
expected = {'jax': '0.4.9', 'jaxlib': '0.4.9+cuda11.cudnn86', 'ott-jax': '0.4.5', 'jaxopt': '0.8.3',
            'torch': '1.11.0+cu113', 'numpy': '1.23.5', 'scipy': '1.10.1', 'h5py': '3.11.0',
            'gym': '0.18.3', 'mujoco-py': '2.1.2.14'}
actual = {name: metadata.version(name) for name in expected}
assert actual == expected, (actual, expected)
with (output / 'ENVIRONMENT_READY.json').open('x') as file:
    json.dump(dict(passed=True, executable=sys.executable, prefix=str(prefix), versions=actual), file, indent=2)
PY
