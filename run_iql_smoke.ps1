$ErrorActionPreference = "Stop"

$mujocoPath = Join-Path $env:USERPROFILE ".mujoco210\mujoco210"
if (-not (Test-Path (Join-Path $mujocoPath "bin"))) {
    throw "MuJoCo bin directory not found: $mujocoPath\bin"
}

$env:MUJOCO_PY_MUJOCO_PATH = $mujocoPath
$env:Path = "$(Join-Path $mujocoPath 'bin');$env:Path"
$env:PATH = "$(Join-Path $mujocoPath 'bin');$env:PATH"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"

conda run -n offdynamics python train.py `
  --policy IQL `
  --env walker2d-kinematic-footjnt `
  --shift_level medium `
  --srctype medium `
  --tartype medium `
  --seed 1 `
  --mode 3 `
  --max_step 1000 `
  --dir runs
