$ErrorActionPreference = "Stop"

# Naive-mix baseline:
# source offline data + target offline data, train IQL on the mixed batches.

$mujocoPath = Join-Path $env:USERPROFILE ".mujoco210\mujoco210"
if (-not (Test-Path (Join-Path $mujocoPath "bin"))) {
    throw "MuJoCo bin directory not found: $mujocoPath\bin"
}

$env:MUJOCO_PY_MUJOCO_PATH = $mujocoPath
$env:Path = "$(Join-Path $mujocoPath 'bin');$env:Path"
$env:PATH = "$(Join-Path $mujocoPath 'bin');$env:PATH"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$startTime = Get-Date
$timer = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "IQL naive-mix started at: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"

conda run -n offdynamics python train.py `
  --policy IQL `
  --env walker2d-kinematic-footjnt `
  --shift_level medium `
  --srctype medium `
  --tartype medium `
  --seed 0 `
  --mode 3 `
  --max_step 500000 `
  --dir runs

$exitCode = $LASTEXITCODE
$timer.Stop()
$endTime = Get-Date
$runtimeText = @"
method=IQL naive-mix
env=walker2d-kinematic-footjnt
shift_level=medium
srctype=medium
tartype=medium
seed=0
max_step=500000
start=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))
end=$($endTime.ToString('yyyy-MM-dd HH:mm:ss'))
elapsed=$($timer.Elapsed.ToString())
elapsed_hours=$([Math]::Round($timer.Elapsed.TotalHours, 4))
exit_code=$exitCode
"@
$runtimeText | Tee-Object -FilePath "$runtimeLogDir\iql_naive_mix_seed0_runtime.txt"
exit $exitCode
