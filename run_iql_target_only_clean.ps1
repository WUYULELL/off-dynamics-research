$ErrorActionPreference = "Stop"

# Target-only baseline:
# use target offline data only, train IQL without source-domain samples.

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
$env:ODRL_TARGET_ONLY = "1"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$startTime = Get-Date
$timer = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "IQL target-only started at: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"

conda run -n offdynamics python train.py `
  --policy IQL `
  --env walker2d-kinematic-footjnt `
  --shift_level medium `
  --srctype medium `
  --tartype medium `
  --seed 0 `
  --mode 3 `
  --max_step 500000 `
  --dir runs_target_only

$exitCode = $LASTEXITCODE
$timer.Stop()
$endTime = Get-Date
$runtimeText = @"
method=IQL target-only
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
$runtimeText | Tee-Object -FilePath "$runtimeLogDir\iql_target_only_seed0_runtime.txt"
exit $exitCode
