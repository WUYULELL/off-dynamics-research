$ErrorActionPreference = "Stop"

# Naive-mix baseline:
# source offline data + target offline data, train BOSA on the mixed batches.

$method = "BOSA naive-mix"
$policy = "BOSA"
$envName = "walker2d-kinematic-footjnt"
$shiftLevel = "medium"
$srcType = "medium"
$tarType = "medium"
$seed = 0
$maxStep = 500000
$outDir = "runs"

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
$safeMethod = "bosa_naive_mix_seed0"
$consoleLogPath = Join-Path $runtimeLogDir "$safeMethod.console.log"
$runtimeLogPath = Join-Path $runtimeLogDir "$safeMethod.runtime.txt"

$startTime = Get-Date
$timer = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "$method started at: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Console log: $consoleLogPath"

$commandArgs = @(
    "run", "-n", "offdynamics", "python", "train.py",
    "--policy", $policy,
    "--env", $envName,
    "--shift_level", $shiftLevel,
    "--srctype", $srcType,
    "--tartype", $tarType,
    "--seed", "$seed",
    "--mode", "3",
    "--max_step", "$maxStep",
    "--dir", $outDir
)

try {
    $ErrorActionPreference = "Continue"
    & conda @commandArgs 2>&1 | ForEach-Object {
        "$_"
    } | Tee-Object -FilePath $consoleLogPath
    $exitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = "Stop"
}

$timer.Stop()
$endTime = Get-Date
$runtimeText = @"
method=$method
env=$envName
shift_level=$shiftLevel
srctype=$srcType
tartype=$tarType
seed=$seed
mode=3
max_step=$maxStep
start=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))
end=$($endTime.ToString('yyyy-MM-dd HH:mm:ss'))
elapsed=$($timer.Elapsed.ToString())
elapsed_hours=$([Math]::Round($timer.Elapsed.TotalHours, 4))
exit_code=$exitCode
console_log=$consoleLogPath
"@
$runtimeText | Tee-Object -FilePath $runtimeLogPath

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "BOSA finished successfully. TensorBoard logs are under:"
    Write-Host "$outDir\$policy\$envName-srcdatatype-$srcType-tardatatype-$tarType-$shiftLevel\r$seed\tb"
} else {
    Write-Host ""
    Write-Host "BOSA failed. Check console log:"
    Write-Host $consoleLogPath
}

exit $exitCode
