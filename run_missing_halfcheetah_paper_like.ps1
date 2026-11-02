$ErrorActionPreference = "Stop"

$Seeds = @(0)
$Policies = @("IGDF", "BOSA")
$MaxStep = 500000
$OutDir = "runs_paper_like_sweep"
$TargetFraction = "0.1"
$Overwrite = $true

$tasks = @(
    @{
        Env = "halfcheetah-kinematic-footjnt"
        Shift = "medium"
        SrcType = "medium"
        TarType = "medium"
    }
)

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
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "missing_halfcheetah_paper_like_status.csv"
"policy,env,shift_level,srctype,tartype,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,console_log" |
    Set-Content -Path $statusPath

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_")
}

foreach ($task in $tasks) {
    $envName = $task.Env
    $shiftLevel = $task.Shift
    $srcType = $task.SrcType
    $tarType = $task.TarType

    foreach ($policy in $Policies) {
        foreach ($seed in $Seeds) {
            $safeRun = "$(ConvertTo-SafeName $policy)_$(ConvertTo-SafeName $envName)_$shiftLevel`_$srcType`_$tarType`_seed$seed"
            $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
            $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

            $startTime = Get-Date
            $timer = [System.Diagnostics.Stopwatch]::StartNew()

            Write-Host ""
            Write-Host "START policy=$policy env=$envName shift=$shiftLevel seed=$seed"
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
                "--max_step", "$MaxStep",
                "--dir", $OutDir
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
            $elapsedHours = [Math]::Round($timer.Elapsed.TotalHours, 4)

            $runtimeText = @"
policy=$policy
env=$envName
shift_level=$shiftLevel
srctype=$srcType
tartype=$tarType
target_fraction=$TargetFraction
seed=$seed
mode=3
max_step=$MaxStep
start=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))
end=$($endTime.ToString('yyyy-MM-dd HH:mm:ss'))
elapsed=$($timer.Elapsed.ToString())
elapsed_hours=$elapsedHours
exit_code=$exitCode
console_log=$consoleLogPath
"@
            $runtimeText | Tee-Object -FilePath $runtimeLogPath

            "$policy,$envName,$shiftLevel,$srcType,$tarType,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,$consoleLogPath" |
                Add-Content -Path $statusPath
        }
    }
}

Write-Host ""
Write-Host "Missing HalfCheetah run finished. Status CSV:"
Write-Host $statusPath
