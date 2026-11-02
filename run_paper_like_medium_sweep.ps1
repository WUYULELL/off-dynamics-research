$ErrorActionPreference = "Stop"

$Seeds = @(0,1,2,3,4)
$Policies = @("IGDF", "BOSA")
$MaxStep = 500000
$OutDir = "runs_paper_like_sweep"
$TargetFraction = "0.1"
$Overwrite = $false

# Paper-like cross-domain offline RL sweep:
# 10% target-domain D4RL data + source-domain ODRL dynamics-shift data.
# Default is seed 0 only. To run paper-style five seeds, change:
#   $Seeds = @(0)
# to:
#   $Seeds = @(0, 1, 2, 3, 4)

$tasks = @(
    @{
        Env = "halfcheetah-kinematic-footjnt"
        Shift = "medium"
        SrcType = "medium"
        TarType = "medium"
    },
    @{
        Env = "hopper-kinematic-legjnt"
        Shift = "medium"
        SrcType = "medium"
        TarType = "medium"
    },
    @{
        Env = "walker2d-kinematic-footjnt"
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

$statusPath = Join-Path $runtimeLogDir "paper_like_medium_sweep_status.csv"
if (-not (Test-Path $statusPath)) {
    "policy,env,shift_level,srctype,tartype,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log" |
        Set-Content -Path $statusPath
}

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_")
}

function Get-RunTbPath {
    param(
        [string]$Policy,
        [string]$EnvName,
        [string]$SrcType,
        [string]$TarType,
        [string]$Shift,
        [int]$Seed
    )
    return Join-Path $OutDir "$Policy\$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$Shift\r$Seed\tb"
}

foreach ($task in $tasks) {
    $envName = $task.Env
    $shiftLevel = $task.Shift
    $srcType = $task.SrcType
    $tarType = $task.TarType

    $datasetName = "$($envName.Replace('-', '_'))_$shiftLevel`_$tarType.hdf5"
    $datasetPath = Join-Path "dataset\mujoco" $datasetName

    foreach ($policy in $Policies) {
        foreach ($seed in $Seeds) {
            $safeRun = "$(ConvertTo-SafeName $policy)_$(ConvertTo-SafeName $envName)_$shiftLevel`_$srcType`_$tarType`_seed$seed"
            $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
            $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"
            $tbPath = Get-RunTbPath $policy $envName $srcType $tarType $shiftLevel $seed

            if (-not (Test-Path $datasetPath)) {
                $now = Get-Date
                $reason = "missing_source_dataset:$datasetPath"
                Write-Host "SKIP $policy $envName seed=$seed -- $reason"
                "$policy,$envName,$shiftLevel,$srcType,$tarType,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,-1,true,$reason,$consoleLogPath" |
                    Add-Content -Path $statusPath
                continue
            }

            $eventFiles = @()
            if (Test-Path $tbPath) {
                $eventFiles = @(Get-ChildItem -Path $tbPath -Filter "events.out.tfevents*" -File -ErrorAction SilentlyContinue)
            }

            if (($eventFiles.Count -gt 0) -and (-not $Overwrite)) {
                $now = Get-Date
                $reason = "existing_tensorboard_event:$tbPath"
                Write-Host "SKIP $policy $envName seed=$seed -- $reason"
                "$policy,$envName,$shiftLevel,$srcType,$tarType,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath" |
                    Add-Content -Path $statusPath
                continue
            }

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
tensorboard_log=$tbPath
"@
            $runtimeText | Tee-Object -FilePath $runtimeLogPath

            "$policy,$envName,$shiftLevel,$srcType,$tarType,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,,$consoleLogPath" |
                Add-Content -Path $statusPath

            if ($exitCode -ne 0) {
                Write-Host "FAILED policy=$policy env=$envName seed=$seed. Continuing to next run."
            } else {
                Write-Host "DONE policy=$policy env=$envName seed=$seed"
            }
        }
    }
}

Write-Host ""
Write-Host "Sweep finished. Status CSV:"
Write-Host $statusPath
