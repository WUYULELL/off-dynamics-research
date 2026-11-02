param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Priority missing experiments for Walker2d paper-like 10% target protocol.
# Existing runs reused by the analysis script:
#   IQL-mix: runs_paper_like_iql_unc
#   IQL_UNC alpha=1.0: runs_paper_like_iql_unc
#   IQL_UNC alpha=2.0: runs_iql_unc_alpha_ablation/alpha_2p0
#   BOSA/IGDF: runs_paper_like_sweep
#
# This script runs only missing baselines:
#   IQL-target
#   TD3_BC-target
#   TD3_BC-mix

$Seeds = @(0, 1, 2, 3, 4)
$MaxStep = 500000
$TargetFraction = "0.1"
$Overwrite = $false

$EnvName = "walker2d-kinematic-footjnt"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"

$Experiments = @(
    @{
        Label = "IQL-target"
        Policy = "IQL"
        OutDir = "runs_paper_like_iql_target_only"
        TargetOnly = $true
    },
    @{
        Label = "TD3_BC-target"
        Policy = "TD3_BC"
        OutDir = "runs_paper_like_td3_bc_target_only"
        TargetOnly = $true
    },
    @{
        Label = "TD3_BC-mix"
        Policy = "TD3_BC"
        OutDir = "runs_paper_like_td3_bc_mix"
        TargetOnly = $false
    }
)

$mujocoPath = Join-Path $env:USERPROFILE ".mujoco210\mujoco210"
$mujocoBin = Join-Path $mujocoPath "bin"
if (-not (Test-Path $mujocoBin)) {
    throw "MuJoCo bin directory not found: $mujocoBin"
}

$datasetName = "$($EnvName.Replace('-', '_'))_$ShiftLevel`_$TarType.hdf5"
$datasetPath = Join-Path "dataset\mujoco" $datasetName
if (-not (Test-Path $datasetPath)) {
    throw "Required paper-like source dataset not found: $datasetPath"
}

$env:MUJOCO_PY_MUJOCO_PATH = $mujocoPath
$basePath = $env:Path
if (-not $basePath) {
    $basePath = @(
        [Environment]::GetEnvironmentVariable("Path", "Machine"),
        [Environment]::GetEnvironmentVariable("Path", "User")
    ) -join ";"
}
$env:Path = "$mujocoBin;$basePath"
$env:PATH = $env:Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"

$condaCommand = "D:\ANACONDA\Scripts\conda.exe"
if (-not (Test-Path $condaCommand)) {
    $condaCommand = $env:CONDA_EXE
}
if (-not $condaCommand -or -not (Test-Path $condaCommand)) {
    throw "Could not find conda.exe."
}

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "priority_walker2d_experiments_status.csv"
if (-not (Test-Path $statusPath)) {
    "label,policy,target_only,seed,env,target_fraction,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_")
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Experiments:"
    foreach ($experiment in $Experiments) {
        Write-Host "  $($experiment.Label): policy=$($experiment.Policy), target_only=$($experiment.TargetOnly), outdir=$($experiment.OutDir)"
    }
    Write-Host "Seeds: $($Seeds -join ', ')"
    Write-Host "Protocol: paper-like, target_fraction=$TargetFraction"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

foreach ($experiment in $Experiments) {
    $label = $experiment.Label
    $policy = $experiment.Policy
    $outDir = $experiment.OutDir
    $targetOnly = [bool]$experiment.TargetOnly

    foreach ($seed in $Seeds) {
        $tbPath = Join-Path $outDir "$policy\$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel\r$seed\tb"
        $eventFiles = @()
        if (Test-Path $tbPath) {
            $eventFiles = @(Get-ChildItem $tbPath -Filter "events.out.tfevents*" -File -ErrorAction SilentlyContinue)
        }

        $safeRun = "$(ConvertTo-SafeName $label)_paper_like_seed$seed"
        $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
        $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

        if (($eventFiles.Count -gt 0) -and (-not $Overwrite)) {
            $now = Get-Date
            $reason = "existing_tensorboard_event:$tbPath"
            Write-Host "SKIP $label seed=$seed -- $reason"
            "$label,$policy,$targetOnly,$seed,$EnvName,$TargetFraction,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                Add-Content $statusPath
            continue
        }

        if ($targetOnly) {
            $env:ODRL_TARGET_ONLY = "1"
        } else {
            Remove-Item Env:ODRL_TARGET_ONLY -ErrorAction SilentlyContinue
        }

        $startTime = Get-Date
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host ""
        Write-Host "START $label policy=$policy target_only=$targetOnly seed=$seed"
        Write-Host "Console log: $consoleLogPath"

        $pythonArgs = @(
            "python", "train.py",
            "--policy", $policy,
            "--env", $EnvName,
            "--shift_level", $ShiftLevel,
            "--srctype", $SrcType,
            "--tartype", $TarType,
            "--seed", "$seed",
            "--mode", "3",
            "--max_step", "$MaxStep",
            "--dir", $outDir
        ) -join " "
        $trainCommand = "set `"PATH=$mujocoBin;%PATH%`" && $pythonArgs"
        $commandArgs = @("run", "-n", "offdynamics", "cmd", "/d", "/c", $trainCommand)

        try {
            $ErrorActionPreference = "Continue"
            & $condaCommand @commandArgs 2>&1 |
                ForEach-Object { "$_" } |
                Tee-Object -FilePath $consoleLogPath
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = "Stop"
            Remove-Item Env:ODRL_TARGET_ONLY -ErrorAction SilentlyContinue
        }

        $timer.Stop()
        $endTime = Get-Date
        $elapsedHours = [Math]::Round($timer.Elapsed.TotalHours, 4)

        @"
label=$label
policy=$policy
target_only=$targetOnly
env=$EnvName
target_fraction=$TargetFraction
paper_like_source_target=1
seed=$seed
max_step=$MaxStep
start=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))
end=$($endTime.ToString('yyyy-MM-dd HH:mm:ss'))
elapsed=$($timer.Elapsed.ToString())
elapsed_hours=$elapsedHours
exit_code=$exitCode
console_log=$consoleLogPath
tensorboard_log=$tbPath
"@ | Set-Content $runtimeLogPath

        "$label,$policy,$targetOnly,$seed,$EnvName,$TargetFraction,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,,$consoleLogPath,$tbPath" |
            Add-Content $statusPath

        if ($exitCode -eq 0) {
            Write-Host "DONE $label seed=$seed"
        } else {
            Write-Host "FAILED $label seed=$seed. Continuing."
        }
    }
}

Write-Host ""
Write-Host "Priority experiments finished. Status CSV: $statusPath"
