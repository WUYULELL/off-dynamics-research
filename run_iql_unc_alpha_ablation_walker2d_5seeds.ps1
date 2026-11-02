param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# IQL_UNC uncertainty-strength ablation.
# alpha=1.0 is reused from runs_paper_like_iql_unc, so this script only runs
# the three missing settings.
$Alphas = @(0.1, 0.5, 2.0)
$Seeds = @(0, 1, 2, 3, 4)
$MaxStep = 500000
$RunRoot = "runs_iql_unc_alpha_ablation"
$TargetFraction = "0.1"
$Overwrite = $false

$EnvName = "walker2d-kinematic-footjnt"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"

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
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"
$env:ODRL_UNC_MIN_WEIGHT = "0.05"
$env:ODRL_SOURCE_LOSS_WEIGHT = "1.0"

$condaCommand = "D:\ANACONDA\Scripts\conda.exe"
if (-not (Test-Path $condaCommand)) {
    $condaCommand = $env:CONDA_EXE
}
if (-not $condaCommand -or -not (Test-Path $condaCommand)) {
    throw "Could not find conda.exe."
}

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "iql_unc_alpha_ablation_walker2d_5seeds_status.csv"
if (-not (Test-Path $statusPath)) {
    "unc_alpha,seed,env,target_fraction,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

function Get-AlphaName {
    param([double]$Alpha)
    return $Alpha.ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Alphas to run: $($Alphas -join ', ')"
    Write-Host "Alpha 1.0: reuse runs_paper_like_iql_unc"
    Write-Host "Seeds: $($Seeds -join ', ')"
    Write-Host "Protocol: paper-like, target_fraction=$TargetFraction"
    Write-Host "Output root: $RunRoot"
    exit 0
}

foreach ($alpha in $Alphas) {
    $alphaText = $alpha.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $alphaName = Get-AlphaName $alpha
    $outDir = "$RunRoot\alpha_$alphaName"
    $env:ODRL_UNC_ALPHA = $alphaText

    foreach ($seed in $Seeds) {
        $tbPath = Join-Path $outDir "IQL_UNC\$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel\r$seed\tb"
        $eventFiles = @()
        if (Test-Path $tbPath) {
            $eventFiles = @(Get-ChildItem $tbPath -Filter "events.out.tfevents*" -File -ErrorAction SilentlyContinue)
        }

        $runName = "iql_unc_alpha_$alphaName`_walker2d_paper_like_seed$seed"
        $consoleLogPath = Join-Path $runtimeLogDir "$runName.console.log"
        $runtimeLogPath = Join-Path $runtimeLogDir "$runName.runtime.txt"

        if (($eventFiles.Count -gt 0) -and (-not $Overwrite)) {
            $now = Get-Date
            $reason = "existing_tensorboard_event:$tbPath"
            Write-Host "SKIP alpha=$alphaText seed=$seed -- $reason"
            "$alphaText,$seed,$EnvName,$TargetFraction,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                Add-Content $statusPath
            continue
        }

        $startTime = Get-Date
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host ""
        Write-Host "START IQL_UNC alpha=$alphaText seed=$seed"
        Write-Host "Console log: $consoleLogPath"

        $pythonArgs = @(
            "python", "train.py",
            "--policy", "IQL_UNC",
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
        }

        $timer.Stop()
        $endTime = Get-Date
        $elapsedHours = [Math]::Round($timer.Elapsed.TotalHours, 4)

        @"
policy=IQL_UNC
unc_alpha=$alphaText
unc_min_weight=0.05
source_loss_weight=1.0
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

        "$alphaText,$seed,$EnvName,$TargetFraction,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,,$consoleLogPath,$tbPath" |
            Add-Content $statusPath

        if ($exitCode -eq 0) {
            Write-Host "DONE alpha=$alphaText seed=$seed"
        } else {
            Write-Host "FAILED alpha=$alphaText seed=$seed. Continuing."
        }
    }
}

Write-Host ""
Write-Host "Alpha ablation finished. Status CSV: $statusPath"
