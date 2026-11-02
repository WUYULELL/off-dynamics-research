$ErrorActionPreference = "Stop"

# Fair comparison for source uncertainty reweighting:
#   IQL mix: 100% source data + 10% target data, source:target batch = 1:1
#   IQL_UNC: same data and batch ratio, source loss reweighted by critic uncertainty

$Policies = @("IQL", "IQL_UNC")
$EnvName = "walker2d-kinematic-footjnt"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"
$Seed = 0
$MaxStep = 500000
$OutDir = "runs_10pct"
$TargetFraction = "0.1"

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
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "0"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "iql_vs_iql_unc_10pct_status.csv"
"policy,env,shift_level,srctype,tartype,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log" |
    Set-Content -Path $statusPath

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_")
}

foreach ($policy in $Policies) {
    $tbPath = Join-Path $OutDir "$policy\$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel\r$Seed\tb"
    $eventFiles = @()
    if (Test-Path $tbPath) {
        $eventFiles = @(Get-ChildItem -Path $tbPath -Filter "events.out.tfevents*" -File -ErrorAction SilentlyContinue)
    }

    $safeRun = "$(ConvertTo-SafeName $policy)_$(ConvertTo-SafeName $EnvName)_10pct_seed$Seed"
    $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
    $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

    if ($eventFiles.Count -gt 0) {
        $now = Get-Date
        $reason = "existing_tensorboard_event:$tbPath"
        Write-Host "SKIP $policy -- $reason"
        "$policy,$EnvName,$ShiftLevel,$SrcType,$TarType,$TargetFraction,$Seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath" |
            Add-Content -Path $statusPath
        continue
    }

    $startTime = Get-Date
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Host ""
    Write-Host "START policy=$policy env=$EnvName target_fraction=$TargetFraction seed=$Seed"
    Write-Host "Console log: $consoleLogPath"

    $commandArgs = @(
        "run", "-n", "offdynamics", "python", "train.py",
        "--policy", $policy,
        "--env", $EnvName,
        "--shift_level", $ShiftLevel,
        "--srctype", $SrcType,
        "--tartype", $TarType,
        "--seed", "$Seed",
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
env=$EnvName
shift_level=$ShiftLevel
srctype=$SrcType
tartype=$TarType
target_fraction=$TargetFraction
seed=$Seed
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

    "$policy,$EnvName,$ShiftLevel,$SrcType,$TarType,$TargetFraction,$Seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,,$consoleLogPath" |
        Add-Content -Path $statusPath

    if ($exitCode -ne 0) {
        Write-Host "FAILED policy=$policy. Stop before running the next policy."
        exit $exitCode
    }
}

Write-Host ""
Write-Host "Finished fair IQL vs IQL_UNC 10pct comparison runs."
Write-Host "Status CSV: $statusPath"
