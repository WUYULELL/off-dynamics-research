param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Paper-like protocol aligned with the existing BOSA/IGDF walker2d sweep:
#   source = ODRL dynamics-shift data
#   target/eval = D4RL normal walker2d data
#   target fraction = 10%
#   seeds = 0,1,2,3,4

$Seeds = @(0, 1, 2, 3, 4)
$Policies = @("IQL", "IQL_UNC")
$MaxStep = 500000
$OutDir = "runs_paper_like_iql_unc"
$TargetFraction = "0.1"
$Overwrite = $false

$EnvName = "walker2d-kinematic-footjnt"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"

$mujocoPath = Join-Path $env:USERPROFILE ".mujoco210\mujoco210"
if (-not (Test-Path (Join-Path $mujocoPath "bin"))) {
    throw "MuJoCo bin directory not found: $mujocoPath\bin"
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
$env:Path = "$(Join-Path $mujocoPath 'bin');$basePath"
$env:PATH = $env:Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"

$condaCommand = $env:CONDA_EXE
if ($condaCommand -and $condaCommand.ToLower().EndsWith(".bat")) {
    $condaExeCandidate = "D:\ANACONDA\Scripts\conda.exe"
    if (Test-Path $condaExeCandidate) {
        $condaCommand = $condaExeCandidate
    }
}
if (-not $condaCommand -or -not (Test-Path $condaCommand)) {
    $condaCandidates = @(
        "D:\ANACONDA\Scripts\conda.exe",
        "D:\ANACONDA\condabin\conda.bat",
        "$env:USERPROFILE\anaconda3\condabin\conda.bat",
        "$env:USERPROFILE\miniconda3\condabin\conda.bat"
    )
    foreach ($candidate in $condaCandidates) {
        if (Test-Path $candidate) {
            $condaCommand = $candidate
            break
        }
    }
}
if (-not $condaCommand -or -not (Test-Path $condaCommand)) {
    throw "Could not find conda. Please run this script from an Anaconda Prompt or update condaCandidates in the script."
}

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null

$statusPath = Join-Path $runtimeLogDir "iql_iql_unc_paper_like_walker2d_5seeds_status.csv"
if (-not (Test-Path $statusPath)) {
    "policy,env,shift_level,srctype,tartype,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log" |
        Set-Content -Path $statusPath
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Policies: $($Policies -join ', ')"
    Write-Host "Seeds: $($Seeds -join ', ')"
    Write-Host "Env: $EnvName"
    Write-Host "OutDir: $OutDir"
    Write-Host "TargetFraction: $TargetFraction"
    Write-Host "PaperLike: $env:ODRL_PAPER_STYLE_SOURCE_TARGET"
    Write-Host "Conda: $condaCommand"
    Write-Host "Dataset: $datasetPath"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_")
}

function Get-RunTbPath {
    param(
        [string]$Policy,
        [int]$Seed
    )
    return Join-Path $OutDir "$Policy\$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel\r$Seed\tb"
}

foreach ($policy in $Policies) {
    foreach ($seed in $Seeds) {
        $safeRun = "$(ConvertTo-SafeName $policy)_$(ConvertTo-SafeName $EnvName)_paper_like_10pct_seed$seed"
        $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
        $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"
        $tbPath = Get-RunTbPath $policy $seed

        $eventFiles = @()
        if (Test-Path $tbPath) {
            $eventFiles = @(Get-ChildItem -Path $tbPath -Filter "events.out.tfevents*" -File -ErrorAction SilentlyContinue)
        }

        if (($eventFiles.Count -gt 0) -and (-not $Overwrite)) {
            $now = Get-Date
            $reason = "existing_tensorboard_event:$tbPath"
            Write-Host "SKIP $policy seed=$seed -- $reason"
            "$policy,$EnvName,$ShiftLevel,$SrcType,$TarType,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath" |
                Add-Content -Path $statusPath
            continue
        }

        $startTime = Get-Date
        $timer = [System.Diagnostics.Stopwatch]::StartNew()

        Write-Host ""
        Write-Host "START policy=$policy env=$EnvName paper_like=1 target_fraction=$TargetFraction seed=$seed"
        Write-Host "Console log: $consoleLogPath"

        $mujocoBin = Join-Path $mujocoPath "bin"
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
            "--dir", $OutDir
        ) -join " "
        $trainCommand = "set `"PATH=$mujocoBin;%PATH%`" && $pythonArgs"

        $commandArgs = @(
            "run", "-n", "offdynamics", "cmd", "/d", "/c", $trainCommand
        )

        try {
            $ErrorActionPreference = "Continue"
            & $condaCommand @commandArgs 2>&1 | ForEach-Object {
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
paper_like_source_target=1
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

        "$policy,$EnvName,$ShiftLevel,$SrcType,$TarType,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,,$consoleLogPath" |
            Add-Content -Path $statusPath

        if ($exitCode -ne 0) {
            Write-Host "FAILED policy=$policy seed=$seed. Continuing to next run."
        } else {
            Write-Host "DONE policy=$policy seed=$seed"
        }
    }
}

Write-Host ""
Write-Host "Finished paper-like IQL/IQL_UNC walker2d 5-seed runs."
Write-Host "Status CSV: $statusPath"
