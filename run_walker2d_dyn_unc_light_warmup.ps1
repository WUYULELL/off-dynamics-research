param(
    [switch]$DryRun,
    [int[]]$Seeds = @(0),
    [int]$MaxStep = 500000,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$CondaEnv = "offdynamics"
$TargetFraction = "0.1"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"
$EnvName = "walker2d-kinematic-footjnt"

$Experiments = @(
    @{
        Label = "IQL_DYN_UNC_LIGHT"
        Policy = "IQL_DYN_UNC_LIGHT"
        OutDir = "runs_iql_dyn_unc_light"
    },
    @{
        Label = "IQL_DYN_UNC_WARMUP"
        Policy = "IQL_DYN_UNC_WARMUP"
        OutDir = "runs_iql_dyn_unc_warmup"
    }
)

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_").Replace(" ", "_")
}

$mujocoCandidates = @(
    (Join-Path $env:USERPROFILE ".mujoco210\mujoco210"),
    (Join-Path $env:USERPROFILE ".mujoco\mujoco210")
)
$mujocoPath = $null
foreach ($candidate in $mujocoCandidates) {
    if (Test-Path (Join-Path $candidate "bin")) {
        $mujocoPath = $candidate
        break
    }
}
if (-not $mujocoPath) {
    throw "MuJoCo 2.1 not found. Expected: $($mujocoCandidates -join ' or ')"
}

$condaCandidates = @(
    "D:\ANACONDA\Scripts\conda.exe",
    $env:CONDA_EXE,
    (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
    (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe")
)
$condaCommand = $null
foreach ($candidate in $condaCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        $condaCommand = $candidate
        break
    }
}
if (-not $condaCommand) {
    throw "Could not find conda.exe."
}

$pythonCheck = & $condaCommand run -n $CondaEnv python -c "import sys; print(sys.version_info[:2] == (3, 8))"
if ($LASTEXITCODE -ne 0 -or ($pythonCheck -join "").Trim() -ne "True") {
    throw "Conda environment '$CondaEnv' is missing or is not Python 3.8."
}

$datasetName = "$($EnvName.Replace('-', '_'))_$ShiftLevel`_$TarType.hdf5"
$datasetPath = Join-Path "dataset\mujoco" $datasetName
if (-not (Test-Path $datasetPath)) {
    throw "Required dataset not found: $datasetPath"
}
& $condaCommand run -n $CondaEnv python -c "import h5py; f=h5py.File(r'$datasetPath','r'); assert len(f.keys()) > 0; f.close()"
if ($LASTEXITCODE -ne 0) {
    throw "Dataset is not a valid readable HDF5 file: $datasetPath"
}

$mujocoBin = Join-Path $mujocoPath "bin"
$basePath = $env:Path
$env:MUJOCO_PY_MUJOCO_PATH = $mujocoPath
$env:Path = "$mujocoBin;$basePath"
$env:PATH = $env:Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "walker2d_dyn_unc_light_warmup_status.csv"
if (-not (Test-Path $statusPath)) {
    "label,policy,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Conda: $condaCommand"
    Write-Host "Conda env: $CondaEnv"
    Write-Host "MuJoCo: $mujocoPath"
    Write-Host "Environment: $EnvName"
    Write-Host "Seeds: $($Seeds -join ', ')"
    Write-Host "Max steps: $MaxStep"
    foreach ($experiment in $Experiments) {
        Write-Host "  $($experiment.Label): policy=$($experiment.Policy), outdir=$($experiment.OutDir)"
    }
    Write-Host "Total runs: $($Experiments.Count * $Seeds.Count)"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

foreach ($experiment in $Experiments) {
    $label = [string]$experiment.Label
    $policy = [string]$experiment.Policy
    $outDir = [string]$experiment.OutDir

    foreach ($seed in $Seeds) {
        $descriptor = "$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel"
        $runDir = Join-Path $outDir "$policy\$descriptor\r$seed"
        $tbPath = Join-Path $runDir "tb"
        $completedPath = Join-Path $runDir ".completed"
        $safeRun = "$(ConvertTo-SafeName $label)_$(ConvertTo-SafeName $EnvName)_seed$seed"
        $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
        $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

        if ((-not $Overwrite) -and (Test-Path $completedPath)) {
            $now = Get-Date
            $reason = "completed_marker:$completedPath"
            Write-Host "SKIP $label $EnvName seed=$seed -- $reason"
            "$label,$policy,$EnvName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                Add-Content -Path $statusPath
            continue
        }

        $startTime = Get-Date
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host ""
        Write-Host "START $label env=$EnvName seed=$seed"
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
        $commandArgs = @("run", "-n", $CondaEnv, "cmd", "/d", "/c", $trainCommand)

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
label=$label
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
"@ | Set-Content -Path $runtimeLogPath

        if ($exitCode -eq 0) {
            New-Item -ItemType File -Force -Path $completedPath | Out-Null
            $reason = "completed"
            Write-Host "DONE $label $EnvName seed=$seed elapsed_hours=$elapsedHours"
        } else {
            $reason = "failed"
            Write-Host "FAILED $label $EnvName seed=$seed. Continuing."
        }

        "$label,$policy,$EnvName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,$reason,$consoleLogPath,$tbPath" |
            Add-Content -Path $statusPath
    }
}

Write-Host ""
Write-Host "Walker2d light/warmup experiments finished. Status CSV: $statusPath"
