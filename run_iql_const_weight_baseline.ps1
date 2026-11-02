param(
    [switch]$DryRun,
    [double]$Weight = 0.33,
    [int]$MaxStep = 500000,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

# Constant-weight control for IQL_UNC:
#   source loss uses the same scalar weight for every source sample.
#   This tests whether uncertainty reweighting is better than merely reducing source strength.

$CondaEnv = "offdynamics"
$Seeds = @(0, 1, 2, 3, 4)
$TargetFraction = "0.1"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"
$OutDir = "runs_cross_env_iql_const_w033"
$Policy = "IQL_UNC"
$Label = "IQL_CONST-w033"

$Environments = @(
    "hopper-kinematic-legjnt",
    "halfcheetah-kinematic-footjnt"
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

foreach ($envName in $Environments) {
    $datasetName = "$($envName.Replace('-', '_'))_$ShiftLevel`_$TarType.hdf5"
    $datasetPath = Join-Path "dataset\mujoco" $datasetName
    if (-not (Test-Path $datasetPath)) {
        throw "Required dataset not found: $datasetPath"
    }
    & $condaCommand run -n $CondaEnv python -c "import h5py; f=h5py.File(r'$datasetPath','r'); assert len(f.keys()) > 0; f.close()"
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset is not a valid readable HDF5 file: $datasetPath"
    }
}

$mujocoBin = Join-Path $mujocoPath "bin"
$basePath = $env:Path
if (-not $basePath) {
    $basePath = @(
        [Environment]::GetEnvironmentVariable("Path", "Machine"),
        [Environment]::GetEnvironmentVariable("Path", "User")
    ) -join ";"
}
$env:MUJOCO_PY_MUJOCO_PATH = $mujocoPath
$env:Path = "$mujocoBin;$basePath"
$env:PATH = $env:Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"
$env:ODRL_CONST_SOURCE_WEIGHT = "$Weight"
$env:ODRL_SOURCE_LOSS_WEIGHT = "1.0"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "iql_const_weight_baseline_status.csv"
if (-not (Test-Path $statusPath)) {
    "label,policy,const_source_weight,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Conda: $condaCommand"
    Write-Host "Conda env: $CondaEnv"
    Write-Host "MuJoCo: $mujocoPath"
    Write-Host "Policy: $Policy"
    Write-Host "Label: $Label"
    Write-Host "Constant source weight: $Weight"
    Write-Host "OutDir: $OutDir"
    Write-Host "Environments: $($Environments -join ', ')"
    Write-Host "Seeds: $($Seeds -join ', ')"
    Write-Host "Protocol: paper-like, source=100%, target_fraction=$TargetFraction"
    Write-Host "Total runs: $($Environments.Count * $Seeds.Count)"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

foreach ($envName in $Environments) {
    foreach ($seed in $Seeds) {
        $descriptor = "$envName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel"
        $runDir = Join-Path $OutDir "$Policy\$descriptor\r$seed"
        $tbPath = Join-Path $runDir "tb"
        $completedPath = Join-Path $runDir ".completed"
        $safeRun = "$(ConvertTo-SafeName $Label)_$(ConvertTo-SafeName $envName)_seed$seed"
        $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
        $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

        if ((-not $Overwrite) -and (Test-Path $completedPath)) {
            $now = Get-Date
            $reason = "completed_marker:$completedPath"
            Write-Host "SKIP $Label $envName seed=$seed -- $reason"
            "$Label,$Policy,$Weight,$envName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                Add-Content -Path $statusPath
            continue
        }

        $startTime = Get-Date
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        Write-Host ""
        Write-Host "START $Label env=$envName seed=$seed weight=$Weight"
        Write-Host "Console log: $consoleLogPath"

        $pythonArgs = @(
            "python", "train.py",
            "--policy", $Policy,
            "--env", $envName,
            "--shift_level", $ShiftLevel,
            "--srctype", $SrcType,
            "--tartype", $TarType,
            "--seed", "$seed",
            "--mode", "3",
            "--max_step", "$MaxStep",
            "--dir", $OutDir
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
label=$Label
policy=$Policy
const_source_weight=$Weight
env=$envName
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
            Write-Host "DONE $Label $envName seed=$seed elapsed_hours=$elapsedHours"
        } else {
            $reason = "failed"
            Write-Host "FAILED $Label $envName seed=$seed. Continuing."
        }

        "$Label,$Policy,$Weight,$envName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,$reason,$consoleLogPath,$tbPath" |
            Add-Content -Path $statusPath
    }
}

Remove-Item Env:ODRL_CONST_SOURCE_WEIGHT -ErrorAction SilentlyContinue
Remove-Item Env:ODRL_SOURCE_LOSS_WEIGHT -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Constant-weight baseline finished. Status CSV: $statusPath"
Write-Host "After it finishes, run: conda run -n offdynamics python .\analyze_cross_env_results.py"
