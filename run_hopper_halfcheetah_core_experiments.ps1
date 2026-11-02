param(
    [switch]$DryRun,
    [int]$MaxStep = 500000,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

# Windows/VS Code PowerShell experiment sweep:
#   Environments: Hopper and HalfCheetah, medium dynamics shift
#   Methods: IQL-target, IQL-mix, IQL_UNC alpha=2.0, IGDF
#   Protocol: paper-like source/target, 100% source + 10% target
#   Seeds: 0,1,2,3,4

$CondaEnv = "offdynamics"
$Seeds = @(0, 1, 2, 3, 4)
$TargetFraction = "0.1"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"

$Environments = @(
    "hopper-kinematic-legjnt",
    "halfcheetah-kinematic-footjnt"
)

$Experiments = @(
    @{
        Label = "IQL-target"
        Policy = "IQL"
        OutDir = "runs_cross_env_iql_target"
        TargetOnly = $true
        UncAlpha = $null
    },
    @{
        Label = "IQL-mix"
        Policy = "IQL"
        OutDir = "runs_cross_env_iql_mix"
        TargetOnly = $false
        UncAlpha = $null
    },
    @{
        Label = "IQL_UNC-a2"
        Policy = "IQL_UNC"
        OutDir = "runs_cross_env_iql_unc_a2"
        TargetOnly = $false
        UncAlpha = "2.0"
    },
    @{
        Label = "IGDF"
        Policy = "IGDF"
        OutDir = "runs_cross_env_igdf"
        TargetOnly = $false
        UncAlpha = $null
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

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "hopper_halfcheetah_core_experiments_status.csv"
if (-not (Test-Path $statusPath)) {
    "label,policy,target_only,unc_alpha,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Conda: $condaCommand"
    Write-Host "Conda env: $CondaEnv"
    Write-Host "MuJoCo: $mujocoPath"
    Write-Host "Environments: $($Environments -join ', ')"
    Write-Host "Seeds: $($Seeds -join ', ')"
    Write-Host "Protocol: paper-like, source=100%, target_fraction=$TargetFraction"
    Write-Host "Shift/data: medium / medium / medium"
    Write-Host "Max steps: $MaxStep"
    foreach ($experiment in $Experiments) {
        $uncText = if ($experiment.UncAlpha) { $experiment.UncAlpha } else { "n/a" }
        Write-Host "  $($experiment.Label): policy=$($experiment.Policy), target_only=$($experiment.TargetOnly), unc_alpha=$uncText, outdir=$($experiment.OutDir)"
    }
    Write-Host "Total runs: $($Environments.Count * $Experiments.Count * $Seeds.Count)"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

foreach ($envName in $Environments) {
    foreach ($experiment in $Experiments) {
        $label = [string]$experiment.Label
        $policy = [string]$experiment.Policy
        $outDir = [string]$experiment.OutDir
        $targetOnly = [bool]$experiment.TargetOnly
        $uncAlpha = $experiment.UncAlpha

        foreach ($seed in $Seeds) {
            $descriptor = "$envName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel"
            $runDir = Join-Path $outDir "$policy\$descriptor\r$seed"
            $tbPath = Join-Path $runDir "tb"
            $completedPath = Join-Path $runDir ".completed"
            $safeRun = "$(ConvertTo-SafeName $label)_$(ConvertTo-SafeName $envName)_seed$seed"
            $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
            $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

            if ((-not $Overwrite) -and (Test-Path $completedPath)) {
                $now = Get-Date
                $reason = "completed_marker:$completedPath"
                Write-Host "SKIP $label $envName seed=$seed -- $reason"
                "$label,$policy,$targetOnly,$uncAlpha,$envName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                    Add-Content -Path $statusPath
                continue
            }

            if ($targetOnly) {
                $env:ODRL_TARGET_ONLY = "1"
            } else {
                Remove-Item Env:ODRL_TARGET_ONLY -ErrorAction SilentlyContinue
            }

            if ($uncAlpha) {
                $env:ODRL_UNC_ALPHA = [string]$uncAlpha
                $env:ODRL_UNC_MIN_WEIGHT = "0.05"
                $env:ODRL_SOURCE_LOSS_WEIGHT = "1.0"
            } else {
                Remove-Item Env:ODRL_UNC_ALPHA -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_UNC_MIN_WEIGHT -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_SOURCE_LOSS_WEIGHT -ErrorAction SilentlyContinue
            }

            $startTime = Get-Date
            $timer = [System.Diagnostics.Stopwatch]::StartNew()
            Write-Host ""
            Write-Host "START $label env=$envName seed=$seed"
            Write-Host "Console log: $consoleLogPath"

            $pythonArgs = @(
                "python", "train.py",
                "--policy", $policy,
                "--env", $envName,
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
                Remove-Item Env:ODRL_TARGET_ONLY -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_UNC_ALPHA -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_UNC_MIN_WEIGHT -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_SOURCE_LOSS_WEIGHT -ErrorAction SilentlyContinue
            }

            $timer.Stop()
            $endTime = Get-Date
            $elapsedHours = [Math]::Round($timer.Elapsed.TotalHours, 4)

            @"
label=$label
policy=$policy
target_only=$targetOnly
unc_alpha=$uncAlpha
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

            "$label,$policy,$targetOnly,$uncAlpha,$envName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,,$consoleLogPath,$tbPath" |
                Add-Content -Path $statusPath

            if ($exitCode -eq 0) {
                New-Item -ItemType Directory -Force -Path $runDir | Out-Null
                New-Item -ItemType File -Force -Path $completedPath | Out-Null
                Write-Host "DONE $label $envName seed=$seed elapsed_hours=$elapsedHours"
            } else {
                Write-Host "FAILED $label $envName seed=$seed. Continuing."
            }
        }
    }
}

Write-Host ""
Write-Host "All Hopper/HalfCheetah experiments finished."
Write-Host "Status CSV: $statusPath"
