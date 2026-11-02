param(
    [switch]$DryRun,
    [string[]]$Seeds = @("0"),
    [int]$MaxStep = 500000,
    [double]$BellmanEta = 1.0,
    [double]$ConstWeight = 0.33,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$CondaEnv = "offdynamics"
$TargetFraction = "0.05"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"

$Environments = @(
    "hopper-kinematic-legjnt",
    "halfcheetah-kinematic-footjnt"
)

$Experiments = @(
    @{
        Label = "IQL-target-5pct"
        Policy = "IQL"
        OutDir = "runs_5pct_iql_target"
        TargetOnly = $true
        ConstWeight = $null
        Bellman = $false
    },
    @{
        Label = "IQL-mix-5pct"
        Policy = "IQL"
        OutDir = "runs_5pct_iql_mix"
        TargetOnly = $false
        ConstWeight = $null
        Bellman = $false
    },
    @{
        Label = "IQL_CONST-w033-5pct"
        Policy = "IQL_UNC"
        OutDir = "runs_5pct_iql_const_w033"
        TargetOnly = $false
        ConstWeight = "$ConstWeight"
        Bellman = $false
    },
    @{
        Label = "IQL_BELLMAN_FB-5pct"
        Policy = "IQL_BELLMAN_FB"
        OutDir = "runs_5pct_iql_bellman_fb"
        TargetOnly = $false
        ConstWeight = $null
        Bellman = $true
    }
)

$SeedValues = @()
foreach ($seedItem in $Seeds) {
    foreach ($part in ([string]$seedItem -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $SeedValues += [int]$trimmed
        }
    }
}
$SeedValues = @($SeedValues | Select-Object -Unique)

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
$statusPath = Join-Path $runtimeLogDir "hopper_halfcheetah_5pct_core_status.csv"
if (-not (Test-Path $statusPath)) {
    "label,policy,target_only,const_weight,bellman_eta,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Conda: $condaCommand"
    Write-Host "Conda env: $CondaEnv"
    Write-Host "MuJoCo: $mujocoPath"
    Write-Host "Environments: $($Environments -join ', ')"
    Write-Host "Seeds: $($SeedValues -join ', ')"
    Write-Host "Target fraction: $TargetFraction"
    Write-Host "Max steps: $MaxStep"
    foreach ($experiment in $Experiments) {
        Write-Host "  $($experiment.Label): policy=$($experiment.Policy), outdir=$($experiment.OutDir)"
    }
    Write-Host "Total runs: $($Environments.Count * $Experiments.Count * $SeedValues.Count)"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

foreach ($envName in $Environments) {
    foreach ($experiment in $Experiments) {
        $label = [string]$experiment.Label
        $policy = [string]$experiment.Policy
        $outDir = [string]$experiment.OutDir
        $targetOnly = [bool]$experiment.TargetOnly
        $constWeightValue = $experiment.ConstWeight
        $isBellman = [bool]$experiment.Bellman

        foreach ($seed in $SeedValues) {
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
                "$label,$policy,$targetOnly,$constWeightValue,$BellmanEta,$envName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                    Add-Content -Path $statusPath
                continue
            }

            if ($targetOnly) {
                $env:ODRL_TARGET_ONLY = "1"
            } else {
                Remove-Item Env:ODRL_TARGET_ONLY -ErrorAction SilentlyContinue
            }
            if ($constWeightValue) {
                $env:ODRL_CONST_SOURCE_WEIGHT = "$constWeightValue"
                $env:ODRL_SOURCE_LOSS_WEIGHT = "1.0"
            } else {
                Remove-Item Env:ODRL_CONST_SOURCE_WEIGHT -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_SOURCE_LOSS_WEIGHT -ErrorAction SilentlyContinue
            }
            if ($isBellman) {
                $env:ODRL_BELLMAN_ETA = "$BellmanEta"
            } else {
                Remove-Item Env:ODRL_BELLMAN_ETA -ErrorAction SilentlyContinue
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
                Remove-Item Env:ODRL_CONST_SOURCE_WEIGHT -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_SOURCE_LOSS_WEIGHT -ErrorAction SilentlyContinue
                Remove-Item Env:ODRL_BELLMAN_ETA -ErrorAction SilentlyContinue
            }

            $timer.Stop()
            $endTime = Get-Date
            $elapsedHours = [Math]::Round($timer.Elapsed.TotalHours, 4)

            @"
label=$label
policy=$policy
target_only=$targetOnly
const_weight=$constWeightValue
bellman_eta=$BellmanEta
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
                Write-Host "DONE $label $envName seed=$seed elapsed_hours=$elapsedHours"
            } else {
                $reason = "failed"
                Write-Host "FAILED $label $envName seed=$seed. Continuing."
            }

            "$label,$policy,$targetOnly,$constWeightValue,$BellmanEta,$envName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,$reason,$consoleLogPath,$tbPath" |
                Add-Content -Path $statusPath
        }
    }
}

Write-Host ""
Write-Host "5 percent target-ratio core experiments finished. Status CSV: $statusPath"
