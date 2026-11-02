param(
    [switch]$DryRun,
    [string[]]$Seeds = @(),
    [int]$MaxStep = 0,
    [string]$OutDir = "",
    [string]$ConfigPath = "config\experiments\iql_bellman_ab_hopper_v4.json",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $RepoRoot

function ConvertTo-InvariantString {
    param([object]$Value)
    return [Convert]::ToString($Value, [Globalization.CultureInfo]::InvariantCulture)
}

function ConvertTo-BoolString {
    param([object]$Value)
    if ([bool]$Value) {
        return "1"
    }
    return "0"
}

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_").Replace(" ", "_")
}

if ([IO.Path]::IsPathRooted($ConfigPath)) {
    $configFullPath = $ConfigPath
} else {
    $configFullPath = Join-Path $RepoRoot $ConfigPath
}
if (-not (Test-Path -LiteralPath $configFullPath)) {
    throw "Experiment config not found: $configFullPath"
}

$config = Get-Content -LiteralPath $configFullPath -Raw | ConvertFrom-Json
$requiredFields = @(
    "experiment_name", "label", "conda_env", "policy", "environment",
    "shift_level", "source_type", "target_type", "mode", "target_fraction",
    "seeds", "max_step", "out_dir", "parameters"
)
foreach ($field in $requiredFields) {
    if ($config.PSObject.Properties.Name -notcontains $field) {
        throw "Missing required config field: $field"
    }
}

$CondaEnv = [string]$config.conda_env
$Policy = [string]$config.policy
$EnvName = [string]$config.environment
$ShiftLevel = [string]$config.shift_level
$SrcType = [string]$config.source_type
$TarType = [string]$config.target_type
$Mode = [int]$config.mode
$TargetFraction = ConvertTo-InvariantString $config.target_fraction
$MaxStepValue = if ($MaxStep -gt 0) { $MaxStep } else { [int]$config.max_step }
$OutDirValue = if ($OutDir.Trim().Length -gt 0) { $OutDir } else { [string]$config.out_dir }

$SeedValues = @()
$seedInputs = if ($Seeds.Count -gt 0) { $Seeds } else { @($config.seeds) }
foreach ($seedItem in $seedInputs) {
    foreach ($part in ([string]$seedItem -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $SeedValues += [int]$trimmed
        }
    }
}
$SeedValues = @($SeedValues | Select-Object -Unique)
if ($SeedValues.Count -eq 0) {
    throw "No seeds were provided by the config or command line."
}

$mujocoCandidates = @(
    (Join-Path $env:USERPROFILE ".mujoco210\mujoco210"),
    (Join-Path $env:USERPROFILE ".mujoco\mujoco210")
)
$mujocoPath = $null
foreach ($candidate in $mujocoCandidates) {
    if (Test-Path -LiteralPath (Join-Path $candidate "bin")) {
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
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
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
$datasetPath = Join-Path $RepoRoot "dataset\mujoco\$datasetName"
if (-not (Test-Path -LiteralPath $datasetPath)) {
    throw "Required dataset not found: $datasetPath"
}
& $condaCommand run -n $CondaEnv python -c "import h5py; f=h5py.File(r'$datasetPath','r'); assert len(f.keys()) > 0; f.close()"
if ($LASTEXITCODE -ne 0) {
    throw "Dataset is not a valid readable HDF5 file: $datasetPath"
}

$p = $config.parameters
$mujocoBin = Join-Path $mujocoPath "bin"
$pythonPathValue = Join-Path $RepoRoot "algo"
if ($env:PYTHONPATH) {
    $pythonPathValue = "$pythonPathValue;$env:PYTHONPATH"
}

$environmentUpdates = [ordered]@{
    "MUJOCO_PY_MUJOCO_PATH" = $mujocoPath
    "PATH" = "$mujocoBin;$env:Path"
    "CUDA_VISIBLE_DEVICES" = "0"
    "PYTHONPATH" = $pythonPathValue
    "D4RL_SUPPRESS_IMPORT_ERROR" = "1"
    "ODRL_TARGET_FRACTION" = $TargetFraction
    "ODRL_PAPER_STYLE_SOURCE_TARGET" = ConvertTo-BoolString $config.paper_style_source_target
    "ODRL_BELLMAN_ALPHA" = ConvertTo-InvariantString $p.bellman_alpha
    "ODRL_BUDGET_MODE" = [string]$p.budget_mode
    "ODRL_FIXED_SOURCE_BUDGET" = ConvertTo-InvariantString $p.fixed_source_budget
    "ODRL_BUDGET_MIN" = ConvertTo-InvariantString $p.budget_min
    "ODRL_BUDGET_MAX" = ConvertTo-InvariantString $p.budget_max
    "ODRL_BUDGET_RATIO_CENTER" = ConvertTo-InvariantString $p.budget_ratio_center
    "ODRL_BUDGET_TEMP" = ConvertTo-InvariantString $p.budget_temp
    "ODRL_BUDGET_EMA_DECAY" = ConvertTo-InvariantString $p.budget_ema_decay
    "ODRL_USE_BUDGET_EMA" = ConvertTo-BoolString $p.use_budget_ema
    "ODRL_ADAPTIVE_BUDGET_UPDATE_INTERVAL" = ConvertTo-InvariantString $p.adaptive_budget_update_interval
    "ODRL_MIN_WEIGHT" = ConvertTo-InvariantString $p.min_weight
    "ODRL_MAX_WEIGHT" = ConvertTo-InvariantString $p.max_weight
    "ODRL_SOURCE_LOSS_WEIGHT" = ConvertTo-InvariantString $p.source_loss_weight
    "ODRL_KNN_K" = ConvertTo-InvariantString $p.knn_k
    "ODRL_TARGET_BANK_SIZE" = ConvertTo-InvariantString $p.target_bank_size
    "ODRL_KNN_CHUNK_SIZE" = ConvertTo-InvariantString $p.knn_chunk_size
    "ODRL_NORMALIZE_BELLMAN" = ConvertTo-BoolString $p.normalize_bellman_mismatch
    "ODRL_EXCLUDE_SELF_IN_TARGET_KNN" = ConvertTo-BoolString $p.exclude_self_in_target_knn
}
if ($p.PSObject.Properties.Name -contains "budget_anchor") {
    $environmentUpdates["ODRL_BUDGET_ANCHOR"] = ConvertTo-InvariantString $p.budget_anchor
}
if ($p.PSObject.Properties.Name -contains "budget_response_scale") {
    $environmentUpdates["ODRL_BUDGET_RESPONSE_SCALE"] = ConvertTo-InvariantString $p.budget_response_scale
}
if ($p.PSObject.Properties.Name -contains "diagnostic_log_interval") {
    $environmentUpdates["ODRL_DIAGNOSTIC_LOG_INTERVAL"] = ConvertTo-InvariantString $p.diagnostic_log_interval
}

$originalEnvironment = @{}
foreach ($name in $environmentUpdates.Keys) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, [string]$environmentUpdates[$name], "Process")
}

$runtimeLogDir = Join-Path $RepoRoot "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusName = "$(ConvertTo-SafeName $config.experiment_name)_status.csv"
$statusPath = Join-Path $runtimeLogDir $statusName
if (-not (Test-Path -LiteralPath $statusPath)) {
    "label,policy,budget_ratio_center,budget_temp,budget_min,budget_max,env,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -LiteralPath $statusPath
}

try {
    if ($DryRun) {
        Write-Host "Dry run OK"
        Write-Host "Config: $configFullPath"
        Write-Host "Conda: $condaCommand"
        Write-Host "Conda env: $CondaEnv"
        Write-Host "MuJoCo: $mujocoPath"
        Write-Host "Dataset: $datasetPath"
        Write-Host "Policy: $Policy"
        Write-Host "Environment: $EnvName"
        Write-Host "Seeds: $($SeedValues -join ', ')"
        Write-Host "Max steps: $MaxStepValue"
        Write-Host "Target fraction: $TargetFraction"
        Write-Host "Budget range: $($p.budget_min) - $($p.budget_max)"
        Write-Host "Budget ratio center: $($p.budget_ratio_center)"
        Write-Host "Budget temperature: $($p.budget_temp)"
        if ($p.PSObject.Properties.Name -contains "budget_anchor") {
            Write-Host "Budget anchor: $($p.budget_anchor)"
        }
        if ($p.PSObject.Properties.Name -contains "budget_response_scale") {
            Write-Host "Budget response scale: $($p.budget_response_scale)"
        }
        if ($p.PSObject.Properties.Name -contains "diagnostic_log_interval") {
            Write-Host "Diagnostic interval: $($p.diagnostic_log_interval)"
        }
        Write-Host "Out dir: $OutDirValue"
        Write-Host "Status CSV: $statusPath"
        exit 0
    }

    foreach ($seed in $SeedValues) {
        $descriptor = "$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel"
        $runDir = Join-Path $RepoRoot "$OutDirValue\$Policy\$descriptor\r$seed"
        $tbPath = Join-Path $runDir "tb"
        $completedPath = Join-Path $runDir ".completed"
        $safeRun = "$(ConvertTo-SafeName $config.experiment_name)_seed$seed"
        $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
        $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

        if ((-not $Overwrite) -and (Test-Path -LiteralPath $completedPath)) {
            $now = Get-Date
            $reason = "completed_marker:$completedPath"
            Write-Host "SKIP $($config.label) seed=$seed -- $reason"
            "$($config.label),$Policy,$($p.budget_ratio_center),$($p.budget_temp),$($p.budget_min),$($p.budget_max),$EnvName,$TargetFraction,$seed,$MaxStepValue,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
                Add-Content -LiteralPath $statusPath
            continue
        }

        $startTime = Get-Date
        $timer = [Diagnostics.Stopwatch]::StartNew()
        Write-Host ""
        Write-Host "START $($config.label) seed=$seed"
        Write-Host "center=$($p.budget_ratio_center) temp=$($p.budget_temp) budget=[$($p.budget_min), $($p.budget_max)]"
        Write-Host "Console log: $consoleLogPath"

        $pythonArgs = @(
            "python", "train.py",
            "--policy", $Policy,
            "--env", $EnvName,
            "--shift_level", $ShiftLevel,
            "--srctype", $SrcType,
            "--tartype", $TarType,
            "--seed", "$seed",
            "--mode", "$Mode",
            "--max_step", "$MaxStepValue",
            "--dir", $OutDirValue
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
label=$($config.label)
policy=$Policy
env=$EnvName
shift_level=$ShiftLevel
srctype=$SrcType
tartype=$TarType
target_fraction=$TargetFraction
paper_like_source_target=$($config.paper_style_source_target)
seed=$seed
mode=$Mode
max_step=$MaxStepValue
budget_mode=$($p.budget_mode)
budget_min=$($p.budget_min)
budget_max=$($p.budget_max)
budget_ratio_center=$($p.budget_ratio_center)
budget_temp=$($p.budget_temp)
budget_anchor=$($p.budget_anchor)
budget_response_scale=$($p.budget_response_scale)
diagnostic_log_interval=$($p.diagnostic_log_interval)
budget_ema_decay=$($p.budget_ema_decay)
bellman_alpha=$($p.bellman_alpha)
start=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))
end=$($endTime.ToString('yyyy-MM-dd HH:mm:ss'))
elapsed=$($timer.Elapsed.ToString())
elapsed_hours=$elapsedHours
exit_code=$exitCode
config=$configFullPath
console_log=$consoleLogPath
tensorboard_log=$tbPath
"@ | Set-Content -LiteralPath $runtimeLogPath

        if ($exitCode -eq 0) {
            New-Item -ItemType File -Force -Path $completedPath | Out-Null
            $reason = "completed"
            Write-Host "DONE $($config.label) seed=$seed elapsed_hours=$elapsedHours"
        } else {
            $reason = "failed"
            Write-Host "FAILED $($config.label) seed=$seed exit_code=$exitCode"
        }

        "$($config.label),$Policy,$($p.budget_ratio_center),$($p.budget_temp),$($p.budget_min),$($p.budget_max),$EnvName,$TargetFraction,$seed,$MaxStepValue,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,$reason,$consoleLogPath,$tbPath" |
            Add-Content -LiteralPath $statusPath
    }
} finally {
    foreach ($name in $environmentUpdates.Keys) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], "Process")
    }
}

Write-Host ""
Write-Host "Hopper Bellman adaptive-budget experiment finished. Status CSV: $statusPath"
