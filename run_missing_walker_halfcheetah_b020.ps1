param(
    [switch]$DryRun,
    [string[]]$Seeds = @("0", "1", "2", "3", "4"),
    [int]$MaxStep = 500000,
    [double]$BellmanEta = 1.0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$CondaEnv = "offdynamics"
$TargetFraction = "0.1"
$ShiftLevel = "medium"
$SrcType = "medium"
$TarType = "medium"
$Policy = "IQL_BELLMAN_FB"
$SourceBudget = 0.20
$BudgetText = $SourceBudget.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
$BudgetName = "b$($BudgetText.Replace('.', 'p'))"

$Tasks = @(
    [pscustomobject]@{
        Label = "Walker2d"
        EnvName = "walker2d-kinematic-footjnt"
        OutDir = "runs_walker2d_bellman_budget_sweep\b0p20"
    },
    [pscustomobject]@{
        Label = "HalfCheetah"
        EnvName = "halfcheetah-kinematic-footjnt"
        OutDir = "runs_budget_ablation_bellman_fb\b0p20"
    }
)

function ConvertTo-SafeName {
    param([string]$Value)
    return $Value.ToLower().Replace("-", "_").Replace(".", "_").Replace(" ", "_")
}

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
if ($SeedValues.Count -eq 0) {
    throw "No seeds were provided."
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

foreach ($task in $Tasks) {
    $datasetName = "$($task.EnvName.Replace('-', '_'))_$ShiftLevel`_$TarType.hdf5"
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
$env:MUJOCO_PY_MUJOCO_PATH = $mujocoPath
$env:Path = "$mujocoBin;$env:Path"
$env:PATH = $env:Path
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"
$env:D4RL_SUPPRESS_IMPORT_ERROR = "1"
$env:ODRL_TARGET_FRACTION = $TargetFraction
$env:ODRL_PAPER_STYLE_SOURCE_TARGET = "1"
$env:ODRL_SOURCE_BUDGET = $SourceBudget.ToString([Globalization.CultureInfo]::InvariantCulture)
$env:ODRL_BELLMAN_ETA = $BellmanEta.ToString([Globalization.CultureInfo]::InvariantCulture)

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "walker_halfcheetah_b020_missing_seeds_status.csv"
$statusHeader = "label,policy,source_budget,bellman_eta,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log"
if (-not (Test-Path $statusPath)) {
    $statusHeader | Set-Content -Path $statusPath
}

$plannedRuns = @()
foreach ($task in $Tasks) {
    $descriptor = "$($task.EnvName)-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel"
    foreach ($seed in $SeedValues) {
        $runDir = Join-Path $task.OutDir "$Policy\$descriptor\r$seed"
        $completedPath = Join-Path $runDir ".completed"
        $plannedRuns += [pscustomobject]@{
            Task = $task
            Seed = $seed
            RunDir = $runDir
            CompletedPath = $completedPath
            AlreadyComplete = Test-Path $completedPath
        }
    }
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Conda: $condaCommand"
    Write-Host "Conda env: $CondaEnv"
    Write-Host "MuJoCo: $mujocoPath"
    Write-Host "Policy: $Policy"
    Write-Host "Source budget: $SourceBudget"
    Write-Host "Bellman eta: $BellmanEta"
    Write-Host "Target fraction: $TargetFraction"
    Write-Host "Seeds checked: $($SeedValues -join ', ')"
    Write-Host "Max steps: $MaxStep"
    Write-Host ""
    foreach ($run in $plannedRuns) {
        $action = if ($run.AlreadyComplete) { "SKIP" } else { "RUN" }
        Write-Host "$action $($run.Task.Label) seed=$($run.Seed) out=$($run.Task.OutDir)"
    }
    Write-Host ""
    Write-Host "Runs remaining: $(@($plannedRuns | Where-Object { -not $_.AlreadyComplete }).Count)"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

$failedRuns = 0
foreach ($run in $plannedRuns) {
    $task = $run.Task
    $seed = $run.Seed
    $runDir = $run.RunDir
    $tbPath = Join-Path $runDir "tb"
    $completedPath = $run.CompletedPath
    $label = "IQL_BELLMAN_FB-$BudgetName"
    $safeRun = "$(ConvertTo-SafeName $label)_$(ConvertTo-SafeName $task.EnvName)_seed$seed"
    $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
    $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

    if ($run.AlreadyComplete) {
        $now = Get-Date
        $reason = "completed_marker:$completedPath"
        Write-Host "SKIP $label $($task.EnvName) seed=$seed -- already completed"
        "$label,$Policy,$SourceBudget,$BellmanEta,$($task.EnvName),$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
            Add-Content -Path $statusPath
        continue
    }

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    New-Item -ItemType Directory -Force -Path $tbPath | Out-Null

    $startTime = Get-Date
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ""
    Write-Host "START $label env=$($task.EnvName) seed=$seed"
    Write-Host "Output: $runDir"
    Write-Host "Console log: $consoleLogPath"

    $pythonArgs = @(
        "python", "train.py",
        "--policy", $Policy,
        "--env", $task.EnvName,
        "--shift_level", $ShiftLevel,
        "--srctype", $SrcType,
        "--tartype", $TarType,
        "--seed", "$seed",
        "--mode", "3",
        "--max_step", "$MaxStep",
        "--dir", $task.OutDir
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
policy=$Policy
source_budget=$SourceBudget
bellman_eta=$BellmanEta
env=$($task.EnvName)
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
        Write-Host "DONE $label $($task.EnvName) seed=$seed elapsed_hours=$elapsedHours"
    } else {
        $failedRuns += 1
        $reason = "failed"
        Write-Host "FAILED $label $($task.EnvName) seed=$seed. Continuing."
    }

    "$label,$Policy,$SourceBudget,$BellmanEta,$($task.EnvName),$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,$reason,$consoleLogPath,$tbPath" |
        Add-Content -Path $statusPath
}

Remove-Item Env:ODRL_SOURCE_BUDGET -ErrorAction SilentlyContinue
Remove-Item Env:ODRL_BELLMAN_ETA -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Walker2d/HalfCheetah Bellman b0.20 completion run finished."
Write-Host "Status CSV: $statusPath"
if ($failedRuns -gt 0) {
    Write-Host "Failed runs: $failedRuns. Re-run the same command after fixing the reported error."
    exit 1
}

Write-Host "All requested runs are complete."
