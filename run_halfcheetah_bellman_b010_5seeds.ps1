param(
    [switch]$DryRun,
    [string[]]$Seeds = @("0", "1", "2", "3", "4"),
    [int]$MaxStep = 500000,
    [double]$BellmanEta = 1.0,
    [switch]$Overwrite
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
$EnvName = "halfcheetah-kinematic-footjnt"
$SourceBudget = 0.10
$BudgetName = "b0p10"
$OutDir = "r_hc_b010"

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
$env:ODRL_SOURCE_BUDGET = "$SourceBudget"
$env:ODRL_BELLMAN_ETA = "$BellmanEta"

$runtimeLogDir = "runs\runtime_logs"
New-Item -ItemType Directory -Force -Path $runtimeLogDir | Out-Null
$statusPath = Join-Path $runtimeLogDir "halfcheetah_bellman_b010_5seeds_status.csv"
if (-not (Test-Path $statusPath)) {
    "label,policy,source_budget,bellman_eta,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" |
        Set-Content -Path $statusPath
}

if ($DryRun) {
    Write-Host "Dry run OK"
    Write-Host "Conda: $condaCommand"
    Write-Host "MuJoCo: $mujocoPath"
    Write-Host "Policy: $Policy"
    Write-Host "Environment: $EnvName"
    Write-Host "Source budget: $SourceBudget"
    Write-Host "Bellman eta: $BellmanEta"
    Write-Host "Seeds: $($SeedValues -join ', ')"
    Write-Host "Target fraction: $TargetFraction"
    Write-Host "Max steps: $MaxStep"
    Write-Host "Out dir: $OutDir"
    Write-Host "Status CSV: $statusPath"
    exit 0
}

foreach ($seed in $SeedValues) {
    $descriptor = "$EnvName-srcdatatype-$SrcType-tardatatype-$TarType-$ShiftLevel"
    $runDir = Join-Path $OutDir "$Policy\$descriptor\r$seed"
    $tbPath = Join-Path $runDir "tb"
    $completedPath = Join-Path $runDir ".completed"
    $label = "IQL_BELLMAN_FB-$BudgetName"
    $safeRun = "$(ConvertTo-SafeName $label)_$(ConvertTo-SafeName $EnvName)_seed$seed"
    $consoleLogPath = Join-Path $runtimeLogDir "$safeRun.console.log"
    $runtimeLogPath = Join-Path $runtimeLogDir "$safeRun.runtime.txt"

    if ((-not $Overwrite) -and (Test-Path $completedPath)) {
        $now = Get-Date
        $reason = "completed_marker:$completedPath"
        Write-Host "SKIP $label $EnvName seed=$seed -- $reason"
        "$label,$Policy,$SourceBudget,$BellmanEta,$EnvName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($now.ToString('yyyy-MM-dd HH:mm:ss')),$($now.ToString('yyyy-MM-dd HH:mm:ss')),0,0,true,$reason,$consoleLogPath,$tbPath" |
            Add-Content -Path $statusPath
        continue
    }

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    New-Item -ItemType Directory -Force -Path $tbPath | Out-Null

    $startTime = Get-Date
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ""
    Write-Host "START $label env=$EnvName seed=$seed"
    Write-Host "Console log: $consoleLogPath"

    $pythonArgs = @(
        "python", "train.py",
        "--policy", $Policy,
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

    "$label,$Policy,$SourceBudget,$BellmanEta,$EnvName,$ShiftLevel,$TargetFraction,$seed,$MaxStep,$($startTime.ToString('yyyy-MM-dd HH:mm:ss')),$($endTime.ToString('yyyy-MM-dd HH:mm:ss')),$elapsedHours,$exitCode,false,$reason,$consoleLogPath,$tbPath" |
        Add-Content -Path $statusPath
}

Remove-Item Env:ODRL_SOURCE_BUDGET -ErrorAction SilentlyContinue
Remove-Item Env:ODRL_BELLMAN_ETA -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "HalfCheetah Bellman b0p10 5-seed run finished. Status CSV: $statusPath"
