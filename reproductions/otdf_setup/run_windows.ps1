[CmdletBinding()]
param(
    [ValidateSet('Status', 'Export', 'Cost', 'Smoke', 'Train')]
    [string]$Stage = 'Status',
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 1,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$')]
    [string]$TrainRunName = 'train_1M',
    [string]$DatasetDir = 'C:\Users\TXin\.d4rl\datasets',
    [string]$MujocoDir = 'C:\Users\TXin\.mujoco210\mujoco210'
)
$ErrorActionPreference = 'Stop'
if ($TrainRunName -ne 'train_1M' -and $Stage -ne 'Train') {
    throw '-TrainRunName may only select a fresh output directory for the Train stage.'
}
$setupDir = $PSScriptRoot
$reproductionsDir = Split-Path -Parent $setupDir
$repoDir = Join-Path $reproductionsDir 'OTDF'
$trainPython = Join-Path $reproductionsDir 'otdf_train\Scripts\python.exe'
$otPython = Join-Path $reproductionsDir 'otdf_ot\Scripts\python.exe'
$taskName = 'walker2d-morph-srcdatatype-medium-tardatatype-medium'
$inputsDir = Join-Path $setupDir "artifacts\$taskName"
$costPath = Join-Path $repoDir "costlogs\$taskName.hdf5"
$sourcePath = Join-Path $DatasetDir 'walker2d_medium-v2.hdf5'
$targetPath = Join-Path $repoDir 'datasets\walker2d_morph_medium.hdf5'

if ($Stage -eq 'Status') {
    Write-Output 'OTDF setup status only: no Python imports, OT, training, rollout or benchmarks are started.'
    @(
        [pscustomobject]@{ Item = 'Training Python'; Present = (Test-Path -LiteralPath $trainPython); Path = $trainPython }
        [pscustomobject]@{ Item = 'OT CPU Python'; Present = (Test-Path -LiteralPath $otPython); Path = $otPython }
        [pscustomobject]@{ Item = 'Source cache'; Present = (Test-Path -LiteralPath $sourcePath); Path = $sourcePath }
        [pscustomobject]@{ Item = 'Official target data'; Present = (Test-Path -LiteralPath $targetPath); Path = $targetPath }
        [pscustomobject]@{ Item = 'Export manifest'; Present = (Test-Path -LiteralPath (Join-Path $inputsDir 'manifest.json')); Path = $inputsDir }
        [pscustomobject]@{ Item = 'OT cost'; Present = (Test-Path -LiteralPath $costPath); Path = $costPath }
    ) | Format-Table -AutoSize | Out-String -Width 240 | Write-Output
    Write-Output 'After the user explicitly asks to start, run stages separately and in order:'
    Write-Output "& '$PSCommandPath' -Stage Export"
    Write-Output "& '$PSCommandPath' -Stage Cost"
    Write-Output "& '$PSCommandPath' -Stage Smoke -Seed 1"
    Write-Output "& '$PSCommandPath' -Stage Train -Seed 1"
    Write-Output 'Smoke is 10000 policy steps PLUS the unchanged 10000 CVAE pretraining steps and evaluation. It is not a paper result.'
    Write-Output 'Train is 1000000 policy steps, one seed. No automatic multi-seed loop or background process is used.'
    return
}

foreach ($path in @($trainPython, $sourcePath, $targetPath, (Join-Path $MujocoDir 'bin'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
if ($Stage -eq 'Cost' -and -not (Test-Path -LiteralPath $otPython)) { throw "Missing OT Python: $otPython" }
$sourceRevision = (& git -C $repoDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sourceRevision -ne '41e2e626844bd6b65e2316a418ce4d6df4bbe775') {
    throw 'Official repository revision differs from the locked source.'
}
& git -C $repoDir diff --quiet HEAD
if ($LASTEXITCODE -ne 0) { throw 'Official tracked source/data files have changed; review before launching.' }

# Process-local environment only; restored even if the stage fails. No existing
# experiment process or machine/user environment is modified.
$savedEnvironment = @{}
$changedNames = @('PATH', 'PYTHONPATH', 'PYTHONHOME', 'D4RL_DATASET_DIR', 'D4RL_SUPPRESS_IMPORT_ERROR', 'MUJOCO_PY_MUJOCO_PATH', 'JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'XLA_PYTHON_CLIENT_PREALLOCATE')
$changedNames += @(Get-ChildItem Env: | Where-Object Name -Like 'ODRL_*' | ForEach-Object Name)
foreach ($name in ($changedNames | Select-Object -Unique)) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    foreach ($name in @($changedNames | Where-Object { $_ -like 'ODRL_*' -or $_ -in @('PYTHONPATH', 'PYTHONHOME') })) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $env:D4RL_DATASET_DIR = $DatasetDir
    $env:D4RL_SUPPRESS_IMPORT_ERROR = '1'
    $env:MUJOCO_PY_MUJOCO_PATH = $MujocoDir
    $env:PATH = "$(Join-Path $MujocoDir 'bin');D:\ANACONDA\envs\offdynamics\Library\bin;D:\ANACONDA\envs\offdynamics;D:\ANACONDA\envs\offdynamics\Scripts;$env:PATH"
    Push-Location -LiteralPath $repoDir
    try {
        if ($Stage -eq 'Export') {
            & $trainPython -u (Join-Path $setupDir 'export_ot_inputs.py') --inputs $inputsDir --dataset-dir $DatasetDir --mujoco-dir $MujocoDir
            if ($LASTEXITCODE -ne 0) { throw "Export failed with exit code $LASTEXITCODE" }
        }
        elseif ($Stage -eq 'Cost') {
            & $otPython -u (Join-Path $setupDir 'compute_ot_cpu.py') --inputs $inputsDir
            if ($LASTEXITCODE -ne 0) { throw "OT cost generation failed with exit code $LASTEXITCODE" }
        }
        else {
            & $trainPython (Join-Path $setupDir 'export_ot_inputs.py') --inputs $inputsDir --dataset-dir $DatasetDir --verify-only --require-cost
            if ($LASTEXITCODE -ne 0) { throw "Input/cost integrity verification failed with exit code $LASTEXITCODE" }
            # The complete cost file is published before this stage. Finite OT
            # outputs alone do not protect the official no-epsilon minmax from
            # degenerate costs, so check fixed diagnostic batches before learning.
            $verifiedInputs = Get-Content -LiteralPath (Join-Path $inputsDir 'manifest.json') -Raw | ConvertFrom-Json
            & $trainPython -u (Join-Path $setupDir 'inspect_cost.py') --expected-count $verifiedInputs.features.source.rows
            if ($LASTEXITCODE -ne 0) { throw "OT cost numerical diagnostics failed with exit code $LASTEXITCODE" }
            $runKind = if ($Stage -eq 'Smoke') { 'smoke_NOT_PAPER_RESULT' } else { $TrainRunName }
            $steps = if ($Stage -eq 'Smoke') { 10000 } else { 1000000 }
            $logRoot = Join-Path $setupDir "runs\$runKind"
            $seedPath = Join-Path $logRoot "otdf\$taskName\r$Seed"
            $invocationPath = Join-Path $setupDir "runs\${runKind}_seed_${Seed}.invocation.json"
            if ((Test-Path -LiteralPath $seedPath) -or (Test-Path -LiteralPath ($seedPath + 'log.txt')) -or (Test-Path -LiteralPath $invocationPath)) {
                throw "Refusing to overwrite existing seed outputs: $seedPath. No resume support is claimed."
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $invocationPath) -Force | Out-Null
            $officialArgs = @('--env', 'walker2d-morph', '--policy', 'OTDF', '--srctype', 'medium', '--tartype', 'medium', '--weight', '--proportion', '0.8', '--reg_weight', '0.5', '--seed', "$Seed", '--steps', "$steps", '--save-model', '--dir', $logRoot)
            $bootstrapPath = Join-Path $setupDir 'run_official.py'
            $trainArgs = @('-u', $bootstrapPath, '--dataset-dir', $DatasetDir, '--mujoco-dir', $MujocoDir, '--') + $officialArgs
            $invocation = [ordered]@{ stage = $Stage; seed = $Seed; run_name = $runKind; initialization = 'from_scratch_no_checkpoint_loaded'; policy_steps = $steps; cvae_steps = 10000; utc_started = [DateTime]::UtcNow.ToString('o'); python = $trainPython; wrapper = $bootstrapPath; arguments = $trainArgs; official_entrypoint = (Join-Path $repoDir 'train_otdf.py'); official_arguments = $officialArgs; dataset_dir = $DatasetDir; mujoco_dir = $MujocoDir; cost_manifest = [string]([IO.Path]::ChangeExtension($costPath, '.manifest.json')); caveat = 'Windows compatibility run; single seed is not full reproduction; Smoke is not a paper result' }
            $json = $invocation | ConvertTo-Json -Depth 5
            $stream = [IO.File]::Open($invocationPath, [IO.FileMode]::CreateNew)
            try {
                $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json)
                $stream.Write($bytes, 0, $bytes.Length)
            } finally { $stream.Dispose() }
            if ($Stage -eq 'Smoke') { Write-Output 'SMOKE ONLY: 10000 policy steps plus 10000 CVAE steps and evaluation; not a paper result.' }
            & $trainPython @trainArgs
            if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE; outputs retained, no automatic retry or resume." }
        }
    }
    finally { Pop-Location }
}
finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
