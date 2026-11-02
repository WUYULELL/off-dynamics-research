[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 5,
    [string]$DatasetDir = 'C:\Users\TXin\.d4rl\datasets',
    [string]$MujocoDir = 'C:\Users\TXin\.mujoco210\mujoco210'
)
$ErrorActionPreference = 'Stop'
$setupDir = $PSScriptRoot
$runner = Join-Path $setupDir 'run_windows.ps1'
$inspector = Join-Path $setupDir 'inspect_run.py'
$aggregator = Join-Path $setupDir 'aggregate_seeds.py'
$studyPath = Join-Path $setupDir 'study_five_seeds.json'
$verifier = Join-Path $setupDir 'export_ot_inputs.py'
$trainPython = Join-Path (Split-Path -Parent $setupDir) 'otdf_train\Scripts\python.exe'
$pwshPath = (Get-Command pwsh.exe -ErrorAction Stop).Source
$taskName = 'walker2d-morph-srcdatatype-medium-tardatatype-medium'
$officialCommit = '41e2e626844bd6b65e2316a418ce4d6df4bbe775'
$formalPipelineDir = Join-Path $setupDir 'runs\pipeline_5seeds_20260921'
$pipelineDir = $(if ($ValidateOnly) { Join-Path $setupDir 'runs\validation_5seeds_20260921' } else { $formalPipelineDir })
$trainRunName = 'train_5seeds'

foreach ($path in @($runner, $inspector, $aggregator, $studyPath, $verifier, $trainPython)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing prerequisite: $path" }
}
if (Test-Path -LiteralPath $pipelineDir) { throw "Refusing to overwrite or resume: $pipelineDir" }
$study = Get-Content -LiteralPath $studyPath -Raw | ConvertFrom-Json
$studyHash = (Get-FileHash -LiteralPath $studyPath -Algorithm SHA256).Hash
if ($study.schema_version -ne 1 -or $study.official_commit -ne $officialCommit -or $study.task -notin @('walker2d-morph', $taskName)) { throw 'Unexpected study schema, source revision, or task.' }
if ($study.train_run_name -ne $trainRunName -or -not $study.pipeline_dir -or [IO.Path]::GetFullPath($study.pipeline_dir) -ne [IO.Path]::GetFullPath($formalPipelineDir)) { throw 'Study training or pipeline output directory differs from the fixed launch contract.' }
if (($study.expected_seeds -join ',') -ne '1,2,3,4,5' -or $study.policy_steps -ne 1000000 -or $study.cvae_steps -ne 10000 -or $study.eval_frequency -ne 10000 -or $study.eval_episodes -ne 10) { throw 'Study seeds or training/evaluation budgets differ from the agreed protocol.' }
$runs = @{}
foreach ($run in $study.seeds) {
    $seedNumber = [int]$run.seed
    if ($seedNumber -notin 1..5 -or $runs.ContainsKey($seedNumber)) { throw 'Unexpected or duplicate seed in study.' }
    $runs[$seedNumber] = $run
}
if ($runs.Count -ne 5) { throw 'Study must contain seeds 1,2,3,4,5.' }

# Future paths are fixed by run_windows.ps1. Check them before reserving the new
# pipeline directory. Existing interrupted/final training outputs are never reused.
foreach ($seedNumber in 2..5) {
    $run = $runs[$seedNumber]
    $expectedRun = Join-Path $setupDir "runs\$trainRunName\otdf\$taskName\r$seedNumber"
    $expectedInvocation = Join-Path $setupDir "runs\${trainRunName}_seed_${seedNumber}.invocation.json"
    $expectedInspection = Join-Path $formalPipelineDir "Seed$seedNumber.inspection.json"
    $expectedStatus = Join-Path $formalPipelineDir 'status.json'
    foreach ($pair in @(@($run.run_dir, $expectedRun), @($run.invocation_path, $expectedInvocation), @($run.inspection_path, $expectedInspection), @($run.completion.status_path, $expectedStatus))) {
        if (-not $pair[0] -or [IO.Path]::GetFullPath($pair[0]) -ne [IO.Path]::GetFullPath($pair[1])) { throw "Study path mismatch for seed $seedNumber" }
    }
    if ($run.completion.train_stage -ne "TrainSeed$seedNumber" -or ($run.completion.validation_path -join '/') -ne "seed_validations/$seedNumber") { throw "Study completion contract mismatch for seed $seedNumber" }
    foreach ($streamName in @('stdout', 'stderr')) {
        $expectedLog = Join-Path $formalPipelineDir "TrainSeed$seedNumber.$streamName.log"
        $configuredLog = $run."${streamName}_path"
        if ($configuredLog -and [IO.Path]::GetFullPath($configuredLog) -ne [IO.Path]::GetFullPath($expectedLog)) { throw "Study $streamName path mismatch for seed $seedNumber" }
    }
    foreach ($existingPath in @($expectedRun, ($expectedRun + 'log.txt'), $expectedInvocation)) {
        if (Test-Path -LiteralPath $existingPath) { throw "Refusing to overwrite existing seed $seedNumber output: $existingPath" }
    }
}
foreach ($reference in @($study.inputs_manifest, $study.cost_manifest)) {
    if (-not (Test-Path -LiteralPath $reference.path -PathType Leaf) -or (Get-FileHash -LiteralPath $reference.path -Algorithm SHA256).Hash -ne $reference.sha256) { throw 'Study input/cost manifest is missing or its SHA256 changed.' }
}

New-Item -ItemType Directory -Path $pipelineDir -ErrorAction Stop | Out-Null
$statusPath = Join-Path $pipelineDir 'status.json'
$stageRecords = [System.Collections.Generic.List[object]]::new()
$state = [ordered]@{
    schema_version = 1; study_path = $studyPath; study_sha256 = $studyHash
    mode = $(if ($ValidateOnly) { 'validation_only' } else { 'serial_seeds_2_to_5' })
    status = 'running'; stage = 'Preparing'; current_seed = 1
    completed_seeds = @(); pending_seeds = @(2, 3, 4, 5)
    pipeline_process_id = $PID; child_process_id = $null
    pipeline_started_utc = [DateTime]::UtcNow.ToString('o')
    stage_started_utc = $null; updated_utc = $null; completed_utc = $null
    last_exit_code = $null; error = $null
    pipeline_dir = $pipelineDir; train_run_name = $trainRunName
    policy_steps_per_seed = 1000000; cvae_steps_per_seed = 10000
    seed_validations = [ordered]@{}; stages = $stageRecords
}

function Write-Status {
    $state.updated_utc = [DateTime]::UtcNow.ToString('o')
    $temporary = Join-Path $pipelineDir 'status.json.new'
    [IO.File]::WriteAllText($temporary, ($state | ConvertTo-Json -Depth 14), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Quote-NativeArgument([string]$Value) {
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-Stage {
    param([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$StdoutName = '')
    $stdoutPath = Join-Path $pipelineDir $(if ($StdoutName) { $StdoutName } else { "$Name.stdout.log" })
    $stderrPath = Join-Path $pipelineDir "$Name.stderr.log"
    $record = [ordered]@{
        stage = $Name; seed = $state.current_seed; executable = $Executable; arguments = $Arguments
        started_utc = [DateTime]::UtcNow.ToString('o'); ended_utc = $null
        child_process_id = $null; exit_code = $null; stdout = $stdoutPath; stderr = $stderrPath
    }
    $stageRecords.Add($record)
    $state.stage = $Name; $state.stage_started_utc = $record.started_utc; $state.child_process_id = $null
    Write-Status
    $quotedArguments = @($Arguments | ForEach-Object { Quote-NativeArgument $_ })
    $child = Start-Process -FilePath $Executable -ArgumentList $quotedArguments -WorkingDirectory $setupDir -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $null = $child.Handle
    $record.child_process_id = $child.Id; $state.child_process_id = $child.Id
    Write-Status
    while (-not $child.WaitForExit($PollSeconds * 1000)) { Write-Status }
    $child.WaitForExit()
    $record.exit_code = [int]$child.ExitCode; $record.ended_utc = [DateTime]::UtcNow.ToString('o')
    $state.last_exit_code = $record.exit_code; $state.child_process_id = $null
    Write-Status
    $child.Dispose()
    if ($record.exit_code -ne 0) { throw "Stage $Name failed with actual exit code $($record.exit_code); no retry. See $stderrPath" }
    return $record
}

function Inspect-Evidence([string]$ReportPath, [string]$RunDirectory) {
    $inspection = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    $problems = [System.Collections.Generic.List[string]]::new()
    if ($inspection.observed_process_exit_code -ne 0 -or $inspection.errors.Count -ne 0) { $problems.Add('Inspection errors or unsuccessful training process') }
    if ($inspection.finite_check.all_recorded_values_finite -ne $true) { $problems.Add('Recorded values are absent or non-finite') }
    if ($inspection.last_policy_step -ne 1000000 -or $inspection.last_logged_policy_step -lt 1000000) { $problems.Add('One million completed/logged policy steps are required') }
    foreach ($domain in @('source', 'target')) {
        $returnRecord = $inspection."latest_${domain}_raw_return"
        if ($null -eq $returnRecord -or $returnRecord.finite -ne $true) { $problems.Add("Missing/non-finite $domain return") }
    }
    foreach ($tag in @('train/adv', 'train/value', 'train/q1', 'test/source_return', 'test/target_return')) {
        $metric = $inspection.scalar_tags.$tag
        if ($null -eq $metric -or $metric.count -le 0 -or $metric.all_finite -ne $true) { $problems.Add("Missing/non-finite scalar: $tag") }
    }
    if ($inspection.scalar_tags.'test/target_return'.latest.step -ne 1000000) { $problems.Add('Final target evaluation at one million steps is absent') }
    $checkpoints = [System.Collections.Generic.List[object]]::new()
    foreach ($suffix in @('critic', 'critic_optimizer', 'value', 'value_optimizer', 'actor', 'actor_optimizer', 'actor_lr_scheduler')) {
        $checkpoint = Join-Path $RunDirectory "models\model_$suffix"
        $exists = Test-Path -LiteralPath $checkpoint -PathType Leaf
        $bytes = $(if ($exists) { (Get-Item -LiteralPath $checkpoint).Length } else { 0 })
        $checkpoints.Add([ordered]@{ path = $checkpoint; exists = $exists; bytes = $bytes })
        if (-not $exists -or $bytes -le 0) { $problems.Add("Missing/empty checkpoint: $checkpoint") }
    }
    return [ordered]@{ passed = ($problems.Count -eq 0); checked_utc = [DateTime]::UtcNow.ToString('o'); expected_policy_steps = 1000000; report = $ReportPath; errors = @($problems.ToArray()); checkpoints = @($checkpoints.ToArray()); checkpoint_check_scope = 'File presence and nonzero length only; no model deserialization' }
}

try {
    Write-Status
    # Only standard-library file hashing: no OT, environment, rollout or training.
    $verifyArgs = @('-u', '-B', $verifier, '--inputs', (Split-Path -Parent $study.inputs_manifest.path), '--dataset-dir', $DatasetDir, '--verify-only', '--require-cost')
    $verification = Invoke-Stage -Name 'VerifyDataHashes' -Executable $trainPython -Arguments $verifyArgs
    $firstRun = $runs[1]
    $oldStatus = Get-Content -LiteralPath $firstRun.completion.status_path -Raw | ConvertFrom-Json
    $oldTraining = @($oldStatus.stages | Where-Object stage -EQ $firstRun.completion.train_stage)
    $oldValidation = $oldStatus
    foreach ($part in $firstRun.completion.validation_path) { $oldValidation = $oldValidation.$part }
    if ($oldStatus.status -ne 'completed' -or $oldTraining.Count -ne 1 -or $oldTraining[0].exit_code -ne 0 -or -not $oldTraining[0].ended_utc -or $oldValidation.passed -ne $true) { throw 'Seed 1 completion evidence is not successful and complete.' }
    $state.seed_validations['1'] = Inspect-Evidence -ReportPath $firstRun.inspection_path -RunDirectory $firstRun.run_dir
    if (-not $state.seed_validations['1'].passed) { throw ('Seed 1 evidence failed: ' + ($state.seed_validations['1'].errors -join '; ')) }
    $state.completed_seeds = @(1)
    Write-Status
    $partial = Invoke-Stage -Name 'AggregateSeed1' -Executable $trainPython -Arguments @('-u', '-B', $aggregator, '--study', $studyPath, '--output', $pipelineDir, '--validate-seed', '1')
    if ($ValidateOnly) {
        $state.status = 'validation_only_passed'; $state.stage = 'ValidationComplete'; $state.current_seed = $null
        $state.completed_utc = [DateTime]::UtcNow.ToString('o'); Write-Status
        exit 0
    }
    foreach ($seedNumber in 2..5) {
        if ((Get-FileHash -LiteralPath $studyPath -Algorithm SHA256).Hash -ne $studyHash) { throw 'Study contract changed after launch; stopping before the next seed.' }
        $state.current_seed = $seedNumber
        $run = $runs[$seedNumber]
        $trainArgs = @('-NoProfile', '-NonInteractive', '-File', $runner, '-Stage', 'Train', '-Seed', [string]$seedNumber, '-TrainRunName', $trainRunName, '-DatasetDir', $DatasetDir, '-MujocoDir', $MujocoDir)
        $training = Invoke-Stage -Name "TrainSeed$seedNumber" -Executable $pwshPath -Arguments $trainArgs
        $inspectArgs = @('-u', '-B', $inspector, '--run-dir', $run.run_dir, '--console-log', $training.stdout, '--console-log', $training.stderr, '--invocation', $run.invocation_path, '--exit-code', [string]$training.exit_code)
        $inspection = Invoke-Stage -Name "InspectSeed$seedNumber" -Executable $trainPython -Arguments $inspectArgs -StdoutName "Seed$seedNumber.inspection.json"
        $state.seed_validations[[string]$seedNumber] = Inspect-Evidence -ReportPath $inspection.stdout -RunDirectory $run.run_dir
        Write-Status  # Aggregator reads this completion evidence before continuing.
        if (-not $state.seed_validations[[string]$seedNumber].passed) { throw ("Seed $seedNumber validation failed: " + ($state.seed_validations[[string]$seedNumber].errors -join '; ')) }
        $aggregate = Invoke-Stage -Name "AggregateSeed$seedNumber" -Executable $trainPython -Arguments @('-u', '-B', $aggregator, '--study', $studyPath, '--output', $pipelineDir, '--validate-seed', [string]$seedNumber)
        $state.completed_seeds = @($state.completed_seeds) + @($seedNumber)
        $state.pending_seeds = @(2..5 | Where-Object { $_ -notin $state.completed_seeds })
        Write-Status
    }
    $state.current_seed = $null
    $finalAggregate = Invoke-Stage -Name 'AggregateFinal' -Executable $trainPython -Arguments @('-u', '-B', $aggregator, '--study', $studyPath, '--output', $pipelineDir)
    $summary = Get-Content -LiteralPath (Join-Path $pipelineDir 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.status -ne 'complete' -or ($state.completed_seeds -join ',') -ne '1,2,3,4,5') { throw 'Final aggregate does not confirm a complete five-seed study.' }
    $state.status = 'completed'; $state.stage = 'Completed'; $state.completed_utc = [DateTime]::UtcNow.ToString('o')
    Write-Status
    exit 0
}
catch {
    $state.status = 'failed'; $state.error = $_.Exception.Message; $state.completed_utc = [DateTime]::UtcNow.ToString('o')
    Write-Status
    [Console]::Error.WriteLine($state.error)
    exit 1
}
