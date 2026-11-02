[CmdletBinding()]
param(
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 5,
    [string]$DatasetDir = 'C:\Users\TXin\.d4rl\datasets',
    [string]$MujocoDir = 'C:\Users\TXin\.mujoco210\mujoco210',
    [switch]$ContinueAfterSmoke,
    [switch]$ValidateSmokeOnly,
    [ValidatePattern('^[A-Za-z0-9_-]{1,87}$')]
    [string]$AttemptName = ''
)
$ErrorActionPreference = 'Stop'
if ($ValidateSmokeOnly -and -not $ContinueAfterSmoke) {
    throw '-ValidateSmokeOnly is allowed only with -ContinueAfterSmoke.'
}
if ($AttemptName -and (-not $ContinueAfterSmoke -or $ValidateSmokeOnly)) {
    throw '-AttemptName is allowed only with -ContinueAfterSmoke and without -ValidateSmokeOnly.'
}
$setupDir = $PSScriptRoot
$runner = Join-Path $setupDir 'run_windows.ps1'
$inspector = Join-Path $setupDir 'inspect_run.py'
$trainPython = Join-Path (Split-Path -Parent $setupDir) 'otdf_train\Scripts\python.exe'
$pwshPath = (Get-Command pwsh.exe -ErrorAction Stop).Source
$taskName = 'walker2d-morph-srcdatatype-medium-tardatatype-medium'
$inputManifest = Join-Path $setupDir "artifacts\$taskName\manifest.json"
$previousPipelineDir = Join-Path $setupDir 'runs\pipeline_20260919_seed1'
$pipelineDir = $previousPipelineDir
if ($ContinueAfterSmoke) {
    $pipelineDir = Join-Path $setupDir $(if ($ValidateSmokeOnly) { 'runs\pipeline_20260921_validation' } else { 'runs\pipeline_20260921_seed1' })
}
$trainRunName = 'train_1M'
$interruptedStatusPath = $null
if ($AttemptName) {
    $pipelineDir = Join-Path $setupDir "runs\pipeline_$AttemptName"
    $trainRunName = "train_1M_$AttemptName"
    $interruptedStatusPath = Join-Path $setupDir 'runs\pipeline_20260921_seed1\status.json'
}
$smokeDir = Join-Path $setupDir "runs\smoke_NOT_PAPER_RESULT\otdf\$taskName\r1"
$smokeInvocation = Join-Path $setupDir 'runs\smoke_NOT_PAPER_RESULT_seed_1.invocation.json'
$trainDir = Join-Path $setupDir "runs\$trainRunName\otdf\$taskName\r1"
$trainInvocation = Join-Path $setupDir "runs\${trainRunName}_seed_1.invocation.json"
$continuationRecord = $null
$previousSmokeRecord = $null

if ($ContinueAfterSmoke) {
    $previousStatusPath = Join-Path $previousPipelineDir 'status.json'
    if (-not (Test-Path -LiteralPath $previousStatusPath -PathType Leaf)) {
        throw "Previous pipeline status is missing: $previousStatusPath"
    }
    $previousStatus = Get-Content -LiteralPath $previousStatusPath -Raw | ConvertFrom-Json
    if ($previousStatus.status -ne 'failed' -or $previousStatus.stage -ne 'InspectSmoke' -or $previousStatus.seed -ne 1) {
        throw 'Continuation requires the seed-1 original pipeline to have failed at InspectSmoke.'
    }
    $previousCostRecords = @($previousStatus.stages | Where-Object stage -EQ 'Cost')
    $previousSmokeRecords = @($previousStatus.stages | Where-Object stage -EQ 'Smoke')
    if ($previousCostRecords.Count -ne 1 -or $previousSmokeRecords.Count -ne 1) {
        throw 'Expected exactly one previous Cost record and one previous Smoke record.'
    }
    if ($previousCostRecords[0].exit_code -ne 0 -or $previousSmokeRecords[0].exit_code -ne 0 -or -not $previousCostRecords[0].ended_utc -or -not $previousSmokeRecords[0].ended_utc) {
        throw 'Previous Cost and Smoke must both have completed with an actual exit code of zero.'
    }
    if (@($previousStatus.stages | Where-Object stage -In @('Train', 'InspectTrain')).Count -ne 0) {
        throw 'This continuation is allowed only before the original pipeline has attempted Train.'
    }
    $previousSmokeRecord = $previousSmokeRecords[0]
    foreach ($streamName in @('stdout', 'stderr')) {
        $expectedLogPath = Join-Path $previousPipelineDir "Smoke.$streamName.log"
        if ([IO.Path]::GetFullPath($previousSmokeRecord.$streamName) -ne [IO.Path]::GetFullPath($expectedLogPath) -or -not (Test-Path -LiteralPath $expectedLogPath -PathType Leaf)) {
            throw "Previous Smoke $streamName log is missing or points outside the expected run."
        }
    }
    if (-not (Test-Path -LiteralPath $smokeInvocation -PathType Leaf) -or -not (Test-Path -LiteralPath $smokeDir -PathType Container)) {
        throw 'Original Smoke invocation or model/log directory is missing.'
    }
    $priorSmokeInvocation = Get-Content -LiteralPath $smokeInvocation -Raw | ConvertFrom-Json
    if ($priorSmokeInvocation.stage -ne 'Smoke' -or $priorSmokeInvocation.seed -ne 1 -or $priorSmokeInvocation.policy_steps -ne 10000) {
        throw 'Original Smoke invocation must be seed 1 with a 10000-step policy budget.'
    }
    $continuationRecord = [ordered]@{
        previous_status_path = $previousStatusPath
        previous_status_sha256 = (Get-FileHash -LiteralPath $previousStatusPath -Algorithm SHA256).Hash
        previous_status = $previousStatus.status
        previous_failed_stage = $previousStatus.stage
        reused_cost = $previousCostRecords[0]
        reused_smoke = $previousSmokeRecord
        smoke_stdout_sha256 = (Get-FileHash -LiteralPath $previousSmokeRecord.stdout -Algorithm SHA256).Hash
        smoke_stderr_sha256 = (Get-FileHash -LiteralPath $previousSmokeRecord.stderr -Algorithm SHA256).Hash
        continuation_scope = 'Reuse completed Cost and Smoke artifacts; rerun inspection; start Train from scratch, not from Smoke weights'
    }
}

# Export is intentionally separate: the parent task completes it before this
# pipeline starts. Do not reserve an output directory until prerequisites exist.
foreach ($requiredPath in @($runner, $inspector, $trainPython, $inputManifest)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Pipeline prerequisite missing: $requiredPath"
    }
}
if (Test-Path -LiteralPath $pipelineDir) {
    throw "Refusing to overwrite or resume an existing pipeline directory: $pipelineDir"
}
if (-not $ValidateSmokeOnly -and ((Test-Path -LiteralPath $trainDir) -or (Test-Path -LiteralPath ($trainDir + 'log.txt')) -or (Test-Path -LiteralPath $trainInvocation))) {
    throw "Refusing to overwrite existing training output or invocation: $trainDir"
}
New-Item -ItemType Directory -Path $pipelineDir -ErrorAction Stop | Out-Null
$statusPath = Join-Path $pipelineDir 'status.json'
$stageRecords = [System.Collections.Generic.List[object]]::new()
$state = [ordered]@{
    schema_version = 1
    pipeline = 'OTDF Windows Cost -> Smoke(10000) -> InspectSmoke -> Train(1000000) -> InspectTrain'
    mode = $(if ($ValidateSmokeOnly) { 'continue_after_smoke_validation_only' } elseif ($ContinueAfterSmoke) { 'continue_after_smoke' } else { 'full_pipeline' })
    continuation = $continuationRecord
    attempt = $AttemptName
    train_run_name = $trainRunName
    restart_from_scratch = [bool]$AttemptName
    interrupted_status_path = $interruptedStatusPath
    seed = 1
    status = 'running'
    stage = 'Preparing'
    pipeline_process_id = $PID
    child_process_id = $null
    child_process_kind = $null
    pipeline_started_utc = [DateTime]::UtcNow.ToString('o')
    stage_started_utc = $null
    updated_utc = $null
    completed_utc = $null
    last_exit_code = $null
    error = $null
    smoke_validation = $null
    train_validation = $null
    pipeline_dir = $pipelineDir
    stages = $stageRecords
}

function Write-PipelineStatus {
    $state.updated_utc = [DateTime]::UtcNow.ToString('o')
    $temporaryStatus = Join-Path $pipelineDir 'status.json.new'
    [IO.File]::WriteAllText($temporaryStatus, ($state | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporaryStatus -Destination $statusPath -Force
}

function ConvertTo-WindowsArgument([string]$Value) {
    # Start-Process joins ArgumentList into a native command line on Windows.
    # Double backslashes before quotes/end according to CommandLineToArgvW rules.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-PipelineProcess {
    param([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$StdoutName = '')
    $stdoutPath = Join-Path $pipelineDir $(if ($StdoutName) { $StdoutName } else { "$Name.stdout.log" })
    $stderrPath = Join-Path $pipelineDir "$Name.stderr.log"
    $record = [ordered]@{
        stage = $Name; executable = $Executable; arguments = $Arguments
        started_utc = [DateTime]::UtcNow.ToString('o'); ended_utc = $null
        child_process_id = $null; exit_code = $null
        stdout = $stdoutPath; stderr = $stderrPath
    }
    $stageRecords.Add($record)
    $state.stage = $Name
    $state.stage_started_utc = $record.started_utc
    $state.child_process_id = $null
    $state.child_process_kind = $(if ($Executable -eq $pwshPath) { 'pwsh stage launcher' } else { 'Python log inspector' })
    Write-PipelineStatus
    $quotedArguments = @($Arguments | ForEach-Object { ConvertTo-WindowsArgument $_ })
    $child = Start-Process -FilePath $Executable -ArgumentList $quotedArguments -WorkingDirectory $setupDir -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    # Retain the process handle while polling so the original ExitCode is
    # available after the child exits, including very short-lived failures.
    $null = $child.Handle
    $record.child_process_id = $child.Id
    $state.child_process_id = $child.Id
    Write-PipelineStatus
    while (-not $child.WaitForExit($PollSeconds * 1000)) {
        Write-PipelineStatus
    }
    $child.WaitForExit()
    $record.exit_code = [int]$child.ExitCode
    $record.ended_utc = [DateTime]::UtcNow.ToString('o')
    $state.last_exit_code = $record.exit_code
    $state.child_process_id = $null
    Write-PipelineStatus
    $child.Dispose()
    if ($record.exit_code -ne 0) {
        throw "Stage $Name failed with actual child-process exit code $($record.exit_code); no retry. See $stderrPath"
    }
    return $record
}

function Invoke-RunnerStage([string]$Name) {
    $arguments = @('-NoProfile', '-NonInteractive', '-File', $runner, '-Stage', $Name, '-Seed', '1', '-DatasetDir', $DatasetDir, '-MujocoDir', $MujocoDir)
    if ($Name -eq 'Train') { $arguments += @('-TrainRunName', $trainRunName) }
    return Invoke-PipelineProcess -Name $Name -Executable $pwshPath -Arguments $arguments
}

function Get-InspectionValidation {
    param([string]$ReportPath, [string]$RunDirectory, [int]$ExpectedSteps)
    $inspection = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    $validationErrors = [System.Collections.Generic.List[string]]::new()
    if ($inspection.observed_process_exit_code -ne 0) { $validationErrors.Add('Training process did not exit successfully') }
    if ($inspection.errors.Count -ne 0) { $validationErrors.Add('Inspector reported errors') }
    if ($inspection.finite_check.all_recorded_values_finite -ne $true) { $validationErrors.Add('Recorded values are absent or non-finite') }
    if ($inspection.last_policy_step -ne $ExpectedSteps -or $inspection.last_logged_policy_step -lt $ExpectedSteps) { $validationErrors.Add("Expected $ExpectedSteps policy steps and a logged step >= $ExpectedSteps") }
    foreach ($domain in @('source', 'target')) {
        $returnRecord = $inspection."latest_${domain}_raw_return"
        if ($null -eq $returnRecord -or $returnRecord.finite -ne $true) { $validationErrors.Add("Missing/non-finite $domain return") }
    }
    foreach ($tag in @('train/adv', 'train/value', 'train/q1', 'test/source_return', 'test/target_return')) {
        $metric = $inspection.scalar_tags.$tag
        if ($null -eq $metric -or $metric.count -le 0 -or $metric.all_finite -ne $true) { $validationErrors.Add("Missing/non-finite required scalar: $tag") }
    }
    if ($inspection.scalar_tags.'test/target_return'.latest.step -lt $ExpectedSteps) { $validationErrors.Add("Target evaluation at step $ExpectedSteps is absent") }
    $checkpointRecords = [System.Collections.Generic.List[object]]::new()
    foreach ($suffix in @('critic', 'critic_optimizer', 'value', 'value_optimizer', 'actor', 'actor_optimizer', 'actor_lr_scheduler')) {
        $checkpointPath = Join-Path $RunDirectory "models\model_$suffix"
        $checkpointExists = Test-Path -LiteralPath $checkpointPath -PathType Leaf
        $checkpointBytes = $(if ($checkpointExists) { (Get-Item -LiteralPath $checkpointPath).Length } else { 0 })
        $checkpointRecords.Add([ordered]@{ path = $checkpointPath; exists = $checkpointExists; bytes = $checkpointBytes })
        if (-not $checkpointExists -or $checkpointBytes -le 0) { $validationErrors.Add("Missing/empty checkpoint: $checkpointPath") }
    }
    return [ordered]@{
        passed = ($validationErrors.Count -eq 0)
        checked_utc = [DateTime]::UtcNow.ToString('o')
        expected_policy_steps = $ExpectedSteps
        report = $ReportPath
        errors = @($validationErrors.ToArray())
        checkpoints = @($checkpointRecords.ToArray())
        checkpoint_check_scope = 'File presence and nonzero length only; model parameter finiteness is not deserialized here'
    }
}

try {
    Write-PipelineStatus
    if ($ContinueAfterSmoke) {
        $smokeRecord = $previousSmokeRecord
    }
    else {
        $costRecord = Invoke-RunnerStage 'Cost'
        # run_windows verifies input/code/cost hashes before it starts Smoke/Train.
        $smokeRecord = Invoke-RunnerStage 'Smoke'
    }
    $inspectionArguments = @('-u', '-B', $inspector, '--run-dir', $smokeDir, '--console-log', $smokeRecord.stdout, '--console-log', $smokeRecord.stderr, '--invocation', $smokeInvocation, '--exit-code', [string]$smokeRecord.exit_code)
    $inspectionRecord = Invoke-PipelineProcess -Name 'InspectSmoke' -Executable $trainPython -Arguments $inspectionArguments -StdoutName 'Smoke.inspection.json'
    $state.smoke_validation = Get-InspectionValidation -ReportPath $inspectionRecord.stdout -RunDirectory $smokeDir -ExpectedSteps 10000
    Write-PipelineStatus
    if (-not $state.smoke_validation.passed) { throw ('Smoke validation failed: ' + ($state.smoke_validation.errors -join '; ')) }
    if ($ValidateSmokeOnly) {
        $state.status = 'validation_only_passed'
        $state.stage = 'SmokeValidationComplete'
        $state.completed_utc = [DateTime]::UtcNow.ToString('o')
        Write-PipelineStatus
        exit 0
    }
    $trainRecord = Invoke-RunnerStage 'Train'
    $finalInspectionArguments = @('-u', '-B', $inspector, '--run-dir', $trainDir, '--console-log', $trainRecord.stdout, '--console-log', $trainRecord.stderr, '--invocation', $trainInvocation, '--exit-code', [string]$trainRecord.exit_code)
    $finalInspectionRecord = Invoke-PipelineProcess -Name 'InspectTrain' -Executable $trainPython -Arguments $finalInspectionArguments -StdoutName 'Train.inspection.json'
    $state.train_validation = Get-InspectionValidation -ReportPath $finalInspectionRecord.stdout -RunDirectory $trainDir -ExpectedSteps 1000000
    Write-PipelineStatus
    if (-not $state.train_validation.passed) { throw ('Train validation failed: ' + ($state.train_validation.errors -join '; ')) }
    $state.status = 'completed'
    $state.stage = 'Completed'
    $state.completed_utc = [DateTime]::UtcNow.ToString('o')
    Write-PipelineStatus
    exit 0
}
catch {
    $state.status = 'failed'
    $state.error = $_.Exception.Message
    $state.completed_utc = [DateTime]::UtcNow.ToString('o')
    Write-PipelineStatus
    [Console]::Error.WriteLine($state.error)
    # Detailed original stage ExitCode is preserved in status.json, independently
    # of PowerShell's normal nonzero pipeline failure exit status.
    exit 1
}
