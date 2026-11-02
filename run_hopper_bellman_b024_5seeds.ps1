param(
    [switch]$DryRun,
    [string[]]$Seeds = @("0", "1", "2", "3", "4"),
    [int]$MaxStep = 500000,
    [double]$BellmanEta = 1.0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$runner = Join-Path (Split-Path -Parent $PSCommandPath) "run_hopper_bellman_b020_5seeds.ps1"
$runnerParams = @{
    Seeds = $Seeds
    MaxStep = $MaxStep
    BellmanEta = $BellmanEta
    SourceBudget = 0.24
    OutRoot = "runs_hopper_bellman_b024_5seeds"
}
if ($DryRun) {
    $runnerParams["DryRun"] = $true
}
if ($Overwrite) {
    $runnerParams["Overwrite"] = $true
}

& $runner @runnerParams
