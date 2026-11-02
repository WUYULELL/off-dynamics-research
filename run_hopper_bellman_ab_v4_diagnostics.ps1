param(
    [switch]$DryRun,
    [string[]]$Seeds = @("3"),
    [int]$MaxStep = 0,
    [string]$OutDir = "",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$runner = Join-Path (Split-Path -Parent $PSCommandPath) "run_hopper_bellman_ab_v4.ps1"
$runnerParams = @{
    ConfigPath = "config\experiments\iql_bellman_ab_hopper_v4_diagnostics.json"
    Seeds = $Seeds
    MaxStep = $MaxStep
    OutDir = $OutDir
}
if ($DryRun) {
    $runnerParams["DryRun"] = $true
}
if ($Overwrite) {
    $runnerParams["Overwrite"] = $true
}

& $runner @runnerParams
