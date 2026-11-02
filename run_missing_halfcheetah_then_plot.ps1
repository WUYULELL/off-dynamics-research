$ErrorActionPreference = "Stop"

Write-Host "Step 1/2: run missing HalfCheetah paper-like experiments"
powershell -ExecutionPolicy Bypass -File .\run_missing_halfcheetah_paper_like.ps1
$trainExitCode = $LASTEXITCODE

if ($trainExitCode -ne 0) {
    Write-Host ""
    Write-Host "Training failed with exit code $trainExitCode. Plotting is skipped."
    exit $trainExitCode
}

Write-Host ""
Write-Host "Step 2/2: generate paper-like sweep plots"
conda run -n offdynamics python .\plot_paper_like_medium_sweep.py
$plotExitCode = $LASTEXITCODE

if ($plotExitCode -ne 0) {
    Write-Host ""
    Write-Host "Plotting failed with exit code $plotExitCode."
    exit $plotExitCode
}

Write-Host ""
Write-Host "All done. Results are under:"
Write-Host "analysis_results\paper_like_medium_sweep"
