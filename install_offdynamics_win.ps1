param(
    [switch]$InsideOffdynamics
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

Invoke-Step "Python version" { python --version }
Invoke-Step "Pip version" { python -m pip --version }

python "$PSScriptRoot\check_offdynamics_context.py"
if ($LASTEXITCODE -ne 0) {
    if ($InsideOffdynamics) {
        throw "Wrong Python environment even after conda run."
    }
    Write-Host ""
    Write-Host "Current shell is not using offdynamics. Relaunching through conda run..."
    conda run -n offdynamics powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\install_offdynamics_win.ps1" -InsideOffdynamics
    exit $LASTEXITCODE
}

$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:ALL_PROXY = "http://127.0.0.1:7890"
$env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:PIP_TRUSTED_HOST = "pypi.tuna.tsinghua.edu.cn"

Invoke-Step "Install pip 22.3.1" { python -m pip install pip==22.3.1 --no-user --proxy $env:HTTP_PROXY }
Invoke-Step "Install build tools" { python -m pip install setuptools==63.2.0 wheel==0.38.4 --no-user --proxy $env:HTTP_PROXY }
Invoke-Step "Install PyTorch cu113" { python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-user --proxy $env:HTTP_PROXY }
Invoke-Step "Install ODRL requirements" { python -m pip install -r "$PSScriptRoot\requirements_win_no_box2d.txt" --no-build-isolation --no-user --proxy $env:HTTP_PROXY }
Invoke-Step "Verify environment" { python "$PSScriptRoot\verify_offdynamics_env.py" }
