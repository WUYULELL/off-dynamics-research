@echo off
setlocal

REM Run this from an Anaconda Prompt or CMD after:
REM   conda activate offdynamics

set _INSIDE_OFFDYNAMICS=
if "%~1"=="--inside-offdynamics" (
  set _INSIDE_OFFDYNAMICS=1
  shift
)

python --version
python -m pip --version

python check_offdynamics_context.py
if errorlevel 1 (
  if not "%_INSIDE_OFFDYNAMICS%"=="1" (
    echo.
    echo Trying to relaunch through: conda run -n offdynamics
    conda run -n offdynamics cmd /c "cd /d "%~dp0" && "%~nx0" --inside-offdynamics"
    exit /b %errorlevel%
  )
  echo.
  echo ERROR: Wrong Python environment. Open an Anaconda Prompt, run:
  echo   conda activate offdynamics
  echo   cd /d "%~dp0"
  echo   install_offdynamics_win.cmd
  exit /b 1
)

REM If your proxy is not 127.0.0.1:7890, edit these three lines.
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set ALL_PROXY=http://127.0.0.1:7890

set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

python -m pip install pip==22.3.1 --no-user --proxy %HTTP_PROXY%
if errorlevel 1 exit /b 1

python -m pip install setuptools==63.2.0 wheel==0.38.4 --no-user --proxy %HTTP_PROXY%
if errorlevel 1 exit /b 1

python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-user --proxy %HTTP_PROXY%
if errorlevel 1 exit /b 1

python -m pip install -r requirements_win_no_box2d.txt --no-build-isolation --no-user --proxy %HTTP_PROXY%
if errorlevel 1 exit /b 1

python verify_offdynamics_env.py
endlocal
