@echo off
REM Called by Windows Task Scheduler every hour.
REM Set PYTHON to the full path of python.exe (Python 3.8+).
set PYTHON=python
cd /d "%~dp0"
if not exist relay_work mkdir relay_work
"%PYTHON%" relay.py --config relay_config.json --once >> relay_work\task_stdout.log 2>&1
