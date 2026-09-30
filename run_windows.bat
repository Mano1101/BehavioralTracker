@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher not found. Install Python 3.10 or newer, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

".venv\Scripts\python.exe" main.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo Setup or application launch failed. Check the error above.
pause
exit /b 1
