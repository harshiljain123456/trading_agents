@echo off
rem Copies the cloud workflow into .github\workflows (one time, before publishing to GitHub).
cd /d "%~dp0"
if not exist ".github\workflows" mkdir ".github\workflows"
copy /Y "deploy\daily-signal.yml" ".github\workflows\daily-signal.yml"
echo.
echo Done: .github\workflows\daily-signal.yml is in place.
pause
