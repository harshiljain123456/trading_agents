@echo off
rem Daily paper signal (Step 7). Runs the Nifty-filter signal and the agent panel, saves the output to logs\.
rem Paper only: nothing is sent to any broker.
cd /d "%~dp0"
if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
echo Running daily signal for %TODAY% ... (output goes to logs\signal_%TODAY%.txt)
python main.py --market nse --real --universe nifty50 --daily-signal --log-level WARNING > "logs\signal_%TODAY%.txt" 2>&1
if errorlevel 1 (echo FAILED - see logs\signal_%TODAY%.txt) else (echo Done.)
