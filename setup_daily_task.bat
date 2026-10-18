@echo off
rem One-time setup: runs run_daily_signal.bat every weekday at 4:30 pm.
rem If the laptop is off or asleep at 4:30, Windows runs it as soon as it is back on.
rem To remove later: run remove_daily_task.bat
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=(Get-Location).Path; $a=New-ScheduledTaskAction -Execute ($d+'\run_daily_signal.bat') -WorkingDirectory $d; $t=New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 4:30pm; $s=New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30); Register-ScheduledTask -TaskName 'TradingAgents Daily Signal' -Action $a -Trigger $t -Settings $s -Force | Out-Null; Get-ScheduledTask -TaskName 'TradingAgents Daily Signal' | Get-ScheduledTaskInfo | Select-Object TaskName,NextRunTime | Format-List"
echo.
echo If you see TaskName and NextRunTime above, the daily task is set up.
pause
