@echo off
rem Moves the GitHub Desktop repository from the inner trading_agents folder up to this folder.
cd /d "%~dp0"
if not exist "trading_agents\.git" (
  echo Nothing to fix: no inner trading_agents\.git folder found.
  pause
  exit /b
)
if exist ".git" (
  echo Stopped: this folder already has a .git folder. Send Claude a screenshot.
  pause
  exit /b
)
attrib -h "trading_agents\.git"
move "trading_agents\.git" ".git"
if exist "trading_agents\.gitattributes" move "trading_agents\.gitattributes" ".gitattributes"
attrib +h ".git"
rmdir "trading_agents"
if exist "trading_agents" (echo Could not remove the inner folder - send Claude a screenshot.) else (echo Done: the repository now covers your whole trading_agents folder.)
pause
