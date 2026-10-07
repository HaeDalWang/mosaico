@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto noinstall
".venv\Scripts\python.exe" run.py %*
goto end
:noinstall
echo Run install.bat first.
:end
if not defined CI pause
