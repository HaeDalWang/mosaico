@echo off
cd /d "%~dp0"
py -3.12 setup_env.py
if errorlevel 1 goto failed
echo.
echo Done. Use run.bat from now on.
goto end
:failed
echo.
echo Install failed.
echo If Python 3.12 is missing: install it from https://www.python.org/downloads/
echo and check "Add python.exe to PATH" on the first screen. Then run this file again.
:end
if not defined CI pause
