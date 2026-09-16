@echo off
rem Launch CASCADE Studio through the Ubuntu WSL distribution and keep the
rem terminal visible if startup fails. Use this diagnostic launcher after
rem setup_env.py; use launch_gui_windows_silent.vbs for normal console-free use.
setlocal
rem CMD cannot retain a UNC working directory.  Keep it on a local Windows
rem directory and give WSL the launcher's absolute UNC path explicitly.
cd /d "%SystemRoot%"
wsl.exe -d Ubuntu --cd "%~dp0" -- /usr/bin/env CASCADE_WINDOWS_LAUNCHER=1 /bin/bash ./launch_gui_linux.sh
if errorlevel 1 pause
endlocal
