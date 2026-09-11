@echo off
setlocal
wsl.exe -d Ubuntu --cd "%~dp0" -- /bin/bash launch_gui_linux.sh
if errorlevel 1 pause
endlocal
