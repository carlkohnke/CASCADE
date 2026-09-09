@echo off
setlocal
wsl.exe -d Ubuntu --cd /home/carl/svv_sweeps/GFM -- flock -n .gfm_gui/gui.instance.lock env PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 .venv/bin/python -u -B -m gfm.gui
if errorlevel 1 pause
endlocal
