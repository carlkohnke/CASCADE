@echo off
setlocal

rem Double-click entry point for the native Windows installer.
rem The PowerShell implementation remains beside the downloaded source tree.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install-cascade.ps1" %*
set "CASCADE_INSTALL_STATUS=%ERRORLEVEL%"

echo.
if "%CASCADE_INSTALL_STATUS%"=="0" (
    echo CASCADE installation finished successfully.
) else (
    echo CASCADE installation failed. Review the installer log shown above.
)
echo.
if /I not "%CASCADE_INSTALLER_NO_PAUSE%"=="1" pause
exit /b %CASCADE_INSTALL_STATUS%
