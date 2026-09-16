Option Explicit

' Launch CASCADE Studio through the Ubuntu WSL distribution without opening a
' console window. Use launch_gui_windows_shell.bat instead when diagnosing a
' startup failure. Both wrappers delegate environment discovery and logging to
' launch_gui_linux.sh in this directory.

Dim shell, command, scriptDir
Set shell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
' Never inherit the launcher's \wsl.localhost UNC directory as the Windows
' process working directory. WSL receives that path through --cd instead.
shell.CurrentDirectory = shell.ExpandEnvironmentStrings("%SystemRoot%")
command = "wsl.exe -d Ubuntu --cd """ & scriptDir & """ -- /usr/bin/env CASCADE_WINDOWS_LAUNCHER=1 /bin/bash ./launch_gui_linux.sh"
shell.Run command, 0, False
