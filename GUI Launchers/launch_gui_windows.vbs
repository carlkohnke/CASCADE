Option Explicit

Dim shell, command, scriptDir
Set shell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
command = "wsl.exe -d Ubuntu --cd """ & scriptDir & """ -- /bin/bash launch_gui.sh"
shell.Run command, 0, False
