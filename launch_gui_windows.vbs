Option Explicit

Dim shell, command
Set shell = CreateObject("WScript.Shell")

' WSL's NVIDIA utility lives outside the non-login shell PATH.  Export it
' explicitly so the GUI hardware probe sees the GPU just as the solver does.
command = "wsl.exe -d Ubuntu -- /bin/bash /home/carl/svv_sweeps/GFM/launch_gui.sh"
shell.Run command, 0, False
