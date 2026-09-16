# WSL installation

CASCADE can run as a Linux application inside WSL 2 and display Studio through
WSLg. This is separate from CASCADE's native Windows installation: it uses
Linux Python, Linux packages, and Linux filesystem/process behavior.

## Prerequisites

- WSL 2 with an Ubuntu distribution and WSLg.
- Python 3.12 with `venv` and `pip` support inside Ubuntu.
- For GPU execution, a Windows NVIDIA driver with WSL CUDA support.

Keep the repository in the Linux filesystem (for example,
`/home/your-name/CASCADE`) for predictable performance. From an Ubuntu shell:

```bash
cd /path/to/CASCADE
python3.12 setup_linux.py --venv .venv --gui
source .venv/bin/activate
cascade doctor --no-gpu-probe
cascade self-test
cascade-gui
```

For NVIDIA GPU support, add `--gpu cu13` to the setup command and validate with
`cascade doctor --require-gpu` and `cascade self-test --require-gpu`.

## Optional WSL launchers

After setup, the supported launcher from an Ubuntu shell is:

```bash
bash scripts/wsl/launch-studio.sh
```

The same directory contains two Windows-side convenience wrappers:

- `launch-from-windows.vbs` starts Studio without a console window.
- `launch-diagnostic.cmd` retains a console so startup failures are visible.

The wrappers currently target a WSL distribution named `Ubuntu`. If the
distribution has another name, launch from its Linux shell or update the
explicit `-d Ubuntu` argument in your local copy.

The WSL launcher prefers WSLg's Wayland transport and CASCADE's software
preview backend for reliability. Advanced users can override
`QT_QPA_PLATFORM` or `CASCADE_RENDER_BACKEND` before launch. Logs are stored
under the configured CASCADE state directory rather than in the source tree.

CLI commands such as `cascade run` work in the activated Ubuntu shell. They do
not become native PowerShell commands because this environment is Linux. For a
native Windows GUI and CLI, instead use
[Install CASCADE for Windows.cmd](../Install%20CASCADE%20for%20Windows.cmd) and
follow the [Windows guide](windows.md).
