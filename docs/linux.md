# Linux installation

CASCADE supports 64-bit Linux with Python 3.12. The repository setup helper
creates a private virtual environment, installs CASCADE and its binary Python
dependencies, and leaves the rest of the machine unchanged.

## Prerequisites

- Python 3.12 with `venv` and `pip` support.
- A Wayland or X11 desktop session for CASCADE Studio.
- For GPU execution, a compatible NVIDIA driver. The CUDA user-space toolkit
  is supplied by the selected Python packages.

## CPU installation

From the extracted or cloned repository root:

```bash
python3.12 setup_linux.py --venv .venv --gui
source .venv/bin/activate
cascade doctor --no-gpu-probe
cascade self-test
cascade-gui
```

The setup is source-based and editable; keep the repository after
installation. Activating the environment is optional: you can launch Studio
directly with `.venv/bin/cascade-gui` or run the CLI as `.venv/bin/cascade`.

## NVIDIA GPU installation

Choose the CUDA package family compatible with the installed NVIDIA driver.
CUDA 13 is the current default for newly qualified systems:

```bash
python3.12 setup_linux.py --venv .venv --gui --gpu cu13
source .venv/bin/activate
cascade doctor --require-gpu
cascade self-test --require-gpu
cascade-gui
```

The helper also accepts `cu11` and `cu12`. Use `--cuda-path` only when the
driver/toolkit layout requires an explicit CUDA target directory; normal
wheel-based installation does not require a compiler or a system CUDA toolkit.

## Updating or rebuilding

After updating the source, rerun the same setup command to refresh the
environment. `--recreate` removes and rebuilds only the exact virtual
environment passed with `--venv`; use it deliberately because any files stored
inside that environment are discarded.

By default, CASCADE stores Linux configuration beneath
`${XDG_CONFIG_HOME:-~/.config}/cascade`, state and logs beneath
`${XDG_STATE_HOME:-~/.local/state}/cascade`, and projects beneath
`~/CASCADE Projects`. These locations can be changed with the documented
`CASCADE_CONFIG_DIR`, `CASCADE_STATE_DIR`, and `CASCADE_PROJECT_DIR`
environment variables.

WSL users should follow the separate [WSL guide](wsl.md), which explains WSLg
and the optional Windows-side convenience launchers. Developers should also
see [development setup](development.md).
