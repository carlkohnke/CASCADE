# Native Windows installation

CASCADE supports native 64-bit Windows 10 and Windows 11 with 64-bit CPython
3.12. The released wheel runs with Windows Python, Qt, filesystem, process, and
NVIDIA driver interfaces. WSL, Git, a C/C++ compiler, and a system CUDA toolkit
are not part of the supported Windows runtime.

## Simplest installation from a GitHub download

Installation path for someone who doesn't want to use a terminal:

1. Install 64-bit Python 3.12 from
   [python.org](https://www.python.org/downloads/windows/). The standard Python
   launcher is sufficient; Python does not have to be added to the system `PATH`.
2. Download the CASCADE source ZIP from GitHub and extract it to an ordinary
   local Windows folder, for example `C:\Users\you\Downloads\CASCADE`.
   Do not run the installer from inside the ZIP, a WSL path, or a network/UNC
   path.
3. Double-click **Install CASCADE for Windows.cmd** in the extracted folder.
4. Wait for the success message, then open **CASCADE Studio** from the desktop
   or Start menu.

The installer builds CASCADE's wheel, creates a private environment beneath
`%LOCALAPPDATA%\Programs\CASCADE`, installs binary dependency wheels, runs
`doctor` and `self-test`, and creates the clearly labeled Studio shortcut. It
automatically qualifies NVIDIA/CUDA acceleration when an NVIDIA device is
present. If GPU qualification fails in automatic mode, it verifies the CPU
path and reports **CPU fallback** rather than leaving a broken installation.

The same installation also enables the command line. Open a *new* PowerShell
or Command Prompt window after installation and run, for example:

```powershell
cascade --version
cascade doctor
cascade self-test
```

Re-running the same installer safely updates the private environment from the
downloaded source. Installation logs are retained in
`%LOCALAPPDATA%\cascade\Logs`.

## Advanced wheel installation

The commands below are the explicit alternative for release engineering or
users who want to choose and manage their own environment. They do not require
a global `PATH` change. Replace the example wheel path with the wheel downloaded
from the CASCADE release.

## CPU installation

In PowerShell:

```powershell
py -3.12 -m venv "$env:USERPROFILE\CASCADE"
& "$env:USERPROFILE\CASCADE\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\CASCADE\Scripts\python.exe" -m pip install "C:\Downloads\cascade_vascular-0.1.0rc5-py3-none-any.whl[gui]"
& "$env:USERPROFILE\CASCADE\Scripts\cascade.exe" doctor --no-gpu-probe
& "$env:USERPROFILE\CASCADE\Scripts\cascade.exe" self-test
```

CASCADE and its scientific dependencies install from binary wheels. A source
build of a scientific or C/C++ dependency is not expected on the supported
Python and platform combination.

## NVIDIA GPU installation

The qualified Windows GPU option is CUDA 13. It requires a supported NVIDIA GPU
and a sufficiently recent Windows NVIDIA driver, but it does not require the
CUDA toolkit to be installed system-wide. CuPy and the runtime compiler and
math libraries are installed as Python wheels:

```powershell
py -3.12 -m venv "$env:USERPROFILE\CASCADE-GPU"
& "$env:USERPROFILE\CASCADE-GPU\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\CASCADE-GPU\Scripts\python.exe" -m pip install "C:\Downloads\cascade_vascular-0.1.0rc5-py3-none-any.whl[gui,gpu-cu13]"
& "$env:USERPROFILE\CASCADE-GPU\Scripts\cascade.exe" doctor --require-gpu
& "$env:USERPROFILE\CASCADE-GPU\Scripts\cascade.exe" self-test --require-gpu
```

Do not copy CUDA DLLs into the environment or add CUDA directories to the
global `PATH`. CASCADE discovers the toolkit components declared by its GPU
extra inside the virtual environment.

## Launching Studio

The normal launcher is a native Windows GUI executable and does not open an
extra console window:

```powershell
& "$env:USERPROFILE\CASCADE\Scripts\cascade-gui.exe"
```

Use the console launcher when diagnosing startup or graphics problems. It runs
the same Studio application but keeps diagnostics visible:

```powershell
& "$env:USERPROFILE\CASCADE\Scripts\cascade-gui-console.exe"
```

Studio also writes unhandled GUI exceptions to
`%LOCALAPPDATA%\cascade\Logs\cascade-studio-errors.log`. Set
`CASCADE_RENDER_BACKEND=software` in the launching PowerShell session to bypass
OpenGL for graphics-driver troubleshooting. Set it to `gpu` to require the
OpenGL renderer and fail visibly when it is unavailable.

The `GUI Launchers` directory in the source repository contains legacy WSL
development helpers. They are not used by the native wheel installation.

## Command-line use

Activation is optional. Calling the environment's executables directly makes
the selected environment explicit:

```powershell
& "$env:USERPROFILE\CASCADE\Scripts\cascade.exe" init-settings case.json
& "$env:USERPROFILE\CASCADE\Scripts\cascade.exe" run --settings case.json
```

With the environment activated, the same commands are available as `cascade`,
`cascade-gui`, and `cascade-gui-console`.

## Files and user state

By default, Studio starts new projects beneath
`%USERPROFILE%\CASCADE Projects`. Projects and simulation outputs are not
stored in the Python environment. CASCADE's application-managed files use:

| Purpose | Default location |
| --- | --- |
| configuration and state | `%LOCALAPPDATA%\cascade` |
| caches | `%LOCALAPPDATA%\cascade\Cache` |
| diagnostic logs | `%LOCALAPPDATA%\cascade\Logs` |

`CASCADE_PROJECT_DIR`, `CASCADE_CONFIG_DIR`, `CASCADE_STATE_DIR`,
`CASCADE_CACHE_DIR`, and `CASCADE_LOG_DIR` can override these locations for an
individual process. `CASCADE_OUTPUT_DIR`, `CASCADE_DOMAIN_CACHE_DIR`, and
`CASCADE_PREPARED_CACHE_DIR` provide narrower workflow overrides. Do not point
runtime state at the installed `site-packages` directory.

## GPU troubleshooting

Run `cascade doctor --require-gpu` first. Common failure categories are:

- **No NVIDIA device:** confirm that Windows Device Manager and `nvidia-smi`
  can see the device, then update the NVIDIA driver if necessary.
- **Driver/runtime mismatch:** update the NVIDIA driver; installing a separate
  CUDA toolkit or copying DLLs does not repair an incompatible driver.
- **Kernel compile or stale-cache error:** close Studio, move the specific
  `%LOCALAPPDATA%\cascade\Cache` directory aside, and rerun
  `cascade self-test --require-gpu`. CASCADE rebuilds disposable caches.
- **OpenGL preview failure:** start with
  `$env:CASCADE_RENDER_BACKEND='software'`; this affects the preview renderer,
  not scientific CUDA execution.

The console launcher and the persistent Studio error log preserve startup
details that the normal console-free launcher cannot display.

## Uninstalling

Remove the package from its environment with:

```powershell
& "$env:USERPROFILE\CASCADE\Scripts\python.exe" -m pip uninstall cascade-vascular
```

If the environment is dedicated to CASCADE, it may instead be removed after
all CASCADE processes are closed. Neither method removes user projects beneath
`%USERPROFILE%\CASCADE Projects`. Application caches and logs beneath
`%LOCALAPPDATA%\cascade` are also retained so uninstalling cannot delete
scientific work or diagnostic evidence.

## Support boundary

Native Windows x86-64, CPython 3.12, CPU execution, Studio, and the CUDA 13 GPU
extra are supported. Windows on ARM, 32-bit Python, Python versions other than
3.12, AMD/Intel GPU compute, Microsoft Store Python aliases, and execution from
a WSL or network/UNC package installation are outside the qualified boundary.
See the [qualification record](windows-qualification.md) for the exact host,
artifacts, tests, and remaining manual UI checks.
