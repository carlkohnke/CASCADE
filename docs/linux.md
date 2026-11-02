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

Setup validates the selected Python version before modifying an environment.
With `--gui`, it constructs every Studio page in a disposable offscreen session
before reporting success. Repeat this check with
`python -m cascade.gui.smoke_test` in the installed environment.

## NVIDIA GPU installation

CASCADE has three GPU package families. Each installs CuPy together with the
NVRTC compiler, cuFFT, and the other CUDA libraries as Python wheels, so a
system CUDA toolkit is not required. Only the NVIDIA driver must be installed.
The driver and the oldest GPU in the machine decide the family. Run
`nvidia-smi` and read **CUDA Version** in the top-right corner of its header;
that is the newest CUDA the driver supports. Run
`nvidia-smi --query-gpu=name,compute_cap --format=csv` for each GPU's compute
capability (7.0 means `sm_70`):

| Family | Extra | CuPy | Driver (`nvidia-smi` CUDA Version) | GPUs |
| --- | --- | --- | --- | --- |
| `--gpu cu13` | `gpu-cu13` | `cupy-cuda13x` 14.x | 13.0 or later (driver 580 or later) | sm_75 or newer (Turing and later) |
| `--gpu cu12` | `gpu-cu12` | `cupy-cuda12x` 14.x | 12.0 or later | sm_50 or newer (Maxwell and later) |
| `--gpu cu11` | `gpu-cu11` | `cupy-cuda11x` 13.x | 11.2 or later | sm_35 or newer (Kepler and later) |

Use the newest family that both columns allow. CUDA 13 is the qualified default
for new systems, but it dropped Maxwell, Pascal and Volta GPUs (sm_50 to sm_72,
for example P100, TITAN Xp and V100); those use `cu12` even with a CUDA 13
driver, and a CUDA 13 NVRTC rejects them with `invalid value for
--gpu-architecture`. A newer driver also runs the older families, but CUDA 13
packages fail with `cudaErrorInsufficientDriver` on a driver that reports CUDA
12.x or earlier. `--gpu auto` reads the driver and every visible GPU and selects
the newest matching family:

```bash
python3.12 setup_linux.py --venv .venv --gui --gpu auto
source .venv/bin/activate
cascade doctor --require-gpu
cascade self-test --require-gpu
cascade-gui
```

`cascade doctor` prints the driver's CUDA version, each GPU's architecture and
compute mode, and the installed CuPy package, and warns when they do not match,
with the commands that switch families. It also flags CUDA component wheels
left over from another family; CASCADE loads the installed family's libraries
first, but uninstalling the leftovers keeps the environment unambiguous. Remove the old CuPy package before installing another family; the
`cupy-cuda*x` packages share one import name and overwrite each other.

On a cluster, the login node often has no NVIDIA driver, so `--gpu auto`
installs CPU-only dependencies there. Run `nvidia-smi` on the GPU nodes you
will use and pass the family that suits all of them explicitly, for example
`--gpu cu12` for a partition that mixes V100 and newer GPUs. Cluster GPUs
often run in Exclusive_Process compute mode, where only one process may hold a
device: start one CASCADE run per GPU, and do not keep a Python session with
CuPy open on the GPU that CASCADE should use. When the driver is older than
the newest 12.x toolkit, you can pin the wheel toolkit to the driver's version
inside the environment:

```bash
python -m pip install "cuda-toolkit[nvjitlink]==12.4.*"
```

`gpu-cu11` uses CuPy 13, the last CuPy release for CUDA 11, with the CUDA 11.8
component wheels. CASCADE loads those libraries before importing CuPy. Use
`--cuda-path` only when the driver/toolkit layout requires an explicit CUDA
target directory; normal wheel-based installation does not require a compiler
or a system CUDA toolkit.

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
