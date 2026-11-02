# Dependency files

All CASCADE pip requirement and constraint files live here.

`conda.yml` is the equivalent minimal Conda environment definition. Create it
from the repository root with `conda env create -f requirements/conda.yml`.

## Install sets

- `base.txt`: editable CASCADE installation and its core dependencies.
- `dev.txt`: base installation plus test/development tools.
- `gui.txt`: base installation plus CASCADE Studio dependencies.
- `gpu-cu11.txt`, `gpu-cu12.txt`, `gpu-cu13.txt`: base installation plus the
  matching optional CuPy/CUDA runtime and PyAMG. Choose the file from the
  **CUDA Version** that `nvidia-smi` reports and the oldest GPU: 13.0 or later
  (driver 580+) with sm_75 or newer GPUs uses `gpu-cu13`; 12.x, or sm_50 to
  sm_72 GPUs such as P100 and V100, uses `gpu-cu12`; 11.2 to 11.8, or Kepler
  GPUs, uses `gpu-cu11`. The CUDA
  12 and 13 files use CuPy 14 with its `ctk` wheel toolkit; the CUDA 11 file uses
  CuPy 13 with the CUDA 11.8 component wheels. They mirror the `gpu-cu*`
  extras in `pyproject.toml`.

Run requirement-file installs from the repository root, for example:

```bash
python -m pip install -r requirements/dev.txt
```

## Reproducible locks

`locks/py312-linux-x86_64-publication-cpu.txt` records the complete dependency
resolution from the fresh Linux/WSL Python 3.12 publication-demo installation
on 5 October 2026, including the GUI extra. Apply it with
`python -m pip install -c requirements/locks/py312-linux-x86_64-publication-cpu.txt ".[gui]"`.
See [tested environments](../docs/tested-environments.md) for the tested machine
and the [publication demo guide](../docs/publication-demo.md) for timings, inputs,
and expected output. It is not a native-Windows or CUDA constraint file.

`locks/` contains fully pinned platform-specific environments used for
reproducible CPU and CUDA installations. Pass these files with pip's `-c`
option; they are constraints, not standalone installation lists. The
`py312-win-amd64-*` files are the binary-only native Windows qualification
candidates. The retained `py39-*` files describe the earlier Linux release
environment and are not compatible with the current Python requirement.

```text
python -m pip install -c requirements/locks/py312-win-amd64-cpu.txt ".[dev,gui]"
```
