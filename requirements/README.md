# Dependency files

All CASCADE pip requirement and constraint files live here.

`conda.yml` is the equivalent minimal Conda environment definition. Create it
from the repository root with `conda env create -f requirements/conda.yml`.

## Install sets

- `base.txt`: editable CASCADE installation and its core dependencies.
- `dev.txt`: base installation plus test/development tools.
- `gui.txt`: base installation plus CASCADE Studio dependencies.
- `gpu-cu11.txt`, `gpu-cu12.txt`, `gpu-cu13.txt`: base installation plus the
  matching optional CuPy/CUDA runtime.

Run requirement-file installs from the repository root, for example:

```bash
python -m pip install -r requirements/dev.txt
```

## Reproducible locks

`locks/` contains the fully pinned Python 3.9 environments used for reproducible
CPU and CUDA installations. Pass these files with pip's `-c` option; they are
constraints, not standalone installation lists.

```bash
python -m pip install -c requirements/locks/py39-cpu.txt '.[dev,gui]'
```
