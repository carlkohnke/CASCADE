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

`locks/` contains fully pinned platform-specific environments used for
reproducible CPU and CUDA installations. Pass these files with pip's `-c`
option; they are constraints, not standalone installation lists. The
`py312-win-amd64-*` files are the binary-only native Windows qualification
candidates. The retained `py39-*` files describe the earlier Linux release
environment and are not compatible with the current Python requirement.

```text
python -m pip install -c requirements/locks/py312-win-amd64-cpu.txt ".[dev,gui]"
```
