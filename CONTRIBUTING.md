# Contributing to CASCADE

Use Python 3.9 and create a development environment from the repository root:

```bash
python setup_env.py --venv .venv --dev --gui --constraints locks/requirements-py39-cpu.txt
source .venv/bin/activate
pytest -q
```

Keep public-`svv` compatibility changes inside `src/cascade/`; do not patch an installed `svv` package. Add tests for numerical behavior, exported schema, and configuration changes. Changes that intentionally alter scientific results require a documented reference case and tolerance.

Build and verify distributions before proposing a release:

```bash
python -m build
python scripts/release_smoke.py --wheel dist/cascade_vascular-*.whl
```
