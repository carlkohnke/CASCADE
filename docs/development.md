# Development setup

CASCADE development is currently centered on Linux and WSL with Python 3.12.
The setup helper creates an editable repository-local environment with the GUI
and development tools:

```bash
python3.12 setup_linux.py --venv .venv --gui --dev
source .venv/bin/activate
```

Run the primary checks from the repository root:

```bash
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82,F601,F811,E741 src setup_linux.py tests
python -m build
```

The editable installation means source changes are immediately visible in the
environment. Rerun the setup helper after dependency or packaging changes.
Use `--recreate` only when a clean environment is required; it removes the
exact virtual-environment directory supplied with `--venv` before rebuilding
it.

Native Windows end users should use the one-click installer described in the
[Windows guide](windows.md). Windows development and release qualification
should use isolated native-Windows environments and an installed wheel; do not
reuse a Linux/WSL virtual environment from PowerShell or point Windows Python
at the source tree over a WSL UNC path.

Platform-specific user setup is documented separately for
[Linux](linux.md), [WSL](wsl.md), and [native Windows](windows.md).
