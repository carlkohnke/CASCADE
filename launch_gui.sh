#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/.cascade_gui"
export PATH="/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin"
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
if [[ -f "$SCRIPT_DIR/.cascade_python" ]]; then
    PYTHON_PATH="$(<"$SCRIPT_DIR/.cascade_python")"
fi
if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "CASCADE environment not found. Run: python setup_env.py --venv .venv --gui" >&2
    exit 1
fi
exec flock -n "$SCRIPT_DIR/.cascade_gui/gui.instance.lock" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 \
    "$PYTHON_PATH" -u -B -m cascade.gui "$@" \
    >> "$SCRIPT_DIR/.cascade_gui/gui.log" 2>&1
