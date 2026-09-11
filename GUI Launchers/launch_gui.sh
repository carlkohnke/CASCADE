#!/usr/bin/env bash
set -euo pipefail
LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd -- "$LAUNCHER_DIR/.." && pwd)"
cd "$REPOSITORY_DIR"
export PATH="/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin"

WORKBENCH_DIR="${CASCADE_WORKBENCH:-$REPOSITORY_DIR/../CASCADE-workbench}"
if [[ -d "$WORKBENCH_DIR" ]]; then
    CONFIG_DIR="${CASCADE_CONFIG_DIR:-$WORKBENCH_DIR/config}"
    STATE_DIR="${CASCADE_STATE_DIR:-$WORKBENCH_DIR/state/gui}"
    export CASCADE_PROJECT_DIR="${CASCADE_PROJECT_DIR:-$WORKBENCH_DIR/projects/CASCADE_Project}"
else
    CONFIG_DIR="${CASCADE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/cascade}"
    STATE_DIR="${CASCADE_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/cascade/gui}"
fi
mkdir -p "$STATE_DIR"
export CASCADE_CONFIG_DIR="$CONFIG_DIR"

PYTHON_PATH="${CASCADE_PYTHON:-$REPOSITORY_DIR/.venv/bin/python}"
if [[ -f "$CONFIG_DIR/.cascade_python" ]]; then
    PYTHON_PATH="$(<"$CONFIG_DIR/.cascade_python")"
elif [[ -f "$REPOSITORY_DIR/.cascade_python" ]]; then
    PYTHON_PATH="$(<"$REPOSITORY_DIR/.cascade_python")"
fi
if [[ -z "${CUDA_PATH:-}" && -f "$CONFIG_DIR/.cascade_cuda_path" ]]; then
    export CUDA_PATH="$(<"$CONFIG_DIR/.cascade_cuda_path")"
fi
if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "CASCADE environment not found. Run: python setup_env.py --venv .venv --gui" >&2
    exit 1
fi
exec flock -n "$STATE_DIR/gui.instance.lock" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 \
    "$PYTHON_PATH" -u -B -m cascade.gui "$@" \
    >> "$STATE_DIR/gui.log" 2>&1
