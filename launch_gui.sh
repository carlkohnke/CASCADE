#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/.gfm_gui"
export PATH="/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin"
exec flock -n "$SCRIPT_DIR/.gfm_gui/gui.instance.lock" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 \
    "$SCRIPT_DIR/.venv/bin/python" -u -B -m gfm.gui "$@" \
    >> "$SCRIPT_DIR/.gfm_gui/gui.log" 2>&1
