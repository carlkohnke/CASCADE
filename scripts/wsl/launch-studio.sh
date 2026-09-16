#!/usr/bin/env bash
# Launch CASCADE Studio from a Linux or WSL checkout.
#
# Run this file directly after `python setup_linux.py --venv .venv --gui`, or let
# one of the Windows wrappers call it under WSL. It resolves the checkout and
# configured Python automatically, writes GUI logs/state outside the source
# tree when a sibling CASCADE-workbench exists, and prevents duplicate Studio
# instances with a per-user lock.
set -euo pipefail
LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd -- "$LAUNCHER_DIR/../.." && pwd)"
cd "$REPOSITORY_DIR"
export PATH="/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin"

# Windows shortcuts mark their WSL process explicitly because WSL_INTEROP is
# not guaranteed to survive every wsl.exe launch configuration.
IS_WSL=0
if [[ -n "${CASCADE_WINDOWS_LAUNCHER:-}" || -n "${WSL_DISTRO_NAME:-}" || -n "${WSL_INTEROP:-}" ]]; then
    IS_WSL=1
fi

# WSLg copy mode can advertise a usable OpenGL context while failing to
# present QOpenGLWidget content. Keep shortcut launches entirely on Qt's
# raster path by default. Users can still opt in with
# CASCADE_RENDER_BACKEND=opengl when their WSLg graphics stack is healthy.
if [[ "$IS_WSL" == "1" && -z "${CASCADE_RENDER_BACKEND:-}" ]]; then
    export CASCADE_RENDER_BACKEND=software
fi
if [[ "$IS_WSL" == "1" && "${CASCADE_RENDER_BACKEND:-}" == "software" ]]; then
    export QT_OPENGL="${QT_OPENGL:-software}"
fi

# Prefer WSLg's native Wayland transport. Forcing XWayland can produce a
# copy-mode window whose client area is entirely black even when CASCADE is
# using its software canvas. An explicit user platform choice still wins.
if [[ "$IS_WSL" == "1" && -z "${QT_QPA_PLATFORM:-}" ]]; then
    export QT_QPA_PLATFORM=wayland
fi
if [[ "${QT_QPA_PLATFORM:-}" == "xcb" && "${CASCADE_RENDER_BACKEND:-}" == "software" ]]; then
    export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
fi

# XWayland otherwise inherits the desktop X cursor size, which WSLg can scale
# a second time and turn into an oversized black pointer. Keep CASCADE at a
# conventional Windows-sized cursor while allowing an explicit user override.
if [[ "${QT_QPA_PLATFORM:-}" == "xcb" ]]; then
    export XCURSOR_THEME="${CASCADE_CURSOR_THEME:-Adwaita}"
    export XCURSOR_SIZE="${CASCADE_CURSOR_SIZE:-16}"
fi

# Prefer the optional workbench for configuration and GUI state files. Project
# locations remain user-owned and portable; CASCADE_PROJECT_DIR is honored only
# when it was explicitly supplied by the user or packaging environment.
WORKBENCH_DIR="${CASCADE_WORKBENCH:-$REPOSITORY_DIR/../CASCADE-workbench}"
if [[ -d "$WORKBENCH_DIR" ]]; then
    CONFIG_DIR="${CASCADE_CONFIG_DIR:-$WORKBENCH_DIR/config}"
    STATE_DIR="${CASCADE_STATE_DIR:-$WORKBENCH_DIR/state/gui}"
else
    CONFIG_DIR="${CASCADE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/cascade}"
    STATE_DIR="${CASCADE_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/cascade/gui}"
fi
mkdir -p "$STATE_DIR"
export CASCADE_CONFIG_DIR="$CONFIG_DIR"

# `setup_linux.py` records its interpreter and optional CUDA toolkit in the
# configuration directory. Explicit environment variables take precedence.
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
    echo "CASCADE environment not found. Run: python setup_linux.py --venv .venv --gui" >&2
    exit 1
fi
exec flock -n "$STATE_DIR/gui.instance.lock" \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 \
    "$PYTHON_PATH" -u -B -m cascade.gui "$@" \
    >> "$STATE_DIR/gui.log" 2>&1
