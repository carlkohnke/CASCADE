"""JSON value conversion shared by GUI pages and window orchestration."""

from __future__ import annotations

import json
from typing import Any


def parse_jsonish(text: str) -> Any:
    """Parse a settings-cell value while preserving ordinary text."""
    value = text.strip()
    try:
        return json.loads(value)
    except Exception:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value


def json_safe(value):
    """Convert NumPy scalars, dtypes, and tuples to JSON-compatible values."""
    try:
        import numpy as np

        if value in (np.float32, np.float64, np.int32, np.int64):
            return np.dtype(value).name
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


# Transitional aliases for callers of the former monolithic GUI module.
_parse_jsonish = parse_jsonish
_json_safe = json_safe

__all__ = ["json_safe", "parse_jsonish"]
