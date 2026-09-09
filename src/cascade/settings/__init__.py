"""Modular runtime settings for CASCADE.

The settings package is the public boundary between JSON run files and the
runtime globals used by the current numerical kernels.
"""

from .registry import (
    SETTINGS_SECTIONS,
    apply_settings,
    collect_config_settings,
    default_settings,
)

__all__ = [
    "SETTINGS_SECTIONS",
    "apply_settings",
    "collect_config_settings",
    "default_settings",
]
