"""Typed-to-legacy runtime configuration bridge."""

from __future__ import annotations

from cascade.vessels._build_common import (
    RunConfig,
    RuntimeConfiguration,
)

def load_runtime_module():
    from cascade.runtime import tissuesim as ts

    return ts


def apply_runtime_settings(ts, config: RunConfig) -> None:
    resolved = RuntimeConfiguration.from_run_config(config)
    report = resolved.apply_compatibility_state(ts)
    config.runtime_setting_overrides = report["applied"]
    config.runtime_setting_warnings = report["warnings"]
    for warning in config.runtime_setting_warnings:
        print(f"Warning: {warning}", flush=True)




__all__ = ('load_runtime_module', 'apply_runtime_settings')
