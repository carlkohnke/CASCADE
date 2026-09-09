from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import cext, growth, hematocrit, hemodynamics, numerics, oxygen, tissue
from .types import SettingSection


SETTINGS_SECTIONS: dict[str, SettingSection] = {
    "growth": SettingSection(
        "growth",
        growth.DEFAULTS,
        aliases=growth.ALIASES,
        prefixes=growth.PREFIXES,
        deprecated=growth.DEPRECATED,
    ),
    "hemodynamics": SettingSection(
        "hemodynamics",
        hemodynamics.DEFAULTS,
        aliases=hemodynamics.ALIASES,
        prefixes=hemodynamics.PREFIXES,
        deprecated=hemodynamics.DEPRECATED,
    ),
    "kirchhoff": SettingSection(
        "kirchhoff",
        {k: v for k, v in hemodynamics.DEFAULTS.items() if k.startswith("KIRCHHOFF_")},
        aliases={k: v for k, v in hemodynamics.ALIASES.items() if v.startswith("KIRCHHOFF_")},
        prefixes=("KIRCHHOFF_",),
    ),
    "hematocrit": SettingSection(
        "hematocrit",
        hematocrit.DEFAULTS,
        aliases=hematocrit.ALIASES,
        prefixes=hematocrit.PREFIXES,
        deprecated=hematocrit.DEPRECATED,
    ),
    "oxygen": SettingSection(
        "oxygen",
        oxygen.DEFAULTS,
        aliases=oxygen.ALIASES,
        prefixes=oxygen.PREFIXES,
        deprecated=oxygen.DEPRECATED,
    ),
    "concentration": SettingSection(
        "concentration",
        {
            key: oxygen.DEFAULTS[key]
            for key in (
                "CONCENTRATION_SOLVER",
                "CONCENTRATION_INLET_BY_FLUID",
                "SOLUTE_DIFFUSIVITY",
                "VMAX_MM",
                "K_M_MM",
                "AXIAL_BLOOD_STEPS",
                "OMEGA",
                "GL_ORDER",
                "GL_ORDER_CEXT",
                "CONC_MAX_FOR_NORMALIZATION",
            )
        },
        aliases={
            key: value
            for key, value in oxygen.ALIASES.items()
            if value
            in {
                "CONCENTRATION_SOLVER",
                "CONCENTRATION_INLET_BY_FLUID",
                "SOLUTE_DIFFUSIVITY",
                "VMAX_MM",
                "K_M_MM",
                "AXIAL_BLOOD_STEPS",
                "OMEGA",
                "GL_ORDER",
                "GL_ORDER_CEXT",
                "CONC_MAX_FOR_NORMALIZATION",
            }
        },
    ),
    "tissue": SettingSection(
        "tissue",
        tissue.DEFAULTS,
        aliases=tissue.ALIASES,
        prefixes=tissue.PREFIXES,
        deprecated=tissue.DEPRECATED,
    ),
    "cext": SettingSection(
        "cext",
        cext.DEFAULTS,
        aliases=cext.ALIASES,
        prefixes=cext.PREFIXES,
        deprecated=cext.DEPRECATED,
    ),
    "numerics": SettingSection(
        "numerics",
        numerics.DEFAULTS,
        aliases=numerics.ALIASES,
        prefixes=numerics.PREFIXES,
        deprecated=numerics.DEPRECATED,
    ),
}

SECTION_ALIASES = {
    "flow": "hemodynamics",
    "pressure": "hemodynamics",
    "kirchhoff_solver": "kirchhoff",
    "hct": "hematocrit",
    "conc": "concentration",
    "tissue_oxygen": "tissue",
    "external": "cext",
    "external_concentration": "cext",
    "runtime": "runtime",
    "legacy": "runtime",
}


def default_settings() -> dict[str, dict[str, Any]]:
    return {name: dict(section.defaults) for name, section in SETTINGS_SECTIONS.items()}


def collect_config_settings(config: Any) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    sim = config.simulation
    growth_cfg = config.growth
    _merge_section(merged, "growth", {"n_equal_bifurcations": growth_cfg.n_equal_bifurcations})
    _merge_section(merged, "growth", dict(growth_cfg.equal_terminal or {}))
    _merge_section(merged, "tissue", {"accel_mode": sim.tissue_accel} if sim.tissue_accel is not None else {})
    _merge_section(
        merged,
        "tissue",
        {"gpu_validate_points": sim.tissue_gpu_validate_points}
        if sim.tissue_gpu_validate_points is not None
        else {},
    )
    _merge_section(merged, "cext", dict(sim.cext or {}))
    _merge_section(merged, "runtime", dict(sim.tissuesim or {}))

    for section_name, values in dict(getattr(config, "runtime_settings", {}) or {}).items():
        _merge_section(merged, section_name, values)
    return merged


def apply_settings(runtime: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    applied: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for raw_section, raw_values in settings.items():
        if raw_values is None:
            continue
        if not isinstance(raw_values, Mapping):
            raise ValueError(f"settings.{raw_section} must be an object.")
        section_name = _section_name(str(raw_section))
        for raw_key, raw_value in raw_values.items():
            if raw_value is None and str(raw_key).lower() not in {"n_equal_bifurcations", "equal_terminal_length"}:
                continue
            const_name, section, warning = _resolve_setting(section_name, str(raw_key), runtime)
            if warning:
                warnings.append(warning)
            default = section.defaults.get(const_name) if section is not None else getattr(runtime, const_name, raw_value)
            value = _coerce_value(const_name, raw_value, default)
            if not hasattr(runtime, const_name):
                raise ValueError(f"Runtime setting {const_name} is not available in the current runtime.")
            setattr(runtime, const_name, value)
            applied.setdefault(section_name, {})[const_name] = _jsonable(value)

    _refresh_derived_runtime_values(runtime, applied)
    return {"applied": applied, "warnings": warnings}


def _merge_section(target: dict[str, dict[str, Any]], section: str, values: Mapping[str, Any]) -> None:
    if not values:
        return
    target.setdefault(str(section), {}).update(dict(values))


def _section_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return SECTION_ALIASES.get(normalized, normalized)


def _resolve_setting(section_name: str, key: str, runtime: Any) -> tuple[str, SettingSection | None, str | None]:
    key_norm = key.strip()
    key_lower = key_norm.lower().replace("-", "_")
    if section_name == "runtime":
        return _resolve_any_setting(key_norm, runtime), None, None

    section = SETTINGS_SECTIONS.get(section_name)
    if section is None:
        raise ValueError(
            f"Unknown settings section '{section_name}'. Valid sections are: "
            + ", ".join(sorted([*SETTINGS_SECTIONS.keys(), "runtime"]))
        )
    const = _resolve_in_section(section, key_norm, key_lower)
    warning = section.deprecated.get(const)
    return const, section, warning


def _resolve_any_setting(key: str, runtime: Any) -> str:
    key_lower = key.lower().replace("-", "_")
    for section in SETTINGS_SECTIONS.values():
        try:
            return _resolve_in_section(section, key, key_lower)
        except ValueError:
            continue
    upper = key.upper()
    if hasattr(runtime, upper):
        return upper
    raise ValueError(f"Unknown runtime setting '{key}'. Move it into a named settings section or use an exact constant name.")


def _resolve_in_section(section: SettingSection, key: str, key_lower: str) -> str:
    upper = key.upper()
    if upper in section.constants:
        return upper
    alias = section.aliases.get(key_lower)
    if alias is not None and alias in section.constants:
        return alias
    for prefix in section.prefixes:
        candidate = prefix + upper
        if candidate in section.constants:
            return candidate
    raise ValueError(f"Unknown settings.{section.name} option '{key}'.")


def _coerce_value(name: str, value: Any, default: Any) -> Any:
    if name == "N_EQUAL_BIFURCATIONS":
        if value is None:
            return None
        value_i = int(value)
        return None if value_i < 0 else value_i
    if name.endswith("_DTYPE"):
        return _dtype_value(value)
    if isinstance(default, bool):
        return _as_bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return tuple(value)
    if isinstance(default, dict):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object.")
        return dict(value)
    return value


def _dtype_value(value: Any) -> Any:
    if value in (np.float32, np.float64, np.int32, np.int64):
        return value
    text = str(value).strip().lower()
    if text in {"float32", "f32"}:
        return np.float32
    if text in {"float64", "f64"}:
        return np.float64
    if text in {"int32", "i32"}:
        return np.int32
    if text in {"int64", "i64"}:
        return np.int64
    return np.dtype(value).type


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean runtime setting from {value!r}.")


def _refresh_derived_runtime_values(runtime: Any, applied: dict[str, dict[str, Any]]) -> None:
    override = getattr(runtime, "LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S", None)
    if override is not None:
        runtime.LUMEN_DIFFUSIVITY_CM2_S = float(override)
    elif hasattr(runtime, "LUMEN_DIFFUSIVITY_BLOOD_CM2_S"):
        runtime.LUMEN_DIFFUSIVITY_CM2_S = float(runtime.LUMEN_DIFFUSIVITY_BLOOD_CM2_S)
    if hasattr(runtime, "NETFLOW_MCV_FL"):
        runtime.NETFLOW_MCV_CORR = (92.0 / float(runtime.NETFLOW_MCV_FL)) ** (1.0 / 3.0)
    if hasattr(runtime, "TREE_DATA_DTYPE"):
        runtime.TREE_DATA_DTYPE_STR = np.dtype(runtime.TREE_DATA_DTYPE).name
    if hasattr(runtime, "TREE_INDEX_DTYPE"):
        runtime.TREE_INDEX_DTYPE_STR = np.dtype(runtime.TREE_INDEX_DTYPE).name


def _jsonable(value: Any) -> Any:
    if value in (np.float32, np.float64, np.int32, np.int64):
        return np.dtype(value).name
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value
