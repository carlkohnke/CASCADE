"""Deterministic invalidation keys for reusable simulation stages."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def geometry_fingerprint(config) -> str:
    """Hash only settings that construct domains, vessels, or sample points.

    Solve-only changes must not evict expensive geometry or spatial indexes.
    The restricted simulation subsection is the important distinction from the
    exact-result fingerprint below.
    """
    raw = config.raw
    network = deepcopy(dict(raw.get("network", {}) or {}))
    network.pop("save_path", None)
    build_simulation = {
        "build_fluid": str(config.simulation.build_fluid),
        "distance_sample_count": int(config.simulation.distance_sample_count),
        "geometry_only": bool(config.simulation.geometry_only),
        "sample_mode": str(config.simulation.sample_mode),
        "sample_points_path": config.simulation.sample_points_path,
        "skip_tissue_oxygen": bool(config.simulation.skip_tissue_oxygen),
        "tissue_grid": deepcopy(config.simulation.tissue_grid),
    }
    if not qin_is_solve_only(config):
        build_simulation["qin_target_ul_min"] = float(
            config.simulation.qin_target_ul_min
        )
        build_simulation["total_qin_ul_min"] = config.simulation.total_qin_ul_min

    payload = {
        "schema_version": raw.get("schema_version"),
        "domain": deepcopy(raw.get("domain", {})),
        "network": network,
        "growth": deepcopy(raw.get("growth", {})),
        "simulation": build_simulation,
        # Runtime settings include vessel-generation parameters and dtypes.
        # Until every setting declares a stage, retain them conservatively.
        "settings": deepcopy(raw.get("settings", raw.get("runtime_settings", {}))),
        "_source_files": source_file_signatures(config),
    }
    return _hash_payload(payload)


def simulation_fingerprint(config) -> str:
    """Hash all scientific and output-shape settings for exact replay."""
    payload = deepcopy(config.raw)
    payload.pop("gui", None)
    payload.pop("sweep", None)
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        outputs.pop("out_dir", None)
        outputs.pop("prefix", None)
    payload["_source_files"] = source_file_signatures(config)
    return _hash_payload(payload)


def qin_is_solve_only(config) -> bool:
    """Return whether inlet flow can change without rebuilding a simple graph."""
    if config.network_mode != "simple":
        return True
    simple = dict(config.network.simple or {})
    return "flow_ul_min" in simple or "qin_ul_min" in simple


def source_file_signatures(config) -> dict[str, dict[str, Any]]:
    """Fingerprint file identity without hashing or loading large contents."""
    values = {
        "domain": config.domain.path,
        "network": config.network.input_path,
        "sample_points": config.simulation.sample_points_path,
        "growth_checkpoint": (
            config.growth.checkpoint_path
            if config.growth.resume_from_checkpoint
            else None
        ),
        "simple_geometry": dict(config.network.simple or {}).get(
            "path", dict(config.network.simple or {}).get("geometry_path")
        ),
    }
    base = (
        config.settings_path.parent
        if config.settings_path is not None
        else Path.cwd()
    )
    signatures: dict[str, dict[str, Any]] = {}
    for name, raw_path in values.items():
        if raw_path is None:
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        signature: dict[str, Any] = {"path": str(path)}
        try:
            stat = path.stat()
        except OSError:
            signature["missing"] = True
        else:
            signature.update(
                {
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        signatures[name] = signature
    return signatures


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


__all__ = [
    "geometry_fingerprint",
    "qin_is_solve_only",
    "simulation_fingerprint",
    "source_file_signatures",
]
