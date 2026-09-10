"""Write solver arrays as individually memory-mappable M2 evidence files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _topology_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=-1.0).astype(np.int64, copy=False)


def save_case_arrays(
    output_dir: Path,
    *,
    tree: Any,
    details: dict[str, Any],
    summary: dict[str, Any],
    fluid: str,
    implementation: str,
) -> Path:
    """Save one solved tree without constructing CSV row dictionaries."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite array evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    segment_count = int(getattr(tree, "segment_count", 0) or 0)
    tree_data = np.asarray(tree.data[:segment_count])
    arrays: dict[str, np.ndarray] = {
        "segment_id": np.arange(segment_count, dtype=np.int64),
        "left_child_id": _topology_array(tree_data[:, 15]),
        "right_child_id": _topology_array(tree_data[:, 16]),
        "parent_id": _topology_array(tree_data[:, 17]),
        "terminal_mask": (
            np.isnan(tree_data[:, 15]) & np.isnan(tree_data[:, 16])
        ).astype(np.uint8, copy=False),
        "starts_cm": np.asarray(details.get("starts", tree_data[:, 0:3])),
        "ends_cm": np.asarray(details.get("ends", tree_data[:, 3:6])),
        "radii_cm": np.asarray(details.get("radii", tree_data[:, 21])),
        "lengths_cm": np.asarray(details.get("lengths", tree_data[:, 20])),
        "flows_cm3_s": np.asarray(details.get("flows", np.empty((0,), dtype=float))),
        # Both implementations solve Kirchhoff pressure in dyn/cm^2.
        "pressures_pa": np.asarray(details.get("pressures", np.empty((0,), dtype=float))) * 0.1,
        "cin": np.asarray(details.get("cin", np.empty((0,), dtype=float))),
        "cout": np.asarray(details.get("cout", np.empty((0,), dtype=float))),
        "discharge_hematocrit": np.asarray(
            details.get("discharge_hematocrit", np.empty((0,), dtype=float))
        ),
        "tube_hematocrit": np.asarray(details.get("tube_hematocrit", np.empty((0,), dtype=float))),
        "tissue_points_cm": np.asarray(details.get("tissue_points", np.empty((0, 3), dtype=float))),
        "tissue_values": np.asarray(details.get("tissue_values", np.empty((0,), dtype=float))),
    }

    manifest_arrays: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = output_dir / f"{name}.npy"
        np.save(path, np.asarray(values), allow_pickle=False)
        manifest_arrays[name] = {
            "path": path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "implementation": implementation,
        "fluid": str(fluid),
        "segment_count": segment_count,
        "tree_data_dtype": str(tree_data.dtype),
        "pressure_source_units": "dyn/cm^2",
        "pressure_evidence_units": "Pa",
        "summary": _jsonable(summary),
        "arrays": manifest_arrays,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value
