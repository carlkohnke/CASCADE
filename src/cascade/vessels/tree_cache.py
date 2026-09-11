"""Vascular-tree cache keys, lookup, loading, and synchronization.

Cache persistence is kept separate from growth and simulation orchestration,
with explicit dependencies for safe direct use.
"""

from __future__ import annotations

import csv
import hashlib
import json
from numbers import Number
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.diagnostics.runtime import _require_tree_class
from cascade.vessels.generation.tree_ops import _apply_equal_bifurcation, set_tree_fluid


def _tree_cache_dir() -> Path:
    root = Path(_state.TREE_CACHE_DIRNAME)
    if root.is_absolute():
        return root
    return Path(__file__).resolve().parent / root


def _tree_cache_index_path() -> Path:
    return _tree_cache_dir() / _state.TREE_CACHE_INDEX_NAME


def _as_float_list(values: Sequence[Number] | np.ndarray | None) -> list[float] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(v) for v in arr]


def _default_tree_params() -> dict:
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._DEFAULT_TREE_PARAMS is None:
        _require_tree_class()
        tree = _state.Tree()
        parms = tree.parameters
        _state._DEFAULT_TREE_PARAMS = {
            "murray_exponent": float(parms.murray_exponent),
            "radius_exponent": float(parms.radius_exponent),
            "length_exponent": float(parms.length_exponent),
            "max_nonconvex_count": int(parms.max_nonconvex_count),
            # "dlp_early_slab_frac": None if parms.dlp_early_slab_frac is None else float(parms.dlp_early_slab_frac),
            # "dlp_early_quota_frac": None if parms.dlp_early_quota_frac is None else float(parms.dlp_early_quota_frac),
        }
    return dict(_state._DEFAULT_TREE_PARAMS)


def _fluid_properties(fluid: str) -> dict[str, float]:
    mode = (fluid or _state.ACTIVE_FLUID).lower()
    if mode in {"water", "cell media", "media"}:
        return {
            "fluid_density": 0.99336,
            "kinematic_viscosity": 0.6959 / 100,
        }
    if mode == "blood":
        mu_plasma_cgs = 0.012
        return {
            "fluid_density": 1.06,
            "kinematic_viscosity": mu_plasma_cgs / 1.06,
        }
    raise ValueError(f"Unknown fluid mode for cache config: {mode}")


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash_config(config: dict) -> str:
    payload = _json_dumps(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _build_tree_cache_config(
    *,
    target_raw: int,
    target_terminals: int,
    side_length: float,
    domain_seed: int | None,
    q_inlet_cm3_s: float,
    terminal_flow_cm3_s: float,
    dlp_enable: bool,
    min_theta: float,
    weighted_sampling: bool,
    ignore_collisions: bool,
    allow_inside_vessels: bool,
    n_closest_vessels: int,
    n_points: int,
) -> tuple[dict, dict, str, str]:
    params = _default_tree_params()
    fluid_props = _fluid_properties(_state.BUILD_FLUID)
    terminal_pressure = (
        float(_state.ROOT_PRESSURE)
        - (abs(_state.ROOT_PRESSURE - _state.TERMINAL_PRESSURE)) * (side_length**3)
        if _state.SCALE_dP_BY_VOLUME
        else float(_state.TERMINAL_PRESSURE)
    )
    base_config = {
        "cube_side_length": float(side_length),
        "domain_shape": "cube",
        "domain_random_seed": int(domain_seed) if domain_seed is not None else None,
        "root_location": _as_float_list(_state.ROOT_LOCATION),
        "root_direction": _as_float_list(_state.ROOT_DIR),
        "root_location_scaled": _as_float_list(_state.ROOT_LOCATION * side_length),
        "root_pressure": float(_state.ROOT_PRESSURE),
        "terminal_pressure": float(terminal_pressure),
        "qin_target_ul_min": float(_state.QIN_TARGET),
        "inlet_flow_cm3_s": float(q_inlet_cm3_s),
        "terminal_flow_cm3_s": float(terminal_flow_cm3_s),
        "scale_nterms_by_volume": bool(_state.SCALE_NTERMS_BY_VOLUME),
        "scale_q_by_volume": bool(_state.SCALE_Q_BY_VOLUME),
        "scale_dp_by_volume": bool(_state.SCALE_dP_BY_VOLUME),
        "build_fluid": str(_state.BUILD_FLUID),
        "build_fluid_density": float(fluid_props["fluid_density"]),
        "build_kinematic_viscosity": float(fluid_props["kinematic_viscosity"]),
        "tree_data_dtype": _state.TREE_DATA_DTYPE_STR,
        "tree_index_dtype": _state.TREE_INDEX_DTYPE_STR,
        # "dlp_enable": bool(dlp_enable),
        # "dlp_min_angle_deg": float(min_theta),
        # "dlp_min_adv": 0.0001,
        # "dlp_build_dir": _as_float_list((-1.0, 1.0, 1.0)),
        # "dlp_parent_below": True,
        # "dlp_build_cap_frac": None,
        # "dlp_early_slab_frac": params["dlp_early_slab_frac"],
        # "dlp_early_quota_frac": params["dlp_early_quota_frac"],
        "weighted_sampling": bool(weighted_sampling),
        "ignore_collisions": bool(ignore_collisions),
        "allow_inside_vessels": bool(allow_inside_vessels),
        "n_closest_vessels": int(n_closest_vessels),
        "n_points": int(n_points),
        "murray_exponent": params["murray_exponent"],
        "radius_exponent": params["radius_exponent"],
        "length_exponent": params["length_exponent"],
        "max_nonconvex_count": params["max_nonconvex_count"],
    }
    full_config = dict(base_config)
    full_config.update(
        {
            "target_raw": int(target_raw),
            "target_terminals": int(target_terminals),
        }
    )
    hash_config = dict(base_config)
    # Tree geometry caches are reusable across inlet-flow targets; QIN is applied
    # during flow/concentration solves after the tree is loaded or grown.
    hash_config.pop("qin_target_ul_min", None)
    hash_config.pop("inlet_flow_cm3_s", None)
    hash_config.pop("terminal_flow_cm3_s", None)
    for key in (
        "n_closest_vessels",
        "n_points",
        "weighted_sampling",
        "allow_inside_vessels",
    ):
        hash_config.pop(key, None)
    # Preserve compatibility with pre-dtype cache config IDs.
    hash_config.pop("tree_data_dtype", None)
    hash_config.pop("tree_index_dtype", None)
    config_id = _hash_config(hash_config)
    legacy_config = dict(base_config)
    legacy_config.pop("qin_target_ul_min", None)
    legacy_config.pop("inlet_flow_cm3_s", None)
    legacy_config.pop("terminal_flow_cm3_s", None)
    for key in (
        "n_closest_vessels",
        "n_points",
        "weighted_sampling",
        "allow_inside_vessels",
    ):
        legacy_config.pop(key, None)
    legacy_config.pop("tree_data_dtype", None)
    legacy_config.pop("tree_index_dtype", None)
    legacy_config_id = _hash_config(legacy_config)
    return base_config, full_config, config_id, legacy_config_id


def _tree_cache_row(
    *,
    config_id: str,
    tree_path: Path,
    base_config: dict,
    full_config: dict,
    target_raw: int,
    target_terminals: int,
) -> dict:
    row = {
        "config_id": config_id,
        "target_terminals": int(target_terminals),
        "target_raw": int(target_raw),
        "tree_path": _tree_cache_relpath(tree_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_json": _json_dumps(full_config),
    }
    for field in _state.TREE_CACHE_FIELDNAMES:
        if field in row:
            continue
        if field in base_config:
            value = base_config[field]
            if isinstance(value, (list, tuple, dict)):
                row[field] = _json_dumps(value)
            elif value is None:
                row[field] = ""
            else:
                row[field] = value
        else:
            row[field] = ""
    return row


def _tree_cache_relpath(path: Path) -> str:
    base = Path(__file__).resolve().parent
    try:
        return path.resolve().relative_to(base).as_posix()
    except Exception:
        return path.as_posix()


def _resolve_tree_cache_path(path_str: str) -> Path:
    normalized = path_str.replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _load_tree_cache_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # Ensure accel uses only cube-domain caches
    filtered = []
    for row in rows:
        shape = (row.get("domain_shape") or "").strip().lower()
        if shape and shape != "cube":
            continue
        filtered.append(row)
    return filtered


def _append_tree_cache_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_state.TREE_CACHE_FIELDNAMES)
    write_header = not path.exists()
    if not write_header:
        try:
            with path.open("r", newline="") as readfile:
                reader = csv.reader(readfile)
                header = next(reader, None)
                if header:
                    fieldnames = header
        except Exception:
            fieldnames = list(_state.TREE_CACHE_FIELDNAMES)
    with path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def _normalize_tree_cache_lookup_config(config: dict, *, legacy: bool) -> dict:
    normalized = dict(config)
    normalized.pop("target_raw", None)
    normalized.pop("target_terminals", None)
    normalized.pop("qin_target_ul_min", None)
    normalized.pop("inlet_flow_cm3_s", None)
    normalized.pop("terminal_flow_cm3_s", None)
    for key in (
        "n_closest_vessels",
        "n_points",
        "weighted_sampling",
        "allow_inside_vessels",
    ):
        normalized.pop(key, None)
    normalized.pop("tree_data_dtype", None)
    normalized.pop("tree_index_dtype", None)
    return normalized


def _tree_cache_row_matches_config_id(
    row: dict[str, str],
    config_id: str,
    *,
    legacy: bool,
) -> bool:
    if row.get("config_id") == config_id:
        return True
    cache_field = "_compat_legacy_config_id" if legacy else "_compat_current_config_id"
    cached_id = row.get(cache_field)
    if cached_id is None:
        config_json = row.get("config_json", "")
        if not config_json:
            row[cache_field] = ""
            return False
        try:
            full_config = json.loads(config_json)
            normalized = _normalize_tree_cache_lookup_config(full_config, legacy=legacy)
            cached_id = _hash_config(normalized)
        except Exception:
            cached_id = ""
        row[cache_field] = cached_id
    return bool(cached_id) and cached_id == config_id


def _find_cached_tree(
    rows: Iterable[dict[str, str]],
    config_id: str,
    target_terminals: int,
) -> tuple[dict[str, str], Path] | None:
    for row in rows:
        if not (
            _tree_cache_row_matches_config_id(row, config_id, legacy=False)
            or _tree_cache_row_matches_config_id(row, config_id, legacy=True)
        ):
            continue
        row_target = _parse_int(row.get("target_terminals"))
        if row_target != int(target_terminals):
            continue
        path_str = row.get("tree_path", "")
        if not path_str:
            continue
        path = _resolve_tree_cache_path(path_str)
        if path.exists():
            return row, path
    return None


def _find_cached_tree_lower(
    rows: Iterable[dict[str, str]],
    config_id: str,
    target_terminals: int,
) -> tuple[dict[str, str], int, Path] | None:
    best = None
    best_target = -1
    best_path = None
    for row in rows:
        if not (
            _tree_cache_row_matches_config_id(row, config_id, legacy=False)
            or _tree_cache_row_matches_config_id(row, config_id, legacy=True)
        ):
            continue
        row_target = _parse_int(row.get("target_terminals"))
        if row_target is None or row_target >= int(target_terminals):
            continue
        path_str = row.get("tree_path", "")
        if not path_str:
            continue
        path = _resolve_tree_cache_path(path_str)
        if not path.exists():
            continue
        if row_target > best_target:
            best = row
            best_target = row_target
            best_path = path
    if best is None or best_path is None:
        return None
    return best, best_target, best_path


def _prepare_loaded_tree(tree: _state.Tree) -> None:
    domain = getattr(tree, "domain", None)
    # Cached trees may be loaded without the Domain/mesh to keep memory usage bounded for large targets.
    if domain is None:
        return
    if domain.mesh is None or getattr(domain, "volume", None) is None:
        domain.build()
    if getattr(domain, "boundary", None) is None:
        domain.get_boundary()
    if domain.mesh is not None:
        prob = domain.mesh.cell_data.get("probability")
        if prob is None:
            prob = domain.mesh.cell_data.get("Normalized_Volume")
        if prob is not None:
            tree.probability = np.array(prob, copy=True)
            domain.cumulative_probability = np.cumsum(tree.probability)


def _sync_loaded_tree_params(
    tree: _state.Tree | None,
    *,
    side_length: float | None,
    terminal_flow_override: float | None,
) -> None:
    if tree is None:
        return
    try:
        params = tree.parameters
    except Exception:
        params = None
    if params is None:
        _require_tree_class()
        params = _state.Tree().parameters
        try:
            tree.parameters = params
        except Exception:
            return

    params.root_pressure = _state.ROOT_PRESSURE
    if _state.SCALE_dP_BY_VOLUME and side_length is not None:
        params.terminal_pressure = _state.ROOT_PRESSURE - (
            abs(_state.ROOT_PRESSURE - _state.TERMINAL_PRESSURE)
        ) * (side_length**3)
        if params.terminal_pressure < 0:
            raise ValueError(
                "Terminal pressure is negative after scaling; aborting run."
            )
    else:
        params.terminal_pressure = _state.TERMINAL_PRESSURE

    defaults = _default_tree_params()
    params.murray_exponent = float(defaults["murray_exponent"])
    params.radius_exponent = float(defaults["radius_exponent"])
    params.length_exponent = float(defaults["length_exponent"])
    params.max_nonconvex_count = int(defaults["max_nonconvex_count"])

    set_tree_fluid(tree, _state.BUILD_FLUID)
    _apply_equal_bifurcation(tree)
    if terminal_flow_override is not None:
        params.terminal_flow = terminal_flow_override


def _ensure_tree_domain(tree: _state.Tree | None, domain: _state.Domain | None) -> None:
    if tree is None or domain is None:
        return
    if getattr(tree, "domain", None) is not None:
        return
    try:
        if hasattr(tree, "set_domain"):
            tree.set_domain(domain)
        else:
            tree.domain = domain
    except Exception:
        return
    _prepare_loaded_tree(tree)


def _fast_tree_path(path: Path) -> Path:
    name = path.name
    if name.endswith(".fast.tree.npz"):
        return path
    if name.endswith(".tree.npz"):
        return path.with_name(name[: -len(".tree.npz")] + ".fast.tree.npz")
    return path.with_suffix(path.suffix + ".fast")


def _infer_loaded_terminal_count(data: np.ndarray) -> int:
    if data.ndim == 2 and data.shape[1] > 16:
        try:
            terminals = np.isnan(data[:, 15]) & np.isnan(data[:, 16])
            count = int(np.count_nonzero(terminals))
            if count > 0:
                return count
        except Exception:
            pass
    return max((int(getattr(data, "shape", (0,))[0]) + 1) // 2, 0)


def _load_tree_analysis_only(path: Path, *, data_dtype, index_dtype) -> _state.Tree:
    """Load an exact cache hit without construction-only spatial indices.

    Analysis needs compact geometry and connectivity, not KDTree, HNSW,
    vessel-map, or preallocation structures. Avoiding those duplicates keeps
    multi-million-segment networks within practical memory bounds.
    """
    _require_tree_class()
    from svv.tree.data.data import TreeData

    resolved_data_dtype = np.dtype(data_dtype)
    resolved_index_dtype = np.dtype(index_dtype)
    with np.load(path, allow_pickle=False) as npz:
        data = npz["data"]
        if data.dtype != resolved_data_dtype:
            data = np.asarray(data, dtype=resolved_data_dtype)
        else:
            data = np.asarray(data)

    tree = _state.Tree(
        preallocation_step=1,
        data_dtype=resolved_data_dtype,
        index_dtype=resolved_index_dtype,
    )
    tree_data = TreeData.from_array(data)
    tree.data = tree_data
    tree.preallocate = tree_data
    tree.segment_count = int(data.shape[0])
    tree.n_terminals = _infer_loaded_terminal_count(data)
    tree.preallocation_step = int(data.shape[0])
    tree.preallocate_midpoints = np.empty((0, 3), dtype=resolved_data_dtype)
    tree.midpoints = tree.preallocate_midpoints
    tree.connectivity = None
    tree.vessel_map = {}
    tree.kdtm = None
    tree.hnsw_tree = None
    tree.hnsw_tree_id = None
    tree.domain = None
    tree.probability = None
    tree._analysis_only_load = True
    return tree


def _load_tree_cached_fast(
    path: Path, *, data_dtype, index_dtype, analysis_only: bool = False
) -> _state.Tree:
    load_path = path
    fast_path = _fast_tree_path(path)
    if _state.TREE_FAST_CACHE_ENABLE and fast_path.exists():
        try:
            if fast_path.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                load_path = fast_path
                print(f"  Fast tree cache: using {fast_path.name}")
        except OSError:
            pass
    if analysis_only:
        print("  Tree load mode: analysis-only (skipping build-time spatial indices)")
        return _load_tree_analysis_only(
            load_path,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
        )
    return _state.Tree.load(
        str(load_path),
        domain_path="",
        data_dtype=data_dtype,
        index_dtype=index_dtype,
    )


__all__ = [
    "_tree_cache_dir",
    "_tree_cache_index_path",
    "_as_float_list",
    "_default_tree_params",
    "_fluid_properties",
    "_json_dumps",
    "_hash_config",
    "_build_tree_cache_config",
    "_tree_cache_row",
    "_tree_cache_relpath",
    "_resolve_tree_cache_path",
    "_load_tree_cache_index",
    "_append_tree_cache_row",
    "_parse_int",
    "_normalize_tree_cache_lookup_config",
    "_tree_cache_row_matches_config_id",
    "_find_cached_tree",
    "_find_cached_tree_lower",
    "_prepare_loaded_tree",
    "_sync_loaded_tree_params",
    "_ensure_tree_domain",
    "_fast_tree_path",
    "_infer_loaded_terminal_count",
    "_load_tree_analysis_only",
    "_load_tree_cached_fast",
]
