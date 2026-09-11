"""Bounded reusable state for repeated interactive CASCADE runs."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cascade.concentration.tissue.cache import build_tissue_cache_from_tree
from cascade.vessels.build import build_or_load_network


class InteractiveRunWorkspace:
    """Retain one compatible geometry/sample set between serial worker jobs.

    The workspace is deliberately single-entry.  A geometry-affecting settings
    change evicts the previous object graph before the replacement is built, so
    a long GUI session cannot accumulate one tree and spatial index per run.
    """

    def __init__(self) -> None:
        self._key: str | None = None
        self._build = None
        self._tissue_caches: list[dict | None] | None = None
        self._simulation_key: str | None = None
        self._simulation = None

    @property
    def populated(self) -> bool:
        return self._build is not None

    def cache_state(self) -> dict[str, Any]:
        """Expose bounded cache occupancy for diagnostics and soak tests."""
        return {
            "geometry": self._build is not None,
            "tissue_spatial": self._tissue_caches is not None,
            "exact_result": self._simulation is not None,
            "exact_result_array_bytes": _array_bytes(self._simulation),
        }

    def resolve_build(self, config):
        key = _interactive_geometry_key(config)
        if key == self._key and self._build is not None:
            print("Reusing interactive geometry and sample points", flush=True)
            return self._build, True

        self.clear()
        build = build_or_load_network(config)
        self._key = key
        self._build = build
        return build, False

    def resolve_tissue_caches(self, config) -> list[dict | None] | None:
        build = self._build
        if build is None:
            raise RuntimeError("interactive workspace has no network build")
        points = np.asarray(
            build.sample_points
            if build.sample_points is not None
            else np.empty((0, 3)),
            dtype=float,
        )
        if (
            not points.size
            or config.simulation.geometry_only
            or config.simulation.skip_tissue_oxygen
            or (
                config.simulation.external_field.enabled
                and config.simulation.external_field.scope == "shared"
            )
        ):
            return None
        if self._tissue_caches is None:
            self._tissue_caches = [
                None
                if getattr(tree, "_cascade_simple_network", False)
                else build_tissue_cache_from_tree(tree, points)
                for tree in build.trees
            ]
        else:
            print("Reusing interactive tissue spatial cache", flush=True)
        return self._tissue_caches

    def resolve_simulation(self, config):
        """Return the last exact result, or evict it before a changed solve."""
        key = _interactive_simulation_key(config)
        if key == self._simulation_key and self._simulation is not None:
            print("Reusing exact interactive simulation result", flush=True)
            return self._simulation, key
        self._simulation = None
        self._simulation_key = None
        return None, key

    def store_simulation(self, config, key: str, simulation) -> None:
        """Retain a lightweight exact result when output shape permits it.

        Solver details contain point fields, quadrature arrays, and references
        back to the tree.  Summary-only replay needs none of those, so keeping
        the original result would pin an entire completed solve in memory.
        """
        outputs = config.outputs
        if (
            outputs.write_points_csv
            or outputs.write_segments_csv
            or outputs.write_paraview
        ):
            return
        from cascade.simulation.engine import SimulationResult, TreeSimulation

        compact_trees = [
            TreeSimulation(
                tree_id=int(item.tree_id),
                tree=None,
                target_count=int(item.target_count),
                summary=dict(item.summary),
                details={},
                elapsed_s=float(item.elapsed_s),
            )
            for item in simulation.tree_results
        ]
        compact = SimulationResult(
            tree_results=compact_trees,
            summary_rows=[dict(row) for row in simulation.summary_rows],
            segment_rows=[],
            point_rows=[],
            sample_points=np.empty((0, 3), dtype=float),
            point_data={},
            sample_meta=dict(simulation.sample_meta),
            timings=dict(simulation.timings),
            interventions=list(simulation.interventions),
            external_field=simulation.external_field,
        )
        limit = max(float(os.environ.get("CASCADE_INTERACTIVE_RESULT_CACHE_MB", "256")), 0.0)
        if _array_bytes(compact) > int(limit * 1024 * 1024):
            return
        self._simulation_key = key
        self._simulation = compact

    def enforce_memory_budget(self, snapshot: dict[str, Any]) -> str | None:
        """Evict reusable state before host RAM headroom becomes dangerous."""
        available = snapshot.get("host_available_bytes")
        total = snapshot.get("host_total_bytes")
        rss = snapshot.get("process_rss_bytes")
        minimum_fraction = min(
            max(
                float(
                    os.environ.get(
                        "CASCADE_INTERACTIVE_MIN_FREE_RAM_FRACTION", "0.15"
                    )
                ),
                0.01,
            ),
            0.90,
        )
        max_rss_mb = max(
            float(os.environ.get("CASCADE_INTERACTIVE_MAX_RSS_MB", "0")), 0.0
        )
        rss_over = bool(max_rss_mb and isinstance(rss, int) and rss > max_rss_mb * 1024 * 1024)
        low = bool(
            isinstance(available, int)
            and isinstance(total, int)
            and total > 0
            and available < minimum_fraction * total
        )
        critical = bool(
            isinstance(available, int)
            and isinstance(total, int)
            and total > 0
            and available < max(0.05, minimum_fraction * 0.5) * total
        )
        if critical:
            self.clear()
            return "all"
        if low or rss_over:
            self._simulation = None
            self._simulation_key = None
            self._tissue_caches = None
            gc.collect()
            return "results-and-spatial-cache"
        return None

    def clear(self) -> None:
        self._tissue_caches = None
        self._simulation = None
        self._simulation_key = None
        self._build = None
        self._key = None
        # A worker can replace a very large tree with another one immediately.
        # Collect here, between dropping the old single-entry cache and
        # constructing its replacement, so both object graphs are never live
        # merely because a cyclic reference has not been collected yet.
        gc.collect()


def _interactive_geometry_key(config) -> str:
    """Hash settings that can change domain, samples, or vessel geometry."""
    payload = deepcopy(config.raw)
    payload.pop("outputs", None)
    payload.pop("gui", None)
    payload.pop("sweep", None)

    network = payload.get("network")
    if isinstance(network, dict):
        network.pop("save_path", None)

    simulation = payload.get("simulation")
    if isinstance(simulation, dict):
        # These are solve-time conditions for SVV trees.  A simple network can
        # inherit its flow from qin, so retain qin in that mode unless the
        # simple geometry declares its own flow explicitly.
        simulation.pop("fluid", None)
        if _qin_is_solve_only(config):
            simulation.pop("qin_target_ul_min", None)
            simulation.pop("qin_target", None)
            simulation.pop("total_qin_ul_min", None)

    payload["_source_files"] = _source_file_signatures(config)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _interactive_simulation_key(config) -> str:
    """Hash all scientific and output-shape settings for exact replay."""
    payload = deepcopy(config.raw)
    payload.pop("gui", None)
    payload.pop("sweep", None)
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        outputs.pop("out_dir", None)
        outputs.pop("prefix", None)
    payload["_source_files"] = _source_file_signatures(config)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _qin_is_solve_only(config) -> bool:
    if config.network_mode != "simple":
        return True
    simple = dict(config.network.simple or {})
    return "flow_ul_min" in simple or "qin_ul_min" in simple


def _source_file_signatures(config) -> dict[str, dict[str, Any]]:
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
        config.settings_path.parent if config.settings_path is not None else Path.cwd()
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


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def _array_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count unique NumPy storage in a small cache candidate."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return _array_bytes(vars(value), seen)
    return 0


__all__ = ["InteractiveRunWorkspace"]

