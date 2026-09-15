"""Bounded reusable state for repeated interactive CASCADE runs."""

from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np

from cascade.concentration.tissue.cache import build_tissue_cache_from_tree
from cascade.runtime.fingerprints import geometry_fingerprint, simulation_fingerprint
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

    def should_presolve(self, config) -> bool:
        """Return whether an immutable GUI case is small enough to solve ahead.

        Pre-solving is deliberately bounded independently of the result byte
        cap.  This prevents merely queueing a very large production case from
        unexpectedly occupying the accelerator for a long time.
        """
        if self._build is None:
            return False
        max_segments = max(
            int(os.environ.get("CASCADE_INTERACTIVE_PRESOLVE_MAX_SEGMENTS", "100000")),
            0,
        )
        max_points = max(
            int(os.environ.get("CASCADE_INTERACTIVE_PRESOLVE_MAX_POINTS", "1000000")),
            0,
        )
        segments = sum(
            int(getattr(tree, "segment_count", 0) or 0) for tree in self._build.trees
        )
        points = int(
            np.asarray(
                self._build.sample_points
                if self._build.sample_points is not None
                else np.empty((0, 3))
            ).shape[0]
        )
        return segments <= max_segments and points <= max_points

    @staticmethod
    def should_prepare(config) -> bool:
        """Bound implicit preloading before any domain or growth work begins."""
        from cascade.runtime.planning import (
            automatic_growth_limit_reason,
            decide_implicit_preparation,
        )

        if automatic_growth_limit_reason(config) is not None:
            return False
        # Raw-only configuration doubles can still exercise the inexpensive
        # growth guard without fabricating every typed configuration section.
        required = ("domain", "network", "simulation")
        if not all(hasattr(config, name) for name in required):
            return True
        return decide_implicit_preparation(config).allowed

    def resolve_build(self, config):
        key = geometry_fingerprint(config)
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
        key = simulation_fingerprint(config)
        if key == self._simulation_key and self._simulation is not None:
            print("Reusing exact interactive simulation result", flush=True)
            return self._simulation, key
        self._simulation = None
        self._simulation_key = None
        return None, key

    def store_simulation(
        self, config, key: str, simulation, *, allow_detailed: bool = False
    ) -> bool:
        """Retain an exact result when its array payload fits the byte cap.

        Solver details contain point fields, quadrature arrays, and references
        back to the tree.  Normal batch runs compact summary-only results;
        Studio may retain a bounded detailed result prepared for immediate VTK
        export, but only when explicitly requested by its preload path.
        """
        outputs = config.outputs
        detailed_outputs = (
            outputs.write_points_csv
            or outputs.write_segments_csv
            or outputs.write_paraview
        )
        if detailed_outputs:
            if not allow_detailed:
                return False
            compact = simulation
        else:
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
        limit = max(
            float(os.environ.get("CASCADE_INTERACTIVE_RESULT_CACHE_MB", "256")), 0.0
        )
        if _array_bytes(compact) > int(limit * 1024 * 1024):
            return False
        self._simulation_key = key
        self._simulation = compact
        return True

    def enforce_memory_budget(self, snapshot: dict[str, Any]) -> str | None:
        """Evict reusable state before host RAM headroom becomes dangerous."""
        available = snapshot.get("host_available_bytes")
        total = snapshot.get("host_total_bytes")
        rss = snapshot.get("process_rss_bytes")
        minimum_fraction = min(
            max(
                float(
                    os.environ.get("CASCADE_INTERACTIVE_MIN_FREE_RAM_FRACTION", "0.15")
                ),
                0.01,
            ),
            0.90,
        )
        max_rss_mb = max(
            float(os.environ.get("CASCADE_INTERACTIVE_MAX_RSS_MB", "0")), 0.0
        )
        rss_over = bool(
            max_rss_mb and isinstance(rss, int) and rss > max_rss_mb * 1024 * 1024
        )
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
