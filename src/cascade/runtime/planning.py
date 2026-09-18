"""Scale-aware resource planning shared by CLI and Studio.

Planning must remain cheap: this module inspects configuration values and file
headers only.  It never imports the numerical runtime, constructs a domain, or
loads a vascular object graph.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cascade.utils.resources import resolve_path
from cascade.vessels.metadata import NetworkMetadata, inspect_array, inspect_network


@dataclass(frozen=True)
class RunResourceEstimate:
    """Conservative, metadata-only estimate used for admission and UI hints."""

    tier: str
    segments: int | None
    tissue_points: int | None
    source_bytes: int
    estimated_host_bytes: int
    network: NetworkMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.network is not None:
            payload["network"] = self.network.to_dict()
        return payload


@dataclass(frozen=True)
class PreparationDecision:
    """Whether Studio may prepare a case opportunistically in the background."""

    allowed: bool
    reason: str
    estimate: RunResourceEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "estimate": self.estimate.to_dict(),
        }


def estimate_run_resources(config) -> RunResourceEstimate:
    """Estimate case scale using settings and archive headers only."""
    base_dir = config.settings_path.parent if config.settings_path else None
    network_meta = _network_metadata(config, base_dir)
    segments = _segment_count(config, network_meta)
    points, sample_bytes = _sample_count(config, base_dir)
    domain_bytes = _source_size(resolve_path(config.domain.path, base_dir=base_dir))
    network_bytes = network_meta.source_bytes if network_meta is not None else 0
    source_bytes = int(network_bytes + sample_bytes + domain_bytes)

    # A loaded tree holds its vessel table plus compact topology and solver
    # working arrays.  Tissue fields need multiple scalar/vector buffers.  The
    # multipliers are intentionally conservative admission estimates, not a
    # promise of exact peak RSS.
    vessel_data = (
        network_meta.vessel_data_bytes
        if network_meta is not None and network_meta.vessel_data_bytes is not None
        else (segments or 0) * 31 * 8
    )
    vessel_working = int(vessel_data * 2.25 + (segments or 0) * 64)
    tissue_working = int((points or 0) * 3 * 8 + (points or 0) * 96)
    host_bytes = int(source_bytes + vessel_working + tissue_working)
    tier = classify_scale(segments, points, source_bytes)
    return RunResourceEstimate(
        tier=tier,
        segments=segments,
        tissue_points=points,
        source_bytes=source_bytes,
        estimated_host_bytes=host_bytes,
        network=network_meta,
    )


def decide_implicit_preparation(config) -> PreparationDecision:
    """Apply explicit, environment-tunable limits to background preparation."""
    estimate = estimate_run_resources(config)
    growth_reason = automatic_growth_limit_reason(config)
    if growth_reason is not None:
        return PreparationDecision(False, growth_reason, estimate)

    max_segments = _env_int("CASCADE_INTERACTIVE_PREPARE_MAX_SEGMENTS", 100_000)
    if estimate.segments is not None and estimate.segments > max_segments:
        return PreparationDecision(
            False,
            f"network exceeds automatic limit ({max_segments:,} segments)",
            estimate,
        )

    max_points = _env_int("CASCADE_INTERACTIVE_PREPARE_MAX_POINTS", 1_000_000)
    if estimate.tissue_points is not None and estimate.tissue_points > max_points:
        return PreparationDecision(
            False,
            f"tissue sampling exceeds automatic limit ({max_points:,} points)",
            estimate,
        )

    max_source_bytes = _env_megabytes(
        "CASCADE_INTERACTIVE_PREPARE_MAX_SOURCE_MB", 512.0
    )
    if estimate.source_bytes > max_source_bytes:
        return PreparationDecision(
            False,
            "inputs exceed automatic read limit "
            f"({max_source_bytes / 1024**2:,.0f} MiB)",
            estimate,
        )

    max_host_bytes = _env_megabytes(
        "CASCADE_INTERACTIVE_PREPARE_MAX_ESTIMATED_RAM_MB", 2_048.0
    )
    if estimate.estimated_host_bytes > max_host_bytes:
        return PreparationDecision(
            False,
            "estimated RAM exceeds automatic limit "
            f"({max_host_bytes / 1024**2:,.0f} MiB)",
            estimate,
        )

    if (
        estimate.network is not None
        and estimate.network.kind == "legacy-forest"
        and estimate.network.segments is None
    ):
        unknown_limit = _env_megabytes(
            "CASCADE_INTERACTIVE_PREPARE_MAX_UNKNOWN_FOREST_MB", 128.0
        )
        if estimate.network.source_bytes > unknown_limit:
            return PreparationDecision(
                False,
                "large legacy forest has no metadata-only segment count",
                estimate,
            )

    return PreparationDecision(True, "within automatic preparation limits", estimate)


def automatic_growth_limit_reason(config) -> str | None:
    """Return a skip reason using only the raw network/growth dictionaries."""
    raw = config.raw
    network = dict(raw.get("network", {}) or {})
    growth = dict(raw.get("growth", {}) or {})

    maximum_terminals = _env_int("CASCADE_INTERACTIVE_PREPARE_MAX_TERMINALS", 500)
    targets = network.get("target_terminal_counts")
    if not isinstance(targets, list):
        targets = [
            network.get(
                "target_total_terminal_count",
                network.get("target_terminal_count"),
            )
        ]
    requested = [int(value) for value in targets if value is not None]
    if (
        bool(growth.get("enabled", False))
        and requested
        and max(requested) > maximum_terminals
    ):
        return (
            "growth count exceeds automatic limit "
            f"({maximum_terminals:,} additions)"
        )
    return None


def classify_scale(
    segments: int | None, tissue_points: int | None, source_bytes: int = 0
) -> str:
    """Map a case onto stable cross-interface scale tiers."""
    s = segments if segments is not None else -1
    p = tissue_points if tissue_points is not None else -1
    if s < 0:
        if source_bytes >= 8 * 1024**3:
            return "heart"
        if source_bytes >= 512 * 1024**2:
            return "very-large"
        return "unknown"
    if s <= 1_000 and p <= 100_000:
        return "instant"
    if s <= 100_000 and p <= 1_000_000:
        return "interactive"
    if s <= 2_000_000:
        return "large"
    if s <= 10_000_000:
        return "very-large"
    return "heart"


def _network_metadata(config, base_dir: Path | None) -> NetworkMetadata | None:
    value = config.network.input_path
    simple = dict(config.network.simple or {})
    if (
        value is None
        and config.network_mode == "simple"
        and str(simple.get("mode", "")).strip().lower() == "custom"
    ):
        value = simple.get("path", simple.get("geometry_path"))
    path = resolve_path(value, base_dir=base_dir)
    if path is None or not path.is_file():
        return None
    try:
        return inspect_network(path)
    except OSError:
        return None


def _segment_count(config, metadata: NetworkMetadata | None) -> int | None:
    if metadata is not None:
        return metadata.segments
    if config.network_mode == "simple":
        simple = dict(config.network.simple or {})
        starts = simple.get("starts")
        if isinstance(starts, list):
            return len(starts)
        return None
    n_trees = max(len(config.network.roots), 1)
    targets = list(config.network.target_terminal_counts or [])
    if targets:
        if len(targets) == 1 and n_trees > 1:
            targets *= n_trees
        return sum(max(2 * int(target) + 1, 1) for target in targets)
    if config.network.target_total_terminal_count is not None:
        total = max(int(config.network.target_total_terminal_count), n_trees)
        return max(2 * total + n_trees, n_trees)
    if config.network.target_terminal_count is not None:
        target = max(int(config.network.target_terminal_count), 1)
        return n_trees * max(2 * target + 1, 1)
    return None


def _sample_count(config, base_dir: Path | None) -> tuple[int | None, int]:
    simulation = config.simulation
    if simulation.geometry_only or simulation.skip_tissue_oxygen:
        return 0, 0
    mode = str(simulation.sample_mode).strip().lower()
    if mode == "grid":
        grid = dict(simulation.tissue_grid or {})
        return (
            max(int(grid.get("nx", 200)), 1)
            * max(int(grid.get("ny", 200)), 1)
            * max(int(grid.get("nz", 200)), 1),
            0,
        )
    if mode == "file":
        path = resolve_path(simulation.sample_points_path, base_dir=base_dir)
        if path is None:
            return None, 0
        size = _source_size(path)
        header = inspect_array(path, ("points", "sample_points"))
        if header is not None and header.shape:
            return int(header.shape[0]), size
        requested = int(simulation.distance_sample_count)
        return (requested if requested > 0 else None), size
    return max(int(simulation.distance_sample_count), 0), 0


def _source_size(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _env_int(name: str, default: int) -> int:
    return max(int(os.environ.get(name, str(default))), 0)


def _env_megabytes(name: str, default: float) -> int:
    return max(int(float(os.environ.get(name, str(default))) * 1024 * 1024), 0)


__all__ = [
    "PreparationDecision",
    "RunResourceEstimate",
    "automatic_growth_limit_reason",
    "classify_scale",
    "decide_implicit_preparation",
    "estimate_run_resources",
]
