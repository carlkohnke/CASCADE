"""Forest assembly, target allocation, and boundary conditions."""

from __future__ import annotations

from typing import Any

import numpy as np

from cascade.configuration.schema import RootConfig, RunConfig
from cascade.vessels.connectivity import repair_trees, validate_trees
from cascade.vessels.generation.svv_adapter import Forest

from cascade.configuration.bridge import (
    load_runtime_module,
)


def _make_forest(config: RunConfig, domain, trees: list[Any]):
    starts = [
        [
            np.asarray(root.start, dtype=float).reshape(1, 3)
            for root in config.network.roots
        ]
    ]
    directions = [
        [
            None
            if root.direction is None
            else np.asarray(root.direction, dtype=float).reshape(1, 3)
            for root in config.network.roots
        ]
    ]
    forest = Forest(
        domain=domain,
        n_networks=1,
        n_trees_per_network=[len(trees)],
        start_points=starts,
        directions=directions,
        physical_clearance=float(config.network.physical_clearance),
        compete=False,
        preallocation_step=1,
    )
    forest.networks = [list(trees)]
    forest.domain = domain
    forest.geodesic = None
    return forest


def _target_counts_for_config(
    config: RunConfig, n_trees: int, *, trees: list[Any] | None = None
) -> list[int]:
    if trees and config.growth.add_per_tree:
        adds = [max(int(v), 0) for v in config.growth.add_per_tree]
        if len(adds) == 1 and n_trees > 1:
            adds = adds * n_trees
        if len(adds) != n_trees:
            raise ValueError(
                f"Expected {n_trees} growth.add_per_tree values, got {len(adds)}."
            )
        current = [_tree_terminal_segments(tree) for tree in trees]
        counts = [max(cur + add, 1) for cur, add in zip(current, adds)]
    elif trees and config.growth.add_total is not None:
        current = [_tree_terminal_segments(tree) for tree in trees]
        adds = _split_total_adds(
            int(config.growth.add_total), trees, mode=config.growth.add_split_mode
        )
        counts = [max(cur + add, 1) for cur, add in zip(current, adds)]
    elif config.network.target_terminal_counts:
        counts = [max(int(v), 1) for v in config.network.target_terminal_counts]
    elif config.network.target_total_terminal_count is not None:
        counts = _split_total_target(
            int(config.network.target_total_terminal_count), n_trees, trees=trees
        )
    elif config.network.target_terminal_count is not None:
        counts = [max(int(config.network.target_terminal_count), 1)]
    elif trees:
        counts = [
            max(int(getattr(tree, "n_terminals", 1) or 1) - 1, 1) for tree in trees
        ]
    else:
        counts = [1]

    if len(counts) == 1 and n_trees > 1:
        counts = counts * n_trees
    if len(counts) != n_trees:
        raise ValueError(f"Expected {n_trees} target counts, got {len(counts)}.")
    return counts


def _tree_terminal_segments(tree: Any) -> int:
    return max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 1)


def _tree_root_flow_for_weight(tree: Any) -> float:
    params = getattr(tree, "parameters", None)
    if params is not None:
        value = getattr(params, "root_flow", None)
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    if data.size:
        value = float(data[0, 22])
        if np.isfinite(value) and value > 0.0:
            return value
    return 1.0


def _split_total_adds(total: int, trees: list[Any], *, mode: str) -> list[int]:
    total = max(int(total), 0)
    n_trees = len(trees)
    if n_trees <= 0:
        return []
    if str(mode).strip().lower() == "flow-proportional":
        weights = np.array(
            [_tree_root_flow_for_weight(tree) for tree in trees], dtype=float
        )
    else:
        weights = np.ones(n_trees, dtype=float)
    return _integer_split(total, weights)


def _split_total_target(
    total: int, n_trees: int, *, trees: list[Any] | None
) -> list[int]:
    total = max(int(total), n_trees)
    if trees:
        weights = np.array(
            [_tree_root_flow_for_weight(tree) for tree in trees], dtype=float
        )
    else:
        weights = np.ones(n_trees, dtype=float)
    return [max(v, 1) for v in _integer_split(total, weights)]


def _integer_split(total: int, weights: np.ndarray) -> list[int]:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return []
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    raw = float(total) * weights / float(np.sum(weights))
    base = np.floor(raw).astype(int)
    remainder = int(total - int(np.sum(base)))
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[int(idx)] += 1
    return [int(v) for v in base]


def _repair_and_validate_if_requested(
    trees: list[Any], config: RunConfig
) -> tuple[list[int], list[dict[str, Any]]]:
    repairs: list[int] = []
    reports: list[dict[str, Any]] = []
    if bool(config.network.repair_connectivity):
        repairs = repair_trees(trees)
    if bool(config.network.validate_connectivity):
        reports = validate_trees(
            trees,
            fail=bool(config.network.fail_connectivity),
            geometry_atol=float(config.network.connectivity_geometry_atol),
        )
    return repairs, reports


def _set_runtime_root(ts, root: RootConfig, *, side_length: float) -> None:
    scale = float(side_length) if side_length else 1.0
    start = np.asarray(root.start, dtype=float).reshape(1, 3) / scale
    ts.ROOT_LOCATION = start
    ts.ROOT_DIR = (
        None
        if root.direction is None
        else np.asarray(root.direction, dtype=float).reshape(1, 3)
    )


def _flow_for_tree(config: RunConfig, tree_id: int, n_trees: int) -> float:
    source = str(config.simulation.flow_source).strip().lower().replace("_", "-")
    condition = _inlet_condition_for_tree(config, tree_id)
    if source == "per-inlet" and condition is not None:
        qin_ul_min = float(condition["flow_ul_min"])
        q_scale = float(config.domain.side_length) ** 3
        return qin_ul_min * 1.0e-3 / 60.0 * q_scale
    qin_ul_min = (
        float(config.simulation.total_qin_ul_min)
        if source == "total-qin-split"
        and config.simulation.total_qin_ul_min is not None
        else float(config.simulation.qin_target_ul_min)
    )
    if source in {"total-split", "total-qin-split"} and n_trees > 0:
        qin_ul_min = qin_ul_min / float(n_trees)
    q_scale = float(config.domain.side_length) ** 3
    return qin_ul_min * 1.0e-3 / 60.0 * q_scale


def _terminal_flow_for_target(
    config: RunConfig, qin_cm3_s: float, target_count: int
) -> float | None:
    # Match the legacy scaled-flow convention: target_count is the
    # number passed to n_add, so terminal sinks are target_count + 1.
    if not getattr(load_runtime_module(), "SCALE_Q_BY_VOLUME", True):
        return None
    return float(qin_cm3_s) / float(max(int(target_count), 1) + 1)


def flow_for_tree(config: RunConfig, tree_id: int, n_trees: int) -> float:
    return _flow_for_tree(config, tree_id, n_trees)


def _inlet_condition_for_tree(
    config: RunConfig, tree_id: int
) -> dict[str, float] | None:
    conditions = list(config.simulation.inlet_conditions or [])
    if 0 <= int(tree_id) < len(conditions):
        return conditions[int(tree_id)]
    return None


def pressures_for_tree(config: RunConfig, tree_id: int) -> tuple[float, float]:
    hemo = config.runtime_settings.get(
        "hemodynamics", config.runtime_settings.get("kirchhoff", {})
    )
    root_pressure = float(hemo.get("root_pressure", hemo.get("ROOT_PRESSURE", 66661.0)))
    terminal_pressure = float(
        hemo.get("terminal_pressure", hemo.get("TERMINAL_PRESSURE", 40000.0))
    )
    condition = _inlet_condition_for_tree(config, tree_id)
    if condition is not None:
        root_pressure = float(condition["inlet_pressure_pa"])
        terminal_pressure = float(condition["outlet_pressure_pa"])
    return root_pressure, terminal_pressure


def inlet_concentration_for_tree(config: RunConfig, tree_id: int) -> float | None:
    condition = _inlet_condition_for_tree(config, tree_id)
    if condition is None:
        return None
    return float(condition["inlet_concentration_mmol_l"])


def _set_runtime_tree_conditions(ts, config: RunConfig, tree_id: int) -> None:
    root_pressure, terminal_pressure = pressures_for_tree(config, tree_id)
    ts.ROOT_PRESSURE = root_pressure
    ts.TERMINAL_PRESSURE = terminal_pressure


def sync_tree_parameters_for_run(
    ts,
    tree,
    config: RunConfig,
    tree_id: int,
    terminal_flow_override: float | None,
) -> None:
    _set_runtime_tree_conditions(ts, config, tree_id)
    ts._sync_loaded_tree_params(
        tree,
        side_length=float(config.domain.side_length),
        terminal_flow_override=terminal_flow_override,
    )


def terminal_flow_for_target(
    config: RunConfig, qin_cm3_s: float, target_count: int
) -> float | None:
    return _terminal_flow_for_target(config, qin_cm3_s, target_count)


__all__ = (
    "_make_forest",
    "_target_counts_for_config",
    "_tree_terminal_segments",
    "_tree_root_flow_for_weight",
    "_split_total_adds",
    "_split_total_target",
    "_integer_split",
    "_repair_and_validate_if_requested",
    "_set_runtime_root",
    "_flow_for_tree",
    "_terminal_flow_for_target",
    "flow_for_tree",
    "_inlet_condition_for_tree",
    "pressures_for_tree",
    "inlet_concentration_for_tree",
    "_set_runtime_tree_conditions",
    "sync_tree_parameters_for_run",
    "terminal_flow_for_target",
)
