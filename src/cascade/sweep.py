from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import RunConfig, parse_config
from .export import _summary_fieldnames, _write_csv
from .execution import release_completed_case_memory
from .growth import (
    NetworkBuildResult,
    _build_configured_trees,
    _extend_trees_to_targets,
    _make_forest,
    _pre_sample_points,
    apply_runtime_settings,
    build_domain,
    load_runtime_module,
    resolve_path,
    save_network_if_requested,
)
from .simulation import run_simulation


def run_sweep(settings_path: str | Path) -> dict[str, str]:
    path = Path(settings_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Settings JSON must contain an object at the top level.")

    sweep = dict(raw.get("sweep", {}) or {})
    targets = _target_values(raw, sweep)
    fluids = _fluid_values(raw, sweep)
    side_lengths = _float_values(
        sweep.get("side_lengths", sweep.get("cube_side_lengths")),
        fallback=[_domain_section(raw).get("side_length", _domain_section(raw).get("side_len", 1.0))],
        name="sweep.side_lengths",
    )
    qin_values = _float_values(
        sweep.get("qin_target_ul_min_values", sweep.get("qin_target_values")),
        fallback=[_simulation_section(raw).get("qin_target_ul_min", _simulation_section(raw).get("qin_target", 900.0))],
        name="sweep.qin_target_ul_min_values",
    )
    distance_counts = _int_values(
        sweep.get("distance_sample_counts"),
        fallback=[_simulation_section(raw).get("distance_sample_count", 1000)],
        name="sweep.distance_sample_counts",
    )
    distance_counts = [max(int(v), 0) for v in distance_counts]

    output_csv = _output_csv_path(raw, sweep, path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    work_dir = resolve_path(
        sweep.get("work_dir", sweep.get("out_dir", _outputs_section(raw).get("out_dir", "cascade_sweep"))),
        base_dir=path.parent,
    )
    assert work_dir is not None
    work_dir.mkdir(parents=True, exist_ok=True)

    write_network = _as_bool(sweep.get("save_final_network"), False)
    legacy_columns_only = _as_bool(sweep.get("legacy_columns_only"), False)
    legacy_single_trial_std_nan = _as_bool(sweep.get("legacy_single_trial_std_nan"), False)
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    t_sweep = perf_counter()

    for side_len in side_lengths:
        base_raw = deepcopy(raw)
        base_raw.pop("sweep", None)
        base_raw.setdefault("domain", {})["side_length"] = float(side_len)
        base_raw.setdefault("simulation", {})["distance_sample_count"] = max(distance_counts) if distance_counts else 0
        base_raw.setdefault("outputs", {})["out_dir"] = str(work_dir / f"side_{_label(side_len)}")
        base_raw.setdefault("outputs", {})["save_network"] = bool(write_network)

        build_config = _config_for(path, base_raw, target=targets[0], fluid=fluids[0], qin=qin_values[0])
        ts = load_runtime_module()
        apply_runtime_settings(ts, build_config)

        t0 = perf_counter()
        domain = build_domain(build_config, ts=ts)
        domain_s = perf_counter() - t0

        t0 = perf_counter()
        sample_points, sample_meta = _pre_sample_points(ts, domain, build_config)
        sample_s = perf_counter() - t0
        sample_points = np.asarray(sample_points if sample_points is not None else np.empty((0, 3)), dtype=float)

        trees: list[Any] | None = None
        forest = None
        previous_target = 0

        for target_raw in targets:
            target = max(int(target_raw), 1)
            target_raw_config = deepcopy(base_raw)
            target_raw_config.setdefault("network", {})["target_terminal_count"] = int(target)
            target_raw_config["network"].pop("target_terminal_counts", None)
            target_raw_config["network"].pop("target_counts", None)

            grow_config = _config_for(path, target_raw_config, target=target, fluid=fluids[0], qin=qin_values[0])
            apply_runtime_settings(ts, grow_config)
            grow_config.simulation.distance_sample_count = int(max(distance_counts) if distance_counts else 0)
            grow_config.outputs.out_dir = str(work_dir / f"side_{_label(side_len)}" / f"target_{target:08d}")

            t0 = perf_counter()
            if trees is None:
                trees = _build_configured_trees(ts, domain, grow_config)
                forest = _make_forest(grow_config, domain, trees) if grow_config.network_mode == "forest" else None
            elif target < previous_target:
                raise ValueError("sweep target_terminal_counts must be non-decreasing for in-memory growth reuse.")
            elif target > previous_target:
                _extend_trees_to_targets(ts, trees, domain, grow_config, [target] * len(trees), forest=forest)
            growth_s = perf_counter() - t0
            previous_target = max(previous_target, target)

            if trees is None:
                raise RuntimeError("Sweep failed to build any trees.")

            target_counts = [target] * len(trees)
            build = NetworkBuildResult(
                domain=domain,
                trees=trees,
                forest=forest,
                target_counts=target_counts,
                build_timings={"domain_s": domain_s, "sample_points_s": sample_s, "growth_s": growth_s},
                sample_points=sample_points,
                sample_meta=sample_meta,
            )

            for qin in qin_values:
                for sample_count in distance_counts:
                    points = sample_points[: int(sample_count)] if sample_points.size else sample_points
                    for fluid in fluids:
                        run_raw = deepcopy(target_raw_config)
                        run_raw.setdefault("simulation", {})["fluid"] = str(fluid)
                        run_raw["simulation"]["qin_target_ul_min"] = float(qin)
                        run_raw["simulation"]["distance_sample_count"] = int(sample_count)
                        run_raw.setdefault("outputs", {})["out_dir"] = str(
                            work_dir
                            / f"side_{_label(side_len)}"
                            / f"target_{target:08d}"
                            / f"qin_{_label(qin)}"
                            / f"M_{int(sample_count)}"
                            / str(fluid)
                        )
                        run_raw["outputs"]["save_network"] = bool(write_network)
                        config = _config_for(path, run_raw, target=target, fluid=fluid, qin=qin)
                        config.simulation.distance_sample_count = int(sample_count)
                        apply_runtime_settings(ts, config)

                        t0 = perf_counter()
                        result = run_simulation(domain, trees, target_counts, config, sample_points=points)
                        sim_s = perf_counter() - t0
                        if legacy_single_trial_std_nan:
                            for row in result.summary_rows:
                                for key in list(row.keys()):
                                    if key.endswith("_std"):
                                        row[key] = float("nan")
                        rows.extend(result.summary_rows)
                        timings.append(
                            {
                                "side_length": float(side_len),
                                "target_terminals": int(target),
                                "qin_target_ul_min": float(qin),
                                "distance_sample_count": int(sample_count),
                                "fluid": str(fluid),
                                "growth_s": float(growth_s),
                                "simulation_s": float(sim_s),
                            }
                        )
                        del result
                        release_completed_case_memory(ts)

            if write_network:
                save_network_if_requested(build, grow_config)

    preferred_fields = _summary_fieldnames()
    write_rows = rows
    if legacy_columns_only and preferred_fields:
        write_rows = [{field: row.get(field, "") for field in preferred_fields} for row in rows]
    _write_csv(output_csv, write_rows, preferred_fields=preferred_fields)
    manifest_path = output_csv.with_name(output_csv.stem + "_manifest.json")
    manifest = {
        "settings_path": str(path),
        "output_csv": str(output_csv),
        "row_count": len(rows),
        "targets_requested": targets,
        "fluids": fluids,
        "side_lengths": side_lengths,
        "qin_target_ul_min_values": qin_values,
        "distance_sample_counts": distance_counts,
        "legacy_columns_only": legacy_columns_only,
        "legacy_single_trial_std_nan": legacy_single_trial_std_nan,
        "elapsed_s": perf_counter() - t_sweep,
        "timings": timings,
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, allow_nan=True), encoding="utf-8")
    return {"summary_csv": str(output_csv), "manifest_json": str(manifest_path)}


def _config_for(path: Path, raw: dict[str, Any], *, target: int, fluid: str, qin: float) -> RunConfig:
    data = deepcopy(raw)
    data.setdefault("network", {})["target_terminal_count"] = max(int(target), 1)
    data["network"].pop("target_terminal_counts", None)
    data["network"].pop("target_counts", None)
    data.setdefault("simulation", {})["fluid"] = str(fluid).lower()
    data["simulation"]["qin_target_ul_min"] = float(qin)
    config = parse_config(data)
    config.settings_path = path
    return config


def _target_values(raw: dict[str, Any], sweep: dict[str, Any]) -> list[int]:
    network = _network_section(raw)
    value = sweep.get(
        "target_terminal_counts",
        sweep.get("target_counts", network.get("target_terminal_counts", network.get("target_counts"))),
    )
    if value is None:
        value = network.get("target_terminal_count", network.get("target_count", 1))
    values = _int_values(value, fallback=[1], name="sweep.target_terminal_counts")
    if not values:
        raise ValueError("Sweep requires at least one target_terminal_count.")
    return [int(v) for v in values]


def _fluid_values(raw: dict[str, Any], sweep: dict[str, Any]) -> list[str]:
    value = sweep.get("fluids", _simulation_section(raw).get("fluid", "blood"))
    if isinstance(value, str):
        if value.strip().lower() == "both":
            return ["water", "blood"]
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = str(item).strip().lower()
            if text == "both":
                out.extend(["water", "blood"])
            elif text:
                out.append(text)
        if out:
            return out
    raise ValueError("sweep.fluids must name at least one fluid.")


def _output_csv_path(raw: dict[str, Any], sweep: dict[str, Any], settings_path: Path) -> Path:
    output = sweep.get("output_csv", sweep.get("summary_csv"))
    if output is None:
        outputs = _outputs_section(raw)
        out_dir = resolve_path(outputs.get("out_dir", "cascade_sweep"), base_dir=settings_path.parent)
        assert out_dir is not None
        prefix = outputs.get("prefix") or "cascade_sweep"
        return (out_dir / f"{prefix}.csv").resolve()
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = settings_path.parent / path
    return path.resolve()


def _int_values(value: Any, *, fallback: list[Any], name: str) -> list[int]:
    if value is None:
        value = fallback
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        return [int(float(part.strip())) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(float(v)) for v in value]
    raise ValueError(f"{name} must be an integer, comma-separated string, or list.")


def _float_values(value: Any, *, fallback: list[Any], name: str) -> list[float]:
    if value is None:
        value = fallback
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    raise ValueError(f"{name} must be a number, comma-separated string, or list.")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _domain_section(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("domain", {}) or {})


def _network_section(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("network", {}) or {})


def _simulation_section(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("simulation", {}) or {})


def _outputs_section(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("outputs", {}) or {})


def _label(value: Any) -> str:
    text = f"{float(value):g}" if isinstance(value, (int, float)) else str(value)
    return text.replace("-", "m").replace(".", "p")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
