"""Reproducible warm CPU benchmark for the exact dense tissue path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cascade.configuration import solver_state as state
from cascade.concentration.properties import get_concentration_inlet
from cascade.concentration.tissue.geometry import (
    _build_tissue_geometry_context,
    _prepare_tissue_geometry,
)
from cascade.concentration.tissue.greens import compute_tissue_samples_greens


def _elapsed(callable_):
    start = perf_counter()
    result = callable_()
    return perf_counter() - start, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrays_dir", type=Path)
    parser.add_argument("--points", type=int, default=50_000)
    args = parser.parse_args()

    root = args.arrays_dir.expanduser().resolve()
    starts = np.load(root / "starts_cm.npy")
    ends = np.load(root / "ends_cm.npy")
    radii = np.load(root / "radii_cm.npy")
    cin = np.load(root / "cin.npy")
    flows = np.load(root / "flows_cm3_s.npy")
    points = np.load(root / "tissue_points_cm.npy", mmap_mode="r")[: args.points]

    state.TISSUE_ACCEL_MODE = "cpu"
    state.ACTIVE_FLUID = "blood"
    state.NEAREST_TISSUE_VESSELS = max(int(starts.shape[0]), 1)
    state.TISSUE_STREAMING_ENABLED = False
    inlet_concentration = get_concentration_inlet("blood")

    # Compile the Numba implementation outside the measured region.
    warm_context = _build_tissue_geometry_context(starts, ends, radii)
    warm_cache = {"dense_fused_cpu": True, "context": warm_context}
    compute_tissue_samples_greens(
        points[: min(64, len(points))],
        starts,
        ends,
        radii,
        cin,
        flows,
        inlet_concentration=inlet_concentration,
        tissue_cache=warm_cache,
    )
    old_warm_cache = _prepare_tissue_geometry(
        points[: min(64, len(points))],
        starts,
        ends,
        radii,
        max_nearby=int(starts.shape[0]),
    )
    compute_tissue_samples_greens(
        points[: min(64, len(points))],
        starts,
        ends,
        radii,
        cin,
        flows,
        inlet_concentration=inlet_concentration,
        tissue_cache=old_warm_cache,
    )

    old_geometry_s, old_cache = _elapsed(
        lambda: _prepare_tissue_geometry(
            points,
            starts,
            ends,
            radii,
            max_nearby=int(starts.shape[0]),
        )
    )
    old_kernel_s, (old_mask, old_values) = _elapsed(
        lambda: compute_tissue_samples_greens(
            points,
            starts,
            ends,
            radii,
            cin,
            flows,
            inlet_concentration=inlet_concentration,
            tissue_cache=old_cache,
        )
    )

    fused_cache = {
        "dense_fused_cpu": True,
        "context": _build_tissue_geometry_context(starts, ends, radii),
    }
    fused_s, (fused_mask, fused_values) = _elapsed(
        lambda: compute_tissue_samples_greens(
            points,
            starts,
            ends,
            radii,
            cin,
            flows,
            inlet_concentration=inlet_concentration,
            tissue_cache=fused_cache,
        )
    )

    common = old_mask & fused_mask
    diff = old_values[common] - fused_values[common]
    reference_norm = float(np.linalg.norm(old_values[common]))
    report = {
        "points": int(points.shape[0]),
        "segments": int(starts.shape[0]),
        "old_geometry_s": float(old_geometry_s),
        "old_kernel_s": float(old_kernel_s),
        "old_total_s": float(old_geometry_s + old_kernel_s),
        "fused_total_s": float(fused_s),
        "speedup": float((old_geometry_s + old_kernel_s) / fused_s),
        "mask_differences": int(np.count_nonzero(old_mask != fused_mask)),
        "relative_l2": float(
            np.linalg.norm(diff) / max(reference_norm, np.finfo(float).tiny)
        ),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

