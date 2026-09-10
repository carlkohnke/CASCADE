"""Run a hash-checked legacy heart exporter while capturing or reusing tissue points.

This wrapper is executed only by the isolated legacy environment. It does not copy
or import the oracle into CASCADE. Arguments after ``--`` are forwarded unchanged
to the frozen exporter.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
from pathlib import Path
import sys

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_points(path: Path) -> np.ndarray:
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected tissue points with shape (N, 3), got {points.shape}")
    points = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("Tissue points contain NaN or infinite coordinates")
    return points


def _resample_cext_state(state: dict, target_order: int) -> dict:
    points = np.asarray(state["gl_points_si"], dtype=np.float32)
    if points.ndim != 3:
        raise ValueError(f"Expected Cext GL points with rank 3, got {points.shape}")
    old_order = int(points.shape[1])
    if old_order == int(target_order):
        return state
    old_nodes, old_weights = np.polynomial.legendre.leggauss(old_order)
    new_nodes, new_weights = np.polynomial.legendre.leggauss(int(target_order))
    old_t = 0.5 * (old_nodes + 1.0)
    new_t = 0.5 * (new_nodes + 1.0)
    vectors = np.asarray(state["segment_vectors"], dtype=np.float32)

    def interpolate(values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != old_order:
            return array
        if old_order == 1:
            return np.repeat(array, int(target_order), axis=1)
        out = np.empty((array.shape[0], int(target_order)), dtype=np.float32)
        for segment_id in range(array.shape[0]):
            out[segment_id] = np.interp(new_t, old_t, array[segment_id])
        return out

    lengths = np.linalg.norm(vectors, axis=1).astype(np.float64, copy=False)
    old_ds = 0.5 * lengths[:, None] * old_weights[None, :]
    new_ds = 0.5 * lengths[:, None] * new_weights[None, :]

    def reweight(values) -> np.ndarray:
        weighted = np.asarray(values, dtype=np.float32)
        density = np.divide(
            weighted,
            old_ds,
            out=np.zeros_like(weighted, dtype=np.float64),
            where=old_ds > 0.0,
        )
        return np.asarray(interpolate(density) * new_ds, dtype=np.float32)

    result = dict(state)
    flow_start = points[:, 0, :] - np.float32(old_t[0]) * vectors
    result["gl_points_si"] = np.asarray(
        flow_start[:, None, :] + new_t[None, :, None] * vectors[:, None, :],
        dtype=np.float32,
    )
    for key in (
        "c_iv_gl",
        "c_bulk_gl",
        "c_wall_gl",
        "c_ext_gl",
        "lambda_iv_gl",
        "k_if_gl",
        "q_line_gl",
    ):
        if key in state:
            result[key] = interpolate(state[key])
    for key in ("q_weighted_gl", "mono2_weight_gl", "dipole2_weight_gl"):
        if key in state:
            result[key] = reweight(state[key])
    result["validation_resampled_from_order"] = old_order
    result["validation_resampled_to_order"] = int(target_order)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-tissue-points", type=Path)
    mode.add_argument("--tissue-points", type=Path)
    parser.add_argument(
        "--independent-tissue-quadrature",
        type=int,
        choices=(5, 9, 20),
        help=(
            "Validation-only correction for the legacy shared-Cext path: resample the "
            "converged GL1 source density onto the requested tissue quadrature before "
            "calling the unmodified legacy GPU tissue evaluator."
        ),
    )
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    oracle = args.oracle.expanduser().resolve()
    actual_hash = _sha256(oracle)
    if actual_hash != args.oracle_sha256.lower():
        raise RuntimeError(
            f"Oracle hash mismatch for {oracle}: expected {args.oracle_sha256}, got {actual_hash}"
        )

    loader = SourceFileLoader("cascade_m2_legacy_heart_oracle", str(oracle))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load oracle module spec: {oracle}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(oracle.parent))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.path and sys.path[0] == str(oracle.parent):
            sys.path.pop(0)
        raise

    original = module._inside_grid_points_chunked
    if args.capture_tissue_points is not None:
        output = args.capture_tissue_points.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite tissue point fixture: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        def capture(domain, boundary, exporter_args):
            points = np.asarray(original(domain, boundary, exporter_args), dtype=np.float64)
            np.save(output, points, allow_pickle=False)
            print(
                f"Captured {points.shape[0]} tissue points at {output} "
                f"(sha256={_sha256(output)})",
                flush=True,
            )
            return points

        module._inside_grid_points_chunked = capture
    else:
        points_path = args.tissue_points.expanduser().resolve()
        points = _load_points(points_path)
        points_hash = _sha256(points_path)

        def fixed_points(_domain, _boundary, _exporter_args):
            print(
                f"Using {points.shape[0]} frozen tissue points from {points_path} "
                f"(sha256={points_hash})",
                flush=True,
            )
            return points

        module._inside_grid_points_chunked = fixed_points

    if args.independent_tissue_quadrature is not None:
        original_compute_tissue = module._compute_tissue
        target_order = int(args.independent_tissue_quadrature)

        def corrected_compute_tissue(ts, cext_ts, combo, domain, exporter_args, **kwargs):
            corrected = dict(combo)
            corrected["cext_state"] = _resample_cext_state(combo["cext_state"], target_order)
            print(
                f"Validation wrapper resampled Cext source state to independent GL{target_order} tissue quadrature",
                flush=True,
            )
            return original_compute_tissue(ts, cext_ts, corrected, domain, exporter_args, **kwargs)

        module._compute_tissue = corrected_compute_tissue

    previous_argv = sys.argv
    sys.argv = [str(oracle), *forwarded]
    try:
        return int(module.main())
    finally:
        sys.argv = previous_argv
        if sys.path and sys.path[0] == str(oracle.parent):
            sys.path.pop(0)


if __name__ == "__main__":
    raise SystemExit(main())
