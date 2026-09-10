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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-tissue-points", type=Path)
    mode.add_argument("--tissue-points", type=Path)
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
