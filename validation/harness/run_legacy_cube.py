"""Run the frozen cube oracle with a hash-checked shared point array."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--points-sha256", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fluid", choices=("both", "blood", "water"), default="both")
    parser.add_argument("--solver", default="topdown")
    parser.add_argument("--tissue-backend", choices=("cpu", "gpu", "auto"), default="cpu")
    args = parser.parse_args()

    oracle = args.oracle.expanduser().resolve()
    points_path = args.points.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    expected = {
        oracle: args.oracle_sha256.lower(),
        points_path: args.points_sha256.lower(),
    }
    for path, expected_hash in expected.items():
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f"Hash mismatch for {path}: expected {expected_hash}, got {actual}")
    if output.exists() or metadata.exists():
        raise FileExistsError("Refusing to overwrite existing legacy result evidence")

    points = np.load(points_path, mmap_mode="r", allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Shared points must have shape (N, 3), got {points.shape}")

    spec = importlib.util.spec_from_file_location(oracle.stem, oracle)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load oracle module spec: {oracle}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(oracle.parent))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(oracle.parent):
            sys.path.pop(0)
    module.FLUID = args.fluid

    def shared_points(_domain, n_points: int) -> np.ndarray:
        if int(n_points) != int(points.shape[0]):
            raise RuntimeError(f"Oracle requested {n_points} points; frozen fixture has {points.shape[0]}")
        return np.asarray(points)

    module.sample_domain_points = shared_points
    unique_name = f"cascade_m2_{uuid.uuid4().hex}.csv"
    transient = oracle.parent / unique_name
    cli_args = [
        str(oracle),
        "--target-counts", str(int(args.target)),
        "--distance-sample-count", str(int(points.shape[0])),
        "--compute-avg-distance-to-channel", "false",
        "--concentration-solver", str(args.solver),
        "--finite-radius-o2-terms", "none" if args.solver == "topdown" else "both",
        "--lumen-wall-closure", "wellmixed" if args.solver == "topdown" else "graetz",
        "--gl-order", "5",
        "--cext-gl-order", "1",
        "--cext-hybrid-bg-grid", "256",
        "--cext-vess-coupling-max-iter", "1",
        "--cext-window-factor", "6",
        "--window-factor", "6",
        "--kirchhoff-solver", "tree",
        "--kirchhoff-bc-mode", "legacy_equal_terminal_flow",
        "--tree-fast-cache", "false",
        "--tissue-accel", args.tissue_backend,
        "--tissue-gpu-validate-points", "0",
        "--n-equal-bifurcations", "-1",
        "--output-csv", unique_name,
    ]
    old_argv = sys.argv
    try:
        sys.argv = cli_args
        module.main()
        if not transient.is_file():
            raise RuntimeError(f"Legacy oracle did not create {transient}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(transient), str(output))
    finally:
        sys.argv = old_argv
        if transient.exists():
            transient.unlink()

    record = {
        "oracle_path": str(oracle),
        "oracle_sha256": expected[oracle],
        "points_path": str(points_path),
        "points_sha256": expected[points_path],
        "points": int(points.shape[0]),
        "target_terminals": int(args.target),
        "fluid": args.fluid,
        "solver": args.solver,
        "tissue_backend": args.tissue_backend,
        "effective_arguments": cli_args[1:],
        "output_path": str(output),
        "output_sha256": sha256(output),
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
