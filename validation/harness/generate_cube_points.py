"""Generate the frozen M2 cube tissue-coordinate pool with the legacy sampler."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1_000_000)
    args = parser.parse_args()

    oracle = args.oracle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    actual_oracle_hash = sha256(oracle)
    if actual_oracle_hash != args.oracle_sha256.lower():
        raise RuntimeError(
            f"Oracle hash mismatch for {oracle}: expected {args.oracle_sha256}, got {actual_oracle_hash}"
        )
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if output.exists() or metadata.exists():
        raise FileExistsError("Refusing to overwrite an existing frozen fixture or metadata file")

    spec = importlib.util.spec_from_file_location("cascade_m2_cube_oracle", oracle)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load oracle module spec: {oracle}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    domain = module.build_domain(1.0)
    points = np.asarray(module.sample_domain_points(domain, int(args.count)), dtype=np.float64)
    if points.shape != (int(args.count), 3):
        raise RuntimeError(f"Unexpected sample shape: {points.shape}")
    if not np.isfinite(points).all():
        raise RuntimeError("Frozen tissue coordinates contain non-finite values")

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, points, allow_pickle=False)
    temporary.replace(output)

    record = {
        "fixture": "M2-CUBE-POINTS-1M",
        "generator": "frozen legacy Domain.get_interior_points via TissueSim_cube_local.sample_domain_points",
        "oracle_path": str(oracle),
        "oracle_sha256": actual_oracle_hash,
        "domain": {"type": "cube", "side_length_cm": 1.0, "random_seed": 42},
        "coordinate_units": "cm",
        "shape": list(points.shape),
        "dtype": str(points.dtype),
        "minimum": points.min(axis=0).tolist(),
        "maximum": points.max(axis=0).tolist(),
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
    }
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
