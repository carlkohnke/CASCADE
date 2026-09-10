"""Record the deterministic 200-cubed bivent3 grid-axis contract without materializing 8M points."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--domain-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    domain_path = args.domain.expanduser().resolve()
    actual_hash = sha256(domain_path)
    if actual_hash != args.domain_sha256.lower():
        raise RuntimeError(
            f"Domain hash mismatch for {domain_path}: expected {args.domain_sha256}, got {actual_hash}"
        )
    if args.size <= 1:
        raise ValueError("--size must be greater than one")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    boundary = pv.read(str(domain_path)).extract_surface()
    if not boundary.is_all_triangles:
        boundary = boundary.triangulate()
    boundary = boundary.clean()
    minimum = np.min(boundary.points, axis=0).astype(np.float64)
    maximum = np.max(boundary.points, axis=0).astype(np.float64)
    axes = [
        np.linspace(minimum[index], maximum[index], int(args.size), dtype=np.float64)
        for index in range(3)
    ]
    combined = np.concatenate(axes)
    record = {
        "fixture": "M2-HEART-GRID-200-CUBED",
        "domain_path": str(domain_path),
        "domain_sha256": actual_hash,
        "coordinate_units": "cm",
        "axis_size": int(args.size),
        "candidate_points": int(args.size) ** 3,
        "ordering": "numpy.meshgrid(indexing='ij'), then C-order reshape",
        "bounds_min": minimum.tolist(),
        "bounds_max": maximum.tolist(),
        "axis_sha256": {name: array_hash(value) for name, value in zip("xyz", axes)},
        "combined_axes_sha256": array_hash(combined),
        "inside_rule": "domain(points) <= -implicit_margin; frozen cases record exact retained coordinates/mask",
        "grid_implementation_sha256": "00a68b9004ee10620666be65171f6277e3548078f473ea29feb8c01c60faf79d",
        "grid_implementation_note": "SHA-256 of the byte-identical grid/filter function blocks in the frozen legacy exporter and CASCADE heart_export at staging time",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
