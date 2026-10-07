"""Bounded, portable storage of exact-matrix AMG preparation results."""

from __future__ import annotations

from hashlib import blake2b
import json
from types import SimpleNamespace
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
import scipy
import scipy.sparse as sp

from cascade.runtime.paths import cache_directory


def _path(matrix, options, version):
    digest = blake2b(digest_size=20)
    digest.update(
        json.dumps(
            {
                "schema": 1,
                "pyamg": version,
                "scipy": scipy.__version__,
                "options": options,
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return cache_directory("pressure-hierarchies") / (digest.hexdigest() + ".npz")


def prepare_hierarchy(matrix, options, *, builder, version, use_cache):
    """Cache numerical transfers only; each solve still validates its residual."""
    if not use_cache:
        return builder(matrix, **options), False
    path = _path(matrix, options, version)
    try:
        with np.load(path, allow_pickle=False) as archive:
            count = int(archive["count"])
            if not 1 <= count <= 32:
                raise ValueError("Invalid hierarchy depth")
            levels = []
            for index in range(count):
                level = SimpleNamespace()
                for name in ("A", "P", "R") if index < count - 1 else ("A",):
                    prefix = f"{index}_{name}_"
                    shape = tuple(archive[prefix + "shape"].tolist())
                    value = sp.csr_matrix(
                        (
                            archive[prefix + "data"],
                            archive[prefix + "indices"],
                            archive[prefix + "indptr"],
                        ),
                        shape=shape,
                    )
                    value.check_format(full_check=True)
                    if not np.all(np.isfinite(value.data)):
                        raise ValueError("Nonfinite hierarchy")
                    setattr(level, name, value)
                levels.append(level)
            for index, level in enumerate(levels):
                if level.A.shape[0] != level.A.shape[1] or np.any(
                    level.A.diagonal() <= 0.0
                ):
                    raise ValueError("Invalid hierarchy operator")
                if index < count - 1 and (
                    level.P.shape != (level.A.shape[0], levels[index + 1].A.shape[0])
                    or level.R.shape != level.P.shape[::-1]
                ):
                    raise ValueError("Invalid hierarchy transfers")
            if levels[0].A.shape != matrix.shape or any(
                not np.array_equal(a, b)
                for a, b in zip(
                    (levels[0].A.indptr, levels[0].A.indices, levels[0].A.data),
                    (matrix.indptr, matrix.indices, matrix.data),
                )
            ):
                raise ValueError("Hierarchy matrix mismatch")
        return SimpleNamespace(levels=levels), True
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        pass
    hierarchy = builder(matrix, **options)
    arrays = {"count": np.asarray(len(hierarchy.levels))}
    for index, level in enumerate(hierarchy.levels):
        for name in ("A", "P", "R"):
            if not hasattr(level, name):
                continue
            value = getattr(level, name).tocsr()
            prefix = f"{index}_{name}_"
            arrays.update(
                {
                    prefix + "shape": np.asarray(value.shape),
                    prefix + "data": value.data,
                    prefix + "indices": value.indices,
                    prefix + "indptr": value.indptr,
                }
            )
    temporary = path.with_name(path.stem + "-" + uuid4().hex + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(path)
        entries = sorted(
            path.parent.glob("*.npz"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        retained_bytes = 0
        for index, entry in enumerate(entries):
            retained_bytes += entry.stat().st_size
            if entry != path and (index >= 8 or retained_bytes > 512 * 1024**2):
                entry.unlink(missing_ok=True)
    except OSError:
        # An unavailable cache must not prevent a valid simulation.
        pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return hierarchy, False
