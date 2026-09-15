"""Persistent, memory-mappable preparation for large tree archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cascade.runtime.paths import cache_directory
from cascade.vessels.metadata import (
    inspect_network,
    read_npy_header,
    read_npy_header_stream,
)

_FORMAT = "cascade_prepared_tree"
_VERSION = 1


@dataclass(frozen=True)
class PreparedTreeFiles:
    """Validated paths belonging to one dtype-specific prepared tree."""

    root: Path
    data: Path
    connectivity: Path
    node_ids: Path
    manifest: Path
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "data": str(self.data),
            "connectivity": str(self.connectivity),
            "node_ids": str(self.node_ids),
            "manifest": str(self.manifest),
            "reused": bool(self.reused),
        }


def prepare_tree_archive(
    source_path: str | Path,
    *,
    data_dtype: Any,
    index_dtype: Any,
    cache_root: str | Path | None = None,
) -> PreparedTreeFiles:
    """Stream one compressed tree into atomic, memory-mappable arrays."""
    source = Path(source_path).expanduser().resolve()
    data_type = _float_dtype(data_dtype)
    index_type = _index_dtype(index_dtype)
    existing = find_prepared_tree(
        source,
        data_dtype=data_type,
        index_dtype=index_type,
        cache_root=cache_root,
    )
    if existing is not None:
        return replace(existing, reused=True)

    metadata = inspect_network(source)
    if metadata.kind != "tree" or metadata.segments is None:
        raise ValueError(
            "Persistent mmap preparation currently requires a .tree.npz archive."
        )
    root = _entry_root(source, data_type, index_type, cache_root)
    root.mkdir(parents=True, exist_ok=True)
    files = _files(root)
    temporary = {
        "data": root / f"data.{os.getpid()}.tmp.npy",
        "connectivity": root / f"connectivity.{os.getpid()}.tmp.npy",
        "node_ids": root / f"node_ids.{os.getpid()}.tmp.npy",
        "manifest": root / f"manifest.{os.getpid()}.tmp.json",
    }
    required = int(metadata.segments) * (
        31 * data_type.itemsize + 5 * index_type.itemsize
    )
    _require_disk_headroom(root, required)
    try:
        shape = _stream_tree_arrays(
            source,
            temporary["data"],
            temporary["connectivity"],
            temporary["node_ids"],
            data_dtype=data_type,
            index_dtype=index_type,
        )
        stat = source.stat()
        payload = {
            "format": _FORMAT,
            "version": _VERSION,
            "source": {
                "path": str(source),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            },
            "data_dtype": str(data_type),
            "index_dtype": str(index_type),
            "shape": list(shape),
            "files": {
                "data": files.data.name,
                "connectivity": files.connectivity.name,
                "node_ids": files.node_ids.name,
            },
        }
        temporary["manifest"].write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        temporary["data"].replace(files.data)
        temporary["connectivity"].replace(files.connectivity)
        temporary["node_ids"].replace(files.node_ids)
        temporary["manifest"].replace(files.manifest)
    finally:
        for path in temporary.values():
            with suppress(OSError):
                path.unlink()

    prepared = find_prepared_tree(
        source,
        data_dtype=data_type,
        index_dtype=index_type,
        cache_root=cache_root,
    )
    if prepared is None:
        raise RuntimeError("Prepared tree cache failed post-write validation.")
    return prepared


def find_prepared_tree(
    source_path: str | Path,
    *,
    data_dtype: Any,
    index_dtype: Any,
    cache_root: str | Path | None = None,
) -> PreparedTreeFiles | None:
    """Return a fresh, complete cache entry without opening its large arrays."""
    source = Path(source_path).expanduser().resolve()
    data_type = _float_dtype(data_dtype)
    index_type = _index_dtype(index_dtype)
    files = _files(_entry_root(source, data_type, index_type, cache_root))
    try:
        payload = json.loads(files.manifest.read_text(encoding="utf-8"))
        if (
            payload.get("format") != _FORMAT
            or int(payload.get("version", 0)) != _VERSION
        ):
            return None
        stat = source.stat()
        signature = dict(payload.get("source", {}) or {})
        if str(signature.get("path")) != str(source):
            return None
        if int(signature.get("size", -1)) != int(stat.st_size):
            return None
        if int(signature.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            return None
        if str(payload.get("data_dtype")) != str(data_type):
            return None
        if str(payload.get("index_dtype")) != str(index_type):
            return None
        required_files = (files.data, files.connectivity, files.node_ids)
        if not all(path.is_file() for path in required_files):
            return None
        shape = tuple(int(value) for value in payload.get("shape", ()))
        if len(shape) != 2 or shape[1] < 20:
            return None
        expected_headers = (
            (files.data, shape, data_type),
            (files.connectivity, (shape[0], 3), index_type),
            (files.node_ids, (shape[0], 2), index_type),
        )
        for path, expected_shape, expected_dtype in expected_headers:
            header = read_npy_header(path)
            if header.shape != expected_shape:
                return None
            if np.dtype(header.dtype) != expected_dtype:
                return None
        return files
    except (EOFError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def default_prepared_cache_root() -> Path:
    """Resolve the user-overridable persistent-cache directory."""
    override = os.environ.get("CASCADE_PREPARED_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return cache_directory("prepared-trees")


def _entry_root(
    source: Path,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
    cache_root: str | Path | None,
) -> Path:
    stat = source.stat()
    encoded = "\0".join(
        (
            str(source),
            str(int(stat.st_size)),
            str(int(stat.st_mtime_ns)),
            str(data_dtype),
            str(index_dtype),
            str(_VERSION),
        )
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:32]
    root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else default_prepared_cache_root()
    )
    return root / key


def _files(root: Path) -> PreparedTreeFiles:
    return PreparedTreeFiles(
        root=root,
        data=root / "data.npy",
        connectivity=root / "connectivity.npy",
        node_ids=root / "node_ids.npy",
        manifest=root / "manifest.json",
    )


def _stream_tree_arrays(
    source: Path,
    data_path: Path,
    connectivity_path: Path,
    node_ids_path: Path,
    *,
    data_dtype: np.dtype,
    index_dtype: np.dtype,
) -> tuple[int, int]:
    with zipfile.ZipFile(source, "r") as archive:
        if "data.npy" not in archive.namelist():
            raise ValueError("Tree archive does not contain data.npy.")
        with archive.open("data.npy", "r") as handle:
            shape, fortran_order, source_dtype = read_npy_header_stream(handle)
            source_dtype = np.dtype(source_dtype)
            if source_dtype.hasobject or source_dtype.kind != "f":
                raise ValueError("Tree vessel data must use a floating dtype.")
            if fortran_order or len(shape) != 2 or int(shape[1]) < 20:
                raise ValueError(
                    "Prepared trees require a C-order N-by-20+ vessel table."
                )
            n_rows, n_cols = int(shape[0]), int(shape[1])
            data = np.lib.format.open_memmap(
                data_path, mode="w+", dtype=data_dtype, shape=(n_rows, n_cols)
            )
            connectivity = np.lib.format.open_memmap(
                connectivity_path,
                mode="w+",
                dtype=index_dtype,
                shape=(n_rows, 3),
            )
            node_ids = np.lib.format.open_memmap(
                node_ids_path, mode="w+", dtype=index_dtype, shape=(n_rows, 2)
            )
            rows_per_chunk = max(
                1, (64 * 1024 * 1024) // max(source_dtype.itemsize * n_cols, 1)
            )
            info = np.iinfo(index_dtype)
            for offset in range(0, n_rows, rows_per_chunk):
                count = min(rows_per_chunk, n_rows - offset)
                byte_count = count * n_cols * source_dtype.itemsize
                block = _read_exact(handle, byte_count)
                source_rows = np.frombuffer(
                    block, dtype=source_dtype, count=count * n_cols
                ).reshape((count, n_cols))
                data[offset : offset + count] = source_rows
                raw_ids = source_rows[:, 15:20]
                finite = np.isfinite(raw_ids)
                finite_values = raw_ids[finite]
                if finite_values.size:
                    if np.any(finite_values != np.rint(finite_values)):
                        raise ValueError(
                            "Tree topology contains non-integral identifiers."
                        )
                    outside_range = (
                        np.min(finite_values) < info.min
                        or np.max(finite_values) > info.max
                    )
                    if outside_range:
                        raise OverflowError(
                            f"Tree topology identifiers do not fit {index_dtype}."
                        )
                exact = np.full(raw_ids.shape, -1, dtype=index_dtype)
                exact[finite] = finite_values.astype(index_dtype, copy=False)
                connectivity[offset : offset + count] = exact[:, 0:3]
                node_ids[offset : offset + count] = exact[:, 3:5]
            data.flush()
            connectivity.flush()
            node_ids.flush()
            del data, connectivity, node_ids
            return n_rows, n_cols


def _read_exact(handle, byte_count: int) -> bytes:
    blocks: list[bytes] = []
    remaining = int(byte_count)
    while remaining:
        block = handle.read(min(remaining, 16 * 1024 * 1024))
        if not block:
            raise EOFError("Tree data ended before the declared NPY payload.")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _require_disk_headroom(root: Path, required_bytes: int) -> None:
    usage = shutil.disk_usage(root)
    minimum = int(
        float(os.environ.get("CASCADE_PREPARED_CACHE_MIN_FREE_GB", "5")) * 1024**3
    )
    proportional = min(
        int(0.10 * usage.total),
        max(2 * int(required_bytes), 1024**3),
    )
    reserve = max(minimum, proportional)
    if usage.free - int(required_bytes) < reserve:
        raise OSError(
            "Insufficient free disk space for prepared tree cache: "
            f"need {required_bytes / 1024**3:.2f} GiB plus "
            f"{reserve / 1024**3:.2f} GiB safety reserve at {root}. "
            "Choose a cache directory on a roomier volume with --cache-dir or "
            "adjust CASCADE_PREPARED_CACHE_MIN_FREE_GB."
        )


def _float_dtype(value: Any) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("prepared tree data dtype must be float32 or float64")
    return dtype


def _index_dtype(value: Any) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        raise ValueError("prepared tree index dtype must be int32 or int64")
    return dtype


__all__ = [
    "PreparedTreeFiles",
    "default_prepared_cache_root",
    "find_prepared_tree",
    "prepare_tree_archive",
]
