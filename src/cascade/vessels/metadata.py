"""Lightweight metadata for saved vascular networks.

The inspection path reads ZIP and NumPy headers only.  It deliberately avoids
materializing vessel tables or unpickling legacy object graphs, which makes it
safe to call from interactive admission checks before a large archive is
loaded.  New CASCADE archives can also carry a small adjacent sidecar so even
legacy forest containers become exactly countable on their next save.
"""

from __future__ import annotations

import json
import csv
import re
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cascade.utils.files import atomic_write_text

_SIDECAR_FORMAT = "cascade_network_metadata"
_SIDECAR_VERSION = 1
_SIMCACHE_DATA = re.compile(r"^tree_\d+_\d+_data\.npy$")


@dataclass(frozen=True)
class ArrayHeader:
    """Shape and dtype obtained without reading an array payload."""

    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class NetworkMetadata:
    """Small, serializable description of a network archive."""

    path: str
    kind: str
    source_bytes: int
    archive_uncompressed_bytes: int
    vessel_data_bytes: int | None
    segments: int | None
    trees: int | None
    data_dtypes: tuple[str, ...] = ()
    exact_counts: bool = False
    from_sidecar: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dtypes"] = list(self.data_dtypes)
        payload["warnings"] = list(self.warnings)
        return payload


def sidecar_path(path: str | Path) -> Path:
    """Return the non-conflicting sidecar path for a network archive."""
    source = Path(path).expanduser().resolve()
    return source.with_name(source.name + ".cascade.json")


def inspect_network(path: str | Path) -> NetworkMetadata:
    """Inspect a saved tree/forest without loading its scientific arrays."""
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    cached = _read_fresh_sidecar(source, stat.st_size, stat.st_mtime_ns)
    if cached is not None:
        return cached

    if source.suffix.lower() == ".csv":
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"start_x", "start_y", "start_z", "end_x", "end_y", "end_z"}
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                missing = ", ".join(sorted(required - fields))
                return NetworkMetadata(
                    path=str(source),
                    kind="custom-csv",
                    source_bytes=int(stat.st_size),
                    archive_uncompressed_bytes=int(stat.st_size),
                    vessel_data_bytes=None,
                    segments=None,
                    trees=1,
                    warnings=(f"custom CSV is missing columns: {missing}",),
                )
            segments = sum(1 for _row in reader)
        return NetworkMetadata(
            path=str(source),
            kind="custom-csv",
            source_bytes=int(stat.st_size),
            archive_uncompressed_bytes=int(stat.st_size),
            vessel_data_bytes=None,
            segments=segments,
            trees=1,
            exact_counts=True,
        )

    if source.suffix.lower() == ".npy":
        header = read_npy_header(source)
        segments = int(header.shape[0]) if header.shape else None
        return NetworkMetadata(
            path=str(source),
            kind="npy",
            source_bytes=int(stat.st_size),
            archive_uncompressed_bytes=int(stat.st_size),
            vessel_data_bytes=header.nbytes,
            segments=segments,
            trees=1,
            data_dtypes=(header.dtype,),
            exact_counts=segments is not None,
        )

    try:
        with zipfile.ZipFile(source, "r") as archive:
            return _inspect_zip_network(source, archive, int(stat.st_size))
    except zipfile.BadZipFile:
        return NetworkMetadata(
            path=str(source),
            kind="unknown",
            source_bytes=int(stat.st_size),
            archive_uncompressed_bytes=int(stat.st_size),
            vessel_data_bytes=None,
            segments=None,
            trees=None,
            warnings=("input is not a supported NumPy/forest ZIP archive",),
        )


def inspect_array(path: str | Path, names: Iterable[str]) -> ArrayHeader | None:
    """Read a named NPY/NPZ array header without loading its payload."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".npy":
        return read_npy_header(source)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            available = set(archive.namelist())
            for name in names:
                member = name if name.endswith(".npy") else f"{name}.npy"
                if member in available:
                    return _read_member_header(archive, member)
    except (OSError, zipfile.BadZipFile, ValueError):
        return None
    return None


def read_npy_header(path: str | Path) -> ArrayHeader:
    """Read one standalone NPY header."""
    with Path(path).open("rb") as handle:
        return _read_header(handle)


def write_network_sidecar(
    path: str | Path,
    trees: Iterable[Any],
    *,
    kind: str | None = None,
) -> Path:
    """Atomically publish exact counts for a newly saved network archive."""
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    tree_list = list(trees)
    inspected = inspect_network(source)
    counts = [max(int(getattr(tree, "segment_count", 0) or 0), 0) for tree in tree_list]
    dtypes = sorted(
        {
            str(np.asarray(getattr(tree, "data", np.empty((0, 0)))).dtype)
            for tree in tree_list
        }
    )
    vessel_bytes = inspected.vessel_data_bytes
    if vessel_bytes is None:
        vessel_bytes = sum(_tree_array_bytes(tree) for tree in tree_list)
    metadata = NetworkMetadata(
        path=str(source),
        kind=kind or ("forest" if len(tree_list) > 1 else "tree"),
        source_bytes=int(stat.st_size),
        archive_uncompressed_bytes=int(inspected.archive_uncompressed_bytes),
        vessel_data_bytes=int(vessel_bytes),
        segments=int(sum(counts)),
        trees=len(tree_list),
        data_dtypes=tuple(dtypes),
        exact_counts=True,
        from_sidecar=True,
    )
    payload = {
        "format": _SIDECAR_FORMAT,
        "version": _SIDECAR_VERSION,
        "source": {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "network": metadata.to_dict(),
    }
    target = sidecar_path(source)
    atomic_write_text(target, json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _inspect_zip_network(
    source: Path, archive: zipfile.ZipFile, source_bytes: int
) -> NetworkMetadata:
    infos = archive.infolist()
    names = {info.filename for info in infos}
    uncompressed = sum(int(info.file_size) for info in infos)
    warnings: list[str] = []

    if "simulation_meta.pkl" in names:
        kind = "forest-simulation-cache"
        data_members = sorted(name for name in names if _SIMCACHE_DATA.match(name))
    elif "trees.npy" in names and "metadata.npy" in names:
        kind = "legacy-forest"
        data_members = []
        warnings.append(
            "legacy forest has no header-level segment counts; save a CASCADE "
            "sidecar or simulation cache"
        )
    elif "data.npy" in names:
        kind = "tree"
        data_members = ["data.npy"]
    elif "starts.npy" in names and "ends.npy" in names:
        kind = "simple-network"
        data_members = sorted(
            name
            for name in names
            if name.endswith(".npy") and name != "metadata.npy"
        )
    else:
        kind = "npz"
        data_members = sorted(name for name in names if name.endswith("_data.npy"))

    headers: list[ArrayHeader] = []
    for member in data_members:
        try:
            headers.append(_read_member_header(archive, member))
        except (EOFError, OSError, ValueError) as exc:
            warnings.append(f"could not inspect {member}: {exc}")

    exact = bool(data_members) and len(headers) == len(data_members)
    if exact and kind == "simple-network":
        starts_index = data_members.index("starts.npy")
        starts_shape = headers[starts_index].shape
        segments = int(starts_shape[0]) if starts_shape else None
    else:
        segments = (
            sum(int(header.shape[0]) for header in headers if header.shape)
            if exact
            else None
        )
    data_bytes = sum(header.nbytes for header in headers) if exact else None
    trees = (1 if kind == "simple-network" else len(headers)) if exact else None
    return NetworkMetadata(
        path=str(source),
        kind=kind,
        source_bytes=source_bytes,
        archive_uncompressed_bytes=uncompressed,
        vessel_data_bytes=data_bytes,
        segments=segments,
        trees=trees,
        data_dtypes=tuple(sorted({header.dtype for header in headers})),
        exact_counts=exact,
        warnings=tuple(warnings),
    )


def _read_member_header(archive: zipfile.ZipFile, member: str) -> ArrayHeader:
    with archive.open(member, "r") as handle:
        return _read_header(handle)


def _read_header(handle) -> ArrayHeader:
    version = np.lib.format.read_magic(handle)
    shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    return ArrayHeader(
        shape=tuple(int(value) for value in shape),
        dtype=str(np.dtype(dtype)),
        fortran_order=bool(fortran_order),
    )


def _read_fresh_sidecar(
    source: Path, source_bytes: int, source_mtime_ns: int
) -> NetworkMetadata | None:
    target = sidecar_path(source)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("format") != _SIDECAR_FORMAT:
            return None
        if int(payload.get("version", 0)) != _SIDECAR_VERSION:
            return None
        signature = dict(payload.get("source", {}) or {})
        if int(signature.get("size", -1)) != int(source_bytes):
            return None
        if int(signature.get("mtime_ns", -1)) != int(source_mtime_ns):
            return None
        raw = dict(payload.get("network", {}) or {})
        return NetworkMetadata(
            path=str(source),
            kind=str(raw.get("kind", "unknown")),
            source_bytes=int(source_bytes),
            archive_uncompressed_bytes=int(
                raw.get("archive_uncompressed_bytes", source_bytes)
            ),
            vessel_data_bytes=_optional_int(raw.get("vessel_data_bytes")),
            segments=_optional_int(raw.get("segments")),
            trees=_optional_int(raw.get("trees")),
            data_dtypes=tuple(str(value) for value in raw.get("data_dtypes", [])),
            exact_counts=bool(raw.get("exact_counts", False)),
            from_sidecar=True,
            warnings=tuple(str(value) for value in raw.get("warnings", [])),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _tree_array_bytes(tree: Any) -> int:
    """Count unique primary arrays without traversing an arbitrary object graph."""
    arrays: list[np.ndarray] = []
    for name in ("data", "starts", "ends", "radii", "lengths", "flows"):
        value = getattr(tree, name, None)
        if value is not None:
            arrays.append(np.asarray(value))
    seen: set[int] = set()
    total = 0
    for array in arrays:
        pointer = int(array.__array_interface__["data"][0]) if array.size else id(array)
        if pointer not in seen:
            seen.add(pointer)
            total += int(array.nbytes)
    return total


__all__ = [
    "ArrayHeader",
    "NetworkMetadata",
    "inspect_array",
    "inspect_network",
    "read_npy_header",
    "sidecar_path",
    "write_network_sidecar",
]
