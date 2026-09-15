from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from cascade.vessels.metadata import inspect_network, read_npy_header
from cascade.vessels.prepared import prepare_tree_archive


def test_npz_network_inspection_uses_public_numpy_header_api(tmp_path: Path) -> None:
    path = tmp_path / "saved.tree.npz"
    np.savez(path, data=np.zeros((3, 31), dtype=np.float64))

    metadata = inspect_network(path)

    assert metadata.kind == "tree"
    assert metadata.segments == 3
    assert metadata.vessel_data_bytes == 3 * 31 * 8
    assert metadata.data_dtypes == ("float64",)
    assert metadata.exact_counts


def test_numpy_v3_unicode_header_is_inspected(tmp_path: Path) -> None:
    path = tmp_path / "unicode-header.npy"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        np.save(path, np.zeros(2, dtype=[("radius_μ", "<f8")]))

    header = read_npy_header(path)

    assert header.shape == (2,)
    assert header.nbytes == 16
    assert header.dtype == "|V8"


def test_tree_archive_prepares_without_private_numpy_api(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "saved.tree.npz"
    data = np.zeros((3, 31), dtype=np.float64)
    data[:, 15:20] = -1
    np.savez(source, data=data)
    monkeypatch.setenv("CASCADE_PREPARED_CACHE_MIN_FREE_GB", "0")

    prepared = prepare_tree_archive(
        source,
        data_dtype=np.float64,
        index_dtype=np.int64,
        cache_root=tmp_path / "prepared cache",
    )

    assert np.load(prepared.data, mmap_mode="r").shape == (3, 31)
    assert np.load(prepared.connectivity, mmap_mode="r").shape == (3, 3)
    assert np.load(prepared.node_ids, mmap_mode="r").shape == (3, 2)
