from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from cascade.vessels.metadata import inspect_network, read_npy_header


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
