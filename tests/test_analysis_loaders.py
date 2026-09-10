from __future__ import annotations

import pickle

import numpy as np

from cascade.svv_adapter import Forest, Tree
from svv.tree.data.data import TreeData


def test_tree_analysis_load_uses_requested_compact_dtypes(tmp_path):
    data = np.arange(3 * 31, dtype=np.float64).reshape(3, 31)
    payload = {
        # Deliberately inconsistent: analysis-only loading must not inflate or
        # trust the potentially enormous pickled growth payload.
        "segment_count": 999,
        "n_terminals": 999,
        "preallocation_step": 10_000,
        "data_dtype": "float64",
        "index_dtype": "int64",
    }
    path = tmp_path / "tiny.tree.npz"
    np.savez_compressed(path, data=data, payload=np.asarray([pickle.dumps(payload)], dtype=object))

    tree = Tree.load(
        str(path),
        data_dtype=np.float32,
        index_dtype=np.int32,
        analysis_only=True,
    )

    assert tree.data.dtype == np.dtype(np.float32)
    assert tree.preallocate is tree.data
    assert tree.segment_count == 3
    assert tree.n_terminals == 2
    assert tree.preallocation_step == 3
    assert tree.preallocate_midpoints.shape == (0, 3)
    assert tree.connectivity is None
    assert tree.kdtm is None
    assert tree.hnsw_tree is None
    assert tree._analysis_only_load is True
    np.testing.assert_allclose(tree.data, data.astype(np.float32))


def test_forest_simcache_streams_into_requested_compact_dtypes(tmp_path):
    source = Forest(n_networks=1, n_trees_per_network=[1], preallocation_step=2)
    source_tree = source.networks[0][0]
    data = np.arange(3 * 31, dtype=np.float64).reshape(3, 31)
    source_tree.data = TreeData.from_array(data)
    source_tree.segment_count = 3
    source_tree.n_terminals = 2
    path = tmp_path / "tiny.forest.simcache"
    source.save_simulation_cache(str(path))

    loaded = Forest.load(
        str(path),
        mode="simulation",
        data_dtype=np.float32,
        index_dtype=np.int32,
    )
    tree = loaded.networks[0][0]

    assert tree.data.dtype == np.dtype(np.float32)
    assert tree.data_dtype == np.dtype(np.float32)
    assert tree.index_dtype == np.dtype(np.int32)
    assert tree.preallocate is tree.data
    assert tree.preallocation_step == 3
    assert tree.preallocate_midpoints.shape == (0, 3)
    assert tree.connectivity is None
    assert tree._analysis_only_load is True
    np.testing.assert_allclose(tree.data, data.astype(np.float32))
