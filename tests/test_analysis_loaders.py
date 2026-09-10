from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from cascade.svv_adapter import Forest, Tree
from svv.forest.connect.forest_connection import ForestConnection
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

    roundtrip = tmp_path / "analysis-roundtrip.tree.npz"
    tree.save(str(roundtrip))
    assert roundtrip.is_file()


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
    np.testing.assert_array_equal(tree.connectivity, data[:, 15:18].astype(np.int32))
    np.testing.assert_array_equal(tree._cascade_node_ids, data[:, 18:20].astype(np.int32))
    assert tree._analysis_only_load is True
    np.testing.assert_allclose(tree.data, data.astype(np.float32))


def test_forest_simcache_preserves_ids_above_float32_exact_range(tmp_path):
    source = Forest(n_networks=1, n_trees_per_network=[1], preallocation_step=2)
    source_tree = source.networks[0][0]
    data = np.zeros((2, 31), dtype=np.float64)
    data[:, 15:18] = [
        [16_777_217, 16_777_219, np.nan],
        [np.nan, np.nan, 16_777_217],
    ]
    data[:, 18:20] = [
        [16_777_217, 16_777_219],
        [16_777_219, 16_777_221],
    ]
    source_tree.data = TreeData.from_array(data)
    source_tree.segment_count = 2
    source_tree.n_terminals = 1
    path = tmp_path / "large-ids.forest.simcache"
    source.save_simulation_cache(str(path))

    loaded = Forest.load(
        str(path), mode="simulation", data_dtype=np.float32, index_dtype=np.int32
    )
    tree = loaded.networks[0][0]

    assert int(tree.data[0, 15]) == 16_777_216  # float32 rounds this value
    np.testing.assert_array_equal(
        tree.connectivity,
        [[16_777_217, 16_777_219, -1], [-1, -1, 16_777_217]],
    )
    np.testing.assert_array_equal(
        tree._cascade_node_ids,
        [[16_777_217, 16_777_219], [16_777_219, 16_777_221]],
    )

    second_path = tmp_path / "large-ids-resaved.forest.simcache"
    loaded.save_simulation_cache(str(second_path))
    reloaded = Forest.load(
        str(second_path), mode="simulation", data_dtype=np.float32, index_dtype=np.int32
    ).networks[0][0]
    np.testing.assert_array_equal(reloaded.connectivity, tree.connectivity)
    np.testing.assert_array_equal(reloaded._cascade_node_ids, tree._cascade_node_ids)


def test_forest_growth_load_restores_saved_connections(tmp_path):
    source = Forest(n_networks=1, n_trees_per_network=[1], preallocation_step=2)
    tree = source.networks[0][0]
    data = np.zeros((1, 31), dtype=np.float64)
    data[0, 15:18] = np.nan
    data[0, 18:20] = [0, 1]
    data[0, 20:22] = [1.0, 0.01]
    tree.data = TreeData.from_array(data)
    tree.segment_count = 1
    tree.n_terminals = 1

    source.connections = ForestConnection(source)
    source.connections.tree_connections = [
        SimpleNamespace(
            network_id=0,
            assignments=np.asarray([0], dtype=np.int64),
            vessels=[[np.asarray([0], dtype=np.int64)]],
            lengths=np.asarray([1.0]),
            curve_type="line",
            connected_network=[tree],
        )
    ]
    path = tmp_path / "with-connections.forest"
    source.save(str(path))

    loaded = Forest.load(str(path), mode="growth")

    assert isinstance(loaded.connections, ForestConnection)
    assert len(loaded.connections.tree_connections) == 1
    connection = loaded.connections.tree_connections[0]
    assert connection.network_id == 0
    np.testing.assert_array_equal(connection.assignments, [0])
