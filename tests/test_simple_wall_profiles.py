from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cascade.configuration.bridge import load_runtime_module
from cascade.exporting.run import _segment_polydata_from_results
from cascade.simulation.simple import solve_simple_network
from cascade.vessels.simple import SimpleNetwork


def test_lattice_graetz_solve_retains_wall_quadrature(monkeypatch):
    starts = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)
    ends = np.asarray([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    lengths = np.linalg.norm(ends - starts, axis=1)
    network = SimpleNetwork(
        starts=starts,
        ends=ends,
        radii=np.full((2,), 0.01, dtype=float),
        lengths=lengths,
        flows=np.zeros((2,), dtype=float),
        cin=np.zeros((2,), dtype=float),
        cout=np.zeros((2,), dtype=float),
        inlet_nodes=[0],
        outlet_nodes=[2],
        prox_ids=np.asarray([0, 1], dtype=np.int64),
        dist_ids=np.asarray([1, 2], dtype=np.int64),
        data=np.zeros((2, 31), dtype=float),
        parameters=None,
        segment_count=2,
        n_terminals=1,
        mode="lattice",
        decay_starts=starts.copy(),
        decay_ends=ends.copy(),
        tissue_starts=starts.copy(),
        tissue_ends=ends.copy(),
        metadata={
            "flow_ul_min": 1.0,
            "concentration_inlet": 1.0,
            "solve_channels_separately": False,
        },
    )
    runtime = load_runtime_module()
    monkeypatch.setattr(runtime, "LUMEN_WALL_CLOSURE", "graetz")
    monkeypatch.setattr(runtime, "KIRCHHOFF_BC_MODE", "terminal_pressure")
    config = SimpleNamespace(
        network=SimpleNamespace(simple={}),
        simulation=SimpleNamespace(fluid="water"),
    )

    solve_simple_network(network, runtime, config)

    quadrature = network.vessel_quadrature
    assert quadrature is not None
    assert quadrature["c_bulk_gl"].shape == quadrature["c_wall_gl"].shape
    assert quadrature["c_wall_gl"].shape[0] == network.segment_count
    assert np.all(np.isfinite(quadrature["c_wall_gl"]))
    assert np.any(np.abs(quadrature["c_wall_gl"] - quadrature["c_bulk_gl"]) > 0.0)

    polydata = _segment_polydata_from_results(
        [
            SimpleNamespace(
                tree_id=0,
                details={
                    "starts": network.starts,
                    "ends": network.ends,
                    "radii": network.radii,
                    "lengths": network.lengths,
                    "flows": network.flows,
                    "pressures": network.pressures,
                    "cin": network.cin,
                    "cout": network.cout,
                    "vessel_quadrature": quadrature,
                },
            )
        ],
        resolution=8,
        float_dtype=np.dtype(np.float32),
        index_dtype=np.dtype(np.int32),
    )
    assert "wall_oxygen" in polydata.point_data
    assert np.all(np.isfinite(polydata.point_data["wall_oxygen"]))
    assert np.ptp(polydata.point_data["wall_oxygen"]) > 0.0
