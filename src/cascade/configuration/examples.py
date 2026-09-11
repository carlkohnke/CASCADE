"""Canonical configuration examples used by the CLI and documentation."""

from __future__ import annotations

from typing import Any


def example_config() -> dict[str, Any]:
    return {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "target_terminal_count": 1,
            "root": {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
        },
        "growth": {
            "n_closest_vessels": 2,
            "n_points": 50,
            "ignore_collisions": True,
            "allow_inside_vessels": True,
            "n_equal_bifurcations": None,
        },
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "concentration_solver": "topdown",
            "distance_sample_count": 64,
            "sample_mode": "random",
            "tissue_accel": "cpu",
            "geometry_only": False,
        },
        "settings": {
            "kirchhoff": {"solver": "tree", "bc_mode": "legacy_equal_terminal_flow"},
            "hematocrit": {"model": "pries_secomb", "flow_iterations": 2},
            "oxygen": {
                "finite_radius_o2_terms": "both",
                "lumen_wall_closure": "graetz",
            },
            "cext": {"accel_mode": "cpu", "vess_coupling_accel": "anderson"},
            "tissue": {"accel_mode": "cpu", "nearest_vessels": 250},
        },
        "outputs": {
            "out_dir": "cascade_run",
            "prefix": "cube_tree",
            "write_paraview": True,
            "save_network": True,
        },
    }
