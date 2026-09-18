from __future__ import annotations

import json
import warnings

import pytest

from cascade.configuration import parse_config
from cascade.vessels.build import build_or_load_network
from cascade.vessels.conditions import terminal_flow_for_target
from cascade.vessels.growth import _save_growth_checkpoint


@pytest.mark.parametrize(
    ("growth_count", "terminal_leaves", "segments"), [(1, 2, 3), (2, 3, 5)]
)
def test_real_svv_growth_uses_two_n_plus_one_segments(
    tmp_path, growth_count, terminal_leaves, segments
):
    config = parse_config(
        {
            "domain": {
                "type": "cube",
                "side_length": 1.0,
                "use_cache": False,
                "random_seed": 42,
            },
            "network": {
                "mode": "tree",
                "target_terminal_count": growth_count,
                "root": {
                    "start": [0.49, -0.49, -0.49],
                    "direction": [-0.49, 0.49, 0.49],
                },
            },
            "growth": {
                "n_closest_vessels": 2,
                "n_points": 20,
                "ignore_collisions": True,
                "allow_inside_vessels": True,
            },
            "simulation": {
                "fluid": "water",
                "build_fluid": "water",
                "qin_target_ul_min": 2.0,
                "distance_sample_count": 0,
                "geometry_only": True,
            },
            "outputs": {
                "out_dir": str(tmp_path / "run"),
                "write_paraview": False,
                "save_network": False,
            },
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tree = build_or_load_network(config).trees[0]

    assert tree.n_terminals == terminal_leaves
    assert tree.segment_count == segments
    assert tree.preallocate.shape[0] == 16
    assert terminal_flow_for_target(config, 1.0, growth_count) == pytest.approx(
        1.0 / terminal_leaves
    )
    assert not any("cannot handle constraints" in str(item.message) for item in caught)


def test_growth_checkpoint_distinguishes_additions_from_terminal_leaves(tmp_path):
    checkpoint = tmp_path / "growth.tree.npz"
    config = parse_config(
        {
            "network": {"target_terminal_count": 2},
            "growth": {"checkpoint_path": str(checkpoint)},
        }
    )

    class FakeTree:
        n_terminals = 3
        vessel_map = {}

        def save(self, path, *, include_domain):
            assert path == str(checkpoint)
            assert include_domain is False
            checkpoint.touch()

    _save_growth_checkpoint(
        config,
        [FakeTree()],
        None,
        completed=2,
        total_add=2,
        targets=[2],
        final=True,
    )

    metadata = json.loads(
        checkpoint.with_name(checkpoint.name + ".json").read_text(encoding="utf-8")
    )
    assert metadata["growth_counts"] == [2]
    assert metadata["terminal_segments"] == [3]
