from __future__ import annotations

import numpy as np
import pytest

from cascade.config import parse_config
from cascade.growth import _load_sample_points, _pre_sample_points


def test_fixed_sample_npz_is_loaded_and_hashed(tmp_path):
    points = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    source = tmp_path / "points.npz"
    np.savez(source, points=points)
    config = parse_config(
        {
            "domain": {"type": "cube"},
            "network": {"mode": "tree", "target_terminal_count": 1},
            "simulation": {
                "sample_mode": "file",
                "sample_points_path": source.name,
                "distance_sample_count": 0,
            },
        }
    )
    config.settings_path = tmp_path / "settings.json"

    loaded, metadata = _pre_sample_points(object(), object(), config)

    np.testing.assert_allclose(loaded, points)
    assert metadata["points"] == 2
    assert metadata["coordinate_units"] == "cm"
    assert len(metadata["sha256"]) == 64


def test_fixed_sample_csv_requires_finite_xyz(tmp_path):
    source = tmp_path / "points.csv"
    source.write_text("x,y,z\n0.1,0.2,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must all be finite"):
        _load_sample_points(source)


def test_fixed_sample_mode_requires_path():
    with pytest.raises(ValueError, match="sample_points_path is required"):
        parse_config(
            {
                "domain": {"type": "cube"},
                "network": {"mode": "tree", "target_terminal_count": 1},
                "simulation": {"sample_mode": "file"},
            }
        )
