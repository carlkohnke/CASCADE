import pytest

from cascade.configuration.bridge import load_runtime_module
from cascade.configuration.schema import parse_config
from cascade.vessels.conditions import flow_for_tree, flow_volume_scale


def _config(side_length, **simulation):
    return parse_config(
        {
            "domain": {"type": "cube", "side_length": side_length},
            "network": {"mode": "tree", "target_terminal_count": 10},
            "simulation": {"qin_target_ul_min": 60.0, **simulation},
        }
    )


@pytest.mark.parametrize("scale, expected_ul_min", [(True, 480.0), (False, 60.0)])
def test_scale_q_by_volume_controls_inlet_flow(monkeypatch, scale, expected_ul_min):
    monkeypatch.setattr(load_runtime_module(), "SCALE_Q_BY_VOLUME", scale)
    config = _config(2.0)

    assert flow_volume_scale(config) == pytest.approx(8.0 if scale else 1.0)
    assert flow_for_tree(config, 0, 1) * 60.0e3 == pytest.approx(expected_ul_min)


@pytest.mark.parametrize("scale, expected_ul_min", [(True, 240.0), (False, 30.0)])
def test_scale_q_by_volume_controls_split_total_flow(
    monkeypatch, scale, expected_ul_min
):
    monkeypatch.setattr(load_runtime_module(), "SCALE_Q_BY_VOLUME", scale)
    config = _config(2.0, flow_source="total_split")

    assert flow_for_tree(config, 0, 2) * 60.0e3 == pytest.approx(expected_ul_min)
