from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from cascade.config import example_config
from cascade.growth import NetworkBuildResult
from cascade.sweep import run_sweep


def test_sweep_uses_explicit_cached_network_without_growth(tmp_path, monkeypatch):
    raw = example_config()
    raw["network"]["input_path"] = str(tmp_path / "frozen.tree.npz")
    raw["network"]["target_terminal_count"] = 1
    raw["growth"]["enabled"] = False
    raw["simulation"]["geometry_only"] = True
    raw["simulation"]["skip_tissue_oxygen"] = True
    raw["outputs"]["out_dir"] = str(tmp_path / "out")
    raw["sweep"] = {
        "target_terminal_counts": [1],
        "fluids": ["blood"],
        "output_csv": str(tmp_path / "summary.csv"),
        "work_dir": str(tmp_path / "work"),
    }
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(raw), encoding="utf-8")

    calls = []
    build = NetworkBuildResult(
        domain=object(),
        trees=[object()],
        forest=None,
        target_counts=[1],
        build_timings={"network_load_s": 0.01},
        sample_points=np.empty((0, 3)),
    )
    monkeypatch.setattr("cascade.sweep.load_runtime_module", lambda: SimpleNamespace())
    monkeypatch.setattr("cascade.sweep.apply_runtime_settings", lambda *_args: None)
    monkeypatch.setattr("cascade.sweep.build_or_load_network", lambda _config: calls.append("load") or build)
    monkeypatch.setattr(
        "cascade.sweep._build_configured_trees",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("growth must not run")),
    )
    monkeypatch.setattr(
        "cascade.sweep.run_simulation",
        lambda *_args, **_kwargs: SimpleNamespace(summary_rows=[]),
    )
    monkeypatch.setattr("cascade.sweep.release_completed_case_memory", lambda *_args, **_kwargs: {})

    outputs = run_sweep(settings)

    assert calls == ["load"]
    assert outputs["summary_csv"] == str(tmp_path / "summary.csv")
