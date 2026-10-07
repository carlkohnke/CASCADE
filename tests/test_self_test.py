"""Exercise the installed-package self-test through its exported results."""

import json


def test_cpu_self_test_runs_without_gpu_preflight(monkeypatch, tmp_path):
    from cascade.accelerators import backend
    from cascade.diagnostics.self_test import _run_case

    def unexpected_probe(*args, **kwargs):
        raise AssertionError("The CPU self-test must not require CUDA")

    monkeypatch.setattr(backend, "probe_gpu_runtime", unexpected_probe)
    report = _run_case(tmp_path, require_gpu=False)
    assert report["status"] == "passed"
    assert report["backend"] == "cpu"
    assert report["vessel_points"] > 0
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["inputs"]["settings_sha256"]
