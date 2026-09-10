from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "validation" / "harness" / "compare_csv.py"
    spec = importlib.util.spec_from_file_location("m2_compare_csv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m2_exact_identity_fields_are_not_relaxed():
    compare = _load_compare_module()
    assert compare.compare_field("total_segments_mean", 201.0, 201.0)["passed"] is True
    assert compare.compare_field("total_segments_mean", 201.0, 202.0)["passed"] is False


def test_m2_fraction_and_relative_rules():
    compare = _load_compare_module()
    assert compare.compare_field("FracAbove1pct_mean", 0.5, 0.5009)["passed"] is True
    assert compare.compare_field("FracAbove1pct_mean", 0.5, 0.5011)["passed"] is False
    assert compare.compare_field("C_tiss_over_Cmax_mean", 10.0, 10.009)["passed"] is True
    assert compare.compare_field("C_tiss_over_Cmax_mean", 10.0, 10.011)["passed"] is False


def test_m2_nonfinite_masks_must_match():
    compare = _load_compare_module()
    assert compare.compare_field("dRnet_mean", float("nan"), float("nan"))["passed"] is True
    assert compare.compare_field("dRnet_mean", float("nan"), 0.0)["passed"] is False
