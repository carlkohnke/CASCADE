from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "validation" / "harness" / "compare_csv.py"
    spec = importlib.util.spec_from_file_location("m2_compare_csv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_array_compare_module():
    path = Path(__file__).resolve().parents[1] / "validation" / "harness" / "compare_arrays.py"
    spec = importlib.util.spec_from_file_location("m2_compare_arrays", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_legacy_heart_runner_module():
    path = Path(__file__).resolve().parents[1] / "validation" / "harness" / "run_legacy_heart.py"
    spec = importlib.util.spec_from_file_location("m2_run_legacy_heart", path)
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


def test_m2_streaming_array_rules_detect_identity_and_tolerance_failures():
    compare = _load_array_compare_module()
    exact = np.arange(24, dtype=np.float32).reshape(8, 3)
    assert compare.compare_exact(exact, exact.copy(), chunk_rows=3)["passed"] is True
    changed = exact.copy()
    changed[6, 1] += 1
    assert compare.compare_exact(exact, changed, chunk_rows=3)["passed"] is False

    reference = np.asarray([0.0, 1.0, 10.0, np.nan])
    accepted = np.asarray([5e-6, 1.0005, 10.009, np.nan])
    rejected = np.asarray([2e-5, 1.002, 10.02, np.nan])
    assert compare.compare_physical(reference, accepted, chunk_rows=2)["passed"] is True
    assert compare.compare_physical(reference, rejected, chunk_rows=2)["passed"] is False


def test_m2_streaming_array_rules_require_matching_finite_masks():
    compare = _load_array_compare_module()
    result = compare.compare_physical(
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, 0.0]),
        chunk_rows=1,
    )
    assert result["passed"] is False
    assert result["finite_mask_mismatches"] == 1


def test_legacy_heart_reference_resampling_preserves_segment_weight():
    runner = _load_legacy_heart_runner_module()
    state = {
        "gl_points_si": np.asarray([[[0.5, 0.0, 0.0]]], dtype=np.float32),
        "segment_vectors": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "q_line_gl": np.asarray([[2.0]], dtype=np.float32),
        "q_weighted_gl": np.asarray([[2.0]], dtype=np.float32),
    }
    result = runner._resample_cext_state(state, 5)
    assert result["gl_points_si"].shape == (1, 5, 3)
    np.testing.assert_allclose(result["q_line_gl"], 2.0)
    np.testing.assert_allclose(np.sum(result["q_weighted_gl"], axis=1), [2.0])
