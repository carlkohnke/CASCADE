"""Compare legacy and CASCADE CSV fields under the frozen M2 tolerance rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_FIELDS = (
    "pressure_in_root",
    "pressure_out_terminals",
    "pressure_drop_mean",
    "total_segments_mean",
    "terminal_segments_mean",
    "total_volume_mean",
    "total_flowrate_mean",
    "total_length_mean",
    "avg_length_mean",
    "avg_radius_mean",
    "radius_min_mean",
    "radius_max_mean",
    "Rnet_mean",
    "dRnet_mean",
    "Qmin_over_Qinlet_mean",
    "C_LQ_over_Cmax_mean",
    "C_tiss_over_Cmax_mean",
    "Damkohler_mean",
    "FracAbove50pct_mean",
    "FracAbove25pct_mean",
    "FracAbove10pct_mean",
    "FracAbove1pct_mean",
    "FracAbove5pct_mean",
)
EXACT_FIELDS = {"total_segments_mean", "terminal_segments_mean"}
FRACTION_FIELDS = {
    "FracAbove50pct_mean",
    "FracAbove25pct_mean",
    "FracAbove10pct_mean",
    "FracAbove5pct_mean",
    "FracAbove1pct_mean",
}
PRESSURE_FIELDS = {"pressure_in_root", "pressure_out_terminals", "pressure_drop_mean"}


def read_row(path: Path, fluid: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("fluid", "").strip().lower() == fluid.lower()]
    if selected:
        return selected[0]
    if len(rows) == 1:
        return rows[0]
    raise RuntimeError(f"Could not select fluid={fluid!r} from {path}")


def number(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or non-numeric field {field!r}: {raw!r}") from exc


def compare_field(
    field: str,
    reference: float,
    candidate: float,
    *,
    reference_case_scale: float | None = None,
) -> dict[str, object]:
    reference_finite = math.isfinite(reference)
    candidate_finite = math.isfinite(candidate)
    finite_mask_equal = reference_finite == candidate_finite
    if not reference_finite or not candidate_finite:
        same_nonfinite = (
            (math.isnan(reference) and math.isnan(candidate))
            or (math.isinf(reference) and math.isinf(candidate) and reference == candidate)
        )
        return {
            "field": field,
            "reference": reference,
            "candidate": candidate,
            "finite_mask_equal": finite_mask_equal,
            "passed": finite_mask_equal and same_nonfinite,
        }

    absolute_error = abs(candidate - reference)
    if field in EXACT_FIELDS:
        absolute_tolerance = 0.0
        relative_error = 0.0 if reference == candidate else math.inf
        passed = reference == candidate
        rule = "exact"
    elif field in FRACTION_FIELDS:
        absolute_tolerance = 0.001
        relative_error = absolute_error / max(abs(reference), 1e-300)
        passed = absolute_error <= absolute_tolerance
        rule = "absolute_fraction"
    else:
        case_scale = abs(reference) if reference_case_scale is None else abs(float(reference_case_scale))
        absolute_tolerance = 1e-6 * case_scale
        relative_error = absolute_error / max(abs(reference), absolute_tolerance)
        passed = absolute_error <= absolute_tolerance or relative_error <= 1e-3
        rule = "relative_or_scale_floor"
    return {
        "field": field,
        "reference": reference,
        "candidate": candidate,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "absolute_tolerance": absolute_tolerance,
        "reference_case_scale": None if field in EXACT_FIELDS or field in FRACTION_FIELDS else case_scale,
        "relative_tolerance": 0.0 if field in EXACT_FIELDS else 1e-3,
        "finite_mask_equal": finite_mask_equal,
        "rule": rule,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--cascade", type=Path, required=True)
    parser.add_argument("--fluid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", nargs="*", default=list(DEFAULT_FIELDS))
    parser.add_argument(
        "--case-scales",
        type=Path,
        help="Optional JSON object mapping physical fields to frozen reference-case scales.",
    )
    args = parser.parse_args()

    legacy_row = read_row(args.legacy, args.fluid)
    cascade_row = read_row(args.cascade, args.fluid)
    case_scales = {}
    if args.case_scales is not None:
        case_scales = json.loads(args.case_scales.read_text(encoding="utf-8"))
        if not isinstance(case_scales, dict):
            raise ValueError("--case-scales must contain a JSON object")
    comparisons = []
    for field in args.fields:
        legacy_value = number(legacy_row, field)
        if field in PRESSURE_FIELDS:
            legacy_value *= 0.1  # frozen oracle emits raw dyn/cm^2; CASCADE public fields are Pa
        item = compare_field(
            field,
            legacy_value,
            number(cascade_row, field),
            reference_case_scale=case_scales.get(field),
        )
        if field in PRESSURE_FIELDS:
            item["units"] = "Pa"
            item["legacy_normalization"] = "dyn/cm^2 * 0.1"
        comparisons.append(item)
    result = {
        "status": "pass" if all(item["passed"] for item in comparisons) else "fail",
        "fluid": args.fluid,
        "legacy_csv": str(args.legacy.resolve()),
        "cascade_csv": str(args.cascade.resolve()),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
