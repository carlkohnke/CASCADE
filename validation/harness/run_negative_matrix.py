#!/usr/bin/env python3
"""Exercise CLI failure paths from an installed CASCADE environment.

The harness creates every malformed input beneath an ignored run directory,
launches each case in a fresh subprocess, and records compact evidence.  A
negative case passes only when the command exits non-zero, reports the expected
root-cause text, and does not leave a successful manifest behind.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _base_settings(run_root: Path, geometry_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": {"type": "box", "dimensions": [1.0, 1.0, 1.0], "random_seed": 42},
        "network": {
            "mode": "simple",
            "simple": {
                "mode": "custom",
                "path": str(geometry_path),
                "flow_ul_min": 24.0,
                "concentration_inlet": 0.2211,
            },
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "water",
            "build_fluid": "water",
            "qin_target_ul_min": 24.0,
            "concentration_solver": "network",
            "distance_sample_count": 8,
            "tissue_accel": "cpu",
            "geometry_only": True,
        },
        "outputs": {
            "out_dir": str(run_root / "valid-output"),
            "prefix": "negative",
            "write_paraview": False,
            "write_summary_csv": False,
            "write_segments_csv": False,
            "write_points_csv": False,
            "save_network": False,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run_case(
    *,
    case_id: str,
    command: list[str],
    expected_text: str,
    output_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr
    manifests = sorted(str(path) for path in output_dir.rglob("manifest.json")) if output_dir.exists() else []
    passed = completed.returncode != 0 and expected_text.lower() in combined.lower() and not manifests
    return {
        "case_id": case_id,
        "command": command,
        "returncode": int(completed.returncode),
        "expected_text": expected_text,
        "expected_text_found": expected_text.lower() in combined.lower(),
        "successful_manifests": manifests,
        "passed": bool(passed),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-8000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade", required=True, help="Installed cascade console entry point.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result).resolve()

    valid_csv = run_root / "valid.csv"
    valid_csv.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm,prox_id,dist_id\n"
        "-0.4,0,0,0,0,0,0.02,0,1\n"
        "0,0,0,0.4,0.2,0,0.01,1,2\n",
        encoding="utf-8",
    )
    malformed_csv = run_root / "malformed.csv"
    malformed_csv.write_text("start_x,start_y\n0,0\n", encoding="utf-8")
    disconnected_csv = run_root / "disconnected.csv"
    disconnected_csv.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm,prox_id,dist_id\n"
        "-0.4,0,0,-0.2,0,0,0.02,0,1\n"
        "0.2,0,0,0.4,0,0,0.02,2,3\n",
        encoding="utf-8",
    )
    corrupt_vtp = run_root / "corrupt.vtp"
    corrupt_vtp.write_text("this is not a VTK XML document\n", encoding="utf-8")
    corrupt_forest = run_root / "corrupt.forest"
    corrupt_forest.write_bytes(b"not a forest archive\n")

    cases: list[tuple[str, dict[str, Any], str]] = []

    unknown_top = _base_settings(run_root, valid_csv)
    unknown_top["simulaton"] = {}
    cases.append(("unknown-top-level", unknown_top, "Unknown CASCADE setting(s): simulaton"))

    unknown_core = _base_settings(run_root, valid_csv)
    unknown_core["simulation"]["distance_sample_cout"] = 8
    cases.append(("unknown-core-setting", unknown_core, "simulation.distance_sample_cout"))

    missing_input = _base_settings(run_root, run_root / "does-not-exist.csv")
    cases.append(("missing-custom-input", missing_input, "custom geometry file not found"))

    malformed = _base_settings(run_root, malformed_csv)
    cases.append(("malformed-custom-csv", malformed, "missing columns"))

    disconnected = _base_settings(run_root, disconnected_csv)
    cases.append(("disconnected-custom-geometry", disconnected, "connected"))

    invalid_grid = _base_settings(run_root, valid_csv)
    invalid_grid["simulation"]["sample_mode"] = "grid"
    invalid_grid["simulation"]["tissue_grid"] = {"nx": 0, "ny": 8, "nz": 8}
    cases.append(("invalid-grid", invalid_grid, "grid"))

    impossible_flow = _base_settings(run_root, valid_csv)
    impossible_flow["simulation"]["flow_source"] = "per_inlet"
    impossible_flow["simulation"]["inlet_conditions"] = []
    cases.append(("impossible-flow-settings", impossible_flow, "per-inlet flow requires"))

    corrupt_domain = _base_settings(run_root, valid_csv)
    corrupt_domain["domain"] = {"type": "file", "path": str(corrupt_vtp)}
    cases.append(("corrupt-vtk-domain", corrupt_domain, "vtk"))

    corrupt_network = _base_settings(run_root, valid_csv)
    corrupt_network["network"] = {"mode": "forest", "input_path": str(corrupt_forest)}
    cases.append(("corrupt-forest", corrupt_network, "forest"))

    reports: list[dict[str, Any]] = []
    for case_id, settings, expected in cases:
        case_root = run_root / case_id
        settings["outputs"]["out_dir"] = str(case_root / "output")
        settings_path = case_root / "settings.json"
        _write_json(settings_path, settings)
        reports.append(
            _run_case(
                case_id=case_id,
                command=[args.cascade, "run", "--settings", str(settings_path)],
                expected_text=expected,
                output_dir=case_root,
            )
        )

    gpu_env = os.environ.copy()
    gpu_env["CUDA_VISIBLE_DEVICES"] = "-1"
    reports.append(
        _run_case(
            case_id="gpu-request-without-visible-device",
            command=[args.cascade, "doctor", "--require-gpu"],
            expected_text="GPU",
            output_dir=run_root / "gpu-request-without-visible-device",
            env=gpu_env,
        )
    )

    payload = {
        "schema_version": 1,
        "python": sys.version,
        "cascade_entry_point": str(Path(args.cascade).resolve()),
        "case_count": len(reports),
        "pass_count": sum(bool(item["passed"]) for item in reports),
        "all_passed": all(bool(item["passed"]) for item in reports),
        "cases": reports,
    }
    _write_json(result_path, payload)
    print(json.dumps({key: payload[key] for key in ("case_count", "pass_count", "all_passed")}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
