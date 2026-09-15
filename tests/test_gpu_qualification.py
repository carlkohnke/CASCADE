from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import pytest
import pyvista as pv
from cascade.gui.model import create_jobs
from cascade.gui.runner import JobRunner
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.skipif(
    os.environ.get("CASCADE_RUN_GPU_QUALIFICATION") != "1",
    reason="set CASCADE_RUN_GPU_QUALIFICATION=1 on a dedicated CUDA runner",
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    instance = QApplication.instance()
    if instance is not None:
        assert isinstance(instance, QApplication)
        return instance
    return QApplication([])


def _wait_until(qapp: QApplication, predicate, timeout_s: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    qapp.processEvents()
    return bool(predicate())


def _process_stopped(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE


def _process_tree(pid: int) -> list[psutil.Process]:
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def _host_memory_bytes(processes: list[psutil.Process]) -> int:
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return total


def _gpu_memory_mib(pids: set[int]) -> tuple[int, str]:
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if process_query.returncode == 0:
        process_values: list[int] = []
        for line in process_query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if (
                len(fields) >= 2
                and fields[0].isdigit()
                and int(fields[0]) in pids
                and fields[1].isdigit()
            ):
                process_values.append(int(fields[1]))
        if process_values:
            return sum(process_values), "process-tree"
    device_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert device_query.returncode == 0, device_query.stderr
    values = [
        int(line.strip())
        for line in device_query.stdout.splitlines()
        if line.strip().isdigit()
    ]
    assert values, device_query.stdout
    return sum(values), "device"


def _run_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "cascade.commands.main", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    return completed


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _generate_tree(root: Path) -> Path:
    output = root / "generated-tree"
    config = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "target_terminal_count": 2,
            "root": {
                "start": [0.49, -0.49, -0.49],
                "direction": [-0.49, 0.49, 0.49],
            },
        },
        "growth": {
            "enabled": True,
            "n_closest_vessels": 2,
            "n_points": 20,
            "ignore_collisions": True,
            "allow_inside_vessels": True,
        },
        "simulation": {
            "fluid": "water",
            "qin_target_ul_min": 2.0,
            "concentration_solver": "topdown",
            "distance_sample_count": 0,
            "tissue_accel": "cpu",
            "geometry_only": True,
        },
        "settings": {
            "hematocrit": {"flow_iterations": 0},
            "cext": {"accel_mode": "cpu"},
            "tissue": {"accel_mode": "cpu"},
        },
        "outputs": {
            "out_dir": str(output),
            "prefix": "qualification",
            "write_paraview": False,
            "save_network": True,
        },
    }
    _run_cli(
        root, "run", "--settings", str(_write_json(root / "generate.json", config))
    )
    tree_path = output / "qualification.tree.npz"
    assert tree_path.is_file()
    return tree_path


def _base_case(tree_path: Path, output: Path) -> dict:
    return {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "input_path": str(tree_path),
            "target_terminal_count": 2,
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "blood",
            "qin_target_ul_min": 900.0,
            "concentration_solver": "topdown_ext",
            "distance_sample_count": 64,
            "tissue_accel": "gpu",
        },
        "settings": {
            "oxygen": {
                "gl_order": 5,
                "gl_order_cext": 2,
                "finite_radius_o2_terms": "both",
                "lumen_wall_closure": "wellmixed",
            },
            "cext": {
                "accel_mode": "gpu",
                "frozen_accel_mode": "gpu",
                "vess_coupling_max_iter": 2,
                "vess_coupling_accel": "none",
                "window_factor": 6.0,
                "hybrid_bg_mode": "fft",
                "hybrid_bg_solver": "fft",
                "hybrid_bg_grid": 16,
                "hybrid_bg_lambda_bins": 2,
                "hybrid_bg_near_radius_mult": 2.0,
                "treecode_leaf_nodes": 1,
                "treecode_lambda_bins": 2,
                "treecode_near_radius_mult": 2.0,
            },
            "tissue": {
                "accel_mode": "gpu",
                "cext_cell_list_enable": True,
                "nearest_vessels": 250,
            },
        },
        "outputs": {
            "out_dir": str(output),
            "prefix": output.name,
            "write_paraview": True,
            "save_network": False,
            "overwrite": True,
            "export_float_dtype": "float32",
            "export_index_dtype": "int32",
        },
    }


def _run_case(root: Path, tree_path: Path, name: str, mutate) -> dict:
    output = root / name
    config = _base_case(tree_path, output)
    mutate(config)
    completed = _run_cli(
        root,
        "run",
        "--settings",
        str(_write_json(root / f"{name}.json", config)),
    )
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = next(csv.DictReader(handle))
    segments = np.atleast_1d(
        np.genfromtxt(output / "segments.csv", delimiter=",", names=True)
    )
    points = np.atleast_1d(
        np.genfromtxt(output / "points.csv", delimiter=",", names=True)
    )
    vessels = pv.read(output / "vessels.vtp")
    oxygen = pv.read(output / "oxygen_points.vtp")
    assert vessels.n_points > 0
    assert oxygen.n_points == 64
    assert {"pressure_pa", "flow_ul_min", "concentration"}.issubset(vessels.point_data)
    assert {"local_concentration", "viability"}.issubset(oxygen.point_data)
    return {
        "stdout": completed.stdout,
        "summary": summary,
        "segments": segments,
        "points": points,
        "vessels": vessels,
        "oxygen": oxygen,
    }


def _set_cpu_direct(config: dict) -> None:
    config["simulation"]["tissue_accel"] = "cpu"
    config["settings"]["cext"].update(
        {"accel_mode": "cpu", "frozen_accel_mode": "cpu", "float_dtype": "float64"}
    )
    config["settings"]["tissue"].update(
        {"accel_mode": "cpu", "cache_force_float64": True}
    )
    config["outputs"].update(
        {"export_float_dtype": "float64", "export_index_dtype": "int64"}
    )


def _set_cpu_topdown(config: dict) -> None:
    _set_cpu_direct(config)
    config["simulation"]["concentration_solver"] = "topdown"


def _set_gpu_direct_cell(config: dict) -> None:
    config["settings"]["cext"]["float_dtype"] = "float32"


def _set_gpu_direct_dense(config: dict) -> None:
    config["settings"]["cext"]["float_dtype"] = "float64"
    config["settings"]["tissue"]["cext_cell_list_enable"] = False
    config["simulation"]["tissue_gpu_validate_points"] = 64
    config["outputs"].update(
        {"export_float_dtype": "float64", "export_index_dtype": "int64"}
    )


def _set_gpu_hybrid_fft(config: dict) -> None:
    config["simulation"]["concentration_solver"] = "topdown_ext_hybrid_bg"


def _set_gpu_hybrid_local(config: dict) -> None:
    _set_gpu_hybrid_fft(config)
    config["settings"]["cext"]["hybrid_bg_mode"] = "hybrid"


def _set_gpu_treecode(config: dict) -> None:
    config["simulation"]["concentration_solver"] = "topdown_ext_treecode"


def _set_gpu_topdown_dense(config: dict) -> None:
    config["simulation"]["concentration_solver"] = "topdown"
    config["simulation"]["tissue_gpu_validate_points"] = 64


def _set_gpu_topdown_sparse(config: dict) -> None:
    _set_gpu_topdown_dense(config)
    config["settings"]["tissue"]["nearest_vessels"] = 2


def _assert_segment_agreement(
    actual: dict, expected: dict, *, oxygen_rtol: float
) -> None:
    for field in ("pressure_pa", "flow_cm3_s"):
        np.testing.assert_allclose(
            actual["segments"][field],
            expected["segments"][field],
            rtol=1e-10,
            atol=1e-12,
        )
    for field in ("cin", "cout"):
        np.testing.assert_allclose(
            actual["segments"][field],
            expected["segments"][field],
            rtol=oxygen_rtol,
            atol=5e-6,
        )


def _assert_tissue_agreement(actual: dict, expected: dict, *, rtol: float) -> None:
    np.testing.assert_array_equal(
        actual["points"]["inside_tissue"], expected["points"]["inside_tissue"]
    )
    np.testing.assert_allclose(
        actual["points"]["local_concentration_raw"],
        expected["points"]["local_concentration_raw"],
        rtol=rtol,
        atol=5e-6,
    )


def test_native_cuda_kernel_families_and_cpu_agreement(tmp_path: Path) -> None:
    tree_path = _generate_tree(tmp_path)
    cpu_direct = _run_case(tmp_path, tree_path, "cpu-direct", _set_cpu_direct)
    cpu_topdown = _run_case(tmp_path, tree_path, "cpu-topdown", _set_cpu_topdown)
    gpu_direct_cell = _run_case(
        tmp_path, tree_path, "gpu-direct-cell", _set_gpu_direct_cell
    )
    gpu_direct_dense = _run_case(
        tmp_path, tree_path, "gpu-direct-dense", _set_gpu_direct_dense
    )
    gpu_hybrid_fft = _run_case(
        tmp_path, tree_path, "gpu-hybrid-fft", _set_gpu_hybrid_fft
    )
    gpu_hybrid_local = _run_case(
        tmp_path, tree_path, "gpu-hybrid-local", _set_gpu_hybrid_local
    )
    gpu_treecode = _run_case(tmp_path, tree_path, "gpu-treecode", _set_gpu_treecode)
    gpu_tissue_dense = _run_case(
        tmp_path, tree_path, "gpu-tissue-dense", _set_gpu_topdown_dense
    )
    gpu_tissue_sparse = _run_case(
        tmp_path, tree_path, "gpu-tissue-sparse", _set_gpu_topdown_sparse
    )

    assert "backend=gpu frozen=gpu" in gpu_direct_cell["stdout"]
    assert "candidates=gpu_direct" in gpu_direct_cell["stdout"]
    assert "candidate_mode=cell_list" in gpu_direct_cell["stdout"]
    assert gpu_direct_cell["summary"]["t_cext_backend"] == "gpu"
    assert gpu_direct_cell["summary"]["t_tissue_backend"] == "cext_gpu_cell_list"

    assert "candidate_mode=all_gpu" in gpu_direct_dense["stdout"]
    assert "Cext tissue GPU validation" in gpu_direct_dense["stdout"]
    assert gpu_direct_dense["summary"]["t_tissue_backend"] == "cext_gpu"

    assert "Concentration topdown_ext_hybrid_bg" in gpu_hybrid_fft["stdout"]
    assert "bg_mode=fft" in gpu_hybrid_fft["stdout"]
    assert int(float(gpu_hybrid_fft["summary"]["cext_hybrid_bg_grid"])) == 16
    for field in (
        "t_cext_hybrid_deposit_s",
        "t_cext_hybrid_fft_s",
        "t_cext_hybrid_o2_terms_s",
        "t_cext_hybrid_self_subtract_s",
        "t_cext_hybrid_sample_s",
    ):
        assert float(gpu_hybrid_fft["summary"][field]) > 0.0, field

    assert "Concentration topdown_ext_hybrid_bg" in gpu_hybrid_local["stdout"]
    assert "bg_mode=hybrid" in gpu_hybrid_local["stdout"]
    for field in (
        "t_cext_hybrid_deposit_s",
        "t_cext_hybrid_fft_s",
        "t_cext_hybrid_local_corr_s",
        "t_cext_hybrid_sample_s",
    ):
        assert float(gpu_hybrid_local["summary"][field]) > 0.0, field

    assert "Concentration topdown_ext_treecode" in gpu_treecode["stdout"]
    assert "backend=gpu frozen=gpu" in gpu_treecode["stdout"]
    assert gpu_treecode["summary"]["t_cext_backend"] == "gpu"

    assert "Tissue GPU solve" in gpu_tissue_dense["stdout"]
    assert "Tissue GPU validation" in gpu_tissue_dense["stdout"]
    assert gpu_tissue_dense["summary"]["t_tissue_backend"] == "gpu_dense_fused"
    assert "Tissue GPU solve" in gpu_tissue_sparse["stdout"]
    assert "Tissue GPU validation" in gpu_tissue_sparse["stdout"]
    assert gpu_tissue_sparse["summary"]["t_tissue_backend"] == "gpu"

    _assert_segment_agreement(gpu_direct_cell, cpu_direct, oxygen_rtol=5e-3)
    _assert_segment_agreement(gpu_direct_dense, cpu_direct, oxygen_rtol=5e-3)
    _assert_segment_agreement(gpu_hybrid_fft, gpu_direct_cell, oxygen_rtol=0.15)
    _assert_segment_agreement(gpu_hybrid_local, gpu_direct_cell, oxygen_rtol=0.15)
    _assert_segment_agreement(gpu_treecode, gpu_direct_cell, oxygen_rtol=0.15)
    _assert_tissue_agreement(gpu_direct_cell, cpu_direct, rtol=3e-2)
    _assert_tissue_agreement(gpu_direct_dense, cpu_direct, rtol=3e-2)
    _assert_segment_agreement(gpu_tissue_dense, cpu_topdown, oxygen_rtol=5e-3)
    _assert_segment_agreement(gpu_tissue_sparse, cpu_topdown, oxygen_rtol=5e-3)
    _assert_tissue_agreement(gpu_tissue_dense, cpu_topdown, rtol=3e-2)
    _assert_tissue_agreement(gpu_tissue_sparse, cpu_topdown, rtol=3e-2)

    assert gpu_direct_cell["vessels"].points.dtype == np.float32
    assert gpu_direct_cell["oxygen"].points.dtype == np.float32
    assert gpu_direct_dense["vessels"].points.dtype == np.float64
    assert gpu_direct_dense["oxygen"].points.dtype == np.float64


def _studio_gpu_config(tree_path: Path, project: Path, name: str, samples: int) -> dict:
    config = _base_case(tree_path, project / "placeholder")
    config["simulation"]["distance_sample_count"] = samples
    config.setdefault("gui", {}).update(
        {"project_name": "Native CUDA qualification", "run_name": name}
    )
    return config


def test_native_cuda_persistent_worker_reuse_cancel_and_shutdown(
    qapp: QApplication, tmp_path: Path
) -> None:
    tree_path = _generate_tree(tmp_path)
    project = tmp_path / "Persistent CUDA Worker"
    project.mkdir()
    runner = JobRunner(project)
    worker_pid = 0
    recovery_pid = 0
    recovery_pids: list[int] = []
    host_samples: list[int] = []
    gpu_samples: list[int] = []
    gpu_sample_mode = ""
    try:
        for index in range(3):
            config = _studio_gpu_config(tree_path, project, f"Repeat {index + 1}", 64)
            job = create_jobs(config, project)[0]
            runner.add([job])
            runner.run([job.id])
            assert _wait_until(qapp, lambda: job.status == "Completed", 240.0), (
                job.error,
                job.stage,
            )
            assert Path(job.manifest_path or "").is_file()
            assert runner.process is not None
            assert runner.process.state() == QProcess.Running
            current_pid = int(runner.process.processId())
            if worker_pid:
                assert current_pid == worker_pid
            else:
                worker_pid = current_pid
            processes = _process_tree(worker_pid)
            host_samples.append(_host_memory_bytes(processes))
            gpu_memory, mode = _gpu_memory_mib({process.pid for process in processes})
            if gpu_sample_mode:
                assert mode == gpu_sample_mode
            else:
                gpu_sample_mode = mode
            gpu_samples.append(gpu_memory)

        host_growth = max(host_samples) - min(host_samples)
        gpu_growth = max(gpu_samples) - min(gpu_samples)
        assert host_growth <= 768 * 1024 * 1024, host_samples
        gpu_limit_mib = 512 if gpu_sample_mode == "process-tree" else 1024
        assert gpu_growth <= gpu_limit_mib, gpu_samples
        print(
            "Persistent CUDA worker memory: "
            f"pid={worker_pid} host_mib={[round(v / 2**20, 1) for v in host_samples]} "
            f"gpu_mib={gpu_samples} gpu_scope={gpu_sample_mode}"
        )

        cancel_config = _studio_gpu_config(
            tree_path, project, "Cancel after CUDA", 2_000_000
        )
        cancelled = create_jobs(cancel_config, project)[0]
        runner.add([cancelled])
        runner.run([cancelled.id])
        assert _wait_until(
            qapp,
            lambda: (
                runner.current is cancelled
                and runner.process is not None
                and runner.process.state() == QProcess.Running
                and cancelled.progress >= 55
            ),
            240.0,
        ), (cancelled.error, cancelled.stage, cancelled.progress)
        cancelled_processes = _process_tree(int(runner.process.processId()))
        cancelled_pids = [process.pid for process in cancelled_processes]
        runner.cancel_current()
        assert _wait_until(
            qapp,
            lambda: cancelled.status == "Cancelled" and runner.current is None,
            30.0,
        )
        assert _wait_until(
            qapp, lambda: all(_process_stopped(pid) for pid in cancelled_pids), 20.0
        )

        recovery_config = _studio_gpu_config(tree_path, project, "Recovery", 64)
        recovery = create_jobs(recovery_config, project)[0]
        runner.add([recovery])
        runner.run([recovery.id])
        assert _wait_until(qapp, lambda: recovery.status == "Completed", 240.0), (
            recovery.error,
            recovery.stage,
        )
        assert Path(recovery.manifest_path or "").is_file()
        assert runner.process is not None
        recovery_pid = int(runner.process.processId())
        assert recovery_pid not in cancelled_pids
        recovery_processes = _process_tree(recovery_pid)
        recovery_pids = [process.pid for process in recovery_processes]
        _gpu_memory_mib(set(recovery_pids))
    finally:
        runner.shutdown()
        qapp.processEvents()
    final_pids = recovery_pids if recovery_pid else [worker_pid]
    assert _wait_until(
        qapp, lambda: all(_process_stopped(pid) for pid in final_pids), 20.0
    )
