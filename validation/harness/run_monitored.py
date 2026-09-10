"""Run one characterization subprocess with host/GPU resource monitoring."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

import psutil


THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_memory_by_pid(pids: set[int]) -> int | None:
    if not pids:
        return 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    total_mib = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            memory_mib = int(parts[1].split()[0])
        except ValueError:
            continue
        if pid in pids:
            total_mib += memory_mib
    return total_mib * 1024 * 1024


def _process_sample(root: psutil.Process) -> tuple[int, int, set[int]]:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rss = 0
    vms = 0
    pids: set[int] = set()
    for process in processes:
        try:
            memory = process.memory_info()
            rss += int(memory.rss)
            vms += int(memory.vms)
            pids.add(int(process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, vms, pids


def _terminate_tree(root: psutil.Process) -> None:
    try:
        children = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for process in reversed(children):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        root.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    _, alive = psutil.wait_procs([*children, root], timeout=10)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=0.25)
    parser.add_argument("--max-rss-gib", type=float, default=45.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")

    metadata_path = args.metadata.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    if metadata_path.exists() or log_path.exists():
        raise FileExistsError("Refusing to overwrite monitored-run evidence")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = Path(tempfile.gettempdir()) / f"cascade-characterization-uid-{os.getuid()}.lock"
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"Another characterization process holds {lock_path}") from exc

    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.perf_counter()
    peak_rss = 0
    peak_vms = 0
    peak_gpu = 0
    gpu_observed = False
    samples = 0
    memory_limit = int(float(args.max_rss_gib) * 1024**3)
    aborted_for_memory = False

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        root = psutil.Process(process.pid)

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        next_status = started + 30.0
        while process.poll() is None:
            rss, vms, pids = _process_sample(root)
            peak_rss = max(peak_rss, rss)
            peak_vms = max(peak_vms, vms)
            gpu = _gpu_memory_by_pid(pids)
            if gpu is not None:
                gpu_observed = True
                peak_gpu = max(peak_gpu, gpu)
            samples += 1
            if rss > memory_limit:
                aborted_for_memory = True
                print(
                    f"Memory safety limit exceeded: {rss / 1024**3:.2f} GiB > {args.max_rss_gib:.2f} GiB",
                    flush=True,
                )
                _terminate_tree(root)
                break
            now = time.perf_counter()
            if now >= next_status:
                print(
                    f"monitor: elapsed={now - started:.1f}s rss={rss / 1024**3:.2f}GiB "
                    f"peak={peak_rss / 1024**3:.2f}GiB",
                    flush=True,
                )
                next_status = now + 30.0
            time.sleep(max(float(args.sample_seconds), 0.05))
        return_code = process.wait()
        reader.join(timeout=10)

    elapsed = time.perf_counter() - started
    record = {
        "schema_version": 1,
        "started_utc": started_utc,
        "command": command,
        "cwd": str(Path.cwd()),
        "return_code": int(return_code),
        "elapsed_wall_s": elapsed,
        "sample_interval_s": float(args.sample_seconds),
        "samples": samples,
        "peak_rss_bytes": peak_rss,
        "peak_vms_bytes": peak_vms,
        "peak_gpu_memory_bytes": peak_gpu if gpu_observed else None,
        "max_rss_bytes": memory_limit,
        "aborted_for_memory": aborted_for_memory,
        "log_path": str(log_path),
        "log_sha256": sha256(log_path),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_bytes": int(psutil.virtual_memory().total),
        },
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return int(return_code) if not aborted_for_memory else 86


if __name__ == "__main__":
    raise SystemExit(main())
