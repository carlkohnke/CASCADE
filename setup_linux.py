"""Create and verify a Linux/WSL repository-local CASCADE environment.

Run this script from the repository root, for example
``python setup_linux.py --venv .venv --gui``. Optional flags add development or
CUDA dependencies, and ``--constraints`` applies one of the reproducible lock
files under ``requirements/``.  The script records the selected Python and
CUDA toolkit paths for the WSL launchers; ``--recreate`` also removes and
rebuilds the requested virtual-environment directory.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _config_dir() -> Path:
    configured = os.environ.get("CASCADE_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    workbench = ROOT.parent / "CASCADE-workbench"
    if workbench.is_dir():
        return workbench / "config"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CASCADE"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "cascade"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a local CASCADE virtual environment and install dependencies."
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Virtual environment directory, relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to create the venv. Defaults to the current Python.",
    )
    parser.add_argument(
        "--gpu",
        choices=("none", "cu11", "cu12", "cu13"),
        default="none",
        help="Install optional CuPy GPU dependencies for the selected CUDA family.",
    )
    parser.add_argument(
        "--cuda-path",
        default=None,
        help=(
            "CUDA toolkit root for CuPy/NVRTC, for example "
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0, "
            "/usr/local/cuda, or /path/to/targets/x86_64-linux. "
            "When omitted, setup tries to auto-detect one."
        ),
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install developer/test dependencies after runtime dependencies.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Install the native CASCADE Studio GUI and external 3D viewer.",
    )
    parser.add_argument(
        "--constraints",
        default=None,
        help=(
            "Optional fully resolved pip constraints file. Use the validated "
            "release lock for a reproducible production environment."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the target venv before creating it.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the final import/CLI verification step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args(argv)

    constraints = None
    if args.constraints:
        constraints = Path(args.constraints).expanduser()
        if not constraints.is_absolute():
            constraints = ROOT / constraints
        if not constraints.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Constraints file not found: {constraints}")

    venv_dir = Path(args.venv).expanduser()
    if not venv_dir.is_absolute():
        venv_dir = ROOT / venv_dir
    venv_dir = venv_dir.absolute()
    venv_python = _venv_python(venv_dir)

    # Check before installing into (or deleting) any existing environment.
    _validate_python312(args.python, args.dry_run)
    if venv_python.exists() and not args.recreate:
        _validate_python312(str(venv_python), args.dry_run)

    if args.recreate and venv_dir.exists():
        venv_dir = _validate_recreate_target(venv_dir)
        venv_python = _venv_python(venv_dir)
        print(f"+ remove {venv_dir}")
        if not args.dry_run:
            shutil.rmtree(venv_dir)

    if not venv_python.exists():
        _run([args.python, "-m", "venv", str(venv_dir)], args.dry_run)

    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        args.dry_run,
    )

    for requirements in _requirements(args.gpu, args.dev, args.gui):
        command = [str(venv_python), "-m", "pip", "install"]
        if constraints is not None:
            command.extend(["-c", str(constraints)])
        command.extend(["-r", str(requirements)])
        _run(command, args.dry_run)

    cuda_path = _resolve_cuda_path(args.cuda_path) if args.gpu != "none" else None
    if cuda_path is None and args.gpu != "none":
        print(
            "No system CUDA toolkit root was detected; setup will verify the "
            "package-provided CUDA headers with a compiled CuPy kernel."
        )

    if not args.skip_verify:
        verify_env = {"CUDA_PATH": str(cuda_path)} if cuda_path is not None else None
        _run(
            [
                str(venv_python),
                "-c",
                (
                    "import svv, cascade; "
                    "print('svv', getattr(svv, '__version__', None), svv.__file__); "
                    "print('cascade', cascade.__file__)"
                ),
            ],
            args.dry_run,
            env=verify_env,
        )
        if args.gpu != "none":
            _run(
                [
                    str(venv_python),
                    "-c",
                    (
                        "import cupy as cp; "
                        "x = cp.full((4, 4), cp.int32(-1), dtype=cp.int32); "
                        "total = int(cp.sum(x).get()); "
                        "assert total == -16, total; "
                        "print('cupy_compiled_kernel', total)"
                    ),
                ],
                args.dry_run,
                env=verify_env,
            )
        _run(
            [str(venv_python), "-m", "cascade.commands.main", "--help"],
            args.dry_run,
        )
        if args.gui:
            _run(
                [
                    str(venv_python),
                    "-c",
                    "from cascade.gui.smoke_test import main; main()",
                ],
                args.dry_run,
                env=verify_env,
            )

    # Publish launcher configuration only after every requested installation and
    # verification step has succeeded. A failed setup therefore leaves the
    # previously working interpreter/toolkit selection intact.
    _write_python_path(venv_python, args.dry_run)
    if cuda_path is not None:
        _write_cuda_path(cuda_path, args.dry_run)

    print()
    print(
        "Dry run complete; environment was not modified."
        if args.dry_run
        else "Environment ready."
    )
    print(f"Python: {venv_python}")
    print("Activate with:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\Activate.ps1")
    else:
        print(f"  source {venv_dir}/bin/activate")
    print("Run CASCADE with:")
    print(f"  {venv_python} -m cascade.commands.main init-settings case.json")
    print(f"  {venv_python} -m cascade.commands.main run --settings case.json")
    if args.gui:
        print("Launch CASCADE Studio with:")
        print(f"  {venv_python} -m cascade.gui")
    return 0


def _requirements(gpu: str, dev: bool, gui: bool) -> list[Path]:
    if gpu == "cu13":
        reqs = [ROOT / "requirements" / "gpu-cu13.txt"]
    elif gpu == "cu12":
        reqs = [ROOT / "requirements" / "gpu-cu12.txt"]
    elif gpu == "cu11":
        reqs = [ROOT / "requirements" / "gpu-cu11.txt"]
    else:
        reqs = [ROOT / "requirements" / "base.txt"]
    if dev:
        reqs.append(ROOT / "requirements" / "dev.txt")
    if gui:
        reqs.append(ROOT / "requirements" / "gui.txt")
    missing = [path for path in reqs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing requirements file(s): " + ", ".join(str(p) for p in missing)
        )
    return reqs


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _validate_python312(python: str, dry_run: bool) -> None:
    if dry_run:
        print(f"+ verify 64-bit CPython 3.12: {python}")
        return
    probe = subprocess.run(
        [
            python, "-c",
            (
                "import struct, sys; "
                "print(sys.version.split()[0]); "
                "sys.exit(0 if sys.implementation.name == 'cpython' "
                "and sys.version_info[:2] == (3, 12) "
                "and struct.calcsize('P') == 8 else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode:
        raise ValueError(
            f"CASCADE requires 64-bit CPython 3.12; {python} reported "
            f"{probe.stdout.strip() or probe.stderr.strip()}. "
            "Run python3.12 setup_linux.py --venv .venv --gui; "
            "use --recreate if the existing environment has another Python version."
        )


def _validate_recreate_target(venv_dir: Path) -> Path:
    """Return a resolved venv path only when recursive removal is demonstrably safe."""
    if venv_dir.is_symlink():
        raise ValueError(
            f"Refusing to recreate symlinked virtual environment: {venv_dir}"
        )
    resolved = venv_dir.resolve(strict=True)
    repository = ROOT.resolve()
    protected = {repository, Path.home().resolve(), Path(resolved.anchor).resolve()}
    if resolved in protected or resolved in repository.parents:
        raise ValueError(f"Refusing to remove protected path: {resolved}")
    if not resolved.is_dir() or not (resolved / "pyvenv.cfg").is_file():
        raise ValueError(
            "Refusing to recursively remove a directory that is not a virtual "
            f"environment (missing pyvenv.cfg): {resolved}"
        )
    return resolved


def _resolve_cuda_path(value: str | None) -> Path | None:
    candidates: list[Path] = []
    if value:
        candidates.append(Path(value).expanduser())
    env_path = None
    try:
        import os

        env_path = os.environ.get("CUDA_PATH")
    except Exception:
        env_path = None
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(sys.prefix) / "targets" / "x86_64-linux")
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            toolkit_root = (
                Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            )
            try:
                candidates.extend(
                    sorted(toolkit_root.glob("v*"), key=_cuda_version_key, reverse=True)
                )
            except OSError:
                pass
    else:
        candidates.extend(
            [
                Path("/usr/local/cuda"),
                Path("/usr/local/cuda-13"),
                Path("/usr/local/cuda-12"),
            ]
        )
    for path in candidates:
        if _is_cuda_root(path):
            return path.resolve()
    return None


def _cuda_version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", path.name))


def _is_cuda_root(path: Path) -> bool:
    if not (path / "include" / "cuda_fp16.h").exists():
        return False
    if sys.platform == "win32":
        return any(path.glob("bin/**/nvrtc*.dll"))
    return (path / "lib" / "libnvrtc.so").exists() or any(
        (path / "lib").glob("libnvrtc.so*")
    )


def _write_cuda_path(cuda_path: Path, dry_run: bool) -> None:
    path_file = _config_dir() / ".cascade_cuda_path"
    print(f"+ write {path_file} = {cuda_path}")
    if dry_run:
        return
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text(str(cuda_path) + "\n", encoding="utf-8")


def _write_python_path(venv_python: Path, dry_run: bool) -> None:
    # Keep each checkout tied to its verified environment when installations
    # share the same user-level configuration directory.
    for path_file in (ROOT / ".cascade_python", _config_dir() / ".cascade_python"):
        print(f"+ write {path_file} = {venv_python}")
        if not dry_run:
            path_file.parent.mkdir(parents=True, exist_ok=True)
            path_file.write_text(str(venv_python) + "\n", encoding="utf-8")


def _run(
    command: list[str] | None, dry_run: bool, *, env: dict[str, str] | None = None
) -> None:
    if command is None:
        return
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={value}" for key, value in env.items()) + " "
    print("+ " + prefix + " ".join(command))
    if dry_run:
        return
    if env:
        import os

        run_env = os.environ.copy()
        run_env.update(env)
    else:
        run_env = None
    subprocess.run(command, cwd=ROOT, check=True, env=run_env)


if __name__ == "__main__":
    raise SystemExit(main())
