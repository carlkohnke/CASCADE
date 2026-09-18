from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import cascade

from cascade.configuration import solver_state
from cascade.domain.cache import default_domain_cache_dir
from cascade.exporting.paths import diagnostic_log_path
from cascade.gui.ui_helpers import _default_project_directory
from cascade.runtime.paths import (
    cache_directory,
    config_directory,
    log_directory,
    project_directory,
    state_directory,
)
from cascade.utils.execution import simulation_lock_path
from cascade.vessels.prepared import default_prepared_cache_root
from cascade.vessels.tree_cache import _tree_cache_dir


def test_pytest_imports_checkout_source_tree() -> None:
    repository = Path(__file__).resolve().parents[1]

    assert Path(cascade.__file__).resolve().is_relative_to(repository / "src")


def test_test_subprocesses_import_checkout_source_tree(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import cascade; print(cascade.__file__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert Path(completed.stdout.strip()).resolve().is_relative_to(
        Path(__file__).resolve().parents[1] / "src"
    )


def test_runtime_location_overrides_are_shared(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "configuration"
    cache = tmp_path / "cache root"
    state = tmp_path / "state"
    logs = tmp_path / "diagnostic logs"
    projects = tmp_path / "projects" / "Untitled"
    monkeypatch.setenv("CASCADE_CONFIG_DIR", str(config))
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(cache))
    monkeypatch.setenv("CASCADE_STATE_DIR", str(state))
    monkeypatch.setenv("CASCADE_LOG_DIR", str(logs))
    monkeypatch.setenv("CASCADE_PROJECT_DIR", str(projects))
    monkeypatch.delenv("CASCADE_DOMAIN_CACHE_DIR", raising=False)
    monkeypatch.delenv("CASCADE_PREPARED_CACHE_DIR", raising=False)
    monkeypatch.setattr(solver_state, "TREE_CACHE_DIRNAME", "trees_cache")

    assert config_directory() == config.resolve()
    assert cache_directory() == cache.resolve()
    assert state_directory() == state.resolve()
    assert log_directory() == logs.resolve()
    assert project_directory() == projects.resolve()
    assert _default_project_directory() == projects.resolve()
    assert default_domain_cache_dir() == cache.resolve() / "domains"
    assert default_prepared_cache_root() == cache.resolve() / "prepared-trees"
    assert _tree_cache_dir() == cache.resolve() / "trees_cache"
    assert simulation_lock_path().parent == state.resolve() / "locks"

    path = diagnostic_log_path("doctor.log")
    assert path == logs.resolve() / "doctor.log"
    assert path.parent.is_dir()


def test_specific_cache_overrides_take_precedence(
    monkeypatch, tmp_path: Path
) -> None:
    shared = tmp_path / "shared"
    domains = tmp_path / "domain override"
    prepared = tmp_path / "prepared override"
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(shared))
    monkeypatch.setenv("CASCADE_DOMAIN_CACHE_DIR", str(domains))
    monkeypatch.setenv("CASCADE_PREPARED_CACHE_DIR", str(prepared))

    assert default_domain_cache_dir() == domains.resolve()
    assert default_prepared_cache_root() == prepared.resolve()
