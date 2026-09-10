from __future__ import annotations

import json

import pytest

from cascade.execution import (
    SimulationAlreadyRunningError,
    guard_simulation,
    single_simulation,
)


def test_single_simulation_rejects_overlapping_owner(tmp_path):
    lock_path = tmp_path / "simulation.lock"
    with single_simulation("first", lock_path=lock_path):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["operation"] == "first"
        with pytest.raises(SimulationAlreadyRunningError, match="one memory-intensive operation"):
            with single_simulation("second", lock_path=lock_path):
                pass

    assert lock_path.read_text(encoding="utf-8") == ""


def test_simulation_guard_releases_lock_after_failure(tmp_path, monkeypatch):
    lock_path = tmp_path / "simulation.lock"
    monkeypatch.setenv("CASCADE_SIMULATION_LOCK_PATH", str(lock_path))

    @guard_simulation("failure-test")
    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        fail()

    with single_simulation("next", lock_path=lock_path):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["operation"] == "next"
