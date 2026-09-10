from __future__ import annotations

import json

import pytest

from cascade.execution import (
    SimulationAlreadyRunningError,
    guard_simulation,
    release_completed_case_memory,
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


def test_release_memory_can_retain_accelerator_pools_for_sequential_reuse():
    class Pool:
        def __init__(self):
            self.calls = 0

        def free_all_blocks(self):
            self.calls += 1

    class Stream:
        def synchronize(self):
            pass

    class CP:
        device_pool = Pool()
        pinned_pool = Pool()

        class cuda:
            class Stream:
                null = Stream()

        @classmethod
        def get_default_memory_pool(cls):
            return cls.device_pool

        @classmethod
        def get_default_pinned_memory_pool(cls):
            return cls.pinned_pool

    class Runtime:
        _cp = CP
        _LAST_CEXT_SOURCE_STATE = {"large": "result"}
        _LAST_CEXT_CONTEXT = {"large": "context"}
        _LAST_TISSUE_TIMINGS = {"time": 1.0}

    report = release_completed_case_memory(Runtime, trim_accelerator_pools=False)

    assert report["cupy_pool_trimmed"] is False
    assert CP.device_pool.calls == CP.pinned_pool.calls == 0
    assert Runtime._LAST_CEXT_SOURCE_STATE is None
    assert Runtime._LAST_CEXT_CONTEXT is None
