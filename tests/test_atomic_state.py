from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cascade.gui.model import JobRecord, QueueStore
from cascade.utils.files import atomic_text_writer, atomic_write_text


def test_atomic_writers_do_not_share_temporary_names(tmp_path: Path) -> None:
    directory = tmp_path / "Path With Spaces" / "研究"
    target = directory / "state.json"
    payloads = [json.dumps({"writer": index}) for index in range(16)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda payload: atomic_write_text(target, payload), payloads))

    assert target.read_text(encoding="utf-8") in payloads
    assert not list(directory.glob(f".{target.name}.*.tmp"))


def test_failed_atomic_write_preserves_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "queue.json"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_text_writer(target) as handle:
            handle.write("incomplete")
            raise RuntimeError("interrupted")

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_queue_state_remains_valid_during_simultaneous_saves(tmp_path: Path) -> None:
    store = QueueStore(tmp_path)
    jobs = [
        JobRecord(str(index), f"job-{index}", "settings.json", "output")
        for index in range(12)
    ]

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda job: store.save([job]), jobs))

    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["schema_version"] > 0
    assert len(raw["jobs"]) == 1
    assert raw["jobs"][0]["id"] in {job.id for job in jobs}
