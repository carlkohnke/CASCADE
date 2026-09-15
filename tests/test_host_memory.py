from __future__ import annotations

from cascade.utils.execution import host_memory_snapshot


def test_native_host_memory_snapshot_is_populated() -> None:
    snapshot = host_memory_snapshot()

    for key in (
        "process_rss_bytes",
        "host_available_bytes",
        "host_total_bytes",
    ):
        assert isinstance(snapshot[key], int)
        assert snapshot[key] > 0
    assert snapshot["host_available_bytes"] <= snapshot["host_total_bytes"]
