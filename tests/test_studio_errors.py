from __future__ import annotations

from pathlib import Path

from cascade.gui.errors import studio_error_log_path, write_studio_exception


def test_studio_exception_is_persisted_outside_package(
    monkeypatch, tmp_path: Path
) -> None:
    logs = tmp_path / "Diagnostic Logs"
    monkeypatch.setenv("CASCADE_LOG_DIR", str(logs))

    try:
        raise RuntimeError("native startup failed")
    except RuntimeError as exc:
        path = write_studio_exception(type(exc), exc, exc.__traceback__)

    assert path == logs.resolve() / "cascade-studio-errors.log"
    text = studio_error_log_path().read_text(encoding="utf-8")
    assert "RuntimeError: native startup failed" in text
    assert "test_studio_exception_is_persisted_outside_package" in text
