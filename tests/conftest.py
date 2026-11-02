from __future__ import annotations

import os
from pathlib import Path

import pytest


if os.environ.get("CASCADE_TEST_INSTALLED") != "1":
    _SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
    _existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_SOURCE_ROOT), *([_existing_pythonpath] if _existing_pythonpath else [])]
    )


@pytest.fixture
def gpu_subprocess_guard() -> None:
    """Skip when a CASCADE subprocess could not open a GPU this process holds."""
    from cascade.accelerators.cuda_driver import cuda_devices, process_has_cuda_context

    if process_has_cuda_context() and any(
        device.exclusive_process for device in cuda_devices()
    ):
        pytest.skip(
            "the GPU is in Exclusive_Process mode and this pytest process already "
            "holds a CUDA context, so a CASCADE subprocess cannot open it; run "
            "this test in its own pytest process"
        )
