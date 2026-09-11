"""CPU/GPU backend discovery and accelerator integration."""

from .backend import GpuProbeResult, gpu_requested, probe_gpu_runtime, require_gpu_runtime

__all__ = ["GpuProbeResult", "gpu_requested", "probe_gpu_runtime", "require_gpu_runtime"]
