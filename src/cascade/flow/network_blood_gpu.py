"""Compatibility entry point for the CUDA blood-hemodynamics pipeline."""

from .network_blood import solve_network_blood as solve_network_blood_gpu

__all__ = ["solve_network_blood_gpu"]
