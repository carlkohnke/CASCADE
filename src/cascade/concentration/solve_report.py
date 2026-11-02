"""Per-solve warnings for network transport convergence and GPU fallbacks.

Warnings are printed and flushed, so they reach the CLI and the CASCADE Studio
run log, which captures worker output.  Inside one concentration solve each
distinct warning is printed once and the totals are merged into the solve
timings; outside a solve every call reports for itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

from cascade.configuration import solver_state as _state

NETWORK_TRANSPORT_TOL = 1.0e-6
# Network transport used to stop here without checking convergence.
LEGACY_NETWORK_ITERATIONS = 100


def network_iteration_limit(nnode: int) -> int:
    """Return the Jacobi safety bound for a cyclic graph with ``nnode`` nodes.

    Each sweep moves junction information about one node downstream, so a
    contracting graph converges well inside ten sweeps per node.  The bound
    only stops non-contracting cycles from iterating forever.
    """
    return max(LEGACY_NETWORK_ITERATIONS, 10 * int(nnode))


def _empty_report() -> dict:
    return {
        "warned": set(),
        "network_calls": 0,
        "network_iterations_max": 0,
        "network_unconverged_calls": 0,
        "network_max_delta": 0.0,
        "gpu_fallbacks": {},
    }


@contextmanager
def concentration_solve_report():
    """Collect network-transport and fallback diagnostics for one solve."""
    outer = _state._CONCENTRATION_SOLVE_REPORT
    report = outer if outer is not None else _empty_report()
    _state._CONCENTRATION_SOLVE_REPORT = report
    try:
        yield report
    finally:
        _state._CONCENTRATION_SOLVE_REPORT = outer


def reports_concentration_solve(func):
    """Run ``func`` as one reported solve and merge totals into its timings."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with concentration_solve_report() as report:
            result = func(*args, **kwargs)
            _state._LAST_CONCENTRATION_TIMINGS.update(solve_report_details(report))
        return result

    return wrapper


def _warn_once(key: object, message: str) -> None:
    report = _state._CONCENTRATION_SOLVE_REPORT
    if report is not None:
        if key in report["warned"]:
            return
        report["warned"].add(key)
    print(f"Warning: {message}", flush=True)


def record_network_transport(
    *, backend: str, iterations: int, max_delta: float, limit: int
) -> bool:
    """Record one fixed-point network solve and warn when it was slow or stalled."""
    converged = bool(max_delta < NETWORK_TRANSPORT_TOL)
    report = _state._CONCENTRATION_SOLVE_REPORT
    if report is not None:
        report["network_calls"] += 1
        report["network_iterations_max"] = max(
            int(report["network_iterations_max"]), int(iterations)
        )
        report["network_max_delta"] = max(
            float(report["network_max_delta"]), float(max_delta)
        )
        report["network_unconverged_calls"] += 0 if converged else 1
    if not converged:
        _warn_once(
            "network_unconverged",
            f"{backend} network oxygen transport did not converge: stopped at the "
            f"safety limit of {int(iterations)} iterations (limit {int(limit)}) with "
            f"residual {float(max_delta):.3e} above tolerance "
            f"{NETWORK_TRANSPORT_TOL:.0e}. Vessel concentrations may be inaccurate.",
        )
    elif iterations > LEGACY_NETWORK_ITERATIONS:
        _warn_once(
            "network_slow",
            f"{backend} network oxygen transport needed {int(iterations)} iterations "
            f"to converge (residual {float(max_delta):.3e}, tolerance "
            f"{NETWORK_TRANSPORT_TOL:.0e}); the former "
            f"{LEGACY_NETWORK_ITERATIONS}-iteration limit would have stopped it "
            "before convergence.",
        )
    return converged


def record_gpu_fallback(stage: str, exc: BaseException) -> str:
    """Warn that ``stage`` left the GPU after ``exc``; return the recorded reason."""
    reason = f"{type(exc).__name__}: {exc}"
    report = _state._CONCENTRATION_SOLVE_REPORT
    if report is not None:
        report["gpu_fallbacks"].setdefault(stage, reason)
    _warn_once(
        ("gpu_fallback", stage, reason),
        f"GPU {stage} failed ({reason}); falling back to CPU.",
    )
    return reason


def solve_report_details(report: dict) -> dict[str, float | int | bool | str]:
    """Return the timing-detail fields summarizing ``report``."""
    return {
        "cext_network_calls": int(report["network_calls"]),
        "cext_network_iterations_max": int(report["network_iterations_max"]),
        "cext_network_max_delta": float(report["network_max_delta"]),
        "cext_network_converged": int(report["network_unconverged_calls"]) == 0,
        "cext_gpu_fallbacks": "; ".join(
            f"{stage}: {reason}" for stage, reason in report["gpu_fallbacks"].items()
        ),
    }


__all__ = [
    "LEGACY_NETWORK_ITERATIONS",
    "NETWORK_TRANSPORT_TOL",
    "concentration_solve_report",
    "network_iteration_limit",
    "record_gpu_fallback",
    "record_network_transport",
    "reports_concentration_solve",
    "solve_report_details",
]
