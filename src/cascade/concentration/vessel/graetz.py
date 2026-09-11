"""Graetz lumen-wall closure.

The closure uses versioned packaged basis data and can also construct a basis
for unsupported parameter combinations.
"""

from __future__ import annotations

import math

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.assets.numerics import (
    GRAETZ_BI_PER_DECADE as _PRECOMPUTED_BI_PER_DECADE,
    GRAETZ_MAX_BI as _PRECOMPUTED_MAX_BI,
    GRAETZ_MIN_BI as _PRECOMPUTED_MIN_BI,
    GRAETZ_N_MODES as _PRECOMPUTED_N_MODES,
    GRAETZ_N_RADIAL as _PRECOMPUTED_N_RADIAL,
    GRAETZ_PROFILES as _PRECOMPUTED_PROFILES,
    GRAETZ_VALIDATION_BASIS_TABLE as _VALIDATION_BASIS_TABLE,
    GRAETZ_VALIDATION_BI_PER_DECADE as _VALIDATION_BI_PER_DECADE,
    GRAETZ_VALIDATION_N_MODES as _VALIDATION_N_MODES,
    GRAETZ_VALIDATION_N_RADIAL as _VALIDATION_N_RADIAL,
)


def _graetz_velocity_profile(rho: np.ndarray, profile_name: str) -> np.ndarray:
    profile = str(profile_name or "poiseuille").strip().lower()
    rho_arr = np.asarray(rho, dtype=float)
    if profile == "plug":
        return np.ones_like(rho_arr, dtype=float)
    if profile == "poiseuille":
        return 2.0 * np.maximum(1.0 - rho_arr * rho_arr, 0.0)
    raise ValueError("--graetz-profile must be 'poiseuille' or 'plug'.")


def _graetz_build_matrices(
    Bi: float,
    n_radial: int,
    profile_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = max(int(n_radial), 2)
    rho_nodes = np.linspace(0.0, 1.0, n, dtype=float)
    K = np.zeros((n, n), dtype=float)
    M = np.zeros((n, n), dtype=float)
    cup = np.zeros((n,), dtype=float)
    gp = np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], dtype=float)
    gw = np.array([1.0, 1.0], dtype=float)
    for elem in range(n - 1):
        x0 = float(rho_nodes[elem])
        x1 = float(rho_nodes[elem + 1])
        h = max(x1 - x0, 1e-30)
        ids = (elem, elem + 1)
        for xi, wi in zip(gp, gw):
            rho = 0.5 * (x0 + x1) + 0.5 * h * float(xi)
            jac = 0.5 * h * float(wi)
            N0 = (x1 - rho) / h
            N1 = (rho - x0) / h
            dN = np.array([-1.0 / h, 1.0 / h], dtype=float)
            N = np.array([N0, N1], dtype=float)
            v = float(
                _graetz_velocity_profile(np.array([rho], dtype=float), profile_name)[0]
            )
            stiff_w = rho * jac
            mass_w = rho * v * jac
            for a in range(2):
                ia = ids[a]
                cup[ia] += 2.0 * mass_w * N[a]
                for b in range(2):
                    ib = ids[b]
                    K[ia, ib] += stiff_w * dN[a] * dN[b]
                    M[ia, ib] += mass_w * N[a] * N[b]
    K[-1, -1] += max(float(Bi), 0.0)
    cup_sum = float(np.sum(cup))
    if cup_sum > 0.0:
        cup /= cup_sum
    return K, M, rho_nodes, cup


def _graetz_get_basis(
    Bi: float,
    n_radial: int,
    n_modes: int,
    profile_name: str,
) -> dict:
    if _state._scipy_linalg is None:
        raise RuntimeError(
            "Graetz closure requires scipy.linalg in the active Python environment."
        )
    n = max(int(n_radial), 2)
    nm = max(1, min(int(n_modes), n))
    K, M, rho, cup = _graetz_build_matrices(float(Bi), n, profile_name)
    eigvals, eigvecs = _state._scipy_linalg.eigh(K, M, check_finite=False)
    order = np.argsort(eigvals)
    mu2 = np.maximum(np.asarray(eigvals[order[:nm]], dtype=float), 0.0)
    phi = np.asarray(eigvecs[:, order[:nm]], dtype=float)
    for j in range(phi.shape[1]):
        norm = float(math.sqrt(max(phi[:, j].T @ M @ phi[:, j], 1e-30)))
        phi[:, j] /= norm
    return {
        "Bi": float(Bi),
        "rho": rho,
        "K": K,
        "M": M,
        "mu2": mu2,
        "phi": phi,
        "project": phi.T @ M,
        "cup_weights": cup,
    }


def _graetz_get_basis_table(
    n_radial: int,
    n_modes: int,
    profile_name: str,
    *,
    bi_per_decade: int | None = None,
    min_bi: float | None = None,
    max_bi: float | None = None,
) -> dict:
    n = max(int(n_radial), 2)
    nm = max(1, min(int(n_modes), n))
    profile = str(profile_name or "poiseuille").strip().lower()
    per_decade = max(
        int(
            bi_per_decade
            if bi_per_decade is not None
            else _state.GRAETZ_BI_CACHE_PER_DECADE
        ),
        1,
    )
    bi_min = max(float(min_bi if min_bi is not None else _state.GRAETZ_MIN_BI), 1e-30)
    bi_max = max(float(max_bi if max_bi is not None else _state.GRAETZ_MAX_BI), bi_min)
    requested = (n, nm, per_decade, bi_min, bi_max)
    production_profile = (
        _PRECOMPUTED_N_RADIAL,
        _PRECOMPUTED_N_MODES,
        _PRECOMPUTED_BI_PER_DECADE,
        _PRECOMPUTED_MIN_BI,
        _PRECOMPUTED_MAX_BI,
    )
    validation_profile = (
        _VALIDATION_N_RADIAL,
        _VALIDATION_N_MODES,
        _VALIDATION_BI_PER_DECADE,
        _PRECOMPUTED_MIN_BI,
        _PRECOMPUTED_MAX_BI,
    )
    if requested == production_profile:
        basis_path = _state.GRAETZ_BASIS_PATH
        table_version = "production-8x4-v1"
    elif requested == validation_profile:
        basis_path = _VALIDATION_BASIS_TABLE
        table_version = "validation-6x3-v1"
    else:
        raise ValueError(
            "CASCADE's Graetz closure accepts the precomputed production "
            f"{_PRECOMPUTED_N_RADIAL}-node/{_PRECOMPUTED_N_MODES}-mode basis "
            f"({_PRECOMPUTED_BI_PER_DECADE} Bi bins per decade, "
            "or the reference-compatible "
            f"{_VALIDATION_N_RADIAL}-node/{_VALIDATION_N_MODES}-mode basis "
            f"({_VALIDATION_BI_PER_DECADE} bins per decade), both over "
            f"{_PRECOMPUTED_MIN_BI:g} <= Bi <= {_PRECOMPUTED_MAX_BI:g}; got {requested}."
        )
    if profile not in _PRECOMPUTED_PROFILES:
        raise ValueError(
            f"Unsupported Graetz velocity profile {profile!r}; choose one of {_PRECOMPUTED_PROFILES}."
        )
    key_min = int(round(math.log10(_PRECOMPUTED_MIN_BI) * per_decade))
    key_max = int(round(math.log10(_PRECOMPUTED_MAX_BI) * per_decade))
    cache_key = (table_version, profile)
    cached = _state._GRAETZ_BASIS_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    n_keys = key_max - key_min + 1
    try:
        with np.load(basis_path, allow_pickle=False) as data:
            metadata = {
                "format_version": int(np.asarray(data["format_version"]).item()),
                "n_radial": int(np.asarray(data["n_radial"]).item()),
                "n_modes": int(np.asarray(data["n_modes"]).item()),
                "per_decade": int(np.asarray(data["per_decade"]).item()),
                "key_min": int(np.asarray(data["key_min"]).item()),
                "key_max": int(np.asarray(data["key_max"]).item()),
            }
            bi_values = np.asarray(data["bi_values"], dtype=np.float32).copy()
            mu2_table = np.asarray(data[f"{profile}_mu2"], dtype=np.float32).copy()
            phi_table = np.asarray(data[f"{profile}_phi"], dtype=np.float32).copy()
            project_table = np.asarray(
                data[f"{profile}_project"], dtype=np.float32
            ).copy()
            cup_table = np.asarray(
                data[f"{profile}_cup_weights"], dtype=np.float32
            ).copy()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load the packaged Graetz basis at {basis_path}: {exc}"
        ) from exc
    expected_metadata = {
        "format_version": 1,
        "n_radial": n,
        "n_modes": nm,
        "per_decade": per_decade,
        "key_min": key_min,
        "key_max": key_max,
    }
    if metadata != expected_metadata:
        raise RuntimeError(
            f"Graetz basis metadata mismatch at {basis_path}: "
            f"expected {expected_metadata}, found {metadata}."
        )
    expected_shapes = {
        "bi_values": (n_keys,),
        "mu2": (n_keys, nm),
        "phi": (n_keys, n, nm),
        "project": (n_keys, nm, n),
        "cup_weights": (n_keys, n),
    }
    arrays = {
        "bi_values": bi_values,
        "mu2": mu2_table,
        "phi": phi_table,
        "project": project_table,
        "cup_weights": cup_table,
    }
    for name, values in arrays.items():
        if values.shape != expected_shapes[name] or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Invalid {name} array in Graetz basis at {basis_path}."
            )
    if not np.all(mu2_table >= 0.0) or not np.all(np.diff(bi_values) > 0.0):
        raise RuntimeError(
            f"Invalid eigenvalues or Bi ordering in {basis_path}."
        )
    table = {
        "profile": profile,
        "n_radial": n,
        "n_modes": nm,
        "per_decade": per_decade,
        "key_min": key_min,
        "key_max": key_max,
        "min_bi": _PRECOMPUTED_MIN_BI,
        "max_bi": _PRECOMPUTED_MAX_BI,
        "bi_values": bi_values,
        "mu2": mu2_table,
        "phi": phi_table,
        "project": project_table,
        "cup_weights": cup_table,
    }
    _state._GRAETZ_BASIS_TABLE_CACHE[cache_key] = table
    return table


__all__ = [
    "_graetz_velocity_profile",
    "_graetz_build_matrices",
    "_graetz_get_basis",
    "_graetz_get_basis_table",
]
