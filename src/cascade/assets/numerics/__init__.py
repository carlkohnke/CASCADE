"""Versioned numerical tables shipped with CASCADE.

The tables in this package are generated offline and validated before a
release. Runtime solvers load them read-only and never regenerate them in a
user's working directory.
"""

from __future__ import annotations

from pathlib import Path

BESSEL_K_TABLE = Path(__file__).with_name("bessel_k_v1.npz")
GRAETZ_BASIS_TABLE = Path(__file__).with_name("graetz_basis_8x4_v1.npz")

GRAETZ_N_RADIAL = 8
GRAETZ_N_MODES = 4
GRAETZ_BI_PER_DECADE = 128
GRAETZ_MIN_BI = 1.0e-8
GRAETZ_MAX_BI = 1.0e6
GRAETZ_PROFILES = ("poiseuille", "plug")

__all__ = [
    "BESSEL_K_TABLE",
    "GRAETZ_BASIS_TABLE",
    "GRAETZ_BI_PER_DECADE",
    "GRAETZ_MAX_BI",
    "GRAETZ_MIN_BI",
    "GRAETZ_N_MODES",
    "GRAETZ_N_RADIAL",
    "GRAETZ_PROFILES",
]
