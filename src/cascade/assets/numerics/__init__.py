"""Versioned numerical tables shipped with CASCADE.

The tables in this package are generated offline and validated before a
release. Runtime solvers load them read-only and never regenerate them in a
user's working directory.
"""

from __future__ import annotations

from pathlib import Path

BESSEL_K_TABLE = Path(__file__).with_name("bessel_k_v1.npz")
GRAETZ_BASIS_TABLE = Path(__file__).with_name("graetz_basis_8x4_v1.npz")
GRAETZ_VALIDATION_BASIS_TABLE = Path(__file__).with_name(
    "graetz_basis_6x3_validation_v1.npz"
)
GRAETZ_HIGH_RESOLUTION_BASIS_TABLE = GRAETZ_BASIS_TABLE

# The production profile is active after completing the locked 6x3 regression
# campaign. The validation profile remains packaged for reproducible reruns.
GRAETZ_N_RADIAL = 8
GRAETZ_N_MODES = 4
GRAETZ_BI_PER_DECADE = 128
GRAETZ_MIN_BI = 1.0e-8
GRAETZ_MAX_BI = 1.0e6
GRAETZ_PROFILES = ("poiseuille", "plug")

GRAETZ_HIGH_RESOLUTION_N_RADIAL = 8
GRAETZ_HIGH_RESOLUTION_N_MODES = 4
GRAETZ_HIGH_RESOLUTION_BI_PER_DECADE = 128

GRAETZ_VALIDATION_N_RADIAL = 6
GRAETZ_VALIDATION_N_MODES = 3
GRAETZ_VALIDATION_BI_PER_DECADE = 16

__all__ = [
    "BESSEL_K_TABLE",
    "GRAETZ_BASIS_TABLE",
    "GRAETZ_BI_PER_DECADE",
    "GRAETZ_HIGH_RESOLUTION_BASIS_TABLE",
    "GRAETZ_HIGH_RESOLUTION_BI_PER_DECADE",
    "GRAETZ_HIGH_RESOLUTION_N_MODES",
    "GRAETZ_HIGH_RESOLUTION_N_RADIAL",
    "GRAETZ_MAX_BI",
    "GRAETZ_MIN_BI",
    "GRAETZ_N_MODES",
    "GRAETZ_N_RADIAL",
    "GRAETZ_PROFILES",
    "GRAETZ_VALIDATION_BASIS_TABLE",
    "GRAETZ_VALIDATION_BI_PER_DECADE",
    "GRAETZ_VALIDATION_N_MODES",
    "GRAETZ_VALIDATION_N_RADIAL",
]
