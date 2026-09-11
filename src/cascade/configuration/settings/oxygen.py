"""Default intravascular oxygen transport and Graetz-model settings."""

from __future__ import annotations

import os

DEFAULTS = {
    # Default intravascular concentration solver used when a run does not specify one.
    "CONCENTRATION_SOLVER": "network_ext",
    # Inlet oxygen concentration by fluid type, in the concentration units used by the solver.
    "CONCENTRATION_INLET_BY_FLUID": {
        # Inlet oxygen concentration for water-filled vessel simulations.
        "water": 0.2211,
        # Inlet oxygen concentration for blood simulations.
        "blood": 0.14,
        # Inlet oxygen concentration for cell-media simulations.
        "cell media": 0.2211,
        # Short alias for cell-media inlet oxygen concentration.
        "media": 0.2211,
    },
    # Reference concentration used to normalize output metrics such as C_tiss_over_Cmax.
    "CONC_MAX_FOR_NORMALIZATION": 0.14,
    # Oxygen diffusivity in tissue, in cm^2/s.
    "SOLUTE_DIFFUSIVITY": 2.41e-5,
    # Michaelis-Menten maximum tissue consumption rate.
    "VMAX_MM": 0.04,
    # Michaelis-Menten concentration where consumption is half of VMAX_MM.
    "K_M_MM": 0.0069,
    # Axial discretization steps used by older blood concentration approximations.
    "AXIAL_BLOOD_STEPS": 5,
    # Relaxation factor for iterative network concentration solves.
    "OMEGA": 0.7,
    # Number of Gauss-Legendre quadrature points per vessel segment for tissue Greens integrals.
    "GL_ORDER": 5,
    # Number of Gauss-Legendre quadrature points per segment for explicit Cext coupling.
    "GL_ORDER_CEXT": 1,
    # Tissue porosity used by legacy transport calculations.
    "POROSITY": 0.9,
    # Finite-radius oxygen correction mode for vessel sources; "both" applies both supported corrections.
    "FINITE_RADIUS_O2_TERMS": os.environ.get("SVV_FINITE_RADIUS_O2_TERMS", "both")
    .strip()
    .lower(),
    # Lumen-wall closure model; "graetz" models radial lumen gradients, "wellmixed" assumes uniform lumen concentration.
    "LUMEN_WALL_CLOSURE": os.environ.get("SVV_LUMEN_WALL_CLOSURE", "graetz")
    .strip()
    .lower(),
    # Fixed radial nodes in the validated, precomputed Graetz basis.
    "GRAETZ_N_RADIAL": 8,
    # Fixed retained modes in the validated, precomputed Graetz basis.
    "GRAETZ_N_MODES": 4,
    # Maximum fixed-point iterations for solving Graetz wall/lumen coupling per segment.
    "GRAETZ_MAX_FP_ITERS": int(os.environ.get("SVV_GRAETZ_MAX_FP_ITERS", "4")),
    # Convergence tolerance for Graetz fixed-point iterations.
    "GRAETZ_FP_TOL": float(os.environ.get("SVV_GRAETZ_FP_TOL", "1e-5")),
    # Velocity profile assumed inside the vessel lumen for Graetz calculations.
    "GRAETZ_VELOCITY_PROFILE": os.environ.get(
        "SVV_GRAETZ_VELOCITY_PROFILE", "poiseuille"
    )
    .strip()
    .lower(),
    # Resolution and range of the validated, precomputed Graetz basis.
    "GRAETZ_BI_CACHE_PER_DECADE": 128,
    "GRAETZ_MIN_BI": 1.0e-8,
    "GRAETZ_MAX_BI": 1.0e6,
    # If true, print extra diagnostics for Graetz basis and fixed-point behavior.
    "GRAETZ_DEBUG_DIAGNOSTICS": os.environ.get("SVV_GRAETZ_DEBUG_DIAGNOSTICS", "0")
    .strip()
    .lower()
    in ("1", "true", "yes", "on"),
    # Oxygen diffusivity inside water/media-filled lumens, in cm^2/s.
    "LUMEN_DIFFUSIVITY_WATER_CM2_S": float(
        os.environ.get("SVV_LUMEN_DIFFUSIVITY_WATER_CM2_S", "3.2e-5")
    ),
    # Oxygen diffusivity inside blood-filled lumens, in cm^2/s.
    "LUMEN_DIFFUSIVITY_BLOOD_CM2_S": float(
        os.environ.get("SVV_LUMEN_DIFFUSIVITY_BLOOD_CM2_S", "2.41e-5")
    ),
    # Optional explicit lumen diffusivity override; when set, it wins over fluid-specific values.
    "LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S": (
        float(os.environ["SVV_LUMEN_DIFFUSIVITY_CM2_S"])
        if "SVV_LUMEN_DIFFUSIVITY_CM2_S" in os.environ
        else None
    ),
    # Effective lumen diffusivity currently used by legacy code paths.
    "LUMEN_DIFFUSIVITY_CM2_S": float(
        os.environ.get(
            "SVV_LUMEN_DIFFUSIVITY_CM2_S",
            os.environ.get("SVV_LUMEN_DIFFUSIVITY_BLOOD_CM2_S", "2.41e-5"),
        )
    ),
    # Hemoglobin-bound oxygen capacity per unit tube hematocrit.
    "O2_CAP_PER_HCT": 20.3,
    # Dissolved oxygen solubility coefficient in blood model units per mmHg.
    "ALPHA_MMHG": 1.408e-3,
    # Oxygen partial pressure where hemoglobin is 50 percent saturated, in mmHg.
    "P50_MMHG": 26.5,
    # Hill exponent controlling the steepness of the hemoglobin saturation curve.
    "N_HILL": 2.7,
    # Deprecated legacy background tissue concentration; current Cext paths compute local extravascular concentration.
    "EXTRAVASCULAR_CONCENTRATION": 0.0,
}

ALIASES = {
    "solver": "CONCENTRATION_SOLVER",
    "concentration_solver": "CONCENTRATION_SOLVER",
    "cmax": "CONC_MAX_FOR_NORMALIZATION",
    "conc_max": "CONC_MAX_FOR_NORMALIZATION",
    "solute_diffusivity": "SOLUTE_DIFFUSIVITY",
    "diffusivity": "SOLUTE_DIFFUSIVITY",
    "vmax": "VMAX_MM",
    "vmax_mm": "VMAX_MM",
    "km": "K_M_MM",
    "k_m": "K_M_MM",
    "omega": "OMEGA",
    "gl_order": "GL_ORDER",
    "gl_order_cext": "GL_ORDER_CEXT",
    "finite_radius_o2_terms": "FINITE_RADIUS_O2_TERMS",
    "lumen_wall_closure": "LUMEN_WALL_CLOSURE",
    "lumen_diffusivity_cm2_s": "LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S",
}

PREFIXES = (
    "GRAETZ_",
    "LUMEN_",
)

DEPRECATED = {
    "EXTRAVASCULAR_CONCENTRATION": (
        "Deprecated: current topdown_ext/Cext paths compute extravascular concentration from vessel sources; "
        "this legacy scalar is retained only for compatibility."
    ),
}
