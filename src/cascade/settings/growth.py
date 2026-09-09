from __future__ import annotations

import os

DEFAULTS = {
    # Terminal count at which growth switches from CCO bifurcation optimization to the fast equal-bifurcation rule.
    "N_EQUAL_BIFURCATIONS": None,
    # If true, equal-bifurcation mode only adds from terminal segments instead of any eligible segment.
    "EQUAL_TERMINAL_ONLY_ENABLE": True,
    # Number of equal-bifurcation additions attempted in one batch before bookkeeping is refreshed.
    "EQUAL_TERMINAL_BATCH_SIZE": 500,
    # Number of candidate branch directions tested for each terminal in equal-bifurcation mode.
    "EQUAL_TERMINAL_N_CANDIDATES": 3,
    # Smallest allowed branch opening angle, in degrees, for equal-bifurcation candidate generation.
    "EQUAL_TERMINAL_ALPHA_MIN_DEG": 5.0,
    # Preferred branch opening angle, in degrees, used when generating equal-bifurcation candidates.
    "EQUAL_TERMINAL_ALPHA_MODE_DEG": 37.5,
    # Largest allowed branch opening angle, in degrees, for equal-bifurcation candidate generation.
    "EQUAL_TERMINAL_ALPHA_MAX_DEG": 85.0,
    # Angular step, in degrees, used when sweeping candidate directions around the parent segment.
    "EQUAL_TERMINAL_PSI_STEP_DEG": 10.0,
    # Maximum angular sweep, in degrees, around the parent segment when searching directions.
    "EQUAL_TERMINAL_PSI_MAX_DEG": 120.0,
    # Extra distance required between a candidate segment and the domain boundary.
    "EQUAL_TERMINAL_DOMAIN_MARGIN": 0.0,
    # If true, check the midpoint of each proposed segment against the domain as well as the endpoint.
    "EQUAL_TERMINAL_CHECK_MIDPOINT": True,
    # If true, delay expensive nearest-neighbor index updates during equal-bifurcation batches.
    "EQUAL_TERMINAL_DEFER_HNSW_UPDATES": True,
    # If true, store per-add timing details for equal-bifurcation debugging.
    "EQUAL_TERMINAL_RECORD_ADD_TIMES": False,
    # If true, print equal-bifurcation timing summaries while growing.
    "EQUAL_TERMINAL_REPORT_TIMINGS": False,
    # Report equal-bifurcation progress every this many batches when timing reports are enabled.
    "EQUAL_TERMINAL_REPORT_EVERY": 1,
    # Worker count used for domain-inside checks during equal-bifurcation growth.
    "EQUAL_TERMINAL_DOMAIN_WORKERS": max(1, (os.cpu_count() or 1) - 2),
    # Candidate chunk size for equal-bifurcation domain checks; 0 lets the code choose.
    "EQUAL_TERMINAL_DOMAIN_CHUNK": 0,
    # Minimum number of candidate points before parallel domain checks are used.
    "EQUAL_TERMINAL_DOMAIN_PARALLEL_MIN_POINTS": 0,
    # Fixed equal-bifurcation child length. None means compute length from the selected length model.
    "EQUAL_TERMINAL_LENGTH": None,
    # Length model used when fixed child length is not supplied.
    "EQUAL_TERMINAL_LENGTH_MODE": "gen_f_mix2",
    # Neighbor count used by density-based equal-bifurcation length estimates.
    "EQUAL_TERMINAL_DENSITY_K": 3,
    # Density-model exponent controlling how local vessel crowding changes branch length.
    "EQUAL_TERMINAL_DENSITY_ALPHA": -0.17,
    # Density-model exponent controlling how generation/depth changes branch length.
    "EQUAL_TERMINAL_DENSITY_BETA": 0.6,
    # Small positive floor that prevents divide-by-zero in density-based length formulas.
    "EQUAL_TERMINAL_DENSITY_EPSILON": 0.99,
    # Neighbor count used by the generation-and-flow mixture length model.
    "EQUAL_TERMINAL_GEN_F_K": 3,
    # Mixture weight between the two log-normal components in the generation-and-flow length model.
    "EQUAL_TERMINAL_GEN_F_MIX_W": 0.19928806732283386,
    # Mean of the first log-normal component for the generation-and-flow length model.
    "EQUAL_TERMINAL_GEN_F_MIX_MU1_LN": -1.4940805101469532,
    # Standard deviation of the first log-normal component for the generation-and-flow length model.
    "EQUAL_TERMINAL_GEN_F_MIX_SIG1_LN": 0.6459130145614477,
    # Mean of the second log-normal component for the generation-and-flow length model.
    "EQUAL_TERMINAL_GEN_F_MIX_MU2_LN": -0.9535671641507659,
    # Standard deviation of the second log-normal component for the generation-and-flow length model.
    "EQUAL_TERMINAL_GEN_F_MIX_SIG2_LN": 0.34151613489071186,
    # Intercept for estimating total network length from terminal count on a log10 scale.
    "EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_INTERCEPT": 0.4795,
    # Slope for estimating total network length from terminal count on a log10 scale.
    "EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_SLOPE": 0.688,
    # Multiplicative scale applied to generated equal-bifurcation segment lengths.
    "EQUAL_TERMINAL_LENGTH_SCALE": 0.01,
    # Power-law exponent used by the simple equal-terminal length model.
    "EQUAL_TERMINAL_LENGTH_POWER": -0.33,
    # Minimum equal-bifurcation segment length.
    "EQUAL_TERMINAL_LENGTH_MIN": 1.0e-4,
    # Maximum equal-bifurcation segment length.
    "EQUAL_TERMINAL_LENGTH_MAX": 1.0,
    # Factor used to shorten a candidate if the original length does not fit in the domain.
    "EQUAL_TERMINAL_LENGTH_SHRINK": 0.7,
    # Minimum bifurcation location fraction along the parent segment in equal-bifurcation updates.
    "EQUAL_BIFURCATION_T_MIN": 1.0e-3,
    # Lower bound on generated child radii in equal-bifurcation mode.
    "EQUAL_BIFURCATION_RADIUS_FLOOR": 1.0e-4,
    # Lower bound on generated child lengths in equal-bifurcation mode.
    "EQUAL_BIFURCATION_LENGTH_FLOOR": 1.0e-4,
    # If true, print low-level vessel-add debug output from svv/CASCADE growth routines.
    "DEBUG_ADD_VESSEL": False,
}

ALIASES = {
    "n_equal_bifurcations": "N_EQUAL_BIFURCATIONS",
    "equal_bifurcations": "N_EQUAL_BIFURCATIONS",
    "equal_terminal_only": "EQUAL_TERMINAL_ONLY_ENABLE",
    "equal_terminal_batch_size": "EQUAL_TERMINAL_BATCH_SIZE",
    "equal_terminal_n_candidates": "EQUAL_TERMINAL_N_CANDIDATES",
    "debug_add_vessel": "DEBUG_ADD_VESSEL",
}

PREFIXES = ("EQUAL_TERMINAL_", "EQUAL_BIFURCATION_")

DEPRECATED = {}
