from __future__ import annotations

import numpy as np

DEFAULTS = {
    # If true, use numba-compiled concentration kernels when numba is installed.
    "CONC_USE_NUMBA": True,
    # If true, collect and print detailed timing fields for flow, concentration, Cext, and tissue steps.
    "SOLVER_TIMING_DETAILS": True,
    # Floating-point dtype used for tree geometry and flow arrays in public-svv-compatible runs.
    "TREE_DATA_DTYPE": np.float64,
    # Integer dtype used for tree connectivity and segment-index arrays.
    "TREE_INDEX_DTYPE": np.int64,
    # If true, prefer a lightweight fast tree cache when loading saved trees.
    "TREE_FAST_CACHE_ENABLE": False,
    # Legacy switch for saving generated trees from the old script-style workflow.
    "SAVE_TREES": False,
    # If true, allow tree cache lookup in legacy-compatible code paths.
    "USE_TREE_CACHE": True,
    # Directory name used by legacy-compatible tree-cache helpers.
    "TREE_CACHE_DIRNAME": "trees_cache",
    # CSV index filename used by legacy-compatible tree-cache helpers.
    "TREE_CACHE_INDEX_NAME": "tree_index.csv",
}

ALIASES = {
    "conc_use_numba": "CONC_USE_NUMBA",
    "solver_timing_details": "SOLVER_TIMING_DETAILS",
    "tree_data_dtype": "TREE_DATA_DTYPE",
    "tree_index_dtype": "TREE_INDEX_DTYPE",
    "tree_fast_cache_enable": "TREE_FAST_CACHE_ENABLE",
    "save_trees": "SAVE_TREES",
    "use_tree_cache": "USE_TREE_CACHE",
}

PREFIXES = ("TREE_",)

DEPRECATED = {}
