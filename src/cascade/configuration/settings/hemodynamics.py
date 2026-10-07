"""Default pressure, flow, viscosity, and Kirchhoff solver settings."""

from __future__ import annotations

DEFAULTS = {
    # Pressure at the root/inlet of the tree, in pascals.
    "ROOT_PRESSURE": 66661.0,
    # Pressure assigned to terminal outlets before any side-length scaling, in pascals.
    "TERMINAL_PRESSURE": 40000.0,
    # Target inlet flow used when a run does not provide a different flow, in microliters per minute.
    "QIN_TARGET": 900.0,
    # Density of a user-defined constant-viscosity perfusate, in g/cm^3.
    "CUSTOM_FLUID_DENSITY_G_CM3": 1.0,
    # Dynamic viscosity of a user-defined perfusate, in centipoise.
    "CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP": 1.0,
    # If true, scale requested terminal counts by the domain volume. Usually left false in CASCADE runs.
    "SCALE_NTERMS_BY_VOLUME": False,
    # If true, scale inlet flow by side_length^3 so larger domains receive proportionally more flow.
    "SCALE_Q_BY_VOLUME": True,
    # If true, scale the pressure drop by side_length^3 before solving vessel flows.
    "SCALE_dP_BY_VOLUME": True,
    # Kirchhoff flow solver family. "tree" uses the fast tree-specialized solver when possible.
    "KIRCHHOFF_SOLVER": "auto",
    # Boundary condition mode for flow. The legacy mode enforces equal terminal flow then reconstructs pressure.
    "KIRCHHOFF_BC_MODE": "legacy_equal_terminal_flow",
    # If true, compare the tree-specialized solver against a sparse solver for debugging.
    "KIRCHHOFF_VALIDATE_TREE": False,
    # Sparse solver used as the reference when validating the tree-specialized solver.
    "KIRCHHOFF_VALIDATE_SPARSE_SOLVER": "spsolve",
    # Sparse Kirchhoff solver used when not using the tree-specialized solver.
    "KIRCHHOFF_SPARSE_SOLVER": "spsolve",
    # Node-count threshold where automatic sparse solving may switch to conjugate gradient.
    "KIRCHHOFF_CG_MIN_NODES": 50000,
    # Relative residual tolerance for conjugate-gradient pressure solves.
    "KIRCHHOFF_CG_RTOL": 1.0e-10,
    # Maximum conjugate-gradient iterations before the solve is considered failed.
    "KIRCHHOFF_CG_MAXITER": 10000,
    # Relative residual tolerance for GMRES pressure solves.
    "KIRCHHOFF_GMRES_RTOL": 1.0e-4,
    # Maximum GMRES iterations before the solve is considered failed.
    "KIRCHHOFF_GMRES_MAXITER": 1000,
    # Number of Krylov vectors kept before GMRES restarts; larger values use more memory.
    "KIRCHHOFF_GMRES_RESTART": 300,
    # Drop tolerance for incomplete-LU preconditioning; smaller values keep more matrix entries.
    "KIRCHHOFF_ILU_DROP_TOL": 1.0e-7,
    # Maximum fill allowed in incomplete-LU preconditioning; larger values can be more robust but use more memory.
    "KIRCHHOFF_ILU_FILL_FACTOR": 300,
    # If true, diagonally rescale the pressure system before GMRES to improve conditioning.
    "KIRCHHOFF_GMRES_EQUILIBRATE": True,
    # Lower bound for diagonal scaling entries, relative to the median positive diagonal.
    "KIRCHHOFF_GMRES_EQ_DIAG_FLOOR_REL": 1.0e-12,
    # Column-ordering methods tried when building the incomplete-LU preconditioner.
    "KIRCHHOFF_ILU_PERMC_SPECS": ("COLAMD", "MMD_AT_PLUS_A", "NATURAL"),
    # Small diagonal shifts tried if incomplete-LU factorization is unstable.
    "KIRCHHOFF_ILU_SHIFT_RELS": (0.0, 1.0e-14, 1.0e-12),
    # If true, retry GMRES without equilibration if the scaled system fails.
    "KIRCHHOFF_GMRES_RETRY_UNSCALED_IF_EQ_FAIL": True,
    # If true, print flow-solver timing and residual diagnostics during runs.
    "KIRCHHOFF_DIAGNOSTICS": True,
}

ALIASES = {
    "root_pressure": "ROOT_PRESSURE",
    "terminal_pressure": "TERMINAL_PRESSURE",
    "qin_target": "QIN_TARGET",
    "qin_target_ul_min": "QIN_TARGET",
    "custom_fluid_density_g_cm3": "CUSTOM_FLUID_DENSITY_G_CM3",
    "custom_fluid_dynamic_viscosity_cp": "CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP",
    "scale_nterms_by_volume": "SCALE_NTERMS_BY_VOLUME",
    "scale_q_by_volume": "SCALE_Q_BY_VOLUME",
    "scale_dp_by_volume": "SCALE_dP_BY_VOLUME",
    "solver": "KIRCHHOFF_SOLVER",
    "bc_mode": "KIRCHHOFF_BC_MODE",
    "validate_tree": "KIRCHHOFF_VALIDATE_TREE",
    "sparse_solver": "KIRCHHOFF_SPARSE_SOLVER",
    "cg_rtol": "KIRCHHOFF_CG_RTOL",
    "cg_maxiter": "KIRCHHOFF_CG_MAXITER",
    "gmres_rtol": "KIRCHHOFF_GMRES_RTOL",
    "gmres_maxiter": "KIRCHHOFF_GMRES_MAXITER",
    "gmres_restart": "KIRCHHOFF_GMRES_RESTART",
    "ilu_drop_tol": "KIRCHHOFF_ILU_DROP_TOL",
    "ilu_fill_factor": "KIRCHHOFF_ILU_FILL_FACTOR",
    "diagnostics": "KIRCHHOFF_DIAGNOSTICS",
}

PREFIXES = ("KIRCHHOFF_",)

DEPRECATED = {}
