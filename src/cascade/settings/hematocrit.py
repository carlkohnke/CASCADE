from __future__ import annotations

DEFAULTS = {
    # Hematocrit model used for blood; "pries_secomb" includes vessel-size-dependent red-cell partitioning.
    "HEMATOCRIT_MODEL": "pries_secomb",
    # Number of times flow and hematocrit are recomputed together to account for viscosity changes.
    "HEMATOCRIT_FLOW_ITERATIONS": 2,
    # Relaxation factor for hematocrit updates; 1.0 fully accepts each new estimate.
    "HEMATOCRIT_RELAXATION": 1.0,
    # Flow-change tolerance for stopping hematocrit-flow iterations, in nanoliters per minute.
    "HEMATOCRIT_QTOL_NL_MIN": 1.0e-3,
    # Hematocrit-change tolerance for stopping hematocrit-flow iterations.
    "HEMATOCRIT_HDTOL": 1.0e-3,
    # Lowest allowed discharge hematocrit after numerical updates.
    "HEMATOCRIT_MIN": 0.0,
    # Highest allowed discharge hematocrit after numerical updates.
    "HEMATOCRIT_MAX": 0.95,
    # If true, print per-iteration hematocrit convergence information.
    "HEMATOCRIT_DIAGNOSTICS": True,
    # Root discharge hematocrit, meaning the red-cell volume fraction entering the root segment.
    "HD_DISCHARGE": 0.42,
    # NetFlow bifurcation-fit coefficient 1 for red-cell splitting at vessel branches.
    "NETFLOW_BIFPAR_1": 0.964,
    # NetFlow bifurcation-fit coefficient 2 for red-cell splitting at vessel branches.
    "NETFLOW_BIFPAR_2": 6.98,
    # NetFlow bifurcation-fit coefficient 3 for red-cell splitting at vessel branches.
    "NETFLOW_BIFPAR_3": -13.29,
    # NetFlow cell-fraction fit coefficient 1 used by the hematocrit/viscosity model.
    "NETFLOW_CPAR_1": 0.80,
    # NetFlow cell-fraction fit coefficient 2 used by the hematocrit/viscosity model.
    "NETFLOW_CPAR_2": -0.075,
    # NetFlow cell-fraction fit coefficient 3 used by the hematocrit/viscosity model.
    "NETFLOW_CPAR_3": -11.0,
    # NetFlow cell-fraction fit coefficient 4 used by the hematocrit/viscosity model.
    "NETFLOW_CPAR_4": 12.0,
    # NetFlow viscosity-fit coefficient 1 for blood apparent viscosity in small vessels.
    "NETFLOW_VISCPAR_1": 6.0,
    # NetFlow viscosity-fit coefficient 2 for blood apparent viscosity in small vessels.
    "NETFLOW_VISCPAR_2": -0.085,
    # NetFlow viscosity-fit coefficient 3 for blood apparent viscosity in small vessels.
    "NETFLOW_VISCPAR_3": 3.2,
    # NetFlow viscosity-fit coefficient 4 for blood apparent viscosity in small vessels.
    "NETFLOW_VISCPAR_4": -2.44,
    # NetFlow viscosity-fit coefficient 5 for blood apparent viscosity in small vessels.
    "NETFLOW_VISCPAR_5": -0.06,
    # NetFlow viscosity-fit coefficient 6 for blood apparent viscosity in small vessels.
    "NETFLOW_VISCPAR_6": 0.645,
    # NetFlow optimal vessel-width scale in micrometers for viscosity corrections.
    "NETFLOW_OPTW_UM": 1.1,
    # Plasma viscosity in centipoise used by the NetFlow hematocrit model.
    "NETFLOW_VPLAS_CP": 1.0466,
    # Mean cell volume in femtoliters used to rescale NetFlow blood-cell geometry.
    "NETFLOW_MCV_FL": 55.0,
}

ALIASES = {
    "model": "HEMATOCRIT_MODEL",
    "flow_iterations": "HEMATOCRIT_FLOW_ITERATIONS",
    "iterations": "HEMATOCRIT_FLOW_ITERATIONS",
    "relaxation": "HEMATOCRIT_RELAXATION",
    "qtol_nl_min": "HEMATOCRIT_QTOL_NL_MIN",
    "hdtol": "HEMATOCRIT_HDTOL",
    "min": "HEMATOCRIT_MIN",
    "max": "HEMATOCRIT_MAX",
    "diagnostics": "HEMATOCRIT_DIAGNOSTICS",
    "hd_discharge": "HD_DISCHARGE",
}

PREFIXES = ("HEMATOCRIT_", "NETFLOW_")

DEPRECATED = {}
