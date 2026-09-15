"""Create and configure individual SVV trees.

The helpers apply CASCADE's active flow and geometry settings before delegating
incremental branch insertion to the installed SVV implementation.
"""

from __future__ import annotations

from typing import Optional

from cascade.configuration import solver_state as _state
from cascade.diagnostics.runtime import _require_tree_class


def grow_tree(
    domain: _state.Domain,
    vessels_to_add: int,
    *,
    dlp_enable: bool,
    min_theta: float,
    side_length: Optional[float] = None,
    terminal_flow_override: Optional[float] = None,
    fluid: str | None = None,
    n_closest_vessels: int = 5,
    n_points: int = 7,
    weighted_sampling: bool = False,
    ignore_collisions: bool = False,
    allow_inside_vessels: bool = False,
) -> _state.Tree:
    _require_tree_class()
    tree = _state.Tree(
        data_dtype=_state.TREE_DATA_DTYPE, index_dtype=_state.TREE_INDEX_DTYPE
    )
    tree.set_domain(domain)
    tree.parameters.root_pressure = _state.ROOT_PRESSURE
    if _state.SCALE_dP_BY_VOLUME and side_length is not None:
        tree.parameters.terminal_pressure = _state.ROOT_PRESSURE - (
            abs(_state.ROOT_PRESSURE - _state.TERMINAL_PRESSURE)
        ) * (side_length**3)
        if tree.parameters.terminal_pressure < 0:
            raise ValueError(
                "Terminal pressure is negative after scaling; aborting run."
            )
    else:
        tree.parameters.terminal_pressure = _state.TERMINAL_PRESSURE
    fluid_mode = (fluid or _state.ACTIVE_FLUID).lower()
    try:
        tree.parameters.fluid = fluid_mode
    except Exception:
        tree.fluid = fluid_mode
    if fluid_mode in {"water", "cell media", "media"}:
        tree.parameters.kinematic_viscosity = 0.6959 / 100
        tree.parameters.fluid_density = 0.99336
    elif fluid_mode == "blood":
        tree.parameters.fluid_density = 1.06
        mu_plasma_cgs = 0.012
        tree.parameters.kinematic_viscosity = (
            mu_plasma_cgs / tree.parameters.fluid_density
        )
    elif fluid_mode == "custom":
        density = float(_state.CUSTOM_FLUID_DENSITY_G_CM3)
        tree.parameters.fluid_density = density
        tree.parameters.kinematic_viscosity = (
            float(_state.CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP) / 100.0 / density
        )
    n_vessels = max(int(vessels_to_add), 1)
    total_terminals = max(n_vessels + 1, 1)
    if terminal_flow_override is not None:
        tree.parameters.terminal_flow = terminal_flow_override
    else:
        tree.parameters.terminal_flow = (
            _state.QIN_TARGET * 0.00001666666666 / total_terminals
        )
    scale = (
        float(side_length)
        if side_length is not None
        else float(getattr(domain, "characteristic_length", 1.0))
    )
    root_loc = _state.ROOT_LOCATION * scale
    if _state.ROOT_DIR is not None:
        tree.set_root(root_loc, _state.ROOT_DIR)
    else:
        tree.set_root(root_loc)
    _apply_equal_bifurcation(tree)
    tree.n_add(
        n_vessels,
        n_closest_vessels=n_closest_vessels,
        n_points=n_points,
        use_random_int=not weighted_sampling,
        ignore_collisions=ignore_collisions,
        allow_inside_vessels=allow_inside_vessels,
        debug_add_vessel=_state.DEBUG_ADD_VESSEL,
    )
    return tree


def _apply_equal_bifurcation(tree: _state.Tree) -> None:
    if tree is None:
        return
    if _state.N_EQUAL_BIFURCATIONS is None:
        return
    try:
        tree.parameters.n_equal_bifurcations = int(_state.N_EQUAL_BIFURCATIONS)
        tree.parameters.equal_bifurcation_radius_floor = float(
            _state.EQUAL_BIFURCATION_RADIUS_FLOOR
        )
        tree.parameters.equal_bifurcation_length_floor = float(
            _state.EQUAL_BIFURCATION_LENGTH_FLOOR
        )

        # Heart-style terminal-only equal-bif method.
        tree.parameters.equal_bifurcation_terminal_only = bool(
            _state.EQUAL_TERMINAL_ONLY_ENABLE
        )
        tree.parameters.equal_terminal_batch_size = int(
            _state.EQUAL_TERMINAL_BATCH_SIZE
        )
        tree.parameters.equal_terminal_n_candidates = int(
            _state.EQUAL_TERMINAL_N_CANDIDATES
        )
        tree.parameters.equal_terminal_alpha_min_deg = float(
            _state.EQUAL_TERMINAL_ALPHA_MIN_DEG
        )
        tree.parameters.equal_terminal_alpha_mode_deg = float(
            _state.EQUAL_TERMINAL_ALPHA_MODE_DEG
        )
        tree.parameters.equal_terminal_alpha_max_deg = float(
            _state.EQUAL_TERMINAL_ALPHA_MAX_DEG
        )
        tree.parameters.equal_terminal_psi_step_deg = float(
            _state.EQUAL_TERMINAL_PSI_STEP_DEG
        )
        tree.parameters.equal_terminal_psi_max_deg = float(
            _state.EQUAL_TERMINAL_PSI_MAX_DEG
        )
        tree.parameters.equal_terminal_domain_margin = float(
            _state.EQUAL_TERMINAL_DOMAIN_MARGIN
        )
        tree.parameters.equal_terminal_check_midpoint = bool(
            _state.EQUAL_TERMINAL_CHECK_MIDPOINT
        )
        tree.parameters.equal_terminal_domain_workers = int(
            _state.EQUAL_TERMINAL_DOMAIN_WORKERS
        )
        tree.parameters.equal_terminal_domain_chunk = int(
            _state.EQUAL_TERMINAL_DOMAIN_CHUNK
        )
        tree.parameters.equal_terminal_domain_parallel_min_points = int(
            _state.EQUAL_TERMINAL_DOMAIN_PARALLEL_MIN_POINTS
        )
        tree.parameters.equal_terminal_length = _state.EQUAL_TERMINAL_LENGTH
        tree.parameters.equal_terminal_length_mode = str(
            _state.EQUAL_TERMINAL_LENGTH_MODE or "power"
        )
        tree.parameters.equal_terminal_total_length_log10_intercept = float(
            _state.EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_INTERCEPT
        )
        tree.parameters.equal_terminal_total_length_log10_slope = float(
            _state.EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_SLOPE
        )
        tree.parameters.equal_terminal_length_scale = float(
            _state.EQUAL_TERMINAL_LENGTH_SCALE
        )
        tree.parameters.equal_terminal_length_power = float(
            _state.EQUAL_TERMINAL_LENGTH_POWER
        )
        tree.parameters.equal_terminal_length_min = float(
            _state.EQUAL_TERMINAL_LENGTH_MIN
        )
        tree.parameters.equal_terminal_length_max = float(
            _state.EQUAL_TERMINAL_LENGTH_MAX
        )
        tree.parameters.equal_terminal_length_shrink = float(
            _state.EQUAL_TERMINAL_LENGTH_SHRINK
        )
        tree.parameters.equal_terminal_density_k = int(_state.EQUAL_TERMINAL_DENSITY_K)
        tree.parameters.equal_terminal_density_alpha = float(
            _state.EQUAL_TERMINAL_DENSITY_ALPHA
        )
        tree.parameters.equal_terminal_density_beta = float(
            _state.EQUAL_TERMINAL_DENSITY_BETA
        )
        tree.parameters.equal_terminal_density_epsilon = float(
            _state.EQUAL_TERMINAL_DENSITY_EPSILON
        )
        tree.parameters.equal_terminal_gen_f_k = int(_state.EQUAL_TERMINAL_GEN_F_K)
        tree.parameters.equal_terminal_gen_f_mix_w = float(
            _state.EQUAL_TERMINAL_GEN_F_MIX_W
        )
        tree.parameters.equal_terminal_gen_f_mix_mu1_ln = float(
            _state.EQUAL_TERMINAL_GEN_F_MIX_MU1_LN
        )
        tree.parameters.equal_terminal_gen_f_mix_sig1_ln = float(
            _state.EQUAL_TERMINAL_GEN_F_MIX_SIG1_LN
        )
        tree.parameters.equal_terminal_gen_f_mix_mu2_ln = float(
            _state.EQUAL_TERMINAL_GEN_F_MIX_MU2_LN
        )
        tree.parameters.equal_terminal_gen_f_mix_sig2_ln = float(
            _state.EQUAL_TERMINAL_GEN_F_MIX_SIG2_LN
        )
        tree.parameters.equal_terminal_report_timings = bool(
            _state.EQUAL_TERMINAL_REPORT_TIMINGS
        )
        tree.parameters.equal_terminal_report_every = int(
            _state.EQUAL_TERMINAL_REPORT_EVERY
        )
    except Exception:
        return


def _apply_run_tree_params(
    tree, side_length: float | None, terminal_flow_override: float | None
) -> None:
    if tree is None:
        return
    # Ensure build-related parameters reflect the current run.
    tree.parameters.root_pressure = _state.ROOT_PRESSURE
    if _state.SCALE_dP_BY_VOLUME and side_length is not None:
        tree.parameters.terminal_pressure = _state.ROOT_PRESSURE - (
            abs(_state.ROOT_PRESSURE - _state.TERMINAL_PRESSURE)
        ) * (side_length**3)
        if tree.parameters.terminal_pressure < 0:
            raise ValueError(
                "Terminal pressure is negative after scaling; aborting run."
            )
    else:
        tree.parameters.terminal_pressure = _state.TERMINAL_PRESSURE
    set_tree_fluid(tree, _state.BUILD_FLUID)
    _apply_equal_bifurcation(tree)
    if terminal_flow_override is not None:
        tree.parameters.terminal_flow = terminal_flow_override


def set_tree_fluid(tree, fluid: str) -> None:
    fluid_mode = (fluid or _state.ACTIVE_FLUID).lower()
    try:
        tree.parameters.fluid = fluid_mode
    except Exception:
        tree.fluid = fluid_mode
    if fluid_mode in {"water", "cell media", "media"}:
        tree.parameters.kinematic_viscosity = 0.6959 / 100
        tree.parameters.fluid_density = 0.99336
    elif fluid_mode == "blood":
        tree.parameters.fluid_density = 1.06
        mu_plasma_cgs = 0.012
        tree.parameters.kinematic_viscosity = (
            mu_plasma_cgs / tree.parameters.fluid_density
        )
    elif fluid_mode == "custom":
        density = float(_state.CUSTOM_FLUID_DENSITY_G_CM3)
        tree.parameters.fluid_density = density
        tree.parameters.kinematic_viscosity = (
            float(_state.CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP) / 100.0 / density
        )


__all__ = [
    "grow_tree",
    "_apply_equal_bifurcation",
    "_apply_run_tree_params",
    "set_tree_fluid",
]
