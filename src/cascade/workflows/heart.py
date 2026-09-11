"""Command-line orchestration for heart exports."""

from __future__ import annotations

from .heart_support import (
    AXIAL_BLOOD_STEPS_DEFAULT,
    CEXT_ACCEL_MODE_DEFAULT,
    CEXT_ACTIVE_SET_ABS_TOL_DEFAULT,
    CEXT_ACTIVE_SET_ENABLE_DEFAULT,
    CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT,
    CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION_DEFAULT,
    CEXT_ACTIVE_SET_REFRESH_PERIOD_DEFAULT,
    CEXT_ACTIVE_SET_REL_TOL_DEFAULT,
    CEXT_ACTIVE_SET_STABLE_ITERS_DEFAULT,
    CEXT_ACTIVE_SET_START_DEFAULT,
    CEXT_APPROX_WINDOW_SCALE_DEFAULT,
    CEXT_BG_ASSIGNMENT_DEFAULT,
    CEXT_BG_GRID_DEFAULT,
    CEXT_BG_LAMBDA_BINS_DEFAULT,
    CEXT_BG_MODE_DEFAULT,
    CEXT_BG_NEAR_RADIUS_MULT_DEFAULT,
    CEXT_BG_SOLVER_DEFAULT,
    CEXT_BG_VCYCLES_DEFAULT,
    CEXT_CONCENTRATION_SOLVER_DEFAULT,
    CEXT_FOREST_MODE_DEFAULT,
    CEXT_FROZEN_ACCEL_MODE_DEFAULT,
    CEXT_GL_ORDER_DEFAULT,
    CEXT_GPU_VALIDATE_SEGMENTS_DEFAULT,
    CEXT_GRID_CELL_FACTOR_DEFAULT,
    CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING_DEFAULT,
    CEXT_HYBRID_FFT_BIN_EPOCH_CACHE_DEFAULT,
    CEXT_HYBRID_FFT_O2_CORRECTION_DEFAULT,
    CEXT_HYBRID_FFT_O2_FUSED_IFFT_DEFAULT,
    CEXT_HYBRID_FFT_O2_MOMENT_BATCH_DEFAULT,
    CEXT_HYBRID_FFT_QUANTILE_BINS_DEFAULT,
    CEXT_HYBRID_FFT_RESPONSE_BATCHED_DEFAULT,
    CEXT_HYBRID_FFT_SELF_SUBTRACT_DEFAULT,
    CEXT_HYBRID_FFT_SELF_SUB_FUSED_DEFAULT,
    CEXT_HYBRID_FFT_SELF_SUB_SCALE_DEFAULT,
    CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING_DEFAULT,
    CEXT_HYBRID_GPU_ITERATION_CACHE_DEFAULT,
    CEXT_HYBRID_GPU_RUNTIME_MOMENTS_DEFAULT,
    CEXT_HYBRID_GPU_RUNTIME_STENCIL_DEFAULT,
    CEXT_HYBRID_GPU_RUNTIME_WEIGHTS_DEFAULT,
    CEXT_INIT_MODE_DEFAULT,
    CEXT_LAMBDA_SOURCE_DEFAULT,
    CEXT_MAX_CANDIDATES_PER_TARGET_DEFAULT,
    CEXT_STREAMING_TARGET_CANDIDATE_SLOTS_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_ABS_TOL_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_ENABLE_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_REL_TOL_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_STABLE_ITERS_DEFAULT,
    CEXT_TARGET_ACTIVE_SET_START_DEFAULT,
    CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_ACCEL_DEFAULT,
    CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_ANDERSON_DEPTH_DEFAULT,
    CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO_DEFAULT,
    CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS_DEFAULT,
    CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_ANDERSON_REG_DEFAULT,
    CEXT_VESS_COUPLING_ANDERSON_START_DEFAULT,
    CEXT_VESS_COUPLING_BEST_REVERT_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_BEST_STALL_ITERS_DEFAULT,
    CEXT_VESS_COUPLING_MAX_ITER_DEFAULT,
    CEXT_VESS_COUPLING_OMEGA_DEFAULT,
    CEXT_VESS_COUPLING_OMEGA_MAX_DEFAULT,
    CEXT_VESS_COUPLING_OMEGA_MIN_DEFAULT,
    CEXT_VESS_COUPLING_REL_TOL_DEFAULT,
    CEXT_VESS_COUPLING_STEP_REJECT_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_STEP_RETRY_FACTOR_DEFAULT,
    CEXT_VESS_COUPLING_TOL_DEFAULT,
    CEXT_VESS_COUPLING_TRUST_ABS_DEFAULT,
    CEXT_VESS_COUPLING_TRUST_REL_DEFAULT,
    CONC_MAX_FOR_NORMALIZATION_DEFAULT,
    DEFAULT_OUT_DIR,
    EXPORT_FLOAT_DTYPE_DEFAULT,
    EXPORT_INDEX_DTYPE_DEFAULT,
    FINITE_RADIUS_O2_TERMS_DEFAULT,
    Forest,
    GL_ORDER_DEFAULT,
    GRAETZ_MAX_FP_ITERS_DEFAULT,
    GRAETZ_N_MODES_DEFAULT,
    GRAETZ_N_RADIAL_DEFAULT,
    GRAETZ_VELOCITY_PROFILE_DEFAULT,
    HEMATOCRIT_FLOW_ITERATIONS_DEFAULT,
    HEMATOCRIT_HDTOL_DEFAULT,
    HEMATOCRIT_MODEL_DEFAULT,
    HEMATOCRIT_QTOL_NL_MIN_DEFAULT,
    HEMATOCRIT_RELAXATION_DEFAULT,
    K_M_MM_DEFAULT,
    LUMEN_WALL_CLOSURE_DEFAULT,
    NEAREST_TISSUE_VESSELS_DEFAULT,
    SOLUTE_DIFFUSIVITY_DEFAULT,
    TISSUE_ACCEL_MODE_DEFAULT,
    TISSUE_GPU_CHUNK_POINTS_DEFAULT,
    TISSUE_GPU_VALIDATE_POINTS_DEFAULT,
    TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT,
    VMAX_MM_DEFAULT,
    WINDOW_FACTOR_DEFAULT,
    WORKING_FLOAT_DTYPE_DEFAULT,
    WORKING_INDEX_DTYPE_DEFAULT,
    _apply_cext_overrides,
    _apply_tissuesim_overrides,
    _bool_arg,
    _default_simulation_cache_path,
    _dt,
    _float_dtype_from_name,
    _int_dtype_from_name,
    _load_cext_tissuesim,
    _load_tissuesim,
    _normalize_path,
    _resolve_forest_path,
    _should_use_simulation_cache,
    argparse,
    gc,
    guard_simulation,
    json,
    np,
    perf_counter,
    preload_cuda_component_libraries,
)

from .heart_flow import (
    _cext_chb_max_for_tree,
    _collect_downstream_segment_ids,
    _flow_inputs,
    _resolve_global_segment_id,
    _restore_solution_radii,
    _run_cext_frozen_step,
    _safe_index_array,
    _solve_tree,
    _solve_tree_cext_prepare,
    _tree_root_flow_cm3_s,
)

from .heart_domain import (
    _attach_domain,
    _build_domain,
    _connectivity_report,
    _domain_cache_dir,
    _domain_cache_path,
    _log,
    _make_forest_analysis_only,
    _make_tree_analysis_only,
    _normalize_forest_dtype_attrs,
    _repair_forest_connectivity,
    _repair_tree_parent_columns_from_children,
    _validate_forest_connectivity,
)

from .heart_output import (
    _build_points_polydata,
    _build_vessel_polydata,
    _compute_tissue,
    _concat_tree_solutions,
    _filter_export_points,
    _get_boundary,
    _grid_axes,
    _grid_points,
    _grid_points_from_axes,
    _inside_grid_points_chunked,
    _inside_mask,
    _load_tissue_points,
    _save_domain_outputs,
    _sha256_file,
)

from .heart_cext import (
    _apply_global_cext_update,
    _compact_cext_state_for_concat,
    _compute_global_gfm_cext,
    _compute_shared_plain_box_cext,
    _concat_global_cext_context,
    _concat_global_cext_state,
    _global_cext_box,
    _global_lambda_bins,
    _make_tree_hybrid_for_global_box,
    _release_cext_transient_gpu_cache,
    _snapshot_sol_cext_state,
    _solve_forest_cext,
    _solve_forest_cext_backend_per_tree,
    _solve_forest_cext_legacy_shared_one_shot,
    _solve_forest_cext_shared_global,
    _solve_shared_cext_fft,
    _solve_tree_cext_backend,
)

@guard_simulation("cascade export-heart")
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cascade export-heart",
        description="Export a two-tree heart .forest plus tissue oxygen grid with optional shared FFT Cext for ParaView/ParaFlow.",
    )
    parser.add_argument("--forest", required=True, help="Input .forest path.")
    parser.add_argument("--no-simulation-cache", action="store_true", help="Disable automatic simulation-cache read/write.")
    parser.add_argument("--domain", dest="domain_path", required=True, help="Heart domain path (.stl or .dmn).")
    parser.add_argument("--domain-stl", dest="domain_path", help="Deprecated alias for --domain.")
    parser.add_argument(
        "--domain-cache-dir",
        default=None,
        help=(
            "Directory for reusable CASCADE .dmn caches of STL/VTP domains. "
            "Defaults to CASCADE_DOMAIN_CACHE_DIR or the platform user cache."
        ),
    )
    parser.add_argument(
        "--no-domain-cache",
        action="store_true",
        help="Rebuild a file-backed domain instead of reading or writing the CASCADE user cache.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to forest stem plus _cext when Cext is enabled.")
    parser.add_argument("--side-length", type=float, default=1.0)
    parser.add_argument("--fluid", default="blood", choices=("blood", "water", "media", "cell media"))
    parser.add_argument(
        "--concentration-solver",
        default="topdown",
        choices=("topdown", "network", "topdown_ext", "topdown_ext_hybrid_bg", "topdown_ext_treecode"),
        help="No-Cext concentration solver. Cext mode uses --cext-concentration-solver where applicable.",
    )
    parser.add_argument("--no-cext", action="store_true", help="Disable Cext and use the packaged per-tree solver.")
    parser.add_argument(
        "--cext-forest-mode",
        default=CEXT_FOREST_MODE_DEFAULT,
        choices=("shared-global", "shared-global-fft", "backend-per-tree", "legacy-shared-one-shot"),
        help=(
            "shared-global uses one combined two-tree CASCADE source/target context for FFT, pairwise, or hybrid modes; "
            "shared-global-fft is a compatibility alias requiring bg_mode=fft; "
            "backend-per-tree solves each tree independently with CASCADE's packaged runtime."
        ),
    )
    parser.add_argument(
        "--cext-concentration-solver",
        default=CEXT_CONCENTRATION_SOLVER_DEFAULT,
        choices=("topdown_ext_hybrid_bg", "topdown_ext_treecode", "topdown_ext"),
        help="Concentration solver used by --cext-forest-mode backend-per-tree.",
    )
    parser.add_argument("--cext-accel", default=CEXT_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--cext-frozen-accel", default=CEXT_FROZEN_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--cext-init-mode", default=CEXT_INIT_MODE_DEFAULT, choices=("zero", "decoupled_greens"))
    parser.add_argument("--cext-lambda-source", default=CEXT_LAMBDA_SOURCE_DEFAULT, choices=("lambda_vv", "lambda_t"))
    parser.add_argument("--cext-window-factor", type=float, default=WINDOW_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-max-iter", type=int, default=CEXT_VESS_COUPLING_MAX_ITER_DEFAULT)
    parser.add_argument("--cext-vess-coupling-tol", type=float, default=CEXT_VESS_COUPLING_TOL_DEFAULT)
    parser.add_argument("--cext-vess-coupling-rel-tol", type=float, default=CEXT_VESS_COUPLING_REL_TOL_DEFAULT)
    parser.add_argument("--cext-vess-coupling-omega", type=float, default=CEXT_VESS_COUPLING_OMEGA_DEFAULT)
    parser.add_argument("--cext-vess-coupling-accel", default=CEXT_VESS_COUPLING_ACCEL_DEFAULT, choices=("none", "aitken", "anderson"))
    parser.add_argument("--cext-vess-coupling-omega-min", type=float, default=CEXT_VESS_COUPLING_OMEGA_MIN_DEFAULT)
    parser.add_argument("--cext-vess-coupling-omega-max", type=float, default=CEXT_VESS_COUPLING_OMEGA_MAX_DEFAULT)
    parser.add_argument("--cext-vess-coupling-trust-abs", type=float, default=CEXT_VESS_COUPLING_TRUST_ABS_DEFAULT)
    parser.add_argument("--cext-vess-coupling-trust-rel", type=float, default=CEXT_VESS_COUPLING_TRUST_REL_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-depth", type=int, default=CEXT_VESS_COUPLING_ANDERSON_DEPTH_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-reg", type=float, default=CEXT_VESS_COUPLING_ANDERSON_REG_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-start", type=int, default=CEXT_VESS_COUPLING_ANDERSON_START_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-gate-ratio", type=float, default=CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-min-stable-iters", type=int, default=CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS_DEFAULT)
    parser.add_argument("--cext-vess-coupling-anderson-omega-gate-factor", type=float, default=CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-accept-factor", type=float, default=CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-step-factor", type=float, default=CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-restart-factor", type=float, default=CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-best-stall-iters", type=int, default=CEXT_VESS_COUPLING_BEST_STALL_ITERS_DEFAULT)
    parser.add_argument("--cext-vess-coupling-best-revert-factor", type=float, default=CEXT_VESS_COUPLING_BEST_REVERT_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-step-reject-factor", type=float, default=CEXT_VESS_COUPLING_STEP_REJECT_FACTOR_DEFAULT)
    parser.add_argument("--cext-vess-coupling-step-retry-factor", type=float, default=CEXT_VESS_COUPLING_STEP_RETRY_FACTOR_DEFAULT)
    parser.add_argument("--cext-active-set-enable", default="true" if CEXT_ACTIVE_SET_ENABLE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-active-set-start", type=int, default=CEXT_ACTIVE_SET_START_DEFAULT)
    parser.add_argument("--cext-active-set-stable-iters", type=int, default=CEXT_ACTIVE_SET_STABLE_ITERS_DEFAULT)
    parser.add_argument("--cext-active-set-rel-tol", type=float, default=CEXT_ACTIVE_SET_REL_TOL_DEFAULT)
    parser.add_argument("--cext-active-set-abs-tol", type=float, default=CEXT_ACTIVE_SET_ABS_TOL_DEFAULT)
    parser.add_argument("--cext-active-set-refresh-period", type=int, default=CEXT_ACTIVE_SET_REFRESH_PERIOD_DEFAULT)
    parser.add_argument("--cext-active-set-min-active-count", type=int, default=CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT)
    parser.add_argument("--cext-active-set-min-active-fraction", type=float, default=CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION_DEFAULT)
    parser.add_argument("--cext-target-active-set-enable", default="true" if CEXT_TARGET_ACTIVE_SET_ENABLE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-target-active-set-start", type=int, default=CEXT_TARGET_ACTIVE_SET_START_DEFAULT)
    parser.add_argument("--cext-target-active-set-stable-iters", type=int, default=CEXT_TARGET_ACTIVE_SET_STABLE_ITERS_DEFAULT)
    parser.add_argument("--cext-target-active-set-rel-tol", type=float, default=CEXT_TARGET_ACTIVE_SET_REL_TOL_DEFAULT)
    parser.add_argument("--cext-target-active-set-abs-tol", type=float, default=CEXT_TARGET_ACTIVE_SET_ABS_TOL_DEFAULT)
    parser.add_argument("--cext-target-active-set-min-active-count", type=int, default=CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT_DEFAULT)
    parser.add_argument("--cext-target-active-set-neighbor-pad", type=int, default=CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD_DEFAULT)
    parser.add_argument("--cext-hybrid-bg-mode", default=CEXT_BG_MODE_DEFAULT, choices=("local_only_nlambda", "local_only", "fft", "hybrid"))
    parser.add_argument("--cext-bg-grid", "--cext-hybrid-bg-grid", dest="cext_hybrid_bg_grid", type=int, default=CEXT_BG_GRID_DEFAULT)
    parser.add_argument("--cext-bg-lambda-bins", "--cext-hybrid-bg-lambda-bins", dest="cext_hybrid_bg_lambda_bins", type=int, default=CEXT_BG_LAMBDA_BINS_DEFAULT)
    parser.add_argument("--cext-bg-assignment", "--cext-hybrid-bg-assignment", dest="cext_hybrid_bg_assignment", default=CEXT_BG_ASSIGNMENT_DEFAULT, choices=("cic", "tsc"))
    parser.add_argument("--cext-bg-solver", "--cext-hybrid-bg-solver", dest="cext_hybrid_bg_solver", default=CEXT_BG_SOLVER_DEFAULT, choices=("auto", "fft", "jacobi"))
    parser.add_argument("--cext-hybrid-bg-near-radius-mult", type=float, default=CEXT_BG_NEAR_RADIUS_MULT_DEFAULT)
    parser.add_argument("--cext-hybrid-bg-vcycles", type=int, default=CEXT_BG_VCYCLES_DEFAULT)
    parser.add_argument("--cext-hybrid-bg-enable-source-freezing", default="true" if CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-quantile-bins", default="true" if CEXT_HYBRID_FFT_QUANTILE_BINS_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-o2-correction", default="true" if CEXT_HYBRID_FFT_O2_CORRECTION_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-self-subtract", default="true" if CEXT_HYBRID_FFT_SELF_SUBTRACT_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-self-sub-target-sampling", default=CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING_DEFAULT, choices=("cic", "tsc", "matched", "assignment", "source"))
    parser.add_argument("--cext-hybrid-fft-self-sub-scale", type=float, default=CEXT_HYBRID_FFT_SELF_SUB_SCALE_DEFAULT)
    parser.add_argument("--cext-hybrid-fft-bin-epoch-cache", default="true" if CEXT_HYBRID_FFT_BIN_EPOCH_CACHE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-response-batched", default="true" if CEXT_HYBRID_FFT_RESPONSE_BATCHED_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-o2-fused-ifft", default="true" if CEXT_HYBRID_FFT_O2_FUSED_IFFT_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-self-sub-fused", default="true" if CEXT_HYBRID_FFT_SELF_SUB_FUSED_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-fft-o2-moment-batch", type=int, default=CEXT_HYBRID_FFT_O2_MOMENT_BATCH_DEFAULT)
    parser.add_argument("--cext-hybrid-gpu-iteration-cache", default="true" if CEXT_HYBRID_GPU_ITERATION_CACHE_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-gpu-runtime-weights", default="true" if CEXT_HYBRID_GPU_RUNTIME_WEIGHTS_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-gpu-runtime-stencil", default="true" if CEXT_HYBRID_GPU_RUNTIME_STENCIL_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-hybrid-gpu-runtime-moments", default="true" if CEXT_HYBRID_GPU_RUNTIME_MOMENTS_DEFAULT else "false", choices=("true", "false"))
    parser.add_argument("--cext-grid-cell-factor", type=float, default=CEXT_GRID_CELL_FACTOR_DEFAULT)
    parser.add_argument("--cext-streaming-target-candidate-slots", type=int, default=CEXT_STREAMING_TARGET_CANDIDATE_SLOTS_DEFAULT)
    parser.add_argument("--cext-gpu-validate-segments", type=int, default=CEXT_GPU_VALIDATE_SEGMENTS_DEFAULT)
    parser.add_argument("--cext-approx-window-scale", type=float, default=CEXT_APPROX_WINDOW_SCALE_DEFAULT)
    parser.add_argument("--cext-max-candidates-per-target", type=int, default=CEXT_MAX_CANDIDATES_PER_TARGET_DEFAULT)
    parser.add_argument("--finite-radius-o2-terms", default=FINITE_RADIUS_O2_TERMS_DEFAULT, choices=("none", "monopole", "dipole", "both"))
    parser.add_argument("--lumen-wall-closure", default=LUMEN_WALL_CLOSURE_DEFAULT, choices=("wellmixed", "graetz"))
    parser.add_argument("--graetz-n-radial", type=int, default=GRAETZ_N_RADIAL_DEFAULT, choices=(8,))
    parser.add_argument("--graetz-n-modes", type=int, default=GRAETZ_N_MODES_DEFAULT, choices=(4,))
    parser.add_argument("--graetz-max-fp-iters", type=int, default=GRAETZ_MAX_FP_ITERS_DEFAULT)
    parser.add_argument("--graetz-profile", default=GRAETZ_VELOCITY_PROFILE_DEFAULT, choices=("poiseuille", "plug"))
    parser.add_argument("--lumen-diffusivity-cm2-s", type=float, default=None)
    parser.add_argument("--cext-gl-order", type=int, default=CEXT_GL_ORDER_DEFAULT, choices=(1, 5, 9, 20))
    parser.add_argument(
        "--kirchhoff-bc-mode",
        default=None,
        choices=("legacy_equal_terminal_flow", "terminal_pressure"),
        help=(
            "Flow boundary condition mode used by the packaged CASCADE heart runtime. "
            "legacy_equal_terminal_flow prescribes total inlet flow and equal terminal sinks; "
            "terminal_pressure prescribes inlet flow with terminal pressures."
        ),
    )
    parser.add_argument(
        "--flow-source",
        default="tree-root-flow",
        choices=("tree-root-flow", "total-qin-split"),
        help=(
            "'tree-root-flow' uses the root_flow saved in each forest tree. "
            "'total-qin-split' overrides those magnitudes by splitting --total-qin-ul-min "
            "across trees in proportion to their saved root_flow."
        ),
    )
    parser.add_argument(
        "--total-qin-ul-min",
        type=float,
        default=None,
        help="Total inlet flow in uL/min, only used with --flow-source total-qin-split.",
    )
    parser.add_argument("--tissue-accel", default=TISSUE_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--tissue-gpu-chunk-points", type=int, default=TISSUE_GPU_CHUNK_POINTS_DEFAULT)
    parser.add_argument("--tissue-gpu-validate-points", type=int, default=TISSUE_GPU_VALIDATE_POINTS_DEFAULT)
    parser.add_argument("--solute-diffusivity", type=float, default=SOLUTE_DIFFUSIVITY_DEFAULT)
    parser.add_argument("--vmax-mm", type=float, default=VMAX_MM_DEFAULT)
    parser.add_argument("--km-mm", type=float, default=K_M_MM_DEFAULT)
    parser.add_argument("--nearest-tissue-vessels", type=int, default=NEAREST_TISSUE_VESSELS_DEFAULT)
    parser.add_argument("--window-factor", type=float, default=WINDOW_FACTOR_DEFAULT)
    parser.add_argument("--gl-order", type=int, default=GL_ORDER_DEFAULT, choices=(5, 9, 20))
    parser.add_argument(
        "--cext-tissue-quadrature-mode",
        default="independent",
        choices=("independent", "legacy_cext"),
        help=(
            "Use independent tissue quadrature (production default), or reuse the "
            "Cext source nodes to reproduce the historical oracle timing/field behavior."
        ),
    )
    parser.add_argument("--tissue-kdtree-candidate-mult", type=int, default=TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT)
    parser.add_argument("--axial-blood-steps", type=int, default=AXIAL_BLOOD_STEPS_DEFAULT)
    parser.add_argument("--conc-max-for-normalization", type=float, default=CONC_MAX_FOR_NORMALIZATION_DEFAULT)
    parser.add_argument(
        "--hematocrit-model",
        default=HEMATOCRIT_MODEL_DEFAULT,
        choices=("uniform_tube", "pries_secomb"),
    )
    parser.add_argument("--hematocrit-flow-iterations", type=int, default=HEMATOCRIT_FLOW_ITERATIONS_DEFAULT)
    parser.add_argument("--hematocrit-relaxation", type=float, default=HEMATOCRIT_RELAXATION_DEFAULT)
    parser.add_argument("--hematocrit-qtol-nl-min", type=float, default=HEMATOCRIT_QTOL_NL_MIN_DEFAULT)
    parser.add_argument("--hematocrit-hdtol", type=float, default=HEMATOCRIT_HDTOL_DEFAULT)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=200)
    parser.add_argument(
        "--tissue-points",
        default=None,
        help=(
            "Optional .npy array with shape (N, 3) containing explicit tissue sample coordinates in cm. "
            "When provided, these points replace generated --nx/--ny/--nz grid filtering."
        ),
    )
    parser.add_argument("--boundary-resolution", type=int, default=28)
    parser.add_argument("--implicit-margin", type=float, default=0.0)
    parser.add_argument("--disable-enclosed-check", action="store_true", default=True)
    parser.add_argument("--enable-enclosed-check", dest="disable_enclosed_check", action="store_false")
    parser.add_argument("--tissue-grid-chunk-points", type=int, default=250_000)
    parser.add_argument("--enclosed-tolerance", type=float, default=1e-6)
    parser.add_argument("--inside-combine-mode", default="and", choices=("and", "or"))
    parser.add_argument("--vessel-resolution", type=int, default=2)
    parser.add_argument(
        "--skip-vessel-output",
        action="store_true",
        help="Skip writing the forest vessel VTP while still computing downstream tissue outputs.",
    )
    parser.add_argument(
        "--fraction-blocked-infarction",
        type=float,
        default=0.0,
        help=(
            "Solve-time infarction severity for the selected global vessel segment. "
            "0.0 = none, 0.5 = 50%% radius reduction, 1.0 = full occlusion "
            "(segment + downstream subtree forced to zero flow/concentration outputs)."
        ),
    )
    parser.add_argument(
        "--infarction-global-segment-id",
        type=int,
        default=17,
        help="Exporter global_segment_id to target for solve-time infarction.",
    )
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--skip-tissue-oxygen", action="store_true")
    parser.add_argument(
        "--report-global-segment-id",
        type=int,
        default=None,
        help="Record and log the solved flow for one global segment id.",
    )
    parser.add_argument("--include-tissue-nearest-fields", action="store_true")
    parser.add_argument("--validate-connectivity", action="store_true", default=False)
    parser.add_argument("--no-validate-connectivity", dest="validate_connectivity", action="store_false")
    parser.add_argument("--no-repair-connectivity", action="store_true", help="Do not repair stale parent columns before validation/solve.")
    parser.add_argument("--connectivity-geometry-atol", type=float, default=1e-6)
    parser.add_argument("--no-fail-connectivity", action="store_true")
    parser.add_argument(
        "--working-float-dtype",
        default=WORKING_FLOAT_DTYPE_DEFAULT,
        choices=("float32", "float64"),
        help="In-memory dtype used while loading and solving the forest.",
    )
    parser.add_argument(
        "--working-index-dtype",
        default=WORKING_INDEX_DTYPE_DEFAULT,
        choices=("int32", "int64"),
        help="In-memory dtype used for reconstructed connectivity arrays.",
    )
    parser.add_argument(
        "--export-float-dtype",
        default=EXPORT_FLOAT_DTYPE_DEFAULT,
        choices=("float32", "float64"),
        help="Floating-point dtype for user-facing exported VTP arrays.",
    )
    parser.add_argument(
        "--export-index-dtype",
        default=EXPORT_INDEX_DTYPE_DEFAULT,
        choices=("int32", "int64"),
        help="Integer dtype for user-facing exported ID arrays. Default keeps vessel IDs as int64.",
    )
    args = parser.parse_args(argv)
    working_float_dtype = _float_dtype_from_name(args.working_float_dtype)
    working_index_dtype = _int_dtype_from_name(args.working_index_dtype)
    export_float_dtype = _float_dtype_from_name(args.export_float_dtype)
    export_index_dtype = _int_dtype_from_name(args.export_index_dtype)
    cext_enabled = not bool(args.no_cext)
    if cext_enabled and str(args.concentration_solver).lower() == "network":
        raise ValueError("Cext mode is topdown-only. Use --no-cext with --concentration-solver network.")
    cext_forest_mode = str(args.cext_forest_mode).lower()
    cext_bg_mode = str(args.cext_hybrid_bg_mode).lower()
    if cext_enabled and cext_forest_mode == "shared-global-fft":
        if cext_bg_mode != "fft":
            raise ValueError("--cext-forest-mode shared-global-fft requires --cext-hybrid-bg-mode fft.")
    if cext_enabled and cext_bg_mode == "fft":
        if str(args.cext_hybrid_bg_solver).lower() == "jacobi":
            raise ValueError("Shared-global heart Cext requires --cext-bg-solver auto or fft.")
    pct_blocked = float(args.fraction_blocked_infarction)
    if not np.isfinite(pct_blocked) or pct_blocked < 0.0 or pct_blocked > 1.0:
        raise ValueError("--fraction-blocked-infarction must be in [0, 1].")
    if pct_blocked > 0.0 and args.infarction_global_segment_id is None:
        raise ValueError("--infarction-global-segment-id is required when --fraction-blocked-infarction > 0.")

    t_start = perf_counter()
    forest_path = _resolve_forest_path(args.forest)
    out_dir = _normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (f"{forest_path.stem}_cext" if cext_enabled else forest_path.stem)

    _log("Loading packaged CASCADE heart runtime...")
    ts = _load_tissuesim()
    solver_parameters = _apply_tissuesim_overrides(ts, args)
    cext_ts = None
    cext_parameters = {"enabled": False}
    if cext_enabled:
        preload_cuda_component_libraries()
        _log("Loading packaged CASCADE Cext runtime...")
        cext_ts = _load_cext_tissuesim()
        if cext_ts._cp is None:
            raise RuntimeError("Cext mode requires GPU/CuPy support. Rerun with --no-cext for the packaged per-tree path.")
        cext_parameters = _apply_cext_overrides(cext_ts, ts, args)
    _log(
        "Oxygen parameters: "
        f"D={ts.SOLUTE_DIFFUSIVITY:g} vmax={ts.VMAX_MM:g} km={ts.K_M_MM:g} "
        f"nearby={ts.NEAREST_TISSUE_VESSELS} window={ts.WINDOW_FACTOR:g} gl={ts.GL_ORDER} "
        f"hematocrit={ts.HEMATOCRIT_MODEL}/{ts.HEMATOCRIT_FLOW_ITERATIONS}"
    )
    _log(f"Tissue accel: mode={ts.TISSUE_ACCEL_MODE} gpu_chunk_points={ts.TISSUE_GPU_CHUNK_POINTS}")
    if cext_enabled:
        _log(
            f"Cext mode: {args.cext_forest_mode} "
            f"solver={args.cext_concentration_solver} bg_mode={args.cext_hybrid_bg_mode} "
            f"grid={args.cext_hybrid_bg_grid} bins={args.cext_hybrid_bg_lambda_bins} "
            f"assignment={args.cext_hybrid_bg_assignment} bg_solver={cext_parameters.get('effective_bg_solver')} "
            f"gl_order={args.cext_gl_order} max_iter={args.cext_vess_coupling_max_iter} "
            f"wall={args.lumen_wall_closure} o2_terms={args.finite_radius_o2_terms}"
        )
    else:
        _log("Cext mode: disabled (--no-cext); using the packaged per-tree concentration/tissue path.")
    if args.kirchhoff_bc_mode is not None:
        _log(f"Kirchhoff BC mode: {ts.KIRCHHOFF_BC_MODE}")

    _log(f"Loading forest: {forest_path}")
    cache_path = _default_simulation_cache_path(forest_path)
    cache_enabled = not bool(args.no_simulation_cache)
    load_source = forest_path
    if cache_enabled and _should_use_simulation_cache(forest_path, cache_path):
        load_source = cache_path
        _log(f"Using simulation cache: {cache_path}")
    elif cache_enabled:
        try:
            _log(f"Building simulation cache without vessel maps: {cache_path}")
            Forest.build_simulation_cache_from_legacy(str(forest_path), str(cache_path), show_progress=True)
            load_source = cache_path
            _log(f"Wrote simulation cache: {cache_path}")
        except Exception as exc:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"Fast simulation-cache build failed ({exc}). "
                "Aborting instead of falling back to the legacy forest loader."
            ) from exc
    forest = Forest.load(
        str(load_source),
        mode="simulation",
        data_dtype=working_float_dtype,
        index_dtype=working_index_dtype,
    )
    _make_forest_analysis_only(forest)
    _log("Forest load mode: analysis-only (skipping build-time spatial indices)")
    if cache_enabled and load_source == forest_path:
        try:
            forest.save_simulation_cache(str(cache_path))
            _log(f"Wrote simulation cache: {cache_path}")
        except Exception as exc:
            _log(f"Warning: failed to write simulation cache ({exc}).")
    input_dtype_summaries = _normalize_forest_dtype_attrs(forest)
    _log(f"Forest loaded: n_networks={forest.n_networks} n_trees_per_network={forest.n_trees_per_network}")

    _log(f"Building/loading domain: {args.domain_path}")
    t0 = perf_counter()
    configured_domain_cache = None
    if not bool(args.no_domain_cache):
        configured_domain_cache = (
            _normalize_path(args.domain_cache_dir)
            if args.domain_cache_dir is not None
            else _domain_cache_dir()
        )
    domain = _build_domain(
        ts,
        _normalize_path(args.domain_path),
        float(args.side_length),
        cache_dir=configured_domain_cache,
    )
    _log(f"Domain loaded in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    _attach_domain(forest, domain)
    _log(f"Domain attached in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    input_dtype_summaries = _normalize_forest_dtype_attrs(forest)
    _log(f"Forest dtype attrs normalized in {perf_counter() - t0:.2f}s")

    parent_repairs = []
    if not bool(args.no_repair_connectivity):
        t0 = perf_counter()
        _log("Repairing connectivity parent columns...")
        parent_repairs = _repair_forest_connectivity(forest)
        _log(f"Connectivity repair completed in {perf_counter() - t0:.2f}s")
        if any(parent_repairs):
            _log(f"Connectivity repair: stale parent refs per tree={parent_repairs}")

    connectivity_reports = []
    if args.validate_connectivity:
        t0 = perf_counter()
        _log("Validating connectivity...")
        connectivity_reports = _validate_forest_connectivity(
            forest,
            fail=not bool(args.no_fail_connectivity),
            geometry_atol=float(args.connectivity_geometry_atol),
        )
        _log(f"Connectivity validation completed in {perf_counter() - t0:.2f}s")

    outputs: dict[str, str] = {}
    meta: dict = {
        "forest": str(forest_path),
        "domain_path": str(args.domain_path),
        "fluid": str(args.fluid),
        "flow_source": str(args.flow_source),
        "connectivity": connectivity_reports,
        "parent_connectivity_repairs": [int(v) for v in parent_repairs],
        "solver_parameters": solver_parameters,
        "cext_parameters": cext_parameters,
        "input_tree_dtypes": input_dtype_summaries,
        "working_float_dtype": str(working_float_dtype),
        "working_index_dtype": str(working_index_dtype),
        "export_float_dtype": str(export_float_dtype),
        "export_index_dtype": str(export_index_dtype),
        "tree_summaries": [],
    }
    if not cext_enabled:
        meta["cext"] = {"enabled": False}
    infarction_meta = {
        "target_global_segment_id": None if args.infarction_global_segment_id is None else int(args.infarction_global_segment_id),
        "target_tree_id": None,
        "target_local_segment_id": None,
        "fraction_blocked_infarction": pct_blocked,
        "applied": False,
        "mode": "none",
        "geometry_only_ignored": bool(args.geometry_only and pct_blocked > 0.0),
        "target_segment_original_radius": None,
        "target_segment_effective_radius": None,
        "blocked_subtree_count": 0,
        "blocked_subtree_preview_global": [],
        "blocked_subtree_preview_local": [],
    }
    if args.geometry_only and pct_blocked > 0.0:
        _log("Infarction requested, but --geometry-only is enabled; ignoring infarction for this run.")
    elif pct_blocked > 0.0:
        target_tree_id, target_local_segment_id = _resolve_global_segment_id(forest, int(args.infarction_global_segment_id))
        infarction_meta["target_tree_id"] = int(target_tree_id)
        infarction_meta["target_local_segment_id"] = int(target_local_segment_id)
        _log(
            f"Infarction target: global_segment_id={args.infarction_global_segment_id} "
            f"-> tree_id={target_tree_id} local_segment_id={target_local_segment_id}"
        )

    tree_solutions: list[dict] = []
    if args.geometry_only:
        _log("Geometry-only mode: skipping flow and concentration solves.")
        for tree_id, tree in enumerate(forest.networks[0]):
            data = np.asarray(tree.data[: int(tree.segment_count)])
            sol = {
                "starts": data[:, 0:3],
                "ends": data[:, 3:6],
                "radii": data[:, 21],
                "lengths": data[:, 20],
                "flows": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "cin": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "cout": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "p_in": float("nan"),
                "p_out": float("nan"),
                "inlet_concentration": float("nan"),
            }
            tree_solutions.append(sol)
            meta["tree_summaries"].append(
                {
                    "tree_id": tree_id,
                    "segments": int(data.shape[0]),
                    "geometry_only": True,
                    "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                    "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                }
            )
    else:
        t0 = perf_counter()
        inlet_flows = _flow_inputs(args, forest)
        _log(f"Flow inputs prepared in {perf_counter() - t0:.2f}s")
        tree_offsets: list[int] = []
        offset = 0
        for tree in forest.networks[0]:
            tree_offsets.append(offset)
            offset += int(getattr(tree, "segment_count", 0) or 0)
        if cext_enabled:
            blocked_tree_id = None
            blocked_subtree_ids = np.empty((0,), dtype=np.int64)
            restore_radii_ids = np.empty((0,), dtype=np.int64)
            restore_radii_values = np.empty((0,), dtype=float)
            if pct_blocked > 0.0:
                blocked_tree_id = int(infarction_meta["target_tree_id"])
                target_tree = forest.networks[0][blocked_tree_id]
                target_id = int(infarction_meta["target_local_segment_id"])
                target_radius = float(target_tree.data[target_id, 21])
                infarction_meta["target_segment_original_radius"] = target_radius
                if pct_blocked < 1.0:
                    effective_radius = target_radius * (1.0 - pct_blocked)
                    infarction_meta["target_segment_effective_radius"] = float(effective_radius)
                    infarction_meta["mode"] = "radius_reduction"
                    infarction_meta["applied"] = True
                    restore_radii_ids = np.array([target_id], dtype=np.int64)
                    restore_radii_values = np.array([target_radius], dtype=float)
                    target_tree.data[target_id, 21] = effective_radius
                    _log(
                        f"Infarction solve override: tree {blocked_tree_id} segment {target_id} radius "
                        f"{target_radius:.9g} -> {effective_radius:.9g}"
                    )
                else:
                    seg_count = int(getattr(target_tree, "segment_count", 0) or 0)
                    blocked_subtree_ids = _collect_downstream_segment_ids(target_tree, target_id, seg_count)
                    blocked_count = int(blocked_subtree_ids.size)
                    infarction_meta["mode"] = "full_occlusion_subtree"
                    infarction_meta["applied"] = True
                    infarction_meta["blocked_subtree_count"] = blocked_count
                    infarction_meta["blocked_subtree_preview_local"] = blocked_subtree_ids[:20].tolist()
                    infarction_meta["blocked_subtree_preview_global"] = (
                        blocked_subtree_ids[:20] + int(tree_offsets[blocked_tree_id])
                    ).tolist()
                    infarction_meta["target_segment_effective_radius"] = 0.0
                    if blocked_count > 0:
                        restore_radii_ids = blocked_subtree_ids
                        restore_radii_values = np.asarray(target_tree.data[restore_radii_ids, 21], dtype=float)
                        target_tree.data[restore_radii_ids, 21] = 0.0
                    _log(
                        f"Infarction full occlusion: tree {blocked_tree_id} segment {target_id} "
                        f"subtree size {blocked_count} set to zero effective radius for solves."
                    )
            try:
                if cext_ts is None:
                    raise RuntimeError("The external-field runtime was not initialized.")
                tree_solutions, cext_meta = _solve_forest_cext(
                    cext_ts,
                    ts,
                    forest,
                    inlet_flows,
                    fluid=str(args.fluid),
                    args=args,
                )
            finally:
                if restore_radii_ids.size > 0 and blocked_tree_id is not None:
                    forest.networks[0][blocked_tree_id].data[restore_radii_ids, 21] = restore_radii_values
            if restore_radii_ids.size > 0 and blocked_tree_id is not None:
                # The radius override is a solve-time boundary condition, not a
                # destructive edit to the anatomical export.  Keep the original
                # geometry in VTP while metadata records the effective radius.
                sol = tree_solutions[int(blocked_tree_id)]
                _restore_solution_radii(sol, restore_radii_ids, restore_radii_values)
            if pct_blocked >= 1.0 and blocked_subtree_ids.size > 0 and blocked_tree_id is not None:
                sol = tree_solutions[int(blocked_tree_id)]
                valid_blocked = blocked_subtree_ids[blocked_subtree_ids < sol["flows"].shape[0]]
                if valid_blocked.size > 0:
                    sol["flows"][valid_blocked] = 0.0
                    sol["cin"][valid_blocked] = 0.0
                    sol["cout"][valid_blocked] = 0.0
                    sol["cext_state"]["cin_seg"][valid_blocked] = 0.0
                    sol["cext_state"]["cout_seg"][valid_blocked] = 0.0
                    sol["cext_state"]["c_iv_gl"][valid_blocked] = 0.0
                    sol["cext_state"]["q_line_gl"][valid_blocked] = 0.0
                    sol["cext_state"]["q_weighted_gl"][valid_blocked] = 0.0
                    sol["cext_state"]["seg_cap_gl"][valid_blocked] = 0.0
                    _log(
                        f"Infarction post-process: forced zero flow/concentration/Cext source on "
                        f"{valid_blocked.size} blocked subtree segments in tree {blocked_tree_id}."
                    )
            meta["cext"] = cext_meta
            for tree_id, (tree, inlet_flow, sol) in enumerate(zip(forest.networks[0], inlet_flows, tree_solutions)):
                meta["tree_summaries"].append(
                    {
                        "tree_id": int(tree_id),
                        "segments": int(sol["starts"].shape[0]),
                        "terminals": int(getattr(tree, "n_terminals", 0)),
                        "inlet_flow_cm3_s": float(inlet_flow),
                        "inlet_flow_ul_min": float(inlet_flow * 60000.0),
                        "pressure_in": float(sol["p_in"]),
                        "pressure_out": float(sol["p_out"]),
                        "cext_reflected_topdown_s": float(sol.get("cext_reflected_topdown_s", 0.0)),
                        "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                        "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                    }
                )
        for tree_id, (tree, inlet_flow) in ([] if cext_enabled else list(enumerate(zip(forest.networks[0], inlet_flows)))):
            t_tree = perf_counter()
            _log(f"Solving tree {tree_id}...")
            blocked_subtree_ids = np.empty((0,), dtype=np.int64)
            restore_radii_ids = np.empty((0,), dtype=np.int64)
            restore_radii_values = np.empty((0,), dtype=float)
            if pct_blocked > 0.0 and int(infarction_meta["target_tree_id"]) == int(tree_id):
                target_id = int(infarction_meta["target_local_segment_id"])
                target_radius = float(tree.data[target_id, 21])
                infarction_meta["target_segment_original_radius"] = target_radius
                if pct_blocked < 1.0:
                    effective_radius = target_radius * (1.0 - pct_blocked)
                    infarction_meta["target_segment_effective_radius"] = float(effective_radius)
                    infarction_meta["mode"] = "radius_reduction"
                    infarction_meta["applied"] = True
                    restore_radii_ids = np.array([target_id], dtype=np.int64)
                    restore_radii_values = np.array([target_radius], dtype=float)
                    tree.data[target_id, 21] = effective_radius
                    _log(
                        f"Infarction solve override: tree {tree_id} segment {target_id} radius "
                        f"{target_radius:.9g} -> {effective_radius:.9g}"
                    )
                else:
                    seg_count = int(getattr(tree, "segment_count", 0) or 0)
                    blocked_subtree_ids = _collect_downstream_segment_ids(tree, target_id, seg_count)
                    blocked_count = int(blocked_subtree_ids.size)
                    infarction_meta["mode"] = "full_occlusion_subtree"
                    infarction_meta["applied"] = True
                    infarction_meta["blocked_subtree_count"] = blocked_count
                    infarction_meta["blocked_subtree_preview_local"] = blocked_subtree_ids[:20].tolist()
                    infarction_meta["blocked_subtree_preview_global"] = (blocked_subtree_ids[:20] + int(tree_offsets[tree_id])).tolist()
                    infarction_meta["target_segment_effective_radius"] = 0.0
                    if blocked_count > 0:
                        restore_radii_ids = blocked_subtree_ids
                        restore_radii_values = np.asarray(tree.data[restore_radii_ids, 21], dtype=float)
                        tree.data[restore_radii_ids, 21] = 0.0
                    _log(
                        f"Infarction full occlusion: tree {tree_id} segment {target_id} "
                        f"subtree size {blocked_count} set to zero effective radius for solves."
                    )
            try:
                sol = _solve_tree(ts, tree, inlet_flow, fluid=str(args.fluid), concentration_solver=args.concentration_solver)
            finally:
                if restore_radii_ids.size > 0:
                    tree.data[restore_radii_ids, 21] = restore_radii_values
            if restore_radii_ids.size > 0:
                _restore_solution_radii(sol, restore_radii_ids, restore_radii_values)
            if pct_blocked >= 1.0 and blocked_subtree_ids.size > 0:
                valid_blocked = blocked_subtree_ids[blocked_subtree_ids < sol["flows"].shape[0]]
                if valid_blocked.size > 0:
                    sol["flows"][valid_blocked] = 0.0
                    sol["cin"][valid_blocked] = 0.0
                    sol["cout"][valid_blocked] = 0.0
                    _log(
                        f"Infarction post-process: forced zero flow/cin/cout on {valid_blocked.size} "
                        f"blocked subtree segments in tree {tree_id}."
                    )
            _log(f"Tree {tree_id} solved in {perf_counter() - t_tree:.2f}s")
            tree_solutions.append(sol)
            meta["tree_summaries"].append(
                {
                    "tree_id": int(tree_id),
                    "segments": int(sol["starts"].shape[0]),
                    "terminals": int(getattr(tree, "n_terminals", 0)),
                    "inlet_flow_cm3_s": float(inlet_flow),
                    "inlet_flow_ul_min": float(inlet_flow * 60000.0),
                    "pressure_in": float(sol["p_in"]),
                    "pressure_out": float(sol["p_out"]),
                    "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                    "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                }
            )

    if args.report_global_segment_id is not None:
        report_tree_id, report_local_segment_id = _resolve_global_segment_id(
            forest, int(args.report_global_segment_id)
        )
        if 0 <= report_tree_id < len(tree_solutions):
            report_flows = np.asarray(tree_solutions[report_tree_id].get("flows", []), dtype=float)
            if 0 <= report_local_segment_id < report_flows.size:
                report_flow = float(report_flows[report_local_segment_id])
                meta["reported_segment"] = {
                    "global_segment_id": int(args.report_global_segment_id),
                    "tree_id": int(report_tree_id),
                    "local_segment_id": int(report_local_segment_id),
                    "flow_cm3_s": report_flow,
                    "flow_ul_min": report_flow * 60000.0,
                }
                _log(
                    "Reported segment flow: "
                    f"global_segment_id={int(args.report_global_segment_id)} "
                    f"tree_id={int(report_tree_id)} local_segment_id={int(report_local_segment_id)} "
                    f"flow_cm3_s={report_flow:.12g} flow_ul_min={report_flow * 60000.0:.12g}"
                )

    t0 = perf_counter()
    combo = _concat_tree_solutions(tree_solutions, index_dtype=export_index_dtype)
    tree_solutions.clear()
    gc.collect()
    if cext_enabled and cext_ts is not None:
        cache_release = _release_cext_transient_gpu_cache(cext_ts)
        meta.setdefault("cext", {}).setdefault("pre_tissue_gpu_cache_release", cache_release)
        _log(
            "Released transient Cext GPU cache before tissue solve: "
            f"cupy_pool_trimmed={cache_release.get('cupy_pool_trimmed')} "
            f"attrs={cache_release.get('module_attrs_cleared', [])}"
        )
    _log(f"Combined tree solution arrays in {perf_counter() - t0:.2f}s")
    _log(f"Combined vessel segments: {combo['starts'].shape[0]}")

    if args.skip_vessel_output:
        _log("Skipping vessel VTP output.")
    else:
        t0 = perf_counter()
        vessels = _build_vessel_polydata(
            combo,
            resolution=int(args.vessel_resolution),
            float_dtype=export_float_dtype,
            index_dtype=export_index_dtype,
        )
        _log(f"Built vessel VTP in {perf_counter() - t0:.2f}s")
        vessels_path = out_dir / f"{prefix}_forest_vessels.vtp"
        _log(f"Saving vessel VTP: {vessels_path}")
        t0 = perf_counter()
        vessels.save(str(vessels_path))
        _log(f"Saved vessel VTP in {perf_counter() - t0:.2f}s")
        outputs["vessels"] = str(vessels_path)
        del vessels
        gc.collect()

    if not args.geometry_only and not args.skip_tissue_oxygen:
        t0 = perf_counter()
        points_poly, tissue_meta = _compute_tissue(
            ts,
            cext_ts,
            combo,
            domain,
            args,
            float_dtype=export_float_dtype,
            index_dtype=export_index_dtype,
        )
        _log(f"Computed tissue oxygen in {perf_counter() - t0:.2f}s")
        points_path = out_dir / f"{prefix}_forest_oxygen_points.vtp"
        _log(f"Saving tissue oxygen VTP: {points_path}")
        t0 = perf_counter()
        points_poly.save(str(points_path))
        _log(f"Saved tissue oxygen VTP in {perf_counter() - t0:.2f}s")
        outputs["oxygen_points"] = str(points_path)
        meta["tissue"] = tissue_meta
        del points_poly
        gc.collect()
    else:
        _log("Skipping tissue oxygen output.")

    outputs.update(_save_domain_outputs(domain, out_dir, prefix))
    meta["infarction"] = infarction_meta
    meta["outputs"] = outputs
    meta["elapsed_s"] = perf_counter() - t_start
    meta["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    meta_path = out_dir / f"{prefix}_forest_paraview_export.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"Metadata: {meta_path}")
    _log(f"Done in {meta['elapsed_s']:.2f}s")
    print("Wrote:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    print(f"  meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ('_normalize_path', '_float_dtype_from_name', '_int_dtype_from_name', '_resolve_forest_path', '_default_simulation_cache_path', '_should_use_simulation_cache', '_load_tissuesim', '_load_cext_tissuesim', '_apply_tissuesim_overrides', '_bool_arg', '_apply_cext_overrides', '_log', '_domain_cache_dir', '_domain_cache_path', '_build_domain', '_attach_domain', '_normalize_forest_dtype_attrs', '_make_tree_analysis_only', '_make_forest_analysis_only', '_repair_tree_parent_columns_from_children', '_repair_forest_connectivity', '_connectivity_report', '_validate_forest_connectivity', '_tree_root_flow_cm3_s', '_flow_inputs', '_safe_index_array', '_collect_downstream_segment_ids', '_restore_solution_radii', '_resolve_global_segment_id', '_solve_tree', '_cext_chb_max_for_tree', '_run_cext_frozen_step', '_solve_tree_cext_prepare', '_global_cext_box', '_global_lambda_bins', '_make_tree_hybrid_for_global_box', '_solve_shared_cext_fft', '_compute_shared_plain_box_cext', '_solve_forest_cext_legacy_shared_one_shot', '_concat_global_cext_context', '_concat_global_cext_state', '_compute_global_gfm_cext', '_apply_global_cext_update', '_snapshot_sol_cext_state', '_solve_tree_cext_backend', '_solve_forest_cext_backend_per_tree', '_release_cext_transient_gpu_cache', '_solve_forest_cext_shared_global', '_solve_forest_cext', '_compact_cext_state_for_concat', '_concat_tree_solutions', '_build_vessel_polydata', '_build_points_polydata', '_filter_export_points', '_get_boundary', '_grid_axes', '_grid_points_from_axes', '_grid_points', '_inside_mask', '_inside_grid_points_chunked', '_sha256_file', '_load_tissue_points', '_compute_tissue', '_save_domain_outputs', 'main')
