"""Scientific visualization and validation plots.

Rendering and plotting live here rather than in solver orchestration.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

from cascade.configuration import _legacy_state as _state
from cascade.concentration.tissue.geometry import get_concentration_inlet
from cascade.concentration.tissue.greens import compute_tissue_samples_greens
from cascade.domain.visualization import _add_domain_outline, _plot_cmap, _show_plotter
from cascade.exporting.statistics import _linear_fit_slope_r2
from cascade.simulation.network_solver import solve_tree_greens

def _build_vessel_polydata(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    scalars_start: np.ndarray,
    scalars_end: np.ndarray,
    *,
    resolution: int = _state.PLOT_LINE_RESOLUTION,
) -> pv.PolyData:
    points = []
    scalars = []
    radii_samples = []
    lines = []
    offset = 0
    for idx, (start, end) in enumerate(zip(starts, ends)):
        pts = np.linspace(start, end, num=max(resolution, 2), endpoint=True)
        vals = np.linspace(scalars_start[idx], scalars_end[idx], num=pts.shape[0])
        points.append(pts)
        scalars.append(vals)
        radii_samples.append(np.full(pts.shape[0], radii[idx]))
        lines.append(np.concatenate(([pts.shape[0]], np.arange(offset, offset + pts.shape[0], dtype=int))))
        offset += pts.shape[0]
    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.vstack(points), lines=np.concatenate(lines))
    mesh["scalar"] = np.concatenate(scalars)
    mesh["radius"] = np.concatenate(radii_samples)
    return mesh


def _plot_flow(
    domain: _state.Domain,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    flows: np.ndarray,
) -> None:
    if starts.size == 0:
        return
    q_abs = np.abs(flows)
    q_abs_ul_min = q_abs * 60000.0
    vessel_lines = _build_vessel_polydata(
        starts,
        ends,
        radii,
        q_abs_ul_min,
        q_abs_ul_min,
        resolution=_state.PLOT_LINE_RESOLUTION,
    )
    if vessel_lines.n_points == 0:
        return
    vessel_mesh = vessel_lines.tube(
        radius=0.0,
        scalars="radius",
        absolute=True,
        n_sides=_state.PLOT_TUBE_SIDES,
        capping=True,
    )
    clim = [float(np.nanmin(q_abs_ul_min)), float(np.nanmax(q_abs_ul_min))]
    plotter = pv.Plotter(window_size=_state.PLOT_WINDOW_SIZE)
    plotter.add_mesh(
        vessel_mesh,
        scalars="scalar",
        cmap=_plot_cmap(),
        clim=clim,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Flow (uL/min)"},
    )
    _add_domain_outline(plotter, domain)
    plotter.camera.Zoom(_state.PLOT_ZOOM)
    _show_plotter(plotter)


def _plot_concentration(
    domain: _state.Domain,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    cout: np.ndarray,
    tissue_points: np.ndarray,
    tissue_values: np.ndarray,
    *,
    show_points: bool,
    inlet_concentration: float,
) -> None:
    if starts.size == 0:
        return
    vessel_lines = _build_vessel_polydata(
        starts,
        ends,
        radii,
        cin,
        cout,
        resolution=_state.PLOT_LINE_RESOLUTION,
    )
    if vessel_lines.n_points == 0:
        return
    vessel_mesh = vessel_lines.tube(
        radius=0.0,
        scalars="radius",
        absolute=True,
        n_sides=_state.PLOT_TUBE_SIDES,
        capping=True,
    )
    clim = [min(float(np.nanmin(cout)), _state.EXTRAVASCULAR_CONCENTRATION), float(inlet_concentration)]
    plotter = pv.Plotter(window_size=_state.PLOT_WINDOW_SIZE)
    plotter.add_mesh(vessel_mesh, scalars="scalar", cmap=_plot_cmap(), clim=clim, show_scalar_bar=True)
    if show_points and tissue_points.size:
        plotter.add_mesh(
            pv.PolyData(tissue_points),
            scalars=tissue_values,
            cmap=_plot_cmap(),
            point_size=8,
            render_points_as_spheres=True,
            clim=clim,
            show_scalar_bar=False,
        )
    _add_domain_outline(plotter, domain)
    plotter.camera.Zoom(_state.PLOT_ZOOM)
    _show_plotter(plotter)


def _plot_viability_points(
    domain: _state.Domain,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    cin: np.ndarray,
    cout: np.ndarray,
    tissue_points: np.ndarray,
    tissue_values: np.ndarray,
    *,
    inlet_concentration: float,
) -> None:
    if tissue_points.size == 0 or tissue_values.size == 0:
        return
    thresh = 0.01 * float(inlet_concentration)
    vessel_mesh = None
    if starts.size != 0:
        alive_start = (cin >= thresh).astype(float)
        alive_end = (cout >= thresh).astype(float)
        vessel_lines = _build_vessel_polydata(
            starts,
            ends,
            radii,
            alive_start,
            alive_end,
            resolution=_state.PLOT_LINE_RESOLUTION,
        )
        if vessel_lines.n_points:
            vessel_mesh = vessel_lines.tube(
                radius=0.0,
                scalars="radius",
                absolute=True,
                n_sides=_state.PLOT_TUBE_SIDES,
                capping=True,
            )
    alive = tissue_values >= thresh
    frac_alive = float(np.count_nonzero(alive) / max(tissue_values.size, 1))
    print(f"Viable (>= 1% inlet): {frac_alive * 100.0:.2f}%")

    plotter = pv.Plotter(window_size=_state.PLOT_WINDOW_SIZE)
    if vessel_mesh is not None and vessel_mesh.n_points:
        plotter.add_mesh(
            vessel_mesh,
            scalars="scalar",
            cmap=["#FF0000", "#048700"],
            clim=[0.0, 1.0],
            show_scalar_bar=False,
        )
    plotter.add_mesh(
        pv.PolyData(tissue_points),
        scalars=alive.astype(float),
        cmap=["#FF0000", "#048700"],
        clim=[0.0, 1.0],
        point_size=_state.PLOT_POINT_SIZE,
        render_points_as_spheres=True,
        show_scalar_bar=False,
        opacity=_state.POINT_OPACITY,
    )
    _add_domain_outline(plotter, domain)
    plotter.camera.Zoom(_state.PLOT_ZOOM)
    plotter.add_text("Viability (red=dead, green=alive)", position="upper_left", font_size=12)
    _show_plotter(plotter)


def _plot_mass_balance(history: Dict[str, List[float]]) -> None:
    if not history.get("iter"):
        return
    import matplotlib.pyplot as plt

    it = np.array(history["iter"], dtype=float)
    m_in = np.array(history["M_in"], dtype=float)
    m_out = np.array(history["M_out"], dtype=float)
    m_drop = np.array(history["M_drop"], dtype=float)
    mb_res = np.array(history["MB_resid"], dtype=float)

    plt.figure()
    plt.plot(it, m_in - m_out, label="M_in - M_out")
    plt.plot(it, m_drop, label="sum(Q*(Cin-Cout))", linestyle="--")
    plt.plot(it, mb_res, label="residual", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("mol/s")
    plt.title("Mass balance convergence")
    plt.legend()
    plt.grid(True)
    plt.show()


def _dnc_histogram_fractions_um(
    dnc_values_um: np.ndarray,
    *,
    bin_max_um: float = 5000.0,
    bin_width_um: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    dnc_values_um = np.asarray(dnc_values_um, dtype=float).reshape(-1)
    bin_max_um = float(bin_max_um)
    bin_width_um = float(bin_width_um)
    if not np.isfinite(bin_max_um) or bin_max_um <= 0.0:
        raise ValueError("bin_max_um must be positive and finite")
    if not np.isfinite(bin_width_um) or bin_width_um <= 0.0:
        raise ValueError("bin_width_um must be positive and finite")

    edges_regular = np.arange(0.0, bin_max_um + bin_width_um, bin_width_um, dtype=float)
    if edges_regular.size < 2 or not np.isclose(edges_regular[-1], bin_max_um):
        edges_regular = np.linspace(0.0, bin_max_um, int(round(bin_max_um / bin_width_um)) + 1, dtype=float)

    finite = dnc_values_um[np.isfinite(dnc_values_um)]
    if finite.size == 0:
        edges_plot = np.concatenate([edges_regular, np.array([bin_max_um + bin_width_um], dtype=float)])
        return edges_plot, np.zeros(edges_regular.size - 1 + 1, dtype=float)

    regular = finite[finite <= bin_max_um]
    overflow = finite[finite > bin_max_um]
    counts_regular, _ = np.histogram(regular, bins=edges_regular) if regular.size else (np.zeros(edges_regular.size - 1, dtype=int), edges_regular)
    overflow_count = int(overflow.size)
    total = int(finite.size)
    fractions = np.concatenate([counts_regular.astype(float), np.array([float(overflow_count)], dtype=float)]) / float(total)

    edges_plot = np.concatenate([edges_regular, np.array([bin_max_um + bin_width_um], dtype=float)])
    return edges_plot, fractions


def _fit_truncnorm_mu_sigma_lower0(values_um: np.ndarray) -> tuple[float, float]:
    values_um = np.asarray(values_um, dtype=float).reshape(-1)
    values_um = values_um[np.isfinite(values_um)]
    if values_um.size < 2:
        return float("nan"), float("nan")
    values_um = values_um[values_um >= 0.0]
    if values_um.size < 2:
        return float("nan"), float("nan")

    x = values_um
    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)

    def _log_survival(z: float) -> float:
        # log(1 - Phi(z)) with some basic stabilization
        p = 0.5 * (1.0 + math.erf(z / sqrt2))
        p = min(max(p, 0.0), 1.0)
        s = 1.0 - p
        if s <= 0.0:
            return -1e300
        return math.log(s)

    def _nll(mu: float, log_sigma: float) -> float:
        sigma = math.exp(log_sigma)
        if not (np.isfinite(sigma) and sigma > 0.0):
            return float("inf")
        z = (x - mu) / sigma
        # log pdf of truncated normal on [0, inf):
        # log(phi(z)) - log(sigma) - log(1 - Phi((0-mu)/sigma))
        log_phi = -0.5 * z * z - math.log(sqrt2pi)
        logZ = _log_survival((0.0 - mu) / sigma)
        if not np.isfinite(logZ):
            return float("inf")
        ll = np.sum(log_phi) - x.size * math.log(sigma) - x.size * logZ
        return float(-ll)

    mu0 = float(np.mean(x))
    sigma0 = float(np.std(x, ddof=0))
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        sigma0 = 1.0
    log_sigma0 = math.log(sigma0)

    best_mu = mu0
    best_log_sigma = log_sigma0
    best = _nll(best_mu, best_log_sigma)

    mu_span = max(6.0 * sigma0, 200.0)
    log_span = math.log(4.0)
    for _ in range(3):
        mu_grid = np.linspace(best_mu - mu_span / 2.0, best_mu + mu_span / 2.0, 61, dtype=float)
        log_grid = np.linspace(best_log_sigma - log_span / 2.0, best_log_sigma + log_span / 2.0, 61, dtype=float)
        for mu in mu_grid:
            for ls in log_grid:
                val = _nll(float(mu), float(ls))
                if val < best:
                    best = val
                    best_mu = float(mu)
                    best_log_sigma = float(ls)
        mu_span *= 0.35
        log_span *= 0.35

    return float(best_mu), float(math.exp(best_log_sigma))


def _init_violin_points_csv(path: Path, *, target_counts: Sequence[int], n_rows: int) -> None:
    headers = [str(int(t)) for t in target_counts]
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        empty_row = ["" for _ in headers]
        for _ in range(int(max(n_rows, 0))):
            writer.writerow(empty_row)


def _update_violin_points_csv(
    path: Path,
    *,
    target_counts: Sequence[int],
    col_idx: int,
    values_um: np.ndarray,
) -> None:
    target_counts = list(target_counts)
    if col_idx < 0 or col_idx >= len(target_counts):
        raise IndexError("col_idx out of range for target_counts")

    values = np.asarray(values_um, dtype=float).reshape(-1)

    with path.open("r", newline="") as csvfile:
        rows = list(csv.reader(csvfile))
    if not rows:
        raise ValueError("violin CSV is empty")

    expected_header = [str(int(t)) for t in target_counts]
    if rows[0] != expected_header:
        raise ValueError(f"violin CSV header mismatch: expected {expected_header}, got {rows[0]}")

    n_rows = len(rows) - 1
    if values.size != n_rows:
        raise ValueError(f"values_um length ({values.size}) does not match CSV rows ({n_rows})")

    for i in range(n_rows):
        v = values[i]
        rows[i + 1][col_idx] = "" if not np.isfinite(v) else f"{float(v):.8g}"

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def _plot_dnc_gaussians(
    *,
    mu_sigma: Sequence[tuple[float, float]],
    target_terminals: Sequence[int],
    title: str,
) -> None:
    if not mu_sigma or not target_terminals:
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    targets = np.asarray(list(target_terminals), dtype=float)
    if targets.size == 0:
        return

    cmapcolorscheme = "jet"
    try:
        cmap = mpl.colormaps.get_cmap(cmapcolorscheme)
    except Exception:  # pragma: no cover
        cmap = mpl.cm.get_cmap(cmapcolorscheme)
    vmin = float(np.nanmin(targets))
    vmax = float(np.nanmax(targets))
    if _state.GAUSSIAN_COLORMAP_LOG10 and np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0.0 and vmax > vmin:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    x = np.linspace(0.0, float(_state.GAUSSIAN_PLOT_XMAX_UM), int(max(_state.GAUSSIAN_PLOT_NPTS, 2)), dtype=float)
    x_log_min = float(_state.GAUSSIAN_LOG_XMIN_UM)
    if not np.isfinite(x_log_min) or x_log_min <= 0.0:
        x_log_min = 0.1
    x_pos = np.linspace(x_log_min, float(_state.GAUSSIAN_PLOT_XMAX_UM), int(max(_state.GAUSSIAN_PLOT_NPTS, 2)), dtype=float)
    y_logx = np.log10(x_pos)

    fig, (ax_lin, ax_log) = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)
    for t, (mu, sigma) in zip(targets, mu_sigma):
        mu = float(mu)
        sigma = float(sigma)
        if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0.0):
            continue
        color = cmap(norm(float(t)))
        y = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax_lin.plot(
            x,
            y,
            color=color,
            alpha=_state.HISTOGRAM_ALPHA,
            linewidth=2.0,
        )
        y_pos = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x_pos - mu) / sigma) ** 2)
        # Plot the same Gaussian values, just against log10(x) (no Jacobian / renormalization).
        y_log = y_pos
        ax_log.plot(
            y_logx,
            y_log,
            color=color,
            alpha=_state.HISTOGRAM_ALPHA,
            linewidth=2.0,
        )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if isinstance(norm, mpl.colors.LogNorm):
        cbar = fig.colorbar(sm, ax=[ax_lin, ax_log], label="N_vessels (target_terminals, log scale)")
        try:
            import matplotlib.ticker as mticker

            cbar.formatter = mticker.LogFormatterMathtext(base=10)
            cbar.update_ticks()
        except Exception:  # pragma: no cover
            pass
    else:
        fig.colorbar(sm, ax=[ax_lin, ax_log], label="N_vessels (target_terminals)")

    ax_lin.set_xlim(0.0, float(_state.GAUSSIAN_PLOT_XMAX_UM))
    ax_lin.set_ylim(bottom=0.0)
    ax_lin.set_xlabel("DNCW (Âµm)")
    ax_lin.set_ylabel("Probability density (1/Âµm)")
    ax_lin.grid(True, alpha=0.3)

    ax_log.set_xlim(float(np.log10(x_log_min)), float(np.log10(_state.GAUSSIAN_PLOT_XMAX_UM)))
    ax_log.set_ylim(bottom=0.0)
    ax_log.set_xlabel("log10(DNCW)")
    ax_log.set_ylabel("Probability density (same as linear; 1/Âµm)")
    ax_log.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.show()


def plot_checker_tree(
    tree: _state.Tree,
    sample_points: np.ndarray,
    *,
    inlet_flow_cm3_s: Optional[float] = None,
    fluid: str | None = None,
) -> None:
    inlet_concentration = get_concentration_inlet(fluid)
    q_inlet = inlet_flow_cm3_s
    if q_inlet is None or not np.isfinite(q_inlet):
        q_inlet = float(_state.QIN_TARGET) * 1e-3 / 60.0
    starts, ends, radii, lengths, flows, cin, cout, _ = solve_tree_greens(
        tree,
        q_inlet,
        fluid=fluid or _state.ACTIVE_FLUID,
        inlet_concentration=inlet_concentration,
    )
    mask, tissue_conc = compute_tissue_samples_greens(
        sample_points,
        starts,
        ends,
        radii,
        cin,
        flows,
        diffusivity=_state.SOLUTE_DIFFUSIVITY,
        vmax=_state.VMAX_MM,
        km=_state.K_M_MM,
        window_factor=_state.WINDOW_FACTOR,
        inlet_concentration=inlet_concentration,
    )
    tissue_pts = sample_points[mask]
    tissue_vals = tissue_conc[mask]
    domain = getattr(tree, "domain", None)
    _plot_concentration(
        domain,
        starts,
        ends,
        radii,
        cin,
        cout,
        tissue_pts,
        tissue_vals,
        show_points=True,
        inlet_concentration=inlet_concentration,
    )


def _load_validation_points(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
    if data.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    names = {name.strip().lower(): name for name in data.dtype.names}
    def _pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            key = cand.lower()
            if key in names:
                return names[key]
        return None

    x_key = _pick("x", "x_m")
    y_key = _pick("y", "y_m")
    z_key = _pick("z", "z_m")
    a_key = _pick("a")
    if not (x_key and y_key and z_key and a_key):
        raise ValueError("CSV must include columns for x,y,z and A (or x_m,y_m,z_m).")

    pts = np.column_stack([data[x_key], data[y_key], data[z_key]])
    vals = np.asarray(data[a_key], dtype=float)
    return pts, vals


def _plot_validation_scatter(x_ref: np.ndarray, y_pred: np.ndarray) -> None:
    if x_ref.size == 0 or y_pred.size == 0:
        return
    import matplotlib.pyplot as plt

    fig, (ax_scatter, ax_bar) = plt.subplots(
        ncols=2,
        figsize=(9, 4),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )
    ax_scatter.scatter(x_ref, y_pred, s=8, alpha=0.7)
    mn = float(min(np.nanmin(x_ref), np.nanmin(y_pred)))
    mx = float(max(np.nanmax(x_ref), np.nanmax(y_pred)))
    ax_scatter.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    ax_scatter.set_xlabel("Validation A")
    ax_scatter.set_ylabel("Greens predicted")
    ax_scatter.set_title("Pointwise comparison")
    ax_scatter.grid(True)

    thresholds = np.array([0.011, 0.002211], dtype=float)
    match_fracs = []
    false_fracs = []
    for thr in thresholds:
        valid_mask = x_ref < thr
        pred_mask = y_pred < thr
        valid_count = int(np.count_nonzero(valid_mask))
        pred_count = int(np.count_nonzero(pred_mask))
        non_valid_count = int(np.count_nonzero(~valid_mask))
        if valid_count == 0:
            match_frac = 0.0
        else:
            match_frac = float(np.count_nonzero(pred_mask & valid_mask) / valid_count)
        if non_valid_count == 0:
            false_frac = 0.0
        else:
            false_frac = float(np.count_nonzero(pred_mask & (~valid_mask)) / non_valid_count)
        match_fracs.append(match_frac)
        false_fracs.append(false_frac)

    fit_lines = []
    for thr in thresholds:
        mask = y_pred >= thr
        slope, intercept, r2 = _linear_fit_slope_r2(x_ref[mask], y_pred[mask])
        n = int(np.count_nonzero(mask))
        print(f"Fit (Greens predicted >= {thr}): slope={slope:.6f}, R2={r2:.6f}, n={n}")
        fit_lines.append((thr, slope, intercept))

    fit_colors = ["#1f77b4", "#9467bd"]
    for (thr, slope, intercept), color in zip(fit_lines, fit_colors):
        if not np.isfinite(slope) or not np.isfinite(intercept):
            continue
        x_line = np.array([mn, mx], dtype=float)
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, color=color, linewidth=1.5, label=f"Fit pred >= {thr}")

    x_pos = np.arange(len(thresholds))
    width = 0.35
    bars_match = ax_bar.bar(x_pos - width / 2.0, match_fracs, width=width, color="#2ca02c")
    bars_false = ax_bar.bar(x_pos + width / 2.0, false_fracs, width=width, color="#d62728")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_ylabel("fraction")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(["<0.011", "<0.002211"], rotation=90)
    ax_bar.set_title("Threshold %")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bar.legend(["match", "false +"], loc="upper right", fontsize=8)
    for bars in (bars_match, bars_false):
        for bar in bars:
            frac = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(1.0, frac + 0.04),
                f"{frac * 100.0:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_scatter.legend(loc="upper left", fontsize=8)
    plt.show()


__all__ = ['_build_vessel_polydata', '_plot_flow', '_plot_concentration', '_plot_viability_points', '_plot_mass_balance', '_dnc_histogram_fractions_um', '_fit_truncnorm_mu_sigma_lower0', '_init_violin_points_csv', '_update_violin_points_csv', '_plot_dnc_gaussians', 'plot_checker_tree', '_load_validation_points', '_plot_validation_scatter']
