"""Domain visualization helpers.

Plot construction is isolated here from domain geometry and sampling logic.
"""

from __future__ import annotations

import pyvista as pv

from cascade.configuration import solver_state as _state


def _plot_cmap():
    # Mutable runtime state is centralized in configuration.solver_state.
    if not _state.USE_TRICOLOR_CMAP:
        return _state.PLOT_CMAP
    if _state.TRICOLOR_CMAP is None:
        from matplotlib.colors import LinearSegmentedColormap

        _state.TRICOLOR_CMAP = LinearSegmentedColormap.from_list(
            "tri_stoplight",
            [
                (0.0, "#0000ff"),
                (0.5, "#800080"),
                (1.0, "#ff0000"),
            ],
        )
    return _state.TRICOLOR_CMAP


def _add_domain_outline(plotter: pv.Plotter, domain: _state.Domain) -> None:
    if domain is None:
        return
    mesh = getattr(domain, "mesh", None)
    if mesh is not None:
        outline = mesh.outline()
        if outline is not None and outline.n_cells:
            plotter.add_mesh(outline, color="gray", opacity=0.6, line_width=2.0)
            return
    bounds = getattr(domain, "bounds", None) or getattr(domain, "bounding_box", None)
    if bounds is not None:
        plotter.add_mesh(
            pv.Cube(bounds=bounds).outline(), color="gray", opacity=0.4, line_width=2.0
        )


def _show_plotter(plotter: pv.Plotter) -> None:
    if _state.PLOT_WINDOW_POSITION is not None:
        try:
            plotter.show(
                window_size=_state.PLOT_WINDOW_SIZE,
                window_position=_state.PLOT_WINDOW_POSITION,
            )
            return
        except TypeError:
            pass
    plotter.show(window_size=_state.PLOT_WINDOW_SIZE)


__all__ = ["_plot_cmap", "_add_domain_outline", "_show_plotter"]
