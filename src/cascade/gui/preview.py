"""Compatibility imports for the modular GUI visualization package."""

from __future__ import annotations

from cascade.gui.visualization.palette import COLORMAPS, _MAP_STOPS

from cascade.gui.visualization.canvas import (
    FlowBackdrop,
    GeometryCanvas,
    _FlowEdge,
)

from cascade.gui.visualization.support import (
    _count_status,
    _create_geometry_canvas,
    _home_icon,
    _opengl_33_available,
    _preview_tree_count,
    _seed_count_status,
    _settings_icon,
)

from cascade.gui.visualization.case import (
    CasePreview,
)

from cascade.gui.visualization.geometry import (
    _analytic_inside,
    _bounded_grid_shape,
    _cached_domain_surface,
    _cached_domain_triangles,
    _cached_domain_wireframe,
    _domain_dimensions,
    _load_uploaded_geometry,
    _mesh_triangles,
    _mesh_wireframe,
    _points_inside_surface,
    _requested_tissue_count,
    _sample_preview_domain_points,
    _simple_geometry,
    _svv_placeholder_geometry,
    domain_surface_triangles,
    domain_wireframe,
    limit_near_inlets,
    network_geometry,
    preview_tissue_geometry,
    select_tissue_points,
    select_vessel_indices,
)

from cascade.gui.visualization.fields import (
    _display_field_values,
    _layer_field_label,
    _legend_tick_values,
    _line_array,
    _map_color,
    _mesh_line_segments,
    _normalize_with_scale,
    _numeric_point_arrays,
    _point_array,
    _polyline_data,
    _scientific,
    _triangle_array,
    _values,
)

__all__ = (
    "COLORMAPS",
    "_MAP_STOPS",
    "FlowBackdrop",
    "_FlowEdge",
    "GeometryCanvas",
    "_preview_tree_count",
    "_count_status",
    "_seed_count_status",
    "_home_icon",
    "_settings_icon",
    "_create_geometry_canvas",
    "_opengl_33_available",
    "CasePreview",
    "domain_wireframe",
    "domain_surface_triangles",
    "network_geometry",
    "_svv_placeholder_geometry",
    "limit_near_inlets",
    "select_vessel_indices",
    "select_tissue_points",
    "_requested_tissue_count",
    "preview_tissue_geometry",
    "_sample_preview_domain_points",
    "_points_inside_surface",
    "_bounded_grid_shape",
    "_domain_dimensions",
    "_analytic_inside",
    "_simple_geometry",
    "_load_uploaded_geometry",
    "_mesh_wireframe",
    "_mesh_triangles",
    "_cached_domain_wireframe",
    "_cached_domain_triangles",
    "_cached_domain_surface",
    "_mesh_line_segments",
    "_polyline_data",
    "_numeric_point_arrays",
    "_line_array",
    "_triangle_array",
    "_point_array",
    "_values",
    "_normalize_with_scale",
    "_scientific",
    "_display_field_values",
    "_layer_field_label",
    "_legend_tick_values",
    "_map_color",
)
