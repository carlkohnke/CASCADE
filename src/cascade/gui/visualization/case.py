"""Simulation-case preview panel and controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cascade.gui.widgets import ChoiceComboBox, IconButton

from cascade.gui.visualization.fields import (
    _display_field_values,
    _layer_field_label,
    _numeric_point_arrays,
    _polyline_data,
)

from cascade.gui.visualization.support import (
    _count_status,
    _create_geometry_canvas,
    _home_icon,
    _preview_tree_count,
    _seed_count_status,
    _settings_icon,
)

from cascade.gui.visualization.geometry import (
    _mesh_triangles,
    _mesh_wireframe,
    _requested_tissue_count,
    domain_display_rotation,
    domain_surface_triangles,
    domain_wireframe,
    network_geometry,
    preview_tissue_geometry,
    select_tissue_points,
    select_vessel_indices,
)


class CasePreview(QFrame):
    """Persistent preview panel shared by setup and analysis pages."""

    result_fields_loaded = Signal(list, list)
    selection_changed = Signal(object)
    view_settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("previewPanel")
        self.setMinimumWidth(240)
        self._result_cache: dict[str, Any] = {}
        self._visible_result_indices = np.empty((0,), dtype=int)
        self._visible_tissue_indices = np.empty((0,), dtype=int)
        self._loading_view_settings = False
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(180)
        self._settings_timer.timeout.connect(self.view_settings_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.title = QLabel("LIVE CASE PREVIEW")
        self.title.setObjectName("previewTitle")
        header.addWidget(self.title)
        header.addStretch()
        self.settings_button = IconButton()
        self.settings_button.setObjectName("previewSettings")
        self.settings_button.setIcon(_settings_icon())
        self.settings_button.setIconSize(QSize(20, 20))
        self.settings_button.setFixedSize(34, 30)
        self.settings_button.setCheckable(True)
        self.settings_button.setToolTip("Viewer settings")
        self.settings_button.setAccessibleName("Viewer settings")
        header.addWidget(self.settings_button)
        self.home_button = IconButton()
        self.home_button.setObjectName("previewHome")
        self.home_button.setIcon(_home_icon())
        self.home_button.setIconSize(QSize(20, 20))
        self.home_button.setFixedSize(34, 30)
        self.home_button.setToolTip("Reset view")
        self.home_button.setAccessibleName("Reset view")
        self.home_button.clicked.connect(self.home)
        header.addWidget(self.home_button)
        layout.addLayout(header)
        self.settings_panel = self._build_settings_panel()
        self.settings_panel.setVisible(False)
        self.settings_button.toggled.connect(self.settings_panel.setVisible)
        layout.addWidget(self.settings_panel)
        self.canvas = _create_geometry_canvas(self)
        self.canvas.selection_changed.connect(self._canvas_selection)
        layout.addWidget(self.canvas, 1)
        view_controls = QHBoxLayout()
        view_controls.setSpacing(8)
        domain_label = QLabel("STYLE")
        domain_label.setObjectName("eyebrow")
        domain_label.setAccessibleName("Domain display style")
        self.domain_view = ChoiceComboBox(self)
        self.domain_view.addItem("Wireframe", "wireframe")
        self.domain_view.addItem("Surface", "surface")
        self.domain_view.addItem("None", "none")
        # "Wireframe" plus the arrow needs a little more than the 94 px
        # compact width; 104 px stays compact without clipping the label.
        self.domain_view.setFixedWidth(104)
        self.domain_view.setToolTip(
            "Switch the domain between sectioned wireframe and translucent surface views."
        )
        self.domain_view.currentIndexChanged.connect(
            lambda: self.canvas.set_domain_mode(self.domain_view.currentData())
        )
        view_controls.addWidget(domain_label)
        view_controls.addWidget(self.domain_view)
        view_controls.addStretch()
        layout.addLayout(view_controls)
        self.status = QLabel("Domain preview")
        self.status.setObjectName("previewStatus")
        self.status.setWordWrap(True)
        self.status.setVisible(False)
        # Keep a stable footprint for the scientific-count strip even when
        # Project/Domain intentionally hide it.  Otherwise the canvas grows
        # and contracts as the user advances through the workflow.
        self._status_slot = QWidget(self)
        self._status_slot.setMinimumHeight(28)
        status_layout = QVBoxLayout(self._status_slot)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status)
        layout.addWidget(self._status_slot)

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("viewerSettingsPanel")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        heading = QLabel("VIEWER SETTINGS")
        heading.setObjectName("eyebrow")
        outer.addWidget(heading)

        rows = QGridLayout()
        rows.setHorizontalSpacing(12)
        rows.setVerticalSpacing(8)
        self.vessel_selection = ChoiceComboBox(panel)
        self.vessel_selection.setObjectName("viewerVesselSelection")
        for label, value in (
            ("Nearest inlets", "near"),
            ("Random sample", "random"),
            ("All", "all"),
            ("None", "none"),
        ):
            self.vessel_selection.addItem(label, value)
        self.vessel_count = QLineEdit("5000", panel)
        self.vessel_count.setObjectName("viewerVesselCount")
        self.vessel_count.setValidator(QIntValidator(1, 10_000_000, self.vessel_count))
        self.vessel_count.setAccessibleName("Vessel display limit")
        self.vessel_count.setToolTip(
            "Display at most this many existing vessels; this does not create vessels"
        )

        self.tissue_selection = ChoiceComboBox(panel)
        self.tissue_selection.setObjectName("viewerTissueSelection")
        for label, value in (
            ("Nearest inlets", "near"),
            ("Random sample", "random"),
            ("All", "all"),
            ("None", "none"),
        ):
            self.tissue_selection.addItem(label, value)
        self.tissue_count = QLineEdit("10000", panel)
        self.tissue_count.setObjectName("viewerTissueCount")
        self.tissue_count.setValidator(QIntValidator(1, 10_000_000, self.tissue_count))
        self.tissue_count.setAccessibleName("Tissue display limit")
        self.tissue_count.setToolTip(
            "Display at most this many existing tissue points; simulation sample count is set under Outputs"
        )

        def field(label: str, widget: QWidget) -> QWidget:
            container = QWidget(panel)
            column = QVBoxLayout(container)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(3)
            caption = QLabel(label)
            caption.setObjectName("fieldLabel")
            column.addWidget(caption)
            column.addWidget(widget)
            return container

        rows.addWidget(field("Vessel selection", self.vessel_selection), 0, 0)
        rows.addWidget(field("Vessel display limit", self.vessel_count), 0, 1)
        rows.addWidget(field("Tissue selection", self.tissue_selection), 1, 0)
        rows.addWidget(field("Tissue display limit", self.tissue_count), 1, 1)
        rows.setColumnStretch(0, 2)
        rows.setColumnStretch(1, 1)
        outer.addLayout(rows)

        reset = QPushButton("Reset defaults", panel)
        reset.setObjectName("viewerSettingsReset")
        reset.setProperty("secondary", True)
        reset.setFixedWidth(132)
        reset.clicked.connect(self.reset_view_settings)
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        reset_row.addWidget(reset)
        outer.addLayout(reset_row)

        self.vessel_selection.currentIndexChanged.connect(self._view_control_changed)
        self.tissue_selection.currentIndexChanged.connect(self._view_control_changed)
        self.vessel_count.textChanged.connect(self._view_count_changed)
        self.tissue_count.textChanged.connect(self._view_count_changed)
        return panel

    @staticmethod
    def _count_value(field: QLineEdit, default: int) -> int:
        text = field.text().strip().replace(",", "")
        try:
            return max(int(text), 1)
        except ValueError:
            return int(default)

    def view_settings(self) -> dict[str, Any]:
        """Return the sampling policy shared by setup and result scenes."""
        return {
            "vessel_mode": str(self.vessel_selection.currentData() or "near"),
            "vessel_limit": self._count_value(self.vessel_count, 5_000),
            "tissue_mode": str(self.tissue_selection.currentData() or "near"),
            "tissue_limit": self._count_value(self.tissue_count, 10_000),
        }

    def _view_control_changed(self, *_args) -> None:
        vessel_uses_count = self.vessel_selection.currentData() not in {"all", "none"}
        tissue_uses_count = self.tissue_selection.currentData() not in {"all", "none"}
        self.vessel_count.setEnabled(vessel_uses_count)
        self.tissue_count.setEnabled(tissue_uses_count)
        if not self._loading_view_settings:
            self._settings_timer.stop()
            self.view_settings_changed.emit()

    def _view_count_changed(self, text: str) -> None:
        if self._loading_view_settings or not text or not text.isdigit():
            return
        self._settings_timer.start()

    def reset_view_settings(self) -> None:
        self._loading_view_settings = True
        try:
            for combo, value in (
                (self.vessel_selection, "near"),
                (self.tissue_selection, "near"),
            ):
                index = combo.findData(value)
                combo.setCurrentIndex(index)
            self.vessel_count.setText("5000")
            self.tissue_count.setText("10000")
        finally:
            self._loading_view_settings = False
        self._view_control_changed()

    def load_view_settings(self, config: dict[str, Any]) -> None:
        gui = config.get("gui", {})
        legacy = gui.get("analysis", {})
        settings = gui.get("viewer", {})
        vessel_limit = settings.get("vessel_limit", legacy.get("max_vessels", 5_000))
        legacy_limit_mode = str(vessel_limit).lower()
        if legacy_limit_mode in {"all", "none"}:
            vessel_limit = 5_000
            vessel_mode = legacy_limit_mode
        else:
            vessel_mode = settings.get("vessel_mode", "near")
            try:
                vessel_limit = max(int(vessel_limit), 1)
            except (TypeError, ValueError):
                vessel_limit = 5_000
        tissue_mode = settings.get("tissue_mode", legacy.get("tissue_mode", "near"))
        tissue_limit = settings.get("tissue_limit", 10_000)
        self._loading_view_settings = True
        try:
            for combo, value, fallback in (
                (self.vessel_selection, vessel_mode, "near"),
                (self.tissue_selection, tissue_mode, "near"),
            ):
                index = combo.findData(value)
                combo.setCurrentIndex(index if index >= 0 else combo.findData(fallback))
            self.vessel_count.setText(str(max(int(vessel_limit), 1)))
            self.tissue_count.setText(str(max(int(tissue_limit), 1)))
        finally:
            self._loading_view_settings = False
        self._sync_count_enabled()

    def write_view_settings(self, config: dict[str, Any]) -> None:
        config.setdefault("gui", {})["viewer"] = self.view_settings()

    def _sync_count_enabled(self) -> None:
        self.vessel_count.setEnabled(
            self.vessel_selection.currentData() not in {"all", "none"}
        )
        self.tissue_count.setEnabled(
            self.tissue_selection.currentData() not in {"all", "none"}
        )

    def home(self) -> None:
        self.canvas.home()

    def _layer_policy(self, kind: str, total: int) -> tuple[str, int]:
        settings = self.view_settings()
        mode = str(settings[f"{kind}_mode"])
        requested = int(settings[f"{kind}_limit"])
        ceiling = int(getattr(self.canvas, f"{kind}_preview_limit", requested))
        if mode == "none":
            return mode, 0
        if mode == "all":
            requested = int(total)
        return mode, min(max(requested, 0), max(ceiling, 0), max(int(total), 0))

    def release(self) -> None:
        self._result_cache.clear()
        self._visible_result_indices = np.empty((0,), dtype=int)
        self._visible_tissue_indices = np.empty((0,), dtype=int)
        if hasattr(self.canvas, "release_gpu_memory"):
            self.canvas.release_gpu_memory()
        else:
            self.canvas.clear()
        self.status.setText("preview memory released")

    def _canvas_selection(self, selection: dict[str, Any]) -> None:
        if not selection or not self._result_cache:
            self.selection_changed.emit({})
            return
        kind = selection.get("kind")
        shown_index = int(selection.get("index", -1))
        if kind == "vessel" and 0 <= shown_index < len(self._visible_result_indices):
            index = int(self._visible_result_indices[shown_index])
            arrays = self._result_cache.get("vessel_arrays", {})
        elif kind == "tissue" and 0 <= shown_index < len(self._visible_tissue_indices):
            index = int(self._visible_tissue_indices[shown_index])
            arrays = self._result_cache.get("tissue_arrays", {})
        else:
            self.selection_changed.emit({})
            return
        values = {
            name: np.asarray(values).reshape(-1)[index]
            for name, values in arrays.items()
            if np.asarray(values).size > index
        }
        self.selection_changed.emit(
            {
                "kind": kind,
                "index": index,
                "id": values.get("segment_id", values.get("local_segment_id", index)),
                "values": values,
            }
        )

    def show_case(
        self,
        config: dict[str, Any],
        *,
        include_network: bool,
        include_tissue: bool = False,
    ) -> None:
        self._visible_result_indices = np.empty((0,), dtype=int)
        self._visible_tissue_indices = np.empty((0,), dtype=int)
        domain_config = config.get("domain", {})
        domain_lines = domain_wireframe(domain_config)
        domain_triangles = domain_surface_triangles(domain_config)
        starts = ends = radii = alpha = inlet_points = outlet_points = None
        tissue_points = tissue_alpha = None
        placeholder = False
        total_vessels = total_tissue = 0
        status = "Domain ready"
        if include_network:
            try:
                (
                    starts,
                    ends,
                    radii,
                    inlet_points,
                    outlet_points,
                    alpha,
                    detail,
                ) = network_geometry(config)
                placeholder = detail.startswith("Preparing the exact hydraulic SVV seed")
                total_vessels = len(starts)
                vessel_mode, vessel_limit = self._layer_policy("vessel", total_vessels)
                vessel_ids = select_vessel_indices(
                    starts,
                    ends,
                    inlet_points,
                    mode=vessel_mode,
                    limit=vessel_limit,
                )
                starts, ends = starts[vessel_ids], ends[vessel_ids]
                radii = radii[vessel_ids] if radii is not None else None
                alpha = alpha[vessel_ids] if alpha is not None else None
                status = detail
            except Exception as exc:
                status = f"Preview unavailable  │  {exc}"
        if include_tissue:
            try:
                tissue_mode, tissue_limit = self._layer_policy(
                    "tissue", _requested_tissue_count(config)
                )
                tissue_points, tissue_alpha, requested = preview_tissue_geometry(
                    config,
                    inlet_points,
                    mode=tissue_mode,
                    limit=tissue_limit,
                )
                total_tissue = requested
            except Exception as exc:
                status += f"  │  tissue preview unavailable: {exc}"
        if include_network and "unavailable" not in status.lower():
            vessel_count = len(starts) if starts is not None else 0
            tree_count = _preview_tree_count(config, inlet_points)
            tissue_count = len(tissue_points) if tissue_points is not None else 0
            status = _count_status(
                vessel_count,
                total_vessels,
                tree_count,
                tissue_count,
                total_tissue,
            )
        self.canvas.tissue_opacity = 0.52
        self.canvas.set_vessel_placeholder(placeholder)
        self.canvas.set_geometry(
            domain_lines=domain_lines,
            domain_triangles=domain_triangles,
            vessel_starts=starts,
            vessel_ends=ends,
            vessel_radii=radii,
            vessel_alpha=alpha,
            inlet_points=inlet_points,
            outlet_points=outlet_points,
            tissue_points=tissue_points,
            tissue_alpha=tissue_alpha,
            model_rotation=domain_display_rotation(domain_config),
        )
        self.status.setText(status)
        self.status.setVisible(include_network or include_tissue)

    def show_seed(
        self,
        config: dict[str, Any],
        geometry_path: str,
        response: dict[str, Any],
        *,
        include_tissue: bool = False,
    ) -> None:
        self.canvas.set_vessel_placeholder(False)
        with np.load(geometry_path) as data:
            starts = np.asarray(data["starts"], dtype=np.float32)
            ends = np.asarray(data["ends"], dtype=np.float32)
            alpha = np.asarray(data["alpha"], dtype=np.float32)
            radii = (
                np.asarray(data["radii"], dtype=np.float32)
                if "radii" in data.files
                else None
            )
        loaded_vessels = len(starts)
        roots = config.get("network", {}).get("roots") or []
        inlets = np.asarray(
            [root.get("start", [0, 0, 0]) for root in roots], dtype=float
        )
        segments = response.get("segments", [])
        requested_segments = response.get("requested_segments", segments)
        total_vessels = (
            sum(int(value) for value in requested_segments)
            if bool(response.get("preview_limited", False))
            else loaded_vessels
        )
        vessel_mode, vessel_limit = self._layer_policy("vessel", total_vessels)
        vessel_ids = select_vessel_indices(
            starts,
            ends,
            inlets,
            mode=vessel_mode,
            limit=vessel_limit,
        )
        starts, ends = starts[vessel_ids], ends[vessel_ids]
        alpha = alpha[vessel_ids]
        radii = radii[vessel_ids] if radii is not None else None
        tissue_points = tissue_alpha = None
        requested = 0
        if include_tissue:
            tissue_mode, tissue_limit = self._layer_policy(
                "tissue", _requested_tissue_count(config)
            )
            tissue_points, tissue_alpha, requested = preview_tissue_geometry(
                config,
                inlets,
                mode=tissue_mode,
                limit=tissue_limit,
            )
        self.canvas.tissue_opacity = 0.52
        self.canvas.set_geometry(
            domain_lines=domain_wireframe(config.get("domain", {})),
            domain_triangles=domain_surface_triangles(config.get("domain", {})),
            vessel_starts=starts,
            vessel_ends=ends,
            vessel_radii=radii,
            vessel_alpha=alpha,
            inlet_points=inlets,
            tissue_points=tissue_points,
            tissue_alpha=tissue_alpha,
            model_rotation=domain_display_rotation(config.get("domain", {})),
        )
        vessel_count = len(starts)
        status = _seed_count_status(
            vessel_count,
            total_vessels,
            len(segments),
            len(tissue_points) if tissue_points is not None else 0,
            requested,
        )
        self.status.setText(status)
        self.status.setVisible(True)

    def show_result(self, manifest_path: str, options: dict[str, Any]) -> None:
        self.canvas.set_vessel_placeholder(False)
        if not manifest_path:
            self.canvas.clear("no result selected")
            self.status.setText("no result selected")
            self.status.setVisible(True)
            return
        try:
            cache = self._load_result(manifest_path)
            vessel_field = str(options.get("vessel_field") or "")
            tissue_field = str(options.get("tissue_field") or "")
            logical_ids = cache["vessel_arrays"].get("global_segment_id")
            if logical_ids is not None and len(logical_ids) == len(cache["starts"]):
                logical_ids = np.asarray(logical_ids, dtype=np.int64)
                unique_ids, first_ids = np.unique(logical_ids, return_index=True)
                _, reverse_first_ids = np.unique(logical_ids[::-1], return_index=True)
                last_ids = len(logical_ids) - 1 - reverse_first_ids
                vessel_mode, vessel_limit = self._layer_policy(
                    "vessel", len(unique_ids)
                )
                selected_ids = select_vessel_indices(
                    cache["starts"][first_ids],
                    cache["ends"][last_ids],
                    cache["inlets"],
                    mode=vessel_mode,
                    limit=vessel_limit,
                )
                selected_logical_ids = unique_ids[selected_ids]
                visible_indices = np.flatnonzero(
                    np.isin(
                        logical_ids, np.asarray(selected_logical_ids, dtype=np.int64)
                    )
                )
                starts = cache["starts"][visible_indices]
                ends = cache["ends"][visible_indices]
                shown_vessels = int(np.unique(logical_ids[visible_indices]).size)
                total_vessels = int(unique_ids.size)
            else:
                vessel_mode, vessel_limit = self._layer_policy(
                    "vessel", len(cache["starts"])
                )
                visible_indices = select_vessel_indices(
                    cache["starts"],
                    cache["ends"],
                    cache["inlets"],
                    mode=vessel_mode,
                    limit=vessel_limit,
                )
                starts = cache["starts"][visible_indices]
                ends = cache["ends"][visible_indices]
                shown_vessels = len(starts)
                total_vessels = len(cache["starts"])
            self._visible_result_indices = np.asarray(visible_indices, dtype=int)
            values, vessel_label = _display_field_values(
                "vessel", vessel_field, cache, options
            )
            values = (
                values[self._visible_result_indices] if values is not None else None
            )
            tissue_mode, tissue_limit = self._layer_policy(
                "tissue", len(cache["tissue_points"])
            )
            tissue_indices, tissue_alpha = select_tissue_points(
                cache["tissue_points"],
                cache["inlets"],
                mode=tissue_mode,
                limit=tissue_limit,
            )
            self._visible_tissue_indices = tissue_indices
            visible_tissue_points = cache["tissue_points"][tissue_indices]
            tissue_values, tissue_label = _display_field_values(
                "tissue", tissue_field, cache, options
            )
            tissue_values = (
                tissue_values[tissue_indices] if tissue_values is not None else None
            )
            radius_source = next(
                (
                    values
                    for name, values in cache["vessel_arrays"].items()
                    if "radius" in name.lower()
                ),
                None,
            )
            visible_radii = (
                np.asarray(radius_source).reshape(-1)[self._visible_result_indices]
                if radius_source is not None
                else None
            )
            if not bool(options.get("show_vessels", True)):
                self._visible_result_indices = np.empty((0,), dtype=int)
                starts = np.empty((0, 3), dtype=np.float32)
                ends = np.empty((0, 3), dtype=np.float32)
                values = None
                visible_radii = None
                shown_vessels = 0
            if not bool(options.get("show_tissue", True)):
                self._visible_tissue_indices = np.empty((0,), dtype=int)
                visible_tissue_points = np.empty((0, 3), dtype=np.float32)
                tissue_values = None
                tissue_alpha = np.empty((0,), dtype=np.float32)
            self.canvas.colormap = str(options.get("colormap", "plasma"))
            self.canvas.vessel_colormap = str(
                options.get("vessel_colormap", self.canvas.colormap)
            )
            self.canvas.tissue_colormap = str(
                options.get("tissue_colormap", self.canvas.vessel_colormap)
            )
            self.canvas.vessel_opacity = float(options.get("vessel_opacity", 1.0))
            self.canvas.tissue_opacity = float(options.get("tissue_opacity", 0.35))
            self.canvas.vessel_range = tuple(options.get("vessel_range", (None, None)))
            self.canvas.tissue_range = tuple(options.get("tissue_range", (None, None)))
            self.canvas.tissue_range_inherit = tuple(
                options.get("tissue_range_inherit", (False, False))
            )
            self.canvas.vessel_scale = str(options.get("vessel_scale", "linear"))
            self.canvas.tissue_scale = str(options.get("tissue_scale", "linear"))
            self.canvas.vessel_label = _layer_field_label("Vessels", vessel_label)
            self.canvas.tissue_label = _layer_field_label("Tissue", tissue_label)
            self.canvas.set_geometry(
                domain_lines=cache["domain_lines"],
                domain_triangles=cache["domain_triangles"],
                vessel_starts=starts,
                vessel_ends=ends,
                vessel_values=values,
                vessel_radii=visible_radii,
                tissue_points=visible_tissue_points,
                tissue_values=tissue_values,
                tissue_alpha=tissue_alpha,
                model_rotation=domain_display_rotation(
                    cache["settings"].get("domain", {})
                ),
            )
            shown_tissue = len(visible_tissue_points)
            total_tissue = len(cache["tissue_points"])
            tree_values = cache["vessel_arrays"].get("tree_id")
            if tree_values is not None and len(tree_values) == len(cache["starts"]):
                all_trees = np.asarray(tree_values)
                tree_count = int(np.unique(all_trees[all_trees >= 0]).size)
            else:
                tree_count = int(cache.get("network", {}).get("tree_count", 0) or 0)
            self.status.setText(
                _count_status(
                    shown_vessels,
                    total_vessels,
                    tree_count,
                    shown_tissue,
                    total_tissue,
                )
            )
            self.status.setVisible(True)
            self.result_fields_loaded.emit(
                list(cache["vessel_arrays"]), list(cache["tissue_arrays"])
            )
        except Exception as exc:
            self.canvas.clear(str(exc))
            self.status.setText(f"Could not load result  │  {exc}")
            self.status.setVisible(True)

    def _load_result(self, manifest_path: str) -> dict[str, Any]:
        path = str(Path(manifest_path).resolve())
        if self._result_cache.get("manifest") == path:
            return self._result_cache
        import pyvista as pv

        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        outputs = manifest.get("outputs", {})

        def read(key):
            raw = outputs.get(key)
            if not raw:
                return None
            target = Path(raw)
            if not target.is_absolute():
                target = Path(path).parent / target
            return pv.read(target) if target.exists() else None

        vessels = read("vessels_vtp")
        tissue = read("oxygen_points_vtp")
        domain = read("domain_boundary_vtp")
        configured_domain = manifest.get("settings", {}).get("domain", {})
        domain_kind = str(
            configured_domain.get("type", configured_domain.get("kind", ""))
        ).lower()
        if domain_kind in {
            "cube",
            "box",
            "rectangular",
            "rectangular_box",
            "sphere",
        }:
            domain_lines = domain_wireframe(configured_domain)
            domain_triangles = domain_surface_triangles(configured_domain)
        else:
            domain_lines = _mesh_wireframe(domain)
            domain_triangles = _mesh_triangles(domain)
        starts, ends, arrays, inlets = _polyline_data(vessels)
        tissue_points = (
            np.asarray(tissue.points, dtype=np.float32)
            if tissue is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        tissue_arrays = _numeric_point_arrays(tissue, len(tissue_points))
        cache = {
            "manifest": path,
            "starts": starts,
            "ends": ends,
            "inlets": inlets,
            "vessel_arrays": arrays,
            "tissue_points": tissue_points,
            "tissue_arrays": tissue_arrays,
            "domain_lines": domain_lines,
            "domain_triangles": domain_triangles,
            "settings": manifest.get("settings", {}),
            "network": manifest.get("network", {}),
        }
        self._result_cache = cache
        return cache


__all__ = ("CasePreview",)
