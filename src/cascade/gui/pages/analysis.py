"""Result analysis page."""

from __future__ import annotations

from PySide6.QtCore import (
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
)
from cascade.gui.model import open_folder
from cascade.gui.preview import COLORMAPS
from cascade.gui.runner import JobRunner
from cascade.gui.widgets import (
    Banner,
)
from pathlib import Path
from cascade.gui.ui_helpers import (
    QMessageBox,
    _combo,
    _double,
    _optional_float,
    _set_combo,
)

from cascade.gui.pages.base import (
    Page,
)


class AnalysisPage(Page):
    render_requested = Signal(str, object)
    use_setup_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(
            "Results",
            "",
            parent,
        )
        select = Card("Result source")
        self.result_job = _combo([])
        self.refresh_btn = QPushButton("Refresh results")
        self.refresh_btn.setProperty("secondary", True)
        self.open_folder_btn = QPushButton("Open results folder")
        self.open_folder_btn.setProperty("secondary", True)
        self.use_setup_btn = QPushButton("Use as current setup")
        self.use_setup_btn.setProperty("secondary", True)
        self.use_setup_btn.setEnabled(False)
        select.add(labeled("Run", self.result_job))
        select.add(
            row_of(self.refresh_btn, self.open_folder_btn, self.use_setup_btn)
        )
        self.column.addWidget(select)

        vessel_render = Card("Vessel network visualization")
        self.vessel_render = vessel_render
        self.vessel_visible = QCheckBox()
        self.vessel_visible.setChecked(True)
        self.vessel_visible.setAccessibleName("Show vessel network visualization")
        self.vessel_visible.setToolTip("Show vessel network visualization")
        self.vessel_visible.setFixedWidth(18)
        vessel_render.heading_row.setContentsMargins(8, 0, 0, 0)
        vessel_render.heading_row.setSpacing(4)
        vessel_render.heading_row.insertWidget(0, self.vessel_visible)
        self.vessel_field = _combo([])
        self.tissue_field = _combo([])
        self.vessel_colormap = _combo(
            [(name.capitalize(), name) for name in COLORMAPS]
        )
        self.tissue_colormap = _combo(
            [("Inherit from vessels", "inherit")]
            + [(name.capitalize(), name) for name in COLORMAPS]
        )
        self.colormap = self.vessel_colormap
        self.vessel_opacity = _double(1.0, 0.0, 1.0, 2, 0.05)
        self.tissue_opacity = _double(0.35, 0.0, 1.0, 2, 0.05)
        self.vessel_min = QLineEdit()
        self.vessel_max = QLineEdit()
        self.tissue_min = QLineEdit()
        self.tissue_max = QLineEdit()
        for field in (
            self.vessel_min,
            self.vessel_max,
            self.tissue_min,
            self.tissue_max,
        ):
            field.setPlaceholderText("Auto")
        self.vessel_scale = _combo([("Linear", "linear"), ("Logarithmic", "log")])
        self.tissue_scale = _combo(
            [
                ("Inherit from vessels", "inherit"),
                ("Linear", "linear"),
                ("Logarithmic", "log"),
            ]
        )
        self.normalize_fields = QCheckBox(
            "Normalize displayed quantities by inlet values"
        )
        self.normalize_fields.setStyleSheet(
            "QCheckBox:disabled { color: #778196; }"
        )
        self.vessel_concentration_unit = _combo(
            [("mol/m³", "concentration"), ("mmHg equivalent", "mmhg")]
        )
        self.tissue_concentration_unit = _combo(
            [("mol/m³", "concentration"), ("mmHg equivalent", "mmhg")]
        )
        self.concentration_unit = self.vessel_concentration_unit
        self.flow_unit = _combo([("μL/min", "ul_min"), ("cm³/s", "cm3_s")])
        self.vessel_quantity_row = labeled("Visualized quantity", self.vessel_field)
        vessel_render.add(self.vessel_quantity_row)
        self.vessel_flow_unit_row = labeled("Display units", self.flow_unit)
        self.vessel_concentration_unit_row = labeled(
            "Display units", self.vessel_concentration_unit
        )
        vessel_render.add(self.vessel_flow_unit_row)
        vessel_render.add(self.vessel_concentration_unit_row)
        self.vessel_colormap_row = labeled("Colormap", self.vessel_colormap)
        self.vessel_min_row = labeled("Plotted minimum", self.vessel_min)
        self.vessel_max_row = labeled("Plotted maximum", self.vessel_max)
        self.vessel_scale_row = labeled("Scale", self.vessel_scale)
        self.vessel_opacity_row = labeled("Opacity", self.vessel_opacity)
        vessel_render.add(self.vessel_colormap_row)
        vessel_render.add(self.vessel_min_row)
        vessel_render.add(self.vessel_max_row)
        vessel_render.add(self.vessel_scale_row)
        vessel_render.add(self.vessel_opacity_row)
        vessel_render.add(self.normalize_fields)
        self.vessel_option_widgets = (
            self.vessel_quantity_row,
            self.vessel_flow_unit_row,
            self.vessel_concentration_unit_row,
            self.vessel_colormap_row,
            self.vessel_min_row,
            self.vessel_max_row,
            self.vessel_scale_row,
            self.vessel_opacity_row,
            self.normalize_fields,
        )
        self.column.addWidget(vessel_render)

        tissue_render = Card("Tissue visualization")
        self.tissue_render = tissue_render
        self.tissue_visible = QCheckBox()
        self.tissue_visible.setChecked(True)
        self.tissue_visible.setAccessibleName("Show tissue visualization")
        self.tissue_visible.setToolTip("Show tissue visualization")
        self.tissue_visible.setFixedWidth(18)
        tissue_render.heading_row.setContentsMargins(8, 0, 0, 0)
        tissue_render.heading_row.setSpacing(4)
        tissue_render.heading_row.insertWidget(0, self.tissue_visible)
        self.tissue_quantity_row = labeled("Visualized quantity", self.tissue_field)
        tissue_render.add(self.tissue_quantity_row)
        self.tissue_concentration_unit_row = labeled(
            "Display units", self.tissue_concentration_unit
        )
        tissue_render.add(self.tissue_concentration_unit_row)
        self.tissue_colormap_row = labeled("Colormap", self.tissue_colormap)
        self.tissue_min_row = labeled("Plotted minimum", self.tissue_min)
        self.tissue_max_row = labeled("Plotted maximum", self.tissue_max)
        self.tissue_scale_row = labeled("Scale", self.tissue_scale)
        self.tissue_opacity_row = labeled("Opacity", self.tissue_opacity)
        tissue_render.add(self.tissue_colormap_row)
        tissue_render.add(self.tissue_min_row)
        tissue_render.add(self.tissue_max_row)
        tissue_render.add(self.tissue_scale_row)
        tissue_render.add(self.tissue_opacity_row)
        self.render_note = Banner()
        self.render_note.setVisible(False)
        tissue_render.add(self.render_note)
        self.tissue_option_widgets = (
            self.tissue_quantity_row,
            self.tissue_concentration_unit_row,
            self.tissue_colormap_row,
            self.tissue_min_row,
            self.tissue_max_row,
            self.tissue_scale_row,
            self.tissue_opacity_row,
            self.render_note,
        )
        self.column.addWidget(tissue_render)

        selection = Card("Selection")
        self.selection_details = QLabel("no selection")
        self.selection_details.setObjectName("muted")
        self.selection_details.setWordWrap(True)
        self.selection_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        selection.add(self.selection_details)
        self.column.addWidget(selection)

        self.finish()
        self.runner: JobRunner | None = None
        self.refresh_btn.clicked.connect(self.refresh)
        self.result_job.currentIndexChanged.connect(self._load_selection)
        for combo in (
            self.vessel_field,
            self.tissue_field,
            self.vessel_colormap,
            self.tissue_colormap,
            self.vessel_scale,
            self.tissue_scale,
            self.vessel_concentration_unit,
            self.tissue_concentration_unit,
            self.flow_unit,
        ):
            combo.currentIndexChanged.connect(self._request_render)
        self.vessel_field.currentIndexChanged.connect(self._update_visualization_rows)
        self.tissue_field.currentIndexChanged.connect(self._update_visualization_rows)
        self.vessel_concentration_unit.currentIndexChanged.connect(
            lambda *_: self._sync_concentration_units(
                self.vessel_concentration_unit, self.tissue_concentration_unit
            )
        )
        self.tissue_concentration_unit.currentIndexChanged.connect(
            lambda *_: self._sync_concentration_units(
                self.tissue_concentration_unit, self.vessel_concentration_unit
            )
        )
        for field in (
            self.vessel_min,
            self.vessel_max,
            self.tissue_min,
            self.tissue_max,
        ):
            field.editingFinished.connect(self._request_render)
        self.vessel_opacity.valueChanged.connect(self._request_render)
        self.tissue_opacity.valueChanged.connect(self._request_render)
        self.normalize_fields.toggled.connect(self._request_render)
        for visibility in (self.vessel_visible, self.tissue_visible):
            visibility.toggled.connect(self._update_visualization_rows)
            visibility.toggled.connect(self._request_render)
        self.open_folder_btn.clicked.connect(self._open_folder)
        self.use_setup_btn.clicked.connect(self._use_as_setup)
        self._update_visualization_rows()

    def set_runner(self, runner):
        self.runner = runner
        runner.jobs_changed.connect(self.refresh)
        self.refresh()

    def load(self, config):
        settings = config.get("gui", {}).get("analysis", {})
        _set_combo(
            self.vessel_colormap,
            settings.get("vessel_colormap", settings.get("colormap", "plasma")),
        )
        _set_combo(self.tissue_colormap, settings.get("tissue_colormap", "inherit"))
        self.vessel_opacity.setValue(float(settings.get("vessel_opacity", 1.0)))
        self.tissue_opacity.setValue(float(settings.get("tissue_opacity", 0.35)))
        self.vessel_min.setText(str(settings.get("vessel_min", "")))
        self.vessel_max.setText(str(settings.get("vessel_max", "")))
        self.tissue_min.setText(str(settings.get("tissue_min", "")))
        self.tissue_max.setText(str(settings.get("tissue_max", "")))
        _set_combo(self.vessel_scale, settings.get("vessel_scale", "linear"))
        _set_combo(self.tissue_scale, settings.get("tissue_scale", "inherit"))
        self.normalize_fields.setChecked(bool(settings.get("normalize_fields", False)))
        concentration_unit = settings.get("concentration_unit", "concentration")
        _set_combo(self.vessel_concentration_unit, concentration_unit)
        _set_combo(self.tissue_concentration_unit, concentration_unit)
        _set_combo(self.flow_unit, settings.get("flow_unit", "ul_min"))
        self.vessel_visible.setChecked(bool(settings.get("show_vessels", True)))
        self.tissue_visible.setChecked(bool(settings.get("show_tissue", True)))
        self._update_visualization_rows()

    def write(self, config):
        config.setdefault("gui", {})["analysis"] = {
            "colormap": self.vessel_colormap.currentData(),
            "vessel_colormap": self.vessel_colormap.currentData(),
            "tissue_colormap": self.tissue_colormap.currentData(),
            "vessel_opacity": self.vessel_opacity.value(),
            "tissue_opacity": self.tissue_opacity.value(),
            "vessel_field": self.vessel_field.currentData(),
            "tissue_field": self.tissue_field.currentData(),
            "vessel_min": self.vessel_min.text().strip(),
            "vessel_max": self.vessel_max.text().strip(),
            "tissue_min": self.tissue_min.text().strip(),
            "tissue_max": self.tissue_max.text().strip(),
            "vessel_scale": self.vessel_scale.currentData(),
            "tissue_scale": self.tissue_scale.currentData(),
            "normalize_fields": self.normalize_fields.isChecked(),
            "concentration_unit": self.vessel_concentration_unit.currentData(),
            "flow_unit": self.flow_unit.currentData(),
            "show_vessels": self.vessel_visible.isChecked(),
            "show_tissue": self.tissue_visible.isChecked(),
        }

    def refresh(self):
        if not self.runner:
            return
        previous = self.result_job.currentData()
        self.result_job.blockSignals(True)
        self.result_job.clear()
        jobs = self.runner.result_jobs()
        if getattr(self.window(), "project_path", None) is None:
            jobs = [
                job
                for job in self.runner.jobs
                if job.manifest_path and Path(job.manifest_path).is_file()
            ]
        for job in jobs:
            self.result_job.addItem(job.name, job.id)
        default_result = jobs[-1].id if jobs else None
        _set_combo(self.result_job, previous or default_result)
        self.result_job.blockSignals(False)
        self._load_selection()

    def showEvent(self, event):
        super().showEvent(event)
        if self.runner:
            self.refresh_btn.click()

    def _load_selection(self):
        if not self.runner:
            return
        job = self._selected_job()
        self.open_folder_btn.setEnabled(bool(job or self.runner))
        self.use_setup_btn.setEnabled(bool(job))
        self._request_render()

    def _use_as_setup(self):
        job = self._selected_job()
        if job:
            self.use_setup_requested.emit(job.id)

    def _sync_concentration_units(self, source, target):
        target.blockSignals(True)
        _set_combo(target, source.currentData())
        target.blockSignals(False)
        self._request_render()

    def _concentration_layers_match(self):
        return self.vessel_field.currentData() in {
            "bulk_concentration",
            "wall_concentration",
        } and self.tissue_field.currentData() == "tissue_concentration"

    def _update_visualization_rows(self, *_):
        vessel_field = self.vessel_field.currentData()
        tissue_field = self.tissue_field.currentData()
        vessel_is_concentration = vessel_field in {
            "bulk_concentration",
            "wall_concentration",
        }
        for widget in self.vessel_option_widgets:
            widget.setEnabled(self.vessel_visible.isChecked())
        self.tissue_render.setEnabled(vessel_is_concentration)
        for widget in self.tissue_option_widgets:
            widget.setEnabled(
                vessel_is_concentration and self.tissue_visible.isChecked()
            )
        self.vessel_flow_unit_row.setVisible(vessel_field == "flow")
        self.vessel_concentration_unit_row.setVisible(
            vessel_is_concentration
        )
        self.tissue_concentration_unit_row.setVisible(
            tissue_field == "tissue_concentration"
        )
        inherited = self._concentration_layers_match()
        placeholder = "Inherit from vessels" if inherited else "Auto"
        self.tissue_min.setPlaceholderText(placeholder)
        self.tissue_max.setPlaceholderText(placeholder)

    def set_render_fields(self, vessel_fields, tissue_fields):
        vessel_fields = list(vessel_fields or [])
        tissue_fields = list(tissue_fields or [])
        vessel_options = [
            ("Flow Rate (μL/min)", "flow") if "flow_ul_min" in vessel_fields else None,
            ("Fluid pressure (mmHg)", "pressure")
            if "pressure_pa" in vessel_fields
            else None,
            ("Bulk concentration (mol/m³)", "bulk_concentration")
            if "concentration" in vessel_fields
            else None,
            ("Wall concentration (mol/m³)", "wall_concentration")
            if "wall_oxygen" in vessel_fields
            else None,
            ("Radius (μm)", "radius") if "radius_cm" in vessel_fields else None,
            ("Length (μm)", "length") if "length_cm" in vessel_fields else None,
            ("Discharge hematocrit", "hematocrit")
            if "discharge_hematocrit" in vessel_fields
            else None,
        ]
        tissue_options = [
            ("Tissue concentration (mol/m³)", "tissue_concentration")
            if "local_concentration" in tissue_fields
            else None,
            ("Viable tissue", "viability") if "viability" in tissue_fields else None,
            ("Distance to nearest vessel (μm)", "distance")
            if "dnc_cm" in tissue_fields
            else None,
        ]
        vessel_options = [option for option in vessel_options if option]
        tissue_options = [option for option in tissue_options if option]
        if [
            (self.vessel_field.itemText(i), self.vessel_field.itemData(i))
            for i in range(self.vessel_field.count())
        ] == vessel_options and [
            (self.tissue_field.itemText(i), self.tissue_field.itemData(i))
            for i in range(self.tissue_field.count())
        ] == tissue_options:
            self._update_visualization_rows()
            return
        old_vessel = self.vessel_field.currentData()
        old_tissue = self.tissue_field.currentData()
        for combo, fields, old, preferred in (
            (
                self.vessel_field,
                vessel_options,
                old_vessel,
                "wall_concentration",
            ),
            (self.tissue_field, tissue_options, old_tissue, "tissue_concentration"),
        ):
            combo.blockSignals(True)
            combo.clear()
            for label, value in fields:
                combo.addItem(label, value)
            _set_combo(
                combo,
                old if any(value == old for _label, value in fields) else preferred,
            )
            combo.blockSignals(False)
        self._update_visualization_rows()
        self._request_render()

    def set_selection(self, details):
        if not details:
            self.selection_details.setText("no selection")
            return
        kind = str(details.get("kind", "selection")).capitalize()
        identifier = details.get("id", details.get("index", "—"))
        lines = [f"{kind} {identifier}"]
        values = details.get("values", {})
        preferred = (
            "radius",
            "length",
            "pressure",
            "flow",
            "velocity",
            "hematocrit",
            "oxygen",
            "concentration",
            "viability",
        )
        ordered = sorted(
            values,
            key=lambda name: (
                next(
                    (i for i, token in enumerate(preferred) if token in name.lower()),
                    99,
                ),
                name.lower(),
            ),
        )
        for name in ordered[:10]:
            value = values[name]
            try:
                rendered = f"{float(value):.6g}"
            except (TypeError, ValueError):
                rendered = str(value)
            lines.append(f"{name.replace('_', ' ')}   {rendered}")
        self.selection_details.setText("\n".join(lines))

    def _request_render(self, *_):
        job = self._selected_job()
        if not job or not Path(job.output_dir, "manifest.json").exists():
            self.render_requested.emit("", {})
            return
        self.render_note.setVisible(False)
        tissue_colormap = self.tissue_colormap.currentData()
        if tissue_colormap == "inherit":
            tissue_colormap = self.vessel_colormap.currentData()
        tissue_scale = self.tissue_scale.currentData()
        if tissue_scale == "inherit":
            tissue_scale = self.vessel_scale.currentData()
        inherit_range = self._concentration_layers_match()
        vessel_is_concentration = self.vessel_field.currentData() in {
            "bulk_concentration",
            "wall_concentration",
        }
        tissue_min_text = self.tissue_min.text().strip()
        tissue_max_text = self.tissue_max.text().strip()
        self.render_requested.emit(
            str(Path(job.output_dir) / "manifest.json"),
            {
                "vessel_field": self.vessel_field.currentData(),
                "tissue_field": self.tissue_field.currentData(),
                "colormap": self.vessel_colormap.currentData(),
                "vessel_colormap": self.vessel_colormap.currentData(),
                "tissue_colormap": tissue_colormap,
                "vessel_opacity": self.vessel_opacity.value(),
                "tissue_opacity": self.tissue_opacity.value(),
                "vessel_range": (
                    _optional_float(self.vessel_min.text())
                    if self.vessel_min.text().strip()
                    else (0.0 if self.normalize_fields.isChecked() else None),
                    _optional_float(self.vessel_max.text())
                    if self.vessel_max.text().strip()
                    else (1.0 if self.normalize_fields.isChecked() else None),
                ),
                "tissue_range": (
                    _optional_float(tissue_min_text)
                    if tissue_min_text
                    else (0.0 if self.normalize_fields.isChecked() else None),
                    _optional_float(tissue_max_text)
                    if tissue_max_text
                    else (1.0 if self.normalize_fields.isChecked() else None),
                ),
                "tissue_range_inherit": (
                    inherit_range and not tissue_min_text,
                    inherit_range and not tissue_max_text,
                ),
                "vessel_scale": self.vessel_scale.currentData(),
                "tissue_scale": self.tissue_scale.currentData(),
                "normalize_fields": self.normalize_fields.isChecked(),
                "concentration_unit": self.vessel_concentration_unit.currentData(),
                "flow_unit": self.flow_unit.currentData(),
                "show_vessels": self.vessel_visible.isChecked(),
                "show_tissue": (
                    self.tissue_visible.isChecked() and vessel_is_concentration
                ),
            },
        )

    def _selected_job(self):
        if not self.runner:
            return None
        selected = self.result_job.currentData()
        return next((job for job in self.runner.result_jobs() if job.id == selected), None)

    def _open_folder(self):
        job = self._selected_job()
        try:
            if job:
                open_folder(job.output_dir)
            elif self.runner:
                open_folder(self.runner.store.root.parent / "results")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open results folder", str(exc))


__all__ = ("AnalysisPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, labeled, row_of
