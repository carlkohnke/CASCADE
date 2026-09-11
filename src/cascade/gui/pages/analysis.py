"""Result analysis page."""

from __future__ import annotations

from cascade.gui._common import (
    Banner,
    COLORMAPS,
    Card,
    JobRunner,
    Path,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    Qt,
    Signal,
    _combo,
    _double,
    _optional_float,
    _set_combo,
    labeled,
    open_folder,
    row_of,
)

from cascade.gui.pages.base import (
    Page,
)

class AnalysisPage(Page):
    render_requested = Signal(str, object)

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
        select.add(labeled("Run", self.result_job))
        select.add(row_of(self.refresh_btn, self.open_folder_btn))
        self.column.addWidget(select)

        render = Card("3D view")
        self.vessel_field = _combo([])
        self.tissue_field = _combo([])
        self.colormap = _combo([(name.capitalize(), name) for name in COLORMAPS])
        self.vessel_opacity = _double(1.0, 0.0, 1.0, 2, 0.05)
        self.tissue_opacity = _double(0.35, 0.0, 1.0, 2, 0.05)
        self.vessel_min = QLineEdit()
        self.vessel_max = QLineEdit()
        self.tissue_min = QLineEdit()
        self.tissue_max = QLineEdit()
        for field in (self.vessel_min, self.vessel_max, self.tissue_min, self.tissue_max):
            field.setPlaceholderText("Auto")
        self.vessel_scale = _combo([("Linear", "linear"), ("Logarithmic", "log")])
        self.tissue_scale = _combo([("Linear", "linear"), ("Logarithmic", "log")])
        self.normalize_fields = QCheckBox("Normalize displayed quantities by inlet values")
        self.concentration_unit = _combo(
            [("mol/m³", "concentration"), ("mmHg equivalent", "mmhg")]
        )
        self.flow_unit = _combo([("μL/min", "ul_min"), ("cm³/s", "cm3_s")])
        render.add(
            row_of(
                labeled("Vessels", self.vessel_field),
                labeled("Tissue", self.tissue_field),
            )
        )
        render.add(labeled("Colormap", self.colormap))
        render.add(
            row_of(
                labeled("Vessel minimum", self.vessel_min),
                labeled("Vessel maximum", self.vessel_max),
                labeled("Vessel scale", self.vessel_scale),
            )
        )
        render.add(
            row_of(
                labeled("Tissue minimum", self.tissue_min),
                labeled("Tissue maximum", self.tissue_max),
                labeled("Tissue scale", self.tissue_scale),
            )
        )
        render.add(
            row_of(
                labeled("Vessel opacity", self.vessel_opacity),
                labeled("Tissue opacity", self.tissue_opacity),
            )
        )
        render.add(row_of(self.normalize_fields, labeled("Flow display", self.flow_unit), labeled("Concentration display", self.concentration_unit)))
        self.render_note = Banner()
        self.render_note.setVisible(False)
        render.add(self.render_note)
        self.column.addWidget(render)

        selection = Card("Selection")
        self.selection_details = QLabel(
            "no selection"
        )
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
            self.colormap,
            self.vessel_scale,
            self.tissue_scale,
            self.concentration_unit,
            self.flow_unit,
        ):
            combo.currentIndexChanged.connect(self._request_render)
        for field in (self.vessel_min, self.vessel_max, self.tissue_min, self.tissue_max):
            field.editingFinished.connect(self._request_render)
        self.vessel_opacity.valueChanged.connect(self._request_render)
        self.tissue_opacity.valueChanged.connect(self._request_render)
        self.normalize_fields.toggled.connect(self._request_render)
        self.open_folder_btn.clicked.connect(self._open_folder)

    def set_runner(self, runner):
        self.runner = runner
        runner.jobs_changed.connect(self.refresh)
        self.refresh()

    def load(self, config):
        settings = config.get("gui", {}).get("analysis", {})
        _set_combo(self.colormap, settings.get("colormap", "plasma"))
        self.vessel_opacity.setValue(float(settings.get("vessel_opacity", 1.0)))
        self.tissue_opacity.setValue(float(settings.get("tissue_opacity", 0.35)))
        self.vessel_min.setText(str(settings.get("vessel_min", "")))
        self.vessel_max.setText(str(settings.get("vessel_max", "")))
        self.tissue_min.setText(str(settings.get("tissue_min", "")))
        self.tissue_max.setText(str(settings.get("tissue_max", "")))
        _set_combo(self.vessel_scale, settings.get("vessel_scale", "linear"))
        _set_combo(self.tissue_scale, settings.get("tissue_scale", "linear"))
        self.normalize_fields.setChecked(bool(settings.get("normalize_fields", False)))
        _set_combo(self.concentration_unit, settings.get("concentration_unit", "concentration"))
        _set_combo(self.flow_unit, settings.get("flow_unit", "ul_min"))

    def write(self, config):
        config.setdefault("gui", {})["analysis"] = {
            "colormap": self.colormap.currentData(),
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
            "concentration_unit": self.concentration_unit.currentData(),
            "flow_unit": self.flow_unit.currentData(),
        }

    def refresh(self):
        if not self.runner:
            return
        previous = self.result_job.currentData()
        self.result_job.blockSignals(True)
        self.result_job.clear()
        jobs = list(self.runner.jobs)
        for job in jobs:
            self.result_job.addItem(f"{job.name}  —  {job.status}", job.id)
        default_result = jobs[-1].id if jobs else None
        _set_combo(self.result_job, previous or default_result)
        self.result_job.blockSignals(False)
        self._load_selection()

    def _load_selection(self):
        if not self.runner:
            return
        job = self._selected_job()
        self.open_folder_btn.setEnabled(bool(job or self.runner))
        self._request_render()

    def set_render_fields(self, vessel_fields, tissue_fields):
        vessel_fields = list(vessel_fields or [])
        tissue_fields = list(tissue_fields or [])
        vessel_options = [
            ("Flow Rate (μL/min)", "flow") if "flow_ul_min" in vessel_fields else None,
            ("Fluid pressure (mmHg)", "pressure") if "pressure_pa" in vessel_fields else None,
            ("Bulk concentration (mol/m³)", "bulk_concentration") if "concentration" in vessel_fields else None,
            ("Wall concentration (mol/m³)", "wall_concentration") if "wall_oxygen" in vessel_fields else None,
            ("Radius (μm)", "radius") if "radius_cm" in vessel_fields else None,
            ("Length (μm)", "length") if "length_cm" in vessel_fields else None,
            ("Discharge hematocrit", "hematocrit") if "discharge_hematocrit" in vessel_fields else None,
        ]
        tissue_options = [
            ("Tissue concentration (mol/m³)", "tissue_concentration") if "local_concentration" in tissue_fields else None,
            ("Viable tissue", "viability") if "viability" in tissue_fields else None,
            ("Distance to nearest vessel (μm)", "distance") if "dnc_cm" in tissue_fields else None,
        ]
        vessel_options = [option for option in vessel_options if option]
        tissue_options = [option for option in tissue_options if option]
        if (
            [(self.vessel_field.itemText(i), self.vessel_field.itemData(i)) for i in range(self.vessel_field.count())]
            == vessel_options
            and [(self.tissue_field.itemText(i), self.tissue_field.itemData(i)) for i in range(self.tissue_field.count())]
            == tissue_options
        ):
            return
        old_vessel = self.vessel_field.currentData()
        old_tissue = self.tissue_field.currentData()
        for combo, fields, old, preferred in (
            (self.vessel_field, vessel_options, old_vessel, "flow"),
            (self.tissue_field, tissue_options, old_tissue, "tissue_concentration"),
        ):
            combo.blockSignals(True)
            combo.clear()
            for label, value in fields:
                combo.addItem(label, value)
            _set_combo(combo, old if any(value == old for _label, value in fields) else preferred)
            combo.blockSignals(False)
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
                next((i for i, token in enumerate(preferred) if token in name.lower()), 99),
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
        self.render_requested.emit(
            str(Path(job.output_dir) / "manifest.json"),
            {
                "vessel_field": self.vessel_field.currentData(),
                "tissue_field": self.tissue_field.currentData(),
                "colormap": self.colormap.currentData(),
                "vessel_opacity": self.vessel_opacity.value(),
                "tissue_opacity": self.tissue_opacity.value(),
                "vessel_range": (
                    _optional_float(self.vessel_min.text()),
                    _optional_float(self.vessel_max.text()),
                ),
                "tissue_range": (
                    _optional_float(self.tissue_min.text()),
                    _optional_float(self.tissue_max.text()),
                ),
                "vessel_scale": self.vessel_scale.currentData(),
                "tissue_scale": self.tissue_scale.currentData(),
                "normalize_fields": self.normalize_fields.isChecked(),
                "concentration_unit": self.concentration_unit.currentData(),
                "flow_unit": self.flow_unit.currentData(),
            },
        )

    def _selected_job(self):
        if not self.runner:
            return None
        selected = self.result_job.currentData()
        return next((job for job in self.runner.jobs if job.id == selected), None)

    def _open_folder(self):
        job = self._selected_job()
        if job:
            open_folder(job.output_dir)
        elif self.runner:
            open_folder(self.runner.store.root.parent / "results")




__all__ = ('AnalysisPage',)
