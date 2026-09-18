"""Output and export configuration page.

The page controls result directories, table/mesh products, tissue annotations,
network persistence, and sweep-specific output options.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.model import (
    flow_from_ul_min,
    flow_to_ul_min,
    oxygen_from_concentration,
    oxygen_to_concentration,
    parse_sweep_values,
)
from cascade.gui.widgets import (
    PathPicker,
)
from cascade.gui.ui_helpers import (
    _combo,
    _set_combo,
    _spin,
)

from cascade.gui.pages.base import (
    Page,
)


class OutputsPage(Page):
    preview_changed = Signal()

    SWEEP_PATHS = [
        ("Inlet flow", "simulation.qin_target_ul_min"),
        ("Growth count N", "network.target_terminal_count"),
        ("Diffusivity", "settings.oxygen.solute_diffusivity"),
        ("Vmax", "settings.oxygen.vmax_mm"),
        ("Km", "settings.oxygen.k_m_mm"),
        ("Discharge hematocrit", "settings.hematocrit.hd_discharge"),
        ("Inlet concentration", "settings.oxygen.conc_max_for_normalization"),
        ("FFT grid", "settings.cext.hybrid_bg_grid"),
        ("Cₑₓₜ iterations", "settings.cext.vess_coupling_max_iter"),
    ]
    SWEEP_UNITS = {
        "simulation.qin_target_ul_min": (
            "µL/min",
            ["µL/min", "cm³/s", "mL/min", "m³/s"],
        ),
        "network.target_terminal_count": ("count", ["count"]),
        "settings.oxygen.solute_diffusivity": ("cm²/s", ["cm²/s", "m²/s"]),
        "settings.oxygen.vmax_mm": ("mol/m³/s", ["mol/m³/s", "mmHg/s"]),
        "settings.oxygen.k_m_mm": ("mol/m³", ["mol/m³", "mmHg"]),
        "settings.hematocrit.hd_discharge": ("fraction", ["fraction", "%"]),
        "settings.oxygen.conc_max_for_normalization": ("mol/m³", ["mol/m³", "mmHg"]),
        "settings.cext.hybrid_bg_grid": ("points", ["points"]),
        "settings.cext.vess_coupling_max_iter": ("iterations", ["iterations"]),
    }

    def __init__(self, parent=None):
        super().__init__(
            "Outputs & sweeps",
            "",
            parent,
        )
        sample = Card(
            "Tissue domain points",
            "Choose independent random points, a structured Cartesian grid, or a frozen coordinate file.",
        )
        self.sample_mode = _combo(
            [
                ("Random points", "random"),
                ("Structured Cartesian grid", "grid"),
                ("Fixed coordinate file", "file"),
            ]
        )
        self.sample_points = _spin(10000, 0, 100_000_000, 1000)
        self.sample_file = PathPicker(
            mode="file",
            caption="Choose fixed tissue sample coordinates",
            file_filter="Coordinate files (*.csv *.npy *.npz);;All files (*)",
        )
        self.grid_x = _spin(20, 2, 2048)
        self.grid_y = _spin(20, 2, 2048)
        self.grid_z = _spin(20, 2, 2048)
        self.grid_fields = row_of(
            labeled("Grid points in X", self.grid_x),
            labeled("Grid points in Y", self.grid_y),
            labeled("Grid points in Z", self.grid_z),
        )
        self.sample_controls = CompactStack()
        self.sample_controls.addWidget(
            labeled("Number of random points", self.sample_points)
        )
        self.sample_controls.addWidget(self.grid_fields)
        self.sample_controls.addWidget(
            labeled("Coordinate file (x, y, z in cm)", self.sample_file)
        )
        self.grid_total = QLabel()
        self.grid_total.setObjectName("fieldHelp")
        sample.add(labeled("Point layout", self.sample_mode, important=True))
        sample.add(self.sample_controls)
        sample.add(self.grid_total)
        self.column.addWidget(sample)

        output = Card(
            "Result files",
            "Summary-only runs minimize peak export memory. Enable detailed files deliberately.",
        )
        self.out_dir = PathPicker(mode="directory", caption="Choose result base folder")
        self.prefix = QLineEdit("cascade_run")
        self.summary_csv = QCheckBox("Summary metrics CSV")
        self.summary_csv.setChecked(True)
        self.combined_sweep_csv = QCheckBox("Combined sweep CSV")
        self.combined_sweep_csv.setChecked(True)
        self.combined_sweep_filename = QLineEdit("sweep_summary.csv")
        self.combined_sweep_filename.setPlaceholderText("sweep_summary.csv")
        self.segments_csv = QCheckBox("Per-vessel CSV")
        self.points_csv = QCheckBox("Tissue-points CSV")
        self.paraview = QCheckBox("ParaView VTK files")
        self.save_network = QCheckBox("Save vessel network")
        self.paraview.setToolTip(
            "Enable for a saved visualization snapshot; summary-only is much faster."
        )
        self.save_network.setToolTip(
            "Enable when the generated network itself must be preserved."
        )
        self.nearest_fields = QCheckBox("Add nearest-vessel fields to tissue points")
        checks = QWidget()
        grid = QGridLayout(checks)
        grid.setContentsMargins(0, 0, 0, 0)
        for index, check in enumerate(
            [
                self.summary_csv,
                self.combined_sweep_csv,
                self.segments_csv,
                self.points_csv,
                self.paraview,
                self.save_network,
                self.nearest_fields,
            ]
        ):
            grid.addWidget(check, index // 2, index % 2)
        self.float_dtype = _combo(
            [("Float32 exports", "float32"), ("Float64 exports", "float64")]
        )
        self.index_dtype = _combo(
            [("Int32 indices", "int32"), ("Int64 indices", "int64")]
        )
        self.vessel_resolution = _spin(2, 2, 32)
        output.add(labeled("Result base folder", self.out_dir))
        output.add(labeled("File prefix", self.prefix))
        output.add(checks)
        output.add(labeled("Combined sweep filename", self.combined_sweep_filename))
        output.add(
            row_of(
                labeled("Floating-point export", self.float_dtype),
                labeled("Index export", self.index_dtype),
                labeled(
                    "Minimum VTK points / vessel",
                    self.vessel_resolution,
                    "Computed external-field quadrature nodes are always retained.",
                ),
            )
        )
        self.column.addWidget(output)

        sweep = Card(
            "Parameter sweeps",
            "Each active dimension forms a Cartesian product. Values are comma-separated and each expanded run is queued independently.",
        )
        self.sweep_rows: list[dict[str, QWidget]] = []
        self.sweep_rows_panel = QFrame()
        self.sweep_rows_panel.setObjectName("sweepRows")
        self.sweep_rows_panel.setMinimumHeight(42)
        self.sweep_rows_layout = QVBoxLayout(self.sweep_rows_panel)
        self.sweep_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.sweep_rows_layout.setSpacing(2)
        self.sweep_rows_layout.addStretch()
        add = QPushButton("+")
        add.setProperty("secondary", True)
        add.setFixedSize(28, 26)
        add.setStyleSheet("padding:0;font-size:18px;font-weight:500;")
        add.setToolTip("Add sweep dimension")
        add.setAccessibleName("Add sweep dimension")
        self.add_sweep_btn = add
        add.clicked.connect(lambda _checked=False: self._add_sweep())
        sweep.heading_row.addWidget(add)
        footer = QWidget()
        row = QHBoxLayout(footer)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch()
        self.job_count = QLabel("1 simulation")
        self.job_count.setStyleSheet("font-weight:800;color:#61C7A4;")
        row.addWidget(self.job_count)
        sweep.add(self.sweep_rows_panel)
        sweep.add(footer)
        self.column.addWidget(sweep)
        self.sample_mode.currentIndexChanged.connect(self._sampling_changed)
        self.sample_mode.currentIndexChanged.connect(
            lambda *_: self.preview_changed.emit()
        )
        self.sample_points.valueChanged.connect(lambda *_: self.preview_changed.emit())
        self.sample_file.changed.connect(lambda *_: self.preview_changed.emit())
        self.combined_sweep_csv.toggled.connect(self._combined_sweep_output_changed)
        self.summary_csv.toggled.connect(self._summary_output_changed)
        for field in (self.grid_x, self.grid_y, self.grid_z):
            field.valueChanged.connect(self._update_grid_total)
            field.valueChanged.connect(lambda *_: self.preview_changed.emit())
        self._update_grid_total()
        self._sampling_changed()
        self._update_combined_sweep_controls()
        self.finish()

    def _combined_sweep_output_changed(self, checked: bool) -> None:
        if checked and not self.summary_csv.isChecked():
            self.summary_csv.setChecked(True)
        self._update_combined_sweep_controls()

    def _summary_output_changed(self, checked: bool) -> None:
        if not checked and self.combined_sweep_csv.isChecked():
            self.combined_sweep_csv.setChecked(False)
        self._update_combined_sweep_controls()

    def _update_combined_sweep_controls(self) -> None:
        has_sweep = any(record["enabled"].isChecked() for record in self.sweep_rows)
        self.combined_sweep_csv.setEnabled(has_sweep)
        self.combined_sweep_filename.setEnabled(
            has_sweep and self.combined_sweep_csv.isChecked()
        )

    def _sampling_changed(self):
        mode = self.sample_mode.currentData()
        grid = mode == "grid"
        self.sample_controls.setCurrentIndex(
            {"random": 0, "grid": 1, "file": 2}.get(mode, 0)
        )
        self.grid_total.setVisible(grid)

    def _update_grid_total(self, *_):
        total = self.grid_x.value() * self.grid_y.value() * self.grid_z.value()
        self.grid_total.setText(
            f"{self.grid_x.value()} × {self.grid_y.value()} × {self.grid_z.value()} = {total:,} tissue points"
        )

    def _add_sweep(self, raw=None):
        # QPushButton.clicked supplies a bool; only mappings are persisted-row
        # data. Keep this defensive normalization for programmatic callers.
        if isinstance(raw, bool):
            raw = None
        used_paths = self._sweep_paths()
        if raw is None:
            available = [
                path for _label, path in self.SWEEP_PATHS if path not in used_paths
            ]
            if not available:
                self.add_sweep_btn.setEnabled(False)
                return
        raw = (
            raw
            if isinstance(raw, dict)
            else {"enabled": True, "path": available[0], "values": []}
        )
        row_widget = QFrame()
        row_widget.setObjectName("sweepRow")
        row_layout = QGridLayout(row_widget)
        row_layout.setContentsMargins(8, 3, 0, 3)
        row_layout.setHorizontalSpacing(8)
        row_layout.setColumnStretch(3, 1)
        enabled = QCheckBox()
        enabled.setChecked(bool(raw.get("enabled", True)))
        combo = _combo(self.SWEEP_PATHS)
        combo.setMinimumWidth(170)
        _set_combo(combo, raw.get("path"))
        path = str(combo.currentData() or "")
        default_unit, unit_options = self.SWEEP_UNITS[path]
        selected_unit = str(raw.get("unit") or default_unit)
        if selected_unit not in unit_options:
            selected_unit = default_unit
        unit = _combo(unit_options)
        unit.setFixedWidth(96)
        _set_combo(unit, selected_unit)
        unit.setEnabled(len(unit_options) > 1)
        display_values = [
            self._sweep_value_from_base(path, value, selected_unit)
            for value in raw.get("values", [])
        ]
        values = QLineEdit(self._format_sweep_values(display_values))
        values.setPlaceholderText("Example: 10, 20, 30")
        remove = QPushButton("×")
        remove.setProperty("secondary", True)
        remove.setFixedSize(30, 26)
        remove.setStyleSheet("padding:0;font-size:16px;font-weight:500;")
        remove.setToolTip("Remove this sweep dimension")
        remove.setAccessibleName(f"Remove {combo.currentText()} sweep")
        row_layout.addWidget(enabled, 0, 0, Qt.AlignHCenter)
        row_layout.addWidget(combo, 0, 1)
        row_layout.addWidget(unit, 0, 2)
        row_layout.addWidget(values, 0, 3)
        row_layout.addWidget(remove, 0, 4)
        record = {
            "widget": row_widget,
            "enabled": enabled,
            "combo": combo,
            "unit": unit,
            "unit_name": selected_unit,
            "values": values,
            "remove": remove,
        }
        self.sweep_rows.append(record)
        self.sweep_rows_layout.insertWidget(
            self.sweep_rows_layout.count() - 1, row_widget
        )
        combo.currentIndexChanged.connect(
            lambda *_: self._sweep_parameter_changed(record)
        )
        unit.currentTextChanged.connect(lambda *_: self._sweep_unit_changed(record))
        values.textChanged.connect(self._count_jobs)
        enabled.toggled.connect(self._count_jobs)
        remove.clicked.connect(lambda _checked=False: self._remove_sweep(record))
        self._count_jobs()

    def _sweep_paths(self, *, exclude=None) -> set[str]:
        paths: set[str] = set()
        for record in self.sweep_rows:
            combo = record["combo"]
            if combo is not None and combo is not exclude and combo.currentData():
                paths.add(str(combo.currentData()))
        return paths

    def _sweep_parameter_changed(self, record) -> None:
        """A Cartesian sweep has one independent axis per parameter."""
        combo = record["combo"]
        path = str(combo.currentData() or "")
        used = self._sweep_paths(exclude=combo)
        if path in used:
            replacement = next(
                (value for _label, value in self.SWEEP_PATHS if value not in used),
                None,
            )
            if replacement is not None:
                combo.blockSignals(True)
                _set_combo(combo, replacement)
                combo.blockSignals(False)
                path = str(combo.currentData() or "")
        self._set_sweep_unit_options(record, path)
        record["remove"].setAccessibleName(f"Remove {combo.currentText()} sweep")
        self.add_sweep_btn.setEnabled(len(self._sweep_paths()) < len(self.SWEEP_PATHS))
        self._count_jobs()

    def _set_sweep_unit_options(
        self, record, path: str, selected: str | None = None
    ) -> None:
        default_unit, options = self.SWEEP_UNITS[path]
        unit = record["unit"]
        unit.blockSignals(True)
        unit.clear()
        unit.addItems(options)
        _set_combo(unit, selected if selected in options else default_unit)
        unit.setEnabled(len(options) > 1)
        unit.blockSignals(False)
        record["unit_name"] = unit.currentText()

    def _sweep_unit_changed(self, record) -> None:
        path = str(record["combo"].currentData() or "")
        old_unit = str(record.get("unit_name") or self.SWEEP_UNITS[path][0])
        new_unit = record["unit"].currentText()
        parsed = parse_sweep_values(record["values"].text())
        converted = [
            self._sweep_value_from_base(
                path,
                self._sweep_value_to_base(path, value, old_unit),
                new_unit,
            )
            for value in parsed
        ]
        record["unit_name"] = new_unit
        record["values"].setText(self._format_sweep_values(converted))

    def _remove_sweep(self, record=None):
        if record is not None and record in self.sweep_rows:
            self.sweep_rows.remove(record)
            record["widget"].setParent(None)
            record["widget"].deleteLater()
        self.add_sweep_btn.setEnabled(True)
        self._count_jobs()

    @staticmethod
    def _format_sweep_values(values) -> str:
        return ", ".join(
            format(float(value), ".12g")
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else str(value)
            for value in values
        )

    @staticmethod
    def _sweep_value_to_base(path: str, value, unit: str):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return value
        if path == "simulation.qin_target_ul_min":
            return flow_to_ul_min(value, unit)
        if path == "settings.oxygen.solute_diffusivity":
            return float(value) * (1e4 if unit == "m²/s" else 1.0)
        if path in {
            "settings.oxygen.vmax_mm",
            "settings.oxygen.k_m_mm",
            "settings.oxygen.conc_max_for_normalization",
        }:
            return oxygen_to_concentration(value, unit)
        if path == "settings.hematocrit.hd_discharge" and unit == "%":
            return float(value) / 100.0
        return value

    @staticmethod
    def _sweep_value_from_base(path: str, value, unit: str):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return value
        if path == "simulation.qin_target_ul_min":
            return flow_from_ul_min(value, unit)
        if path == "settings.oxygen.solute_diffusivity":
            return float(value) / 1e4 if unit == "m²/s" else float(value)
        if path in {
            "settings.oxygen.vmax_mm",
            "settings.oxygen.k_m_mm",
            "settings.oxygen.conc_max_for_normalization",
        }:
            return oxygen_from_concentration(value, unit)
        if path == "settings.hematocrit.hd_discharge" and unit == "%":
            return float(value) * 100.0
        return value

    def _count_jobs(self, *_):
        count = 1
        for record in self.sweep_rows:
            enabled = record["enabled"]
            values = record["values"]
            if enabled.isChecked():
                count *= max(
                    len(parse_sweep_values(values.text() if values else "")), 1
                )
        self.job_count.setText(f"{count:,} simulation{'s' if count != 1 else ''}")
        self._update_combined_sweep_controls()

    def set_resource(self, estimate):
        # Resource estimates are deliberately not a persistent UI element.
        # Retain this hook while status refreshes are shared across pages.
        return None

    def load(self, config):
        sim = config.get("simulation", {})
        outputs = config.get("outputs", {})
        _set_combo(self.sample_mode, sim.get("sample_mode", "random"))
        self.sample_points.setValue(int(sim.get("distance_sample_count", 10000)))
        self.sample_file.setText(str(sim.get("sample_points_path") or ""))
        grid = sim.get("tissue_grid", {})
        shape = list(grid.get("shape", grid.get("dimensions", [20, 20, 20])) or [])
        shape = (shape + [20, 20, 20])[:3]
        shape = [
            grid.get("nx", shape[0]),
            grid.get("ny", shape[1]),
            grid.get("nz", shape[2]),
        ]
        self.grid_x.setValue(int(shape[0]))
        self.grid_y.setValue(int(shape[1]))
        self.grid_z.setValue(int(shape[2]))
        self._sampling_changed()
        self.out_dir.setText(str(outputs.get("out_dir", "results")))
        self.prefix.setText(str(outputs.get("prefix") or "cascade_run"))
        self.summary_csv.setChecked(bool(outputs.get("write_summary_csv", True)))
        self.combined_sweep_csv.setChecked(
            bool(outputs.get("write_combined_sweep_csv", True))
        )
        self.combined_sweep_filename.setText(
            str(outputs.get("combined_sweep_filename", "sweep_summary.csv"))
        )
        self.segments_csv.setChecked(bool(outputs.get("write_segments_csv", False)))
        self.points_csv.setChecked(bool(outputs.get("write_points_csv", False)))
        self.paraview.setChecked(bool(outputs.get("write_paraview", True)))
        self.save_network.setChecked(bool(outputs.get("save_network", True)))
        self.nearest_fields.setChecked(
            bool(outputs.get("include_tissue_nearest_fields", False))
        )
        _set_combo(self.float_dtype, outputs.get("export_float_dtype", "float32"))
        _set_combo(self.index_dtype, outputs.get("export_index_dtype", "int32"))
        self.vessel_resolution.setValue(int(outputs.get("vessel_resolution", 2)))
        for record in self.sweep_rows:
            record["widget"].setParent(None)
            record["widget"].deleteLater()
        self.sweep_rows.clear()
        seen_paths: set[str] = set()
        for raw in config.get("gui", {}).get("sweeps", []):
            if raw.get("path") not in seen_paths:
                self._add_sweep(raw)
                seen_paths.add(raw.get("path"))
        self.add_sweep_btn.setEnabled(len(self._sweep_paths()) < len(self.SWEEP_PATHS))
        self._count_jobs()

    def write(self, config):
        sim = config.setdefault("simulation", {})
        sim["sample_mode"] = self.sample_mode.currentData()
        sim["distance_sample_count"] = (
            0 if sim["sample_mode"] == "file" else self.sample_points.value()
        )
        if sim["sample_mode"] == "grid":
            grid = dict(sim.get("tissue_grid", {}))
            grid.pop("shape", None)
            grid.pop("dimensions", None)
            grid.update(
                nx=self.grid_x.value(),
                ny=self.grid_y.value(),
                nz=self.grid_z.value(),
            )
            sim["tissue_grid"] = grid
        if sim["sample_mode"] == "file":
            sim["sample_points_path"] = self.sample_file.text()
        else:
            sim.pop("sample_points_path", None)
        outputs = config.setdefault("outputs", {})
        outputs.update(
            {
                "out_dir": self.out_dir.text() or "results",
                "prefix": self.prefix.text().strip() or "cascade_run",
                "write_summary_csv": self.summary_csv.isChecked(),
                "write_combined_sweep_csv": self.combined_sweep_csv.isChecked(),
                "combined_sweep_filename": (
                    self.combined_sweep_filename.text().strip() or "sweep_summary.csv"
                ),
                "write_segments_csv": self.segments_csv.isChecked(),
                "write_points_csv": self.points_csv.isChecked(),
                "write_paraview": self.paraview.isChecked(),
                "save_network": self.save_network.isChecked(),
                "include_tissue_nearest_fields": self.nearest_fields.isChecked(),
                "export_float_dtype": self.float_dtype.currentData(),
                "export_index_dtype": self.index_dtype.currentData(),
                "vessel_resolution": self.vessel_resolution.value(),
            }
        )
        sweeps = []
        for record in self.sweep_rows:
            enabled = record["enabled"]
            combo = record["combo"]
            values = record["values"]
            unit = record["unit"]
            path = str(combo.currentData() or "")
            sweeps.append(
                {
                    "enabled": enabled.isChecked(),
                    "path": path,
                    "unit": unit.currentText(),
                    "values": [
                        self._sweep_value_to_base(path, value, unit.currentText())
                        for value in parse_sweep_values(values.text() if values else "")
                    ],
                }
            )
        config.setdefault("gui", {})["sweeps"] = sweeps


__all__ = ("OutputsPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, CompactStack, labeled, row_of
