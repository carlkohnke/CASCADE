"""Vessel architecture configuration page.

Users can select generated forests, loaded networks, lattices, or simple custom
graphs and configure the construction or growth inputs required by each mode.
"""

from __future__ import annotations

import math
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QLineEdit,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.model import (
    format_points,
    parse_points,
)
from cascade.gui.widgets import (
    Banner,
    FocusPlainTextEdit,
    PathPicker,
)
from cascade.gui.ui_helpers import (
    _combo,
    _double,
    _set_combo,
    _spin,
)

from cascade.gui.pages.base import (
    Page,
)


class _CurrentPageStack(QStackedWidget):
    """A stack whose height follows the visible source panel, not the largest one."""

    def sizeHint(self):
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self):
        current = self.currentWidget()
        return current.minimumSizeHint() if current is not None else super().minimumSizeHint()

    def setCurrentIndex(self, index):
        super().setCurrentIndex(index)
        self.updateGeometry()


class VesselsPage(Page):
    roots_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(
            "Vessel network",
            "",
            parent,
        )
        card = Card("Network source")
        self.source = _combo(
            [
                ("Generate with svVascularize", "svv_generated"),
                ("Upload saved svVascularize network", "uploaded"),
                ("Generate lattice network", "lattice"),
                ("Simple channel geometry", "simple"),
                ("Import custom CSV / NPZ network", "custom"),
            ]
        )
        card.add(labeled("How should vessels be created?", self.source, important=True))
        self.stack = CompactStack()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stack.addWidget(self._generated())
        self.stack.addWidget(self._uploaded())
        self.stack.addWidget(self._lattice())
        self.stack.addWidget(self._simple())
        self.stack.addWidget(self._custom())
        self.stack.sync_height()
        card.add(self.stack)
        self.column.addWidget(card)
        export = Card(
            "Network export",
            "Preserve the exact generated, lattice, channel, or imported network with each completed run.",
        )
        self.save_network = QCheckBox("Save a reusable network file with each run")
        self.network_save_path = PathPicker(
            mode="save",
            caption="Choose network export file",
            file_filter="CASCADE network (*.npz *.forest);;All files (*)",
        )
        export.add(self.save_network)
        export.add(
            labeled(
                "Network file (optional)",
                self.network_save_path,
                "Leave blank to save beside the run results with an automatic name.",
            )
        )
        self.save_network.toggled.connect(self.network_save_path.setEnabled)
        self.column.addWidget(export)
        self.network_note = Banner()
        self.column.addWidget(self.network_note)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.finish()

    def _generated(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.topology = _combo(
            [("One tree", "tree"), ("Forest / multiple roots", "forest")]
        )
        self.inlet_count = _spin(1, 1, 64)
        self.inlet_count.setEnabled(False)
        self.terminals = _spin(100, 2, 10_000_000)
        self.root_rows = []
        self.root_rows_panel = QWidget()
        self.root_rows_layout = QVBoxLayout(self.root_rows_panel)
        self.root_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.root_rows_layout.setSpacing(0)
        self.auto_roots = QCheckBox("Place inlets on the domain boundary automatically")
        self.auto_roots.setStyleSheet("padding-left: 8px;")
        self.auto_roots.setChecked(True)
        self.auto_roots.toggled.connect(self._root_mode_changed)
        self.topology.currentIndexChanged.connect(self._topology_changed)
        self.inlet_count.valueChanged.connect(self._sync_root_rows)
        self._root_domain = {"type": "cube", "side_length": 1.0}
        layout.addWidget(
            row_of(
                labeled("Topology", self.topology),
                labeled("Number of trees / inlets", self.inlet_count),
                labeled("Final terminal vessels per tree", self.terminals, important=True),
            )
        )
        layout.addWidget(self.auto_roots)
        layout.addWidget(self.root_rows_panel)
        self._sync_root_rows(1)
        self._set_root_values(
            [{"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]}]
        )
        return panel

    def _topology_changed(self, *_):
        forest = self.topology.currentData() == "forest"
        if not forest:
            self.inlet_count.setValue(1)
        self.inlet_count.setEnabled(forest and self.auto_roots.isChecked())
        self._sync_root_rows(self.inlet_count.value() if forest else 1)
        self._root_mode_changed()
        if hasattr(self, "stack"):
            self.stack.sync_height()

    def _root_mode_changed(self, *_):
        automatic = self.auto_roots.isChecked()
        for record in self.root_rows:
            record["proximal"].setDisabled(automatic)
            record["distal"].setDisabled(automatic)
        self.inlet_count.setEnabled(
            automatic and self.topology.currentData() == "forest"
        )

    def configured_inlet_count(self) -> int:
        if self.source.currentData() != "svv_generated":
            return 1
        if self.topology.currentData() == "tree":
            return 1
        if self.auto_roots.isChecked():
            return self.inlet_count.value()
        return max(1, len(self.root_rows))

    def _sync_root_rows(self, count):
        count = max(1, int(count))
        defaults = self._automatic_roots(self._root_domain, count)
        while len(self.root_rows) > count:
            record = self.root_rows.pop()
            record["widget"].setParent(None)
            record["widget"].deleteLater()
        while len(self.root_rows) < count:
            index = len(self.root_rows) + 1
            proximal = QLineEdit()
            distal = QLineEdit()
            proximal.setPlaceholderText("x, y, z")
            distal.setPlaceholderText("x, y, z")
            row = row_of(
                labeled(f"Inlet {index} proximal end (cm)", proximal, important=True),
                labeled(f"Inlet {index} distal end (cm)", distal, important=True),
            )
            proximal.textChanged.connect(self.roots_changed)
            distal.textChanged.connect(self.roots_changed)
            self.root_rows_layout.addWidget(row)
            self.root_rows.append(
                {"widget": row, "proximal": proximal, "distal": distal}
            )
        if self.auto_roots.isChecked():
            for record, root in zip(self.root_rows, defaults):
                start = root["start"]
                direction = root["direction"]
                distal = [start[axis] + direction[axis] for axis in range(3)]
                for field in (record["proximal"], record["distal"]):
                    field.blockSignals(True)
                record["proximal"].setText(
                    ", ".join(format(value, ".8g") for value in start)
                )
                record["distal"].setText(
                    ", ".join(format(value, ".8g") for value in distal)
                )
                for field in (record["proximal"], record["distal"]):
                    field.blockSignals(False)
        self._root_mode_changed()

    @staticmethod
    def _parse_point(text, label):
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
        if len(values) != 3:
            raise ValueError(f"{label} requires three comma-separated coordinates.")
        return values

    def _roots_from_fields(self):
        roots = []
        for index, record in enumerate(self.root_rows, start=1):
            start = self._parse_point(record["proximal"].text(), f"Inlet {index} proximal end")
            distal = self._parse_point(record["distal"].text(), f"Inlet {index} distal end")
            direction = [distal[axis] - start[axis] for axis in range(3)]
            if math.sqrt(sum(value * value for value in direction)) <= 1.0e-12:
                raise ValueError(f"Inlet {index} proximal and distal ends must differ.")
            roots.append({"start": start, "direction": direction})
        return roots

    def _set_root_values(self, roots):
        roots = list(roots or [])
        self._sync_root_rows(max(1, len(roots)))
        for record, root in zip(self.root_rows, roots):
            start = [float(value) for value in root.get("start", [0.0, 0.0, 0.0])]
            direction = [float(value) for value in root.get("direction", [1.0, 0.0, 0.0])]
            distal = [start[axis] + direction[axis] for axis in range(3)]
            record["proximal"].setText(", ".join(format(value, ".8g") for value in start))
            record["distal"].setText(", ".join(format(value, ".8g") for value in distal))

    @staticmethod
    def _automatic_roots(domain, count=1):
        kind = str(domain.get("type", "cube")).lower()
        center = [float(value) for value in domain.get("center", [0.0, 0.0, 0.0])]
        directions = []
        count = max(1, int(count))
        if count == 1:
            directions = [[1.0, -1.0, -1.0]]
        elif count == 2:
            directions = [[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]]
        else:
            golden = math.pi * (3.0 - math.sqrt(5.0))
            for index in range(count):
                y = 1.0 - 2.0 * (index + 0.5) / count
                radial = math.sqrt(max(0.0, 1.0 - y * y))
                angle = golden * index
                directions.append(
                    [radial * math.cos(angle), y, radial * math.sin(angle)]
                )

        roots = []
        side = float(domain.get("side_length", 1.0))
        dims = [
            float(domain.get("x_length", side)),
            float(domain.get("y_length", side)),
            float(domain.get("z_length", side)),
        ]
        for raw in directions:
            norm = math.sqrt(sum(value * value for value in raw)) or 1.0
            unit = [value / norm for value in raw]
            if kind == "sphere":
                scale = float(domain.get("radius", 0.5)) * 0.98
            elif kind in {"cylinder", "disk"}:
                radius = float(domain.get("radius", 0.5))
                height = float(domain.get("height", domain.get("z_length", 1.0)))
                radial = math.sqrt(unit[0] * unit[0] + unit[1] * unit[1])
                limits = []
                if radial > 1e-12:
                    limits.append(radius / radial)
                if abs(unit[2]) > 1e-12:
                    limits.append((0.5 * height) / abs(unit[2]))
                # Cylinders are reconstructed as an implicit surface from a
                # faceted mesh. Keep automatic roots comfortably inside that
                # approximation rather than relying on a near-boundary value.
                scale = min(limits) * 0.90
            else:
                scale = min(
                    0.49 * dims[index] / abs(unit[index])
                    for index in range(3)
                    if abs(unit[index]) > 1e-12
                )
            start = [center[index] + scale * unit[index] for index in range(3)]
            direction = [center[index] - start[index] for index in range(3)]
            roots.append({"start": start, "direction": direction})
        return roots

    def _uploaded(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.upload_kind = _combo([("Tree", "tree"), ("Forest", "forest")])
        self.network_path = PathPicker(
            caption="Choose saved vascular network",
            file_filter="svVascularize network (*.tree.npz *.forest *.simcache *.npz);;All files (*)",
        )
        self.extend_uploaded = QCheckBox("Continue growing this network")
        self.upload_target = _spin(100, 1, 10_000_000)
        layout.addWidget(labeled("Saved object type", self.upload_kind))
        layout.addWidget(labeled("Network file", self.network_path, important=True))
        layout.addWidget(self.extend_uploaded)
        layout.addWidget(labeled("Target terminals after growth", self.upload_target))
        self.extend_uploaded.toggled.connect(self.upload_target.setEnabled)
        return panel

    def _lattice(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.lattice_type = _combo(
            [
                ("Simple cubic", "cubic"),
                ("Tetrahedral / diamond cubic", "diamond"),
                ("Body-centered cubic", "bcc"),
                ("Octet", "octet"),
            ]
        )
        self.lattice_sizing = _combo(
            [
                ("Cells along X", "cells"),
                ("Unit-cell spacing along X", "spacing"),
            ]
        )
        self.lattice_cells = _spin(4, 1, 500)
        self.lattice_spacing = _double(0.25, 1e-6, 1e6, 6, 0.01)
        self._lattice_domain_x_cm = 1.0
        self.lattice_anisotropy_yx = _double(1.0, 0.001, 1000.0, 4, 0.1)
        self.lattice_anisotropy_zx = _double(1.0, 0.001, 1000.0, 4, 0.1)
        self.lattice_radius = _double(0.0005, 1e-8, 10.0, 8, 0.0001)
        self.radius_expression = QLineEdit()
        self.radius_expression.setPlaceholderText(
            "Optional, e.g. abs(r0 * (1 + 0.25*x/L)) + 1e-8"
        )
        self.subdivisions = _spin(1, 1, 100)
        self.inlet_points = FocusPlainTextEdit()
        self.inlet_points.setMaximumHeight(74)
        self.inlet_points.setPlaceholderText("One x, y, z point per line")
        self.outlet_points = FocusPlainTextEdit()
        self.outlet_points.setMaximumHeight(74)
        self.outlet_points.setPlaceholderText("One x, y, z point per line")
        layout.addWidget(
            row_of(
                labeled("Lattice family", self.lattice_type, important=True),
                labeled("Lattice sizing", self.lattice_sizing, important=True),
            )
        )
        layout.addWidget(
            row_of(
                labeled(
                    "Cells along X",
                    self.lattice_cells,
                    "Defines the X unit-cell size from the domain width.",
                    important=True,
                ),
                labeled(
                    "X unit-cell spacing (cm)",
                    self.lattice_spacing,
                    "Used when spacing-based sizing is selected.",
                    important=True,
                ),
            )
        )
        layout.addWidget(
            row_of(
                labeled(
                    "Y:X anisotropy",
                    self.lattice_anisotropy_yx,
                    "Above 1 stretches unit cells along Y; below 1 compresses them.",
                ),
                labeled(
                    "Z:X anisotropy",
                    self.lattice_anisotropy_zx,
                    "Above 1 stretches unit cells along Z; below 1 compresses them.",
                ),
            )
        )
        layout.addWidget(
            row_of(
                labeled("Base vessel radius (cm)", self.lattice_radius, important=True),
                labeled("Subsegments per strut", self.subdivisions),
            )
        )
        layout.addWidget(
            labeled(
                "Spatial radius law",
                self.radius_expression,
                "x, y, z = segment-midpoint coordinates (cm); r0 = Base vessel radius above (cm); "
                "L = longest domain dimension (cm). abs(...) is supported. Because abs(0) is still zero, "
                "add a small floor such as + 1e-8 when an expression may cross zero. Both ^ and ** mean exponentiation.",
            )
        )
        layout.addWidget(
            row_of(
                labeled(
                    "Inlet locations (cm)",
                    self.inlet_points,
                    "Blank uses the minimum corner; points snap to nearest nodes.",
                ),
                labeled(
                    "Outlet locations (cm)",
                    self.outlet_points,
                    "Blank uses the maximum corner; multiple lines are allowed.",
                ),
            )
        )
        self.lattice_sizing.currentIndexChanged.connect(self._lattice_sizing_changed)
        self.lattice_cells.valueChanged.connect(self._lattice_cells_changed)
        self._lattice_sizing_changed()
        return panel

    def _lattice_sizing_changed(self, *_):
        use_cells = self.lattice_sizing.currentData() == "cells"
        self.lattice_cells.setEnabled(use_cells)
        self.lattice_spacing.setEnabled(not use_cells)

    def _lattice_cells_changed(self, cells):
        if self.lattice_sizing.currentData() == "cells":
            self.lattice_spacing.setValue(
                self._lattice_domain_x_cm / max(int(cells), 1)
            )

    def _simple(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.simple_mode = _combo(
            [
                ("Single straight channel", "onechannel"),
                ("Parallel channels", "multichannel"),
                ("Serpentine channel", "snake"),
            ]
        )
        self.simple_radius = _double(0.015, 1e-8, 10.0, 6)
        self.simple_axis = _combo([("X axis", "x"), ("Y axis", "y")])
        self.simple_offsets = QLineEdit("-0.2, 0, 0.2")
        layout.addWidget(
            row_of(
                labeled("Geometry", self.simple_mode),
                labeled("Radius (cm)", self.simple_radius, important=True),
            )
        )
        layout.addWidget(
            row_of(
                labeled("Flow axis", self.simple_axis),
                labeled("Parallel offsets (cm)", self.simple_offsets),
            )
        )
        self.snake_arc_segments = _spin(5, 2, 100)
        self.snake_straight_segments = _spin(5, 1, 100)
        layout.addWidget(
            row_of(
                labeled("Segments per bend", self.snake_arc_segments),
                labeled("Segments per straight run", self.snake_straight_segments),
            )
        )
        return panel

    def _custom(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.custom_path = PathPicker(
            caption="Choose vessel network",
            file_filter="Vessel graph (*.csv *.npz);;All files (*)",
        )
        self.custom_radius = _double(0.015, 1e-8, 10.0, 6)
        layout.addWidget(
            labeled(
                "Network file",
                self.custom_path,
                "CSV columns: start_x, start_y, start_z, end_x, end_y, end_z; radius_cm and node IDs are optional. NPZ files use starts and ends arrays.",
                important=True,
            )
        )
        layout.addWidget(
            labeled(
                "Default radius (cm)",
                self.custom_radius,
                "Used only when the imported file does not provide radii.",
            )
        )
        return panel

    def _source_changed(self, index):
        self.stack.setCurrentIndex(index)
        messages = {
            "svv_generated": (
                "The reusable SVV seed builds automatically and continues during the full run.",
                "info",
            ),
            "uploaded": (
                "The uploaded network will be validated before simulation.",
                "success",
            ),
            "lattice": (
                "Compatible network solvers are selected automatically.",
                "warning",
            ),
            "simple": (
                "Channel geometry is ready for physical setup.",
                "info",
            ),
            "custom": (
                "The imported graph is previewed and simulated with its original segment geometry.",
                "success",
            ),
        }
        self.network_note.set_message(*messages[self.source.currentData()])
        self.custom_radius.parentWidget().setVisible(False)

    def load(self, config):
        network = config.get("network", {})
        gui = config.get("gui", {})
        self._root_domain = config.get("domain", {})
        source = gui.get("network_source")
        simple = network.get("simple", {})
        if not source:
            source = (
                "custom"
                if simple.get("mode") == "custom"
                else "lattice"
                if simple.get("mode") == "lattice"
                else "simple"
                if network.get("mode") == "simple"
                else "uploaded"
                if network.get("input_path")
                else "svv_generated"
            )
        _set_combo(self.source, source)
        self._source_changed(self.source.currentIndex())
        _set_combo(self.topology, network.get("mode", "tree"))
        self.auto_roots.setChecked(bool(gui.get("auto_svv_roots", True)))
        roots = network.get("roots") or (
            [network["root"]] if network.get("root") else []
        )
        self.inlet_count.setValue(max(2, len(roots)))
        self._topology_changed()
        self.terminals.setValue(int(network.get("target_terminal_count") or 100))
        if not roots:
            roots = self._automatic_roots(config.get("domain", {}), self.inlet_count.value())
        self._set_root_values(roots)
        _set_combo(self.upload_kind, network.get("mode", "tree"))
        self.network_path.setText(str(network.get("input_path") or ""))
        self.extend_uploaded.setChecked(
            bool(
                config.get("growth", {}).get("enabled", False)
                and network.get("input_path")
            )
        )
        self.upload_target.setValue(int(network.get("target_terminal_count") or 100))
        self.upload_target.setEnabled(self.extend_uploaded.isChecked())
        _set_combo(self.lattice_type, simple.get("lattice_type", "cubic"))
        sizing_mode = simple.get("sizing_mode")
        if not sizing_mode:
            sizing_mode = (
                "spacing" if simple.get("cell_spacing_cm") is not None else "cells"
            )
        _set_combo(self.lattice_sizing, sizing_mode)
        try:
            from cascade.gui.visualization.geometry import _domain_dimensions

            dims, _center = _domain_dimensions(config.get("domain", {}))
            self._lattice_domain_x_cm = float(dims[0])
        except Exception:
            self._lattice_domain_x_cm = 1.0
        lattice_cells = int(simple.get("cells", 4))
        self.lattice_cells.setValue(lattice_cells)
        spacing = simple.get("cell_spacing_cm")
        if spacing is None:
            spacing = self._lattice_domain_x_cm / max(lattice_cells, 1)
        self.lattice_spacing.setValue(float(spacing))
        self.lattice_anisotropy_yx.setValue(float(simple.get("anisotropy_yx", 1.0)))
        self.lattice_anisotropy_zx.setValue(float(simple.get("anisotropy_zx", 1.0)))
        self._lattice_sizing_changed()
        self.lattice_radius.setValue(float(simple.get("radius_cm", 0.0005)))
        self.radius_expression.setText(str(simple.get("radius_expression") or ""))
        self.subdivisions.setValue(int(simple.get("subdivisions", 1)))
        self.inlet_points.setPlainText(format_points(simple.get("inlet_points_cm")))
        self.outlet_points.setPlainText(format_points(simple.get("outlet_points_cm")))
        _set_combo(
            self.simple_mode,
            "onechannel" if simple.get("mode") == "custom" else simple.get("mode", "onechannel"),
        )
        self.custom_path.setText(
            str(simple.get("path", simple.get("geometry_path", "")) or "")
        )
        self.custom_radius.setValue(float(simple.get("radius_cm", 0.015)))
        self.simple_radius.setValue(float(simple.get("radius_cm", 0.015)))
        _set_combo(self.simple_axis, simple.get("axis", "x"))
        self.simple_offsets.setText(
            ", ".join(str(v) for v in simple.get("y_offsets_cm", [-0.2, 0, 0.2]))
        )
        self.snake_arc_segments.setValue(int(simple.get("snake_arc_segments", 5)))
        self.snake_straight_segments.setValue(
            int(simple.get("snake_straight_segments", 5))
        )
        self.save_network.setChecked(
            bool(config.get("outputs", {}).get("save_network", True))
        )
        self.network_save_path.setText(str(network.get("save_path") or ""))
        self.network_save_path.setEnabled(self.save_network.isChecked())

    def write(self, config):
        source = self.source.currentData()
        config.setdefault("gui", {})["network_source"] = source
        growth = config.setdefault("growth", {})
        if source == "svv_generated":
            auto_roots = self.auto_roots.isChecked()
            count = (
                self.inlet_count.value()
                if self.topology.currentData() == "forest"
                else 1
            )
            roots = (
                self._automatic_roots(config.get("domain", {}), count)
                if auto_roots
                else self._roots_from_fields()
            )
            config.setdefault("gui", {})["auto_svv_roots"] = auto_roots
            config["network"] = {
                "mode": self.topology.currentData(),
                "target_terminal_count": self.terminals.value(),
                "roots": roots,
            }
            growth["enabled"] = True
            growth["n_equal_bifurcations"] = None
        elif source == "uploaded":
            network = {
                "mode": self.upload_kind.currentData(),
                "input_path": self.network_path.text(),
            }
            if self.extend_uploaded.isChecked():
                network["target_terminal_count"] = self.upload_target.value()
            config["network"] = network
            growth["enabled"] = self.extend_uploaded.isChecked()
        elif source == "lattice":
            inlet = parse_points(self.inlet_points.toPlainText())
            outlet = parse_points(self.outlet_points.toPlainText())
            simple = {
                "mode": "lattice",
                "lattice_type": self.lattice_type.currentData(),
                "sizing_mode": self.lattice_sizing.currentData(),
                "cells": self.lattice_cells.value(),
                "cell_spacing_cm": self.lattice_spacing.value(),
                "anisotropy_yx": self.lattice_anisotropy_yx.value(),
                "anisotropy_zx": self.lattice_anisotropy_zx.value(),
                "radius_cm": self.lattice_radius.value(),
                "radius_expression": self.radius_expression.text().strip(),
                "subdivisions": self.subdivisions.value(),
            }
            if inlet:
                simple["inlet_points_cm"] = inlet
            if outlet:
                simple["outlet_points_cm"] = outlet
            config["network"] = {
                "mode": "simple",
                "target_terminal_count": 1,
                "simple": simple,
            }
            growth["enabled"] = False
            config.setdefault("settings", {}).setdefault("hemodynamics", {})[
                "kirchhoff_solver"
            ] = "spsolve"
            config.setdefault("simulation", {})["concentration_solver"] = "network_ext"
        elif source == "custom":
            config["network"] = {
                "mode": "simple",
                "target_terminal_count": 1,
                "simple": {
                    "mode": "custom",
                    "path": self.custom_path.text(),
                    "radius_cm": self.custom_radius.value(),
                },
            }
            growth["enabled"] = False
            config.setdefault("settings", {}).setdefault("hemodynamics", {})[
                "kirchhoff_solver"
            ] = "spsolve"
            config.setdefault("simulation", {})["concentration_solver"] = "network_ext"
        else:
            offsets = [
                float(v.strip())
                for v in self.simple_offsets.text().split(",")
                if v.strip()
            ]
            simple_mode = self.simple_mode.currentData()
            simple_config = {
                "mode": simple_mode,
                "radius_cm": self.simple_radius.value(),
                "axis": self.simple_axis.currentData(),
                "y_offsets_cm": offsets,
                "snake_arc_segments": self.snake_arc_segments.value(),
                "snake_straight_segments": self.snake_straight_segments.value(),
            }
            config["network"] = {
                "mode": "simple",
                "target_terminal_count": 1,
                "simple": simple_config,
            }
            growth["enabled"] = False
        save_path = self.network_save_path.text()
        if save_path:
            config["network"]["save_path"] = save_path
        else:
            config["network"].pop("save_path", None)
        config.setdefault("outputs", {})["save_network"] = self.save_network.isChecked()

    @staticmethod
    def _parse_roots(text):
        roots = []
        for line in text.splitlines():
            if not line.strip():
                continue
            sides = line.replace("→", "->").split("->")
            start = [float(v.strip()) for v in sides[0].split(",")]
            if len(start) != 3:
                raise ValueError("Each vessel root needs three start coordinates.")
            root = {"start": start}
            if len(sides) > 1 and sides[1].strip():
                direction = [float(v.strip()) for v in sides[1].split(",")]
                if len(direction) != 3:
                    raise ValueError("Each root direction needs three coordinates.")
                root["direction"] = direction
            roots.append(root)
        return roots or [
            {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]}
        ]

    @staticmethod
    def _format_roots(roots):
        return "\n".join(
            ", ".join(str(v) for v in r.get("start", []))
            + (
                " → " + ", ".join(str(v) for v in r.get("direction", []))
                if r.get("direction")
                else ""
            )
            for r in roots
        )


__all__ = ("VesselsPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, CompactStack, labeled, row_of
