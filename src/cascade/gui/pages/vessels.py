"""Vessel architecture configuration page.

Users can select generated forests, loaded networks, lattices, or simple custom
graphs and configure the construction or growth inputs required by each mode.
"""

from __future__ import annotations

import math
from PySide6.QtCore import (
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
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
    Card,
    FocusPlainTextEdit,
    PathPicker,
    labeled,
    row_of,
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


class VesselsPage(Page):
    open_physics_requested = Signal()

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
            ]
        )
        card.add(labeled("How should vessels be created?", self.source, important=True))
        self.stack = QStackedWidget()
        self.stack.addWidget(self._generated())
        self.stack.addWidget(self._uploaded())
        self.stack.addWidget(self._lattice())
        self.stack.addWidget(self._simple())
        card.add(self.stack)
        self.column.addWidget(card)
        self.network_note = Banner()
        self.column.addWidget(self.network_note)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.finish()

    def _generated(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        self.topology = _combo(
            [("One tree", "tree"), ("Forest / multiple roots", "forest")]
        )
        self.inlet_count = _spin(2, 2, 64)
        self.inlet_count.setEnabled(False)
        self.terminals = _spin(100, 1, 10_000_000)
        self.roots = FocusPlainTextEdit()
        self.roots.setMaximumHeight(92)
        self.roots.setPlaceholderText("One root per line: x, y, z → dx, dy, dz")
        self.roots.setPlainText("0.49, -0.49, -0.49 → -0.49, 0.49, 0.49")
        self.auto_roots = QCheckBox("Place inlets on the domain boundary automatically")
        self.auto_roots.setChecked(True)
        self.roots.setEnabled(False)
        self.auto_roots.toggled.connect(self._root_mode_changed)
        self.topology.currentIndexChanged.connect(self._topology_changed)
        layout.addWidget(
            row_of(
                labeled("Topology", self.topology),
                labeled("Number of trees / inlets", self.inlet_count),
                labeled("Terminal vessels per tree", self.terminals, important=True),
            )
        )
        layout.addWidget(self.auto_roots)
        layout.addWidget(
            labeled(
                "Root locations and directions (cm)",
                self.roots,
                "For a forest, enter one root per line. The arrow may be written as '->' or '→'.",
            )
        )
        radius_box = QFrame()
        radius_box.setObjectName("radiusSizing")
        radius_layout = QVBoxLayout(radius_box)
        radius_layout.setContentsMargins(12, 10, 12, 10)
        radius_layout.setSpacing(7)
        radius_title = QLabel("Hydraulic radius sizing")
        radius_title.setObjectName("cardTitle")
        self.radius_summary = QLabel("radii pending flow and pressure")
        self.radius_summary.setObjectName("fieldHelp")
        self.radius_summary.setWordWrap(True)
        self.radius_physics_btn = QPushButton("Set flow and pressure")
        self.radius_physics_btn.setProperty("secondary", True)
        self.radius_physics_btn.clicked.connect(self.open_physics_requested)
        radius_layout.addWidget(radius_title)
        radius_layout.addWidget(self.radius_summary)
        radius_layout.addWidget(self.radius_physics_btn, 0, Qt.AlignLeft)
        layout.addWidget(radius_box)
        return panel

    def _topology_changed(self, *_):
        forest = self.topology.currentData() == "forest"
        if not forest:
            self.inlet_count.setValue(2)
        self.inlet_count.setEnabled(forest and self.auto_roots.isChecked())
        self._root_mode_changed()

    def _root_mode_changed(self, *_):
        automatic = self.auto_roots.isChecked()
        self.roots.setDisabled(automatic)
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
        return max(
            1,
            len(
                [line for line in self.roots.toPlainText().splitlines() if line.strip()]
            ),
        )

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

    def set_radius_summary(self, response=None, *, calculating=False):
        if calculating:
            self.radius_summary.setText("calculating radii…")
            return
        radii = (response or {}).get("root_radii_cm", [])
        if radii:
            rendered = ", ".join(f"{float(radius) * 10_000:.2f} µm" for radius in radii)
            self.radius_summary.setText(f"r_in  {rendered}")
        else:
            self.radius_summary.setText("radii pending flow and pressure")

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
        self.lattice_cells = _spin(4, 1, 500)
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
                labeled("Cells per axis", self.lattice_cells, important=True),
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
        return panel

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
        }
        self.network_note.set_message(*messages[self.source.currentData()])

    def load(self, config):
        network = config.get("network", {})
        gui = config.get("gui", {})
        source = gui.get("network_source")
        simple = network.get("simple", {})
        if not source:
            source = (
                "lattice"
                if simple.get("mode") == "lattice"
                else (
                    "simple"
                    if network.get("mode") == "simple"
                    else ("uploaded" if network.get("input_path") else "svv_generated")
                )
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
        self.roots.setPlainText(self._format_roots(roots))
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
        self.lattice_cells.setValue(int(simple.get("cells", 4)))
        self.lattice_radius.setValue(float(simple.get("radius_cm", 0.0005)))
        self.radius_expression.setText(str(simple.get("radius_expression") or ""))
        self.subdivisions.setValue(int(simple.get("subdivisions", 1)))
        self.inlet_points.setPlainText(format_points(simple.get("inlet_points_cm")))
        self.outlet_points.setPlainText(format_points(simple.get("outlet_points_cm")))
        _set_combo(self.simple_mode, simple.get("mode", "onechannel"))
        self.simple_radius.setValue(float(simple.get("radius_cm", 0.015)))
        _set_combo(self.simple_axis, simple.get("axis", "x"))
        self.simple_offsets.setText(
            ", ".join(str(v) for v in simple.get("y_offsets_cm", [-0.2, 0, 0.2]))
        )
        self.snake_arc_segments.setValue(int(simple.get("snake_arc_segments", 5)))
        self.snake_straight_segments.setValue(
            int(simple.get("snake_straight_segments", 5))
        )

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
                else self._parse_roots(self.roots.toPlainText())
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
                "cells": self.lattice_cells.value(),
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
        else:
            offsets = [
                float(v.strip())
                for v in self.simple_offsets.text().split(",")
                if v.strip()
            ]
            config["network"] = {
                "mode": "simple",
                "target_terminal_count": 1,
                "simple": {
                    "mode": self.simple_mode.currentData(),
                    "radius_cm": self.simple_radius.value(),
                    "axis": self.simple_axis.currentData(),
                    "y_offsets_cm": offsets,
                    "snake_arc_segments": self.snake_arc_segments.value(),
                    "snake_straight_segments": self.snake_straight_segments.value(),
                },
            }
            growth["enabled"] = False

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
