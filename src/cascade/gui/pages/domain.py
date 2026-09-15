"""Domain configuration page."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.widgets import (
    Card,
    PathPicker,
    labeled,
    row_of,
)
from pathlib import Path
from typing import Any
from cascade.gui.ui_helpers import (
    _combo,
    _double,
    _set_combo,
    _spin,
)

from cascade.gui.pages.base import (
    Page,
)


class DomainPage(Page):
    def __init__(self, parent=None):
        super().__init__(
            "Domain",
            "",
            parent,
        )
        card = Card(
            "Tissue volume",
            "The domain is centered at the origin unless an uploaded mesh defines otherwise.",
        )
        self.kind = _combo(
            [
                ("Box", "box"),
                ("Sphere", "sphere"),
                ("Cylinder / disk", "cylinder"),
                ("Biventricular heart", "bivent3"),
                ("Upload mesh / .dmn", "file"),
            ]
        )
        card.add(labeled("Domain source", self.kind, important=True))
        self.stack = CompactStack()
        self.box_x = _double(1.0, 1e-6, 1e6)
        self.box_y = _double(1.0, 1e-6, 1e6)
        self.box_z = _double(1.0, 1e-6, 1e6)
        box_panel = QWidget()
        box_layout = QVBoxLayout(box_panel)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.addWidget(
            row_of(
                labeled("X length (cm)", self.box_x),
                labeled("Y length (cm)", self.box_y),
                labeled("Z length (cm)", self.box_z),
            )
        )
        self.stack.addWidget(box_panel)
        self.sphere_radius = _double(0.5, 1e-6, 1e6)
        self.sphere_detail = _combo(
            [
                ("Coarse (recommended)", "12x8"),
                ("Fine", "64x64"),
            ]
        )
        sphere_panel = QWidget()
        sphere_layout = QVBoxLayout(sphere_panel)
        sphere_layout.setContentsMargins(0, 0, 0, 0)
        sphere_layout.addWidget(labeled("Radius (cm)", self.sphere_radius))
        sphere_layout.addWidget(
            labeled(
                "Surface detail retention",
                self.sphere_detail,
                "Coarse is the default and is about 55× smaller than Fine, which matches the reference sphere resolution.",
            )
        )
        self.stack.addWidget(sphere_panel)
        self.cylinder_radius = _double(0.5, 1e-6, 1e6)
        self.cylinder_height = _double(1.0, 1e-6, 1e6)
        cylinder_panel = QWidget()
        cylinder_layout = QVBoxLayout(cylinder_panel)
        cylinder_layout.setContentsMargins(0, 0, 0, 0)
        cylinder_layout.addWidget(
            row_of(
                labeled("Radius (cm)", self.cylinder_radius),
                labeled("Height (cm)", self.cylinder_height),
            )
        )
        self.stack.addWidget(cylinder_panel)
        heart_panel = QWidget()
        heart_layout = QVBoxLayout(heart_panel)
        heart_layout.setContentsMargins(0, 0, 0, 0)
        heart_description = QLabel(
            "Use the packaged bivent3 STL heart surface. CASCADE resolves the "
            "mesh from the installed package, so saved projects remain portable."
        )
        heart_description.setWordWrap(True)
        heart_description.setMinimumWidth(0)
        heart_layout.addWidget(heart_description)
        self.stack.addWidget(heart_panel)
        self.domain_path = PathPicker(
            caption="Choose tissue domain",
            file_filter="Domain and mesh (*.dmn *.stl *.vtk *.vtp *.vtu *.ply *.obj);;All files (*)",
        )
        self.stack.addWidget(
            labeled(
                "Domain file",
                self.domain_path,
                "Use a watertight surface/volume mesh or a verified CASCADE .dmn file.",
            )
        )
        self.stack.sync_height()
        card.add(self.stack)
        self.seed = _spin(42, 0, 2_147_483_647)
        card.add(
            labeled(
                "Random seed",
                self.seed,
                "Controls domain sampling and reproducible vessel growth.",
            )
        )
        self.column.addWidget(card)
        self.kind.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.finish()

    def load(self, config):
        domain = config.get("domain", {})
        kind = domain.get("type", domain.get("kind", "cube"))
        path_name = Path(str(domain.get("path") or "")).name.lower()
        visible_kind = (
            "box"
            if kind == "cube"
            else "bivent3"
            if kind == "file" and path_name == "bivent3.stl"
            else kind
        )
        _set_combo(
            self.kind,
            visible_kind
            if visible_kind in {"box", "sphere", "cylinder", "bivent3"}
            else "file",
        )
        self.stack.setCurrentIndex(self.kind.currentIndex())
        self.box_x.setValue(
            float(domain.get("x_length", domain.get("side_length", 1.0)))
        )
        self.box_y.setValue(
            float(domain.get("y_length", domain.get("side_length", 1.0)))
        )
        self.box_z.setValue(
            float(domain.get("z_length", domain.get("side_length", 1.0)))
        )
        self.sphere_radius.setValue(float(domain.get("radius", 0.5)))
        self.cylinder_radius.setValue(float(domain.get("radius", 0.5)))
        self.cylinder_height.setValue(
            float(domain.get("height", domain.get("z_length", 1.0)))
        )
        theta = int(domain.get("theta_resolution", 12))
        phi = int(domain.get("phi_resolution", 8))
        detail = f"{theta}x{phi}"
        if detail not in {"12x8", "64x64"}:
            detail = "12x8"
        _set_combo(self.sphere_detail, detail)
        self.domain_path.setText(str(domain.get("path") or ""))
        self.seed.setValue(int(domain.get("random_seed", 42)))

    def write(self, config):
        kind = self.kind.currentData()
        domain: dict[str, Any] = {"type": kind, "random_seed": self.seed.value()}
        if kind == "box":
            domain.update(
                {
                    "side_length": max(
                        self.box_x.value(), self.box_y.value(), self.box_z.value()
                    ),
                    "x_length": self.box_x.value(),
                    "y_length": self.box_y.value(),
                    "z_length": self.box_z.value(),
                }
            )
        elif kind == "sphere":
            theta, phi = (
                int(value)
                for value in str(self.sphere_detail.currentData() or "12x8").split("x")
            )
            domain.update(
                {
                    "side_length": 2 * self.sphere_radius.value(),
                    "radius": self.sphere_radius.value(),
                    "center": [0.0, 0.0, 0.0],
                    "theta_resolution": theta,
                    "phi_resolution": phi,
                }
            )
        elif kind == "cylinder":
            radius = self.cylinder_radius.value()
            height = self.cylinder_height.value()
            domain.update(
                {
                    "side_length": max(2 * radius, height),
                    "radius": radius,
                    "height": height,
                    "x_length": 2 * radius,
                    "y_length": 2 * radius,
                    "z_length": height,
                    "center": [0.0, 0.0, 0.0],
                }
            )
        elif kind == "bivent3":
            domain.update(
                {
                    "type": "file",
                    "path": "bivent3.stl",
                }
            )
        else:
            domain.update(
                {
                    "type": "file",
                    "side_length": max(
                        self.box_x.value(), self.box_y.value(), self.box_z.value()
                    ),
                    "path": self.domain_path.text(),
                }
            )
        config["domain"] = domain


__all__ = ("DomainPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, CompactStack, labeled, row_of
