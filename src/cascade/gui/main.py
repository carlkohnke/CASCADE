from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from PySide6.QtCore import QEvent, QPointF, QProcess, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QMessageBox as QtMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cascade.settings import SETTINGS_SECTIONS

from .model import (
    ALPHA_MMHG,
    create_jobs,
    default_project,
    estimate_resources,
    flow_from_ul_min,
    flow_to_ul_min,
    format_points,
    hardware_info,
    human_bytes,
    load_project,
    oxygen_from_concentration,
    oxygen_to_concentration,
    parse_points,
    parse_sweep_values,
    pressure_from_pa,
    pressure_to_pa,
    save_project,
    validate_project,
    open_folder,
    open_path,
)
from .runner import JobRunner
from .preview import COLORMAPS, CasePreview, FlowBackdrop
from .theme import Tokens, stylesheet
from .widgets import (
    Banner,
    Card,
    FocusPlainTextEdit,
    NumberInput,
    PathPicker,
    StatusPill,
    Select,
    UnitValue,
    cancel_native_pickers,
    choose_native_path,
    labeled,
    row_of,
    title_label,
)


def _default_project_directory() -> Path:
    configured = os.environ.get("CASCADE_PROJECT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "CASCADE Projects" / "Untitled").resolve()


class QMessageBox:
    """CASCADE-styled replacement for Qt's native message dialogs.

    Native Windows message boxes retain a light title bar even when the
    application is dark.  Using a frameless Qt dialog keeps every alert and
    confirmation visually part of the studio.
    """

    Yes = QtMessageBox.Yes
    No = QtMessageBox.No

    @staticmethod
    def _show(parent, icon, title: str, text: str, buttons=QtMessageBox.Ok, default=None):
        dialog = QDialog(parent)
        dialog.setObjectName("cascadeMessageDialog")
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setMinimumWidth(410)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("dialogHeading")
        heading.setWordWrap(True)
        message = QLabel(text)
        message.setObjectName("dialogMessage")
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(heading)
        layout.addWidget(message)
        button_row = QHBoxLayout()
        button_row.addStretch()
        layout.addLayout(button_row)

        if buttons & QtMessageBox.Yes:
            choices = [("Yes", QtMessageBox.Yes)]
        else:
            choices = []
        if buttons & QtMessageBox.No:
            choices.append(("No", QtMessageBox.No))
        if buttons & QtMessageBox.Ok:
            choices.append(("OK", QtMessageBox.Ok))
        selected = [default or (QtMessageBox.No if buttons & QtMessageBox.No else QtMessageBox.Ok)]
        for label, answer in choices:
            button = QPushButton(label)
            button.setDefault(answer == selected[0])
            button.clicked.connect(
                lambda _checked=False, choice=answer: (selected.__setitem__(0, choice), dialog.accept())
            )
            button_row.addWidget(button)

        accent = {
            QtMessageBox.Warning: "#F6C64E",
            QtMessageBox.Critical: "#FF7C87",
            QtMessageBox.Information: Tokens.SUCCESS,
            QtMessageBox.Question: Tokens.VIOLET,
        }.get(icon, Tokens.SUCCESS)
        dialog.setStyleSheet(
            f"""
            QDialog#cascadeMessageDialog {{ background:{Tokens.SURFACE_1}; color:{Tokens.TEXT};
                border:1px solid {Tokens.BORDER_ACTIVE}; }}
            QLabel#dialogHeading {{ color:{accent}; font-size:15px; font-weight:650; }}
            QLabel#dialogMessage {{ color:{Tokens.TEXT}; min-width:360px; }}
            QDialog#cascadeMessageDialog QPushButton {{ background:{Tokens.SURFACE_2}; color:{Tokens.TEXT};
                border:1px solid {Tokens.BORDER}; border-radius:4px; padding:5px 12px; }}
            QDialog#cascadeMessageDialog QPushButton:hover {{ background:{Tokens.VIOLET}; color:#FFFFFF;
                border-color:{Tokens.VIOLET}; }}
            """
        )
        dialog.exec()
        return selected[0]

    @classmethod
    def information(cls, parent, title: str, text: str):
        return cls._show(parent, QtMessageBox.Information, title, text)

    @classmethod
    def warning(cls, parent, title: str, text: str):
        return cls._show(parent, QtMessageBox.Warning, title, text)

    @classmethod
    def critical(cls, parent, title: str, text: str):
        return cls._show(parent, QtMessageBox.Critical, title, text)

    @classmethod
    def question(cls, parent, title: str, text: str, buttons, default=None):
        return cls._show(parent, QtMessageBox.Question, title, text, buttons, default)


APP_STYLE = stylesheet()


def _combo(items: list[tuple[str, str]] | list[str]) -> QComboBox:
    box = Select()
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(8)
    for item in items:
        if isinstance(item, tuple):
            box.addItem(item[0], item[1])
        else:
            box.addItem(item, item)
    return box


def _spin(value=0, minimum=0, maximum=10_000_000, step=1) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setSingleStep(step)
    widget.setValue(value)
    widget.setGroupSeparatorShown(True)
    return widget


def _double(
    value=0.0, minimum=0.0, maximum=1e12, decimals=7, step=0.01
) -> QDoubleSpinBox:
    widget = NumberInput()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


def _optional_float(text: str) -> float | None:
    """Read an optional scalar range bound; blank or invalid means automatic."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _concentration_unit(value: Any, *, rate: bool = False) -> str:
    """Normalize legacy mmol/L labels to the equivalent SI mol/m³ labels."""
    unit = str(value or "").strip()
    if unit.lower() in {"mmol/l", "mmol/l/s"}:
        return "mol/m³/s" if rate else "mol/m³"
    return unit or ("mol/m³/s" if rate else "mol/m³")


def _queue_action_icon(kind: str) -> QIcon:
    """Small, dependency-free action icons for the queue toolbar."""
    def draw(color_name: str) -> QPixmap:
        pixmap = QPixmap(22, 18)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(color_name)
        painter.setPen(QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(color)

        if kind in {"add", "add_run"}:
            center = 8.0 if kind == "add" else 5.0
            painter.drawLine(
                QPointF(center - 3.5, 9.0), QPointF(center + 3.5, 9.0)
            )
            painter.drawLine(QPointF(center, 5.5), QPointF(center, 12.5))
        if kind == "add_run":
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(
                QPolygonF(
                    (QPointF(12.0, 4.5), QPointF(19.0, 9.0), QPointF(12.0, 13.5))
                )
            )
        elif kind == "play_circle":
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(9.0, 9.0), 7.0, 7.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygonF(
                    (QPointF(7.5, 5.5), QPointF(13.0, 9.0), QPointF(7.5, 12.5))
                )
            )
        elif kind == "fast_forward":
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(
                QPolygonF(
                    (QPointF(2.0, 4.5), QPointF(9.0, 9.0), QPointF(2.0, 13.5))
                )
            )
            painter.drawPolygon(
                QPolygonF(
                    (QPointF(9.0, 4.5), QPointF(16.0, 9.0), QPointF(9.0, 13.5))
                )
            )
        elif kind == "stop":
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(4.0, 4.0, 10.0, 10.0), 1.4, 1.4)

        painter.end()
        return pixmap

    icon = QIcon(draw("#EBECF0"))
    icon.addPixmap(draw("#41444C"), QIcon.Disabled, QIcon.Off)
    return icon


def _workflow_icon(kind: str) -> QIcon:
    """Consistent line icons for the workflow rail."""

    def draw(color_name: str) -> QPixmap:
        pixmap = QPixmap(28, 28)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(1.4, 1.4)
        color = QColor(color_name)
        painter.setPen(QPen(color, 1.45, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        if kind == "project":
            painter.drawRoundedRect(QRectF(4.0, 2.5, 12.0, 15.0), 1.2, 1.2)
            painter.drawLine(QPointF(7.0, 7.0), QPointF(13.0, 7.0))
            painter.drawLine(QPointF(7.0, 10.0), QPointF(13.0, 10.0))
            painter.drawLine(QPointF(7.0, 13.0), QPointF(11.0, 13.0))
        elif kind == "domain":
            front = QPolygonF(
                (QPointF(4.0, 7.0), QPointF(10.0, 3.5), QPointF(16.0, 7.0),
                 QPointF(16.0, 14.0), QPointF(10.0, 17.5), QPointF(4.0, 14.0))
            )
            painter.drawPolygon(front)
            painter.drawLine(QPointF(4.0, 7.0), QPointF(10.0, 10.5))
            painter.drawLine(QPointF(16.0, 7.0), QPointF(10.0, 10.5))
            painter.drawLine(QPointF(10.0, 10.5), QPointF(10.0, 17.5))
        elif kind == "network":
            painter.drawLine(QPointF(5.0, 5.0), QPointF(10.0, 10.0))
            painter.drawLine(QPointF(10.0, 10.0), QPointF(15.0, 5.0))
            painter.drawLine(QPointF(10.0, 10.0), QPointF(15.0, 15.0))
            painter.setBrush(color)
            for point in (QPointF(5.0, 5.0), QPointF(10.0, 10.0), QPointF(15.0, 5.0), QPointF(15.0, 15.0)):
                painter.drawEllipse(point, 1.7, 1.7)
        elif kind == "physics":
            painter.drawEllipse(QRectF(4.0, 4.0, 12.0, 12.0))
            painter.drawLine(QPointF(10.0, 1.8), QPointF(10.0, 4.0))
            painter.drawLine(QPointF(10.0, 16.0), QPointF(10.0, 18.2))
            painter.drawLine(QPointF(1.8, 10.0), QPointF(4.0, 10.0))
            painter.drawLine(QPointF(16.0, 10.0), QPointF(18.2, 10.0))
        elif kind == "solver":
            for y, x in ((5.0, 7.0), (10.0, 13.0), (15.0, 9.0)):
                painter.drawLine(QPointF(3.0, y), QPointF(17.0, y))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x, y), 1.8, 1.8)
                painter.setBrush(Qt.NoBrush)
        elif kind == "outputs":
            painter.drawLine(QPointF(10.0, 3.0), QPointF(10.0, 13.0))
            painter.drawLine(QPointF(6.5, 9.5), QPointF(10.0, 13.0))
            painter.drawLine(QPointF(13.5, 9.5), QPointF(10.0, 13.0))
            painter.drawLine(QPointF(4.0, 16.0), QPointF(16.0, 16.0))
        elif kind == "run":
            painter.drawEllipse(QRectF(2.5, 2.5, 15.0, 15.0))
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(QPolygonF((QPointF(8.0, 6.5), QPointF(14.0, 10.0), QPointF(8.0, 13.5))))
        elif kind == "results":
            painter.drawLine(QPointF(3.0, 17.0), QPointF(17.0, 17.0))
            painter.drawRect(QRectF(4.0, 11.0, 2.5, 6.0))
            painter.drawRect(QRectF(8.75, 7.0, 2.5, 10.0))
            painter.drawRect(QRectF(13.5, 3.0, 2.5, 14.0))

        painter.end()
        return pixmap

    icon = QIcon(draw(Tokens.TEXT_3))
    icon.addPixmap(draw(Tokens.TEXT), QIcon.Selected, QIcon.Off)
    return icon


def _set_combo(box: QComboBox, data: Any) -> None:
    index = box.findData(data)
    if index < 0:
        index = box.findText(str(data))
    if index >= 0:
        box.setCurrentIndex(index)


def _value(mapping: dict[str, Any], section: str, key: str, default: Any) -> Any:
    values = mapping.get("settings", {}).get(section, {})
    return values.get(key, values.get(key.upper(), default))


class Page(QScrollArea):
    changed = Signal()

    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body.setObjectName("page")
        self.column = QVBoxLayout(self.body)
        self.column.setContentsMargins(20, 20, 20, 24)
        self.column.setSpacing(12)
        self.column.addWidget(title_label(title, subtitle))
        self.setWidget(self.body)

    def finish(self):
        self.column.addStretch(1)

    def load(self, config: dict[str, Any]) -> None:
        pass

    def write(self, config: dict[str, Any]) -> None:
        pass


class OverviewPage(Page):
    def __init__(self, hardware, parent=None):
        super().__init__(
            "Project overview",
            "",
            parent,
        )
        intro = Card("Study")
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Perfused construct oxygen sweep")
        intro.add(labeled("Project name", self.name, important=True))
        self.column.addWidget(intro)

        hw = Card("Available compute")
        self.vtk_memory_note = QLabel("VTK export can increase peak RAM.")
        self.vtk_memory_note.setObjectName("muted")
        hw.add(self.vtk_memory_note)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        stats = [
            ("System RAM", human_bytes(hardware.total_ram_bytes)),
            ("Available now", human_bytes(hardware.available_ram_bytes)),
            ("CPU threads", str(hardware.cpu_count)),
            ("GPU", hardware.gpu_name),
            (
                "GPU memory",
                human_bytes(hardware.gpu_memory_bytes)
                if hardware.gpu_memory_bytes
                else "Not detected",
            ),
            ("Platform", hardware.platform_name),
        ]
        for index, (label, value) in enumerate(stats):
            tile = QFrame()
            tile.setObjectName("computeTile")
            lay = QVBoxLayout(tile)
            key = QLabel(label.upper())
            key.setObjectName("eyebrow")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet(
                f"font-family:'JetBrains Mono','Consolas',monospace;font-size:12px;font-weight:650;color:{Tokens.TEXT};"
            )
            lay.addWidget(key)
            lay.addWidget(val)
            grid.addWidget(tile, index // 2, index % 2)
        wrap = QWidget()
        wrap.setLayout(grid)
        hw.add(wrap)
        self.column.addWidget(hw)

        self.finish()

    def load(self, config):
        self.name.setText(str(config.get("gui", {}).get("project_name", "")))
        self.vtk_memory_note.setVisible(
            bool(config.get("outputs", {}).get("write_paraview", True))
        )

    def write(self, config):
        config.setdefault("gui", {})["project_name"] = (
            self.name.text().strip() or "Untitled CASCADE project"
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
                ("Biventricular heart (bivent3)", "bivent3"),
                ("Upload mesh / .dmn", "file"),
            ]
        )
        card.add(labeled("Domain source", self.kind, important=True))
        self.stack = QStackedWidget()
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
                ("Coarse (about 144 surface patches)", "12x8"),
                ("Fine (about 7,936 surface patches)", "64x64"),
            ]
        )
        sphere_panel = QWidget()
        sphere_layout = QVBoxLayout(sphere_panel)
        sphere_layout.setContentsMargins(0, 0, 0, 0)
        sphere_layout.addWidget(labeled("Radius (cm)", self.sphere_radius))
        sphere_layout.addWidget(
            labeled(
                "Surface detail",
                self.sphere_detail,
                "Coarse is the default and is about 55× smaller than Fine, which matches the former sphere resolution.",
            )
        )
        self.stack.addWidget(sphere_panel)
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
            visible_kind if visible_kind in {"box", "sphere", "bivent3"} else "file",
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
        layout.addWidget(
            self.auto_roots
        )
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
        self.radius_summary = QLabel(
            "radii pending flow and pressure"
        )
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
        return max(1, len([line for line in self.roots.toPlainText().splitlines() if line.strip()]))

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
                directions.append([radial * math.cos(angle), y, radial * math.sin(angle)])

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
        roots = network.get("roots") or ([network["root"]] if network.get("root") else [])
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
        self.snake_straight_segments.setValue(int(simple.get("snake_straight_segments", 5)))

    def write(self, config):
        source = self.source.currentData()
        config.setdefault("gui", {})["network_source"] = source
        growth = config.setdefault("growth", {})
        if source == "svv_generated":
            auto_roots = self.auto_roots.isChecked()
            count = self.inlet_count.value() if self.topology.currentData() == "forest" else 1
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


class PhysicsPage(Page):
    def __init__(self, parent=None):
        super().__init__(
            "Physical setup",
            "",
            parent,
        )
        bc = Card("Boundary conditions")
        self.bc_mode = _combo(
            [
                ("Inlet flow + outlet pressure", "flow_pressure"),
                (
                    "Equal outlet flows + pressure reference (legacy)",
                    "legacy_equal_flow",
                ),
                ("Inlet pressure + outlet pressure", "pressure_pressure"),
            ]
        )
        self.flow = UnitValue(
            ["µL/min", "mL/min", "m³/s"], 100.0, minimum=0, maximum=1e12, decimals=7
        )
        self.inlet_pressure = UnitValue(
            ["mmHg", "Pa"], 500.0, minimum=-1e9, maximum=1e9, decimals=5
        )
        self.outlet_pressure = UnitValue(
            ["mmHg", "Pa"], 300.0, minimum=-1e9, maximum=1e9, decimals=5
        )
        bc.add(labeled("Boundary-condition mode", self.bc_mode, important=True))
        bc.add(
            row_of(
                labeled("Shared inlet flow", self.flow, important=True),
                labeled(
                    "Outlet / reference pressure", self.outlet_pressure, important=True
                ),
                labeled(
                    "Nominal inlet pressure",
                    self.inlet_pressure,
                    "Used for reporting and growth; current flow solves are driven by inlet flow.",
                ),
            )
        )
        self.bc_banner = Banner()
        bc.add(self.bc_banner)
        self.column.addWidget(bc)

        self.inlet_card = Card(
            "Inlet-specific conditions",
            "Use different boundary conditions for each tree in a forest.",
        )
        self.use_inlet_conditions = QCheckBox("Set conditions separately for each inlet")
        self.inlet_selector = _combo([])
        self.inlet_condition_stack = QStackedWidget()
        self._inlet_widgets: list[dict[str, UnitValue]] = []
        self._inlet_count = 1
        self.inlet_card.add(self.use_inlet_conditions)
        self.inlet_card.add(labeled("Editing", self.inlet_selector))
        self.inlet_card.add(self.inlet_condition_stack)
        self.column.addWidget(self.inlet_card)
        self.inlet_selector.currentIndexChanged.connect(
            self.inlet_condition_stack.setCurrentIndex
        )
        self.use_inlet_conditions.toggled.connect(
            self.inlet_condition_stack.setEnabled
        )
        self.use_inlet_conditions.toggled.connect(self.inlet_selector.setEnabled)
        self.use_inlet_conditions.toggled.connect(lambda *_: self.changed.emit())
        self.inlet_card.setVisible(False)

        oxy = Card("Diffusion and consumption")
        # TODO(M2+): expose a validated custom tissue oxygen-consumption law.
        # M0 preserves the current Michaelis-Menten Vmax/Km contract.
        self.diffusivity = UnitValue(
            ["cm²/s", "m²/s"], 2.41e-5, minimum=0, maximum=1e6, decimals=10
        )
        self.vmax = UnitValue(
            ["mol/m³/s", "mmHg/s"], 0.001, minimum=0, maximum=1e6, decimals=8
        )
        self.km = UnitValue(
            ["mol/m³", "mmHg"], 0.005, minimum=0, maximum=1e6, decimals=8
        )
        self.inlet_o2 = UnitValue(
            ["mmHg", "mol/m³"], 100.0, minimum=0, maximum=1e9, decimals=6
        )
        self.viability_enabled = QCheckBox("Classify viable tissue")
        self.viability = UnitValue(
            ["mmHg", "mol/m³"], 1.0, minimum=0, maximum=1e9, decimals=6
        )
        self.viability_enabled.setChecked(True)
        self.viability_enabled.toggled.connect(self.viability.setEnabled)
        oxy.add(
            row_of(
                labeled("Tissue diffusivity", self.diffusivity, important=True),
                labeled("Inlet concentration", self.inlet_o2, important=True),
            )
        )
        oxy.add(
            row_of(
                labeled("Vmax", self.vmax, important=True),
                labeled("Km", self.km, important=True),
            )
        )
        oxy.add(
            row_of(
                self.viability_enabled,
                labeled(
                    "Viability threshold",
                    self.viability,
                    "Optional; used in tissue point exports.",
                ),
            )
        )
        self.column.addWidget(oxy)

        blood = Card("Fluid and blood model")
        self.fluid = _combo([("Blood", "blood"), ("Water / cell media", "water")])
        self.hematocrit_model = _combo(
            [
                ("Pries–Secomb phase separation", "pries_secomb"),
                ("Uniform discharge hematocrit", "constant"),
            ]
        )
        self.hematocrit = _double(0.42, 0, 0.95, 4, 0.01)
        self.hb_capacity = _double(20.3, 0, 1000, 4, 0.1)
        blood.add(
            row_of(
                labeled("Perfusate", self.fluid, important=True),
                labeled("Hematocrit model", self.hematocrit_model),
            )
        )
        blood.add(
            row_of(
                labeled("Discharge hematocrit", self.hematocrit, important=True),
                labeled(
                "Hemoglobin O₂ capacity per Hct",
                self.hb_capacity,
                    "Oxygen-carrying capacity normalized by hematocrit.",
                ),
            )
        )
        density = QLineEdit("Derived by the selected blood/media model")
        density.setEnabled(False)
        blood.add(
            labeled(
                "Density and viscosity",
                density,
                "The selected microvascular hematocrit model determines effective segment viscosity.",
            )
        )
        self.column.addWidget(blood)
        self.fluid.currentIndexChanged.connect(self._update_blood_controls)
        self._bind_units(self.flow, flow_to_ul_min, flow_from_ul_min)
        self._bind_units(self.inlet_pressure, pressure_to_pa, pressure_from_pa)
        self._bind_units(self.outlet_pressure, pressure_to_pa, pressure_from_pa)
        self._bind_units(
            self.inlet_o2, oxygen_to_concentration, oxygen_from_concentration
        )
        self._bind_units(
            self.viability, oxygen_to_concentration, oxygen_from_concentration
        )
        self._bind_units(self.vmax, oxygen_to_concentration, oxygen_from_concentration)
        self._bind_units(self.km, oxygen_to_concentration, oxygen_from_concentration)
        self._bind_units(
            self.diffusivity,
            lambda value, unit: value * (1e4 if unit == "m²/s" else 1.0),
            lambda value, unit: value / (1e4 if unit == "m²/s" else 1.0),
        )
        self.set_inlet_count(1)
        self.bc_mode.currentIndexChanged.connect(self._update_bc)
        self._update_bc()
        self._update_blood_controls()
        self.finish()

    def _update_blood_controls(self, *_):
        """Blood-specific parameters do not apply to water or cell media."""
        is_blood = self.fluid.currentData() == "blood"
        for widget in (self.hematocrit_model, self.hematocrit, self.hb_capacity):
            widget.setEnabled(is_blood)

    @staticmethod
    def _bind_units(widget, to_base, from_base):
        widget.setProperty("previousUnit", widget.unit())

        def changed(new_unit):
            old_unit = widget.property("previousUnit") or new_unit
            base_value = to_base(widget.value(), str(old_unit))
            widget.spin.blockSignals(True)
            widget.setValue(from_base(base_value, new_unit))
            widget.spin.blockSignals(False)
            widget.setProperty("previousUnit", new_unit)

        widget.unitChanged.connect(changed)

    def _update_bc(self):
        mode = self.bc_mode.currentData()
        pressure_only = mode == "pressure_pressure"
        self.flow.setEnabled(not pressure_only)
        if pressure_only:
            self.bc_banner.set_message(
                "Inlet flow is required. Choose “Inlet flow + outlet pressure.”",
                "danger",
            )
        elif mode == "legacy_equal_flow":
            self.bc_banner.set_message(
                "Fully specified  │  equal outlet flow",
                "warning",
            )
        else:
            self.bc_banner.set_message(
                "Fully specified",
                "success",
            )

    def _make_inlet_condition_panel(self, index: int) -> dict[str, UnitValue]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(9)
        fields = {
            "flow": UnitValue(
                ["µL/min", "mL/min", "m³/s"],
                self.flow.value(),
                minimum=0,
                maximum=1e12,
                decimals=7,
            ),
            "inlet_pressure": UnitValue(
                ["mmHg", "Pa"],
                self.inlet_pressure.value(),
                minimum=-1e9,
                maximum=1e9,
                decimals=5,
            ),
            "outlet_pressure": UnitValue(
                ["mmHg", "Pa"],
                self.outlet_pressure.value(),
                minimum=-1e9,
                maximum=1e9,
                decimals=5,
            ),
            "inlet_o2": UnitValue(
                ["mmHg", "mol/m³"],
                self.inlet_o2.value(),
                minimum=0,
                maximum=1e9,
                decimals=6,
            ),
        }
        fields["flow"].setUnit(self.flow.unit())
        fields["inlet_pressure"].setUnit(self.inlet_pressure.unit())
        fields["outlet_pressure"].setUnit(self.outlet_pressure.unit())
        fields["inlet_o2"].setUnit(self.inlet_o2.unit())
        self._bind_units(fields["flow"], flow_to_ul_min, flow_from_ul_min)
        self._bind_units(fields["inlet_pressure"], pressure_to_pa, pressure_from_pa)
        self._bind_units(fields["outlet_pressure"], pressure_to_pa, pressure_from_pa)
        self._bind_units(
            fields["inlet_o2"], oxygen_to_concentration, oxygen_from_concentration
        )
        layout.addWidget(labeled("Inlet flow", fields["flow"], important=True))
        layout.addWidget(labeled("Inlet pressure", fields["inlet_pressure"], important=True))
        layout.addWidget(labeled("Outlet / reference pressure", fields["outlet_pressure"], important=True))
        layout.addWidget(labeled("Inlet concentration", fields["inlet_o2"], important=True))
        for widget in fields.values():
            widget.valueChanged.connect(lambda *_: self.changed.emit())
            widget.unitChanged.connect(lambda *_: self.changed.emit())
        fields["panel"] = panel
        return fields

    def _canonical_inlet_conditions(self) -> list[dict[str, float]]:
        conditions = []
        for fields in self._inlet_widgets:
            conditions.append(
                {
                    "flow_ul_min": flow_to_ul_min(fields["flow"].value(), fields["flow"].unit()),
                    "inlet_pressure_pa": pressure_to_pa(
                        fields["inlet_pressure"].value(), fields["inlet_pressure"].unit()
                    ),
                    "outlet_pressure_pa": pressure_to_pa(
                        fields["outlet_pressure"].value(), fields["outlet_pressure"].unit()
                    ),
                    "inlet_concentration_mmol_l": oxygen_to_concentration(
                        fields["inlet_o2"].value(), fields["inlet_o2"].unit()
                    ),
                }
            )
        return conditions

    def set_inlet_count(self, count: int) -> None:
        count = max(1, int(count))
        if count == self._inlet_count and len(self._inlet_widgets) == count:
            self.inlet_card.setVisible(count > 1)
            return
        preserved = self._canonical_inlet_conditions() if self._inlet_widgets else []
        while self.inlet_condition_stack.count():
            widget = self.inlet_condition_stack.widget(0)
            self.inlet_condition_stack.removeWidget(widget)
            widget.deleteLater()
        self._inlet_widgets = []
        self.inlet_selector.blockSignals(True)
        self.inlet_selector.clear()
        for index in range(count):
            fields = self._make_inlet_condition_panel(index)
            self._inlet_widgets.append(fields)
            self.inlet_condition_stack.addWidget(fields["panel"])
            self.inlet_selector.addItem(f"Inlet {index + 1}", index)
            if index < len(preserved):
                self._set_inlet_condition(fields, preserved[index])
        self.inlet_selector.blockSignals(False)
        self.inlet_selector.setCurrentIndex(0)
        self.inlet_condition_stack.setCurrentIndex(0)
        self._inlet_count = count
        self.inlet_card.setVisible(count > 1)
        enabled = self.use_inlet_conditions.isChecked()
        self.inlet_selector.setEnabled(enabled)
        self.inlet_condition_stack.setEnabled(enabled)

    @staticmethod
    def _set_inlet_condition(fields, condition) -> None:
        fields["flow"].setValue(
            flow_from_ul_min(float(condition.get("flow_ul_min", 100.0)), fields["flow"].unit())
        )
        fields["inlet_pressure"].setValue(
            pressure_from_pa(float(condition.get("inlet_pressure_pa", 66661.0)), fields["inlet_pressure"].unit())
        )
        fields["outlet_pressure"].setValue(
            pressure_from_pa(float(condition.get("outlet_pressure_pa", 40000.0)), fields["outlet_pressure"].unit())
        )
        fields["inlet_o2"].setValue(
            oxygen_from_concentration(
                float(condition.get("inlet_concentration_mmol_l", 0.14)),
                fields["inlet_o2"].unit(),
            )
        )

    def load(self, config):
        gui = config.get("gui", {})
        bc = gui.get("boundary_conditions", {})
        _set_combo(self.bc_mode, bc.get("mode", "flow_pressure"))
        self._update_bc()
        pressure_unit = bc.get("pressure_unit", "mmHg")
        flow_unit = bc.get("flow_unit", "µL/min")
        self.flow.setUnit(flow_unit)
        self.flow.setValue(
            flow_from_ul_min(
                float(config.get("simulation", {}).get("qin_target_ul_min", 100.0)),
                flow_unit,
            )
        )
        root_pa = float(_value(config, "hemodynamics", "root_pressure", 66661.0))
        terminal_pa = float(
            _value(config, "hemodynamics", "terminal_pressure", 40000.0)
        )
        self.inlet_pressure.setUnit(pressure_unit)
        self.inlet_pressure.setValue(pressure_from_pa(root_pa, pressure_unit))
        self.outlet_pressure.setUnit(pressure_unit)
        self.outlet_pressure.setValue(pressure_from_pa(terminal_pa, pressure_unit))
        oxy_unit = _concentration_unit(gui.get("oxygen_input_unit", "mmHg"))
        self.inlet_o2.setUnit(oxy_unit)
        inlet_map = _value(
            config,
            "oxygen",
            "concentration_inlet_by_fluid",
            {"blood": 0.14, "water": 0.2211, "cell media": 0.2211, "media": 0.2211},
        )
        fluid = config.get("simulation", {}).get("fluid", "blood")
        inlet_c = (
            float(inlet_map.get(fluid, 0.14))
            if isinstance(inlet_map, dict)
            else float(inlet_map)
        )
        self.inlet_o2.setValue(oxygen_from_concentration(inlet_c, oxy_unit))
        diff = float(_value(config, "oxygen", "solute_diffusivity", 2.41e-5))
        diff_unit = gui.get("diffusivity_unit", "cm²/s")
        self.diffusivity.setUnit(diff_unit)
        self.diffusivity.setValue(diff / 1e4 if diff_unit == "m²/s" else diff)
        vmax_unit = _concentration_unit(gui.get("vmax_unit", "mol/m³/s"), rate=True)
        km_unit = _concentration_unit(gui.get("km_unit", "mol/m³"))
        self.vmax.setUnit(vmax_unit)
        self.km.setUnit(km_unit)
        self.vmax.setValue(
            oxygen_from_concentration(
                float(_value(config, "oxygen", "vmax_mm", 0.001)), self.vmax.unit()
            )
        )
        self.km.setValue(
            oxygen_from_concentration(
                float(_value(config, "oxygen", "k_m_mm", 0.005)), self.km.unit()
            )
        )
        _set_combo(self.fluid, fluid)
        _set_combo(
            self.hematocrit_model, _value(config, "hematocrit", "model", "pries_secomb")
        )
        self.hematocrit.setValue(
            float(_value(config, "hematocrit", "hd_discharge", 0.42))
        )
        self.hb_capacity.setValue(
            float(_value(config, "oxygen", "o2_cap_per_hct", 20.3))
        )
        threshold = config.get("simulation", {}).get("viability_threshold")
        self.viability_enabled.setChecked(threshold is not None)
        self.viability.setUnit(_concentration_unit(gui.get("viability_unit", "mmHg")))
        self.viability.setValue(
            oxygen_from_concentration(float(threshold or ALPHA_MMHG), self.viability.unit())
        )
        conditions = list(config.get("simulation", {}).get("inlet_conditions", []) or [])
        self.set_inlet_count(max(1, len(conditions)))
        self.use_inlet_conditions.setChecked(bool(conditions))
        for fields, condition in zip(self._inlet_widgets, conditions):
            self._set_inlet_condition(fields, condition)
        enabled = self.use_inlet_conditions.isChecked()
        self.inlet_selector.setEnabled(enabled)
        self.inlet_condition_stack.setEnabled(enabled)
        self._update_blood_controls()

    def write(self, config):
        gui = config.setdefault("gui", {})
        bc = gui.setdefault("boundary_conditions", {})
        bc.update(
            {
                "mode": self.bc_mode.currentData(),
                "pressure_unit": self.outlet_pressure.unit(),
                "flow_unit": self.flow.unit(),
            }
        )
        gui["oxygen_input_unit"] = self.inlet_o2.unit()
        gui["vmax_unit"] = self.vmax.unit()
        gui["km_unit"] = self.km.unit()
        gui["viability_unit"] = self.viability.unit()
        gui["diffusivity_unit"] = self.diffusivity.unit()
        sim = config.setdefault("simulation", {})
        sim["qin_target_ul_min"] = flow_to_ul_min(self.flow.value(), self.flow.unit())
        if self.use_inlet_conditions.isChecked() and self._inlet_count > 1:
            sim["inlet_conditions"] = self._canonical_inlet_conditions()
            sim["flow_source"] = "per_inlet"
        else:
            sim.pop("inlet_conditions", None)
            if sim.get("flow_source") == "per_inlet":
                sim["flow_source"] = "per_tree"
        sim["fluid"] = self.fluid.currentData()
        sim["build_fluid"] = self.fluid.currentData()
        sim["viability_threshold"] = (
            oxygen_to_concentration(self.viability.value(), self.viability.unit())
            if self.viability_enabled.isChecked()
            else None
        )
        settings = config.setdefault("settings", {})
        hemo = settings.setdefault("hemodynamics", {})
        hemo.update(
            {
                "root_pressure": pressure_to_pa(
                    self.inlet_pressure.value(), self.inlet_pressure.unit()
                ),
                "terminal_pressure": pressure_to_pa(
                    self.outlet_pressure.value(), self.outlet_pressure.unit()
                ),
                "kirchhoff_bc_mode": "legacy_equal_terminal_flow"
                if self.bc_mode.currentData() == "legacy_equal_flow"
                else "terminal_pressure",
            }
        )
        oxygen = settings.setdefault("oxygen", {})
        inlet = oxygen_to_concentration(self.inlet_o2.value(), self.inlet_o2.unit())
        oxygen.update(
            {
                "concentration_inlet_by_fluid": {
                    "blood": inlet,
                    "water": inlet,
                    "cell media": inlet,
                    "media": inlet,
                },
                "conc_max_for_normalization": inlet,
                "solute_diffusivity": self.diffusivity.value()
                * (1e4 if self.diffusivity.unit() == "m²/s" else 1.0),
                "vmax_mm": oxygen_to_concentration(self.vmax.value(), self.vmax.unit()),
                "k_m_mm": oxygen_to_concentration(self.km.value(), self.km.unit()),
                "o2_cap_per_hct": self.hb_capacity.value(),
            }
        )
        settings.setdefault("hematocrit", {}).update(
            {
                "model": self.hematocrit_model.currentData(),
                "hd_discharge": self.hematocrit.value(),
            }
        )


class SolverPage(Page):
    def __init__(self, parent=None):
        super().__init__(
            "Solver decisions",
            "",
            parent,
        )
        choose = Card("Solver path")
        self._lattice_mode = False
        self.preset = _combo(
            [
                ("Paper-aligned balanced", "paper"),
                ("Fast preview", "fast"),
                ("Higher accuracy", "accurate"),
                ("Custom", "custom"),
            ]
        )
        self.conc_solver = _combo(
            [
                ("Network direct solver (best for large domains)", "network_ext"),
                ("Network FFT solver (best for primitive-shape domains)", "network_ext_hybrid_bg"),
                ("Top-down direct solver (for tree structures only)", "topdown_ext"),
                ("Top-down FFT solver (for tree structures only)", "topdown_ext_hybrid_bg"),
                ("Network, no vessel-vessel coupling (faster but less accurate)", "network"),
                ("Top-down, no vessel-vessel coupling (faster but less accurate)", "topdown"),
            ]
        )
        self.conc_solver.setMinimumWidth(520)
        self.flow_solver = _combo(
            [
                ("Tree-specialized", "tree"),
                ("Automatic sparse", "auto"),
                ("Sparse direct (spsolve)", "spsolve"),
                ("Conjugate gradient", "cg"),
                ("GMRES + ILU", "gmres"),
            ]
        )
        choose.add(
            row_of(
                labeled("Starting preset", self.preset),
                labeled(
                    "Vessel oxygen solver", self.conc_solver, important=True
                ),
            )
        )
        choose.add(labeled("Flow solver", self.flow_solver, important=True))
        self.solver_hint = Banner()
        self.solver_hint.setVisible(False)
        choose.add(self.solver_hint)
        self.column.addWidget(choose)

        primary = Card("Important numerical choices")
        self.closure = _combo(
            [
                ("Resolved intralumen radial transport (recommended)", "graetz"),
                ("Well mixed lumen", "wellmixed"),
                ("Custom wall exchange expression", "custom_kappa"),
            ]
        )
        self.finite_radius = _combo(
            [
                ("Both monopole and dipole terms", "both"),
                ("Monopole term only", "monopole"),
                ("Dipole term only", "dipole"),
                ("Neither (centerline approximation)", "none"),
            ]
        )
        self.kappa = QLineEdit()
        self.kappa.setPlaceholderText("e.g. 0.12 or 0.12 * (1 + x/L)")
        self.tissue_gl_order = _spin(5, 1, 32)
        # Retain the old attribute for extensions that addressed the original
        # single control directly.
        self.gl_order = self.tissue_gl_order
        self.cext_gl_order = _spin(1, 1, 32)
        self.axial_steps = _spin(5, 1, 100)
        self.hct_iterations = _spin(2, 0, 1000)
        self.hct_tol = _double(0.001, 0, 1, 8, 0.0001)
        primary.add(
            row_of(
                labeled("Lumen-to-wall closure", self.closure, important=True),
                labeled("Finite-radius Green's terms", self.finite_radius),
            )
        )
        self.kappa_row = labeled(
            "Custom κ for q = κ(C − Cₑₓₜ)",
            self.kappa,
            "Constant or position-dependent wall exchange coefficient.",
        )
        self.kappa_row.setVisible(False)
        primary.add(self.kappa_row)
        primary.add(
            row_of(
                labeled("Tissue quadrature", self.tissue_gl_order),
                labeled("External-field quadrature", self.cext_gl_order),
                labeled("Axial blood steps", self.axial_steps),
            )
        )
        primary.add(
            row_of(
                labeled("Hct / flow iterations", self.hct_iterations),
                labeled("Hct tolerance", self.hct_tol),
            )
        )
        self.column.addWidget(primary)

        self.cext_card = Card("External concentration / tissue coupling")
        self.backend = _combo([("Automatic", "auto"), ("GPU", "gpu"), ("CPU", "cpu")])
        self.cext_backend = _combo(
            [("Automatic", "auto"), ("GPU", "gpu"), ("CPU", "cpu")]
        )
        self.precision = _combo(
            [
                ("Float32 accelerator work arrays", "float32"),
                ("Float64 accelerator work arrays", "float64"),
            ]
        )
        self.cext_mode = _combo(
            [
                ("FFT background + local correction", "fft"),
                ("Local screened interactions", "local_only_nlambda"),
                ("Hybrid", "hybrid"),
            ]
        )
        self.cext_grid = _spin(256, 16, 1024, 16)
        self.lambda_bins = _spin(5, 1, 64)
        self.window = _double(6.0, 0.1, 100, 3, 0.5)
        self.cext_iters = _spin(1, 1, 10000)
        self.cext_tol = _double(1e-3, 0, 1e9, 10, 1e-4)
        self.cext_accel = _combo(
            [("Anderson", "anderson"), ("Aitken", "aitken"), ("None", "none")]
        )
        self.cext_card.add(
            row_of(
                labeled("Tissue backend", self.backend, important=True),
                labeled("Cₑₓₜ backend", self.cext_backend, important=True),
            )
        )
        self.cext_card.add(
            labeled(
                "Accelerator working precision",
                self.precision,
                "Precision for accelerated tissue calculations.",
            )
        )
        self.cext_card.add(
            row_of(
                labeled("Cₑₓₜ method", self.cext_mode),
                labeled("Interaction window λ", self.window),
            )
        )
        self.cext_card.add(
            row_of(
                labeled("FFT grid per axis", self.cext_grid, important=True),
                labeled("Screening bins", self.lambda_bins),
            )
        )
        self.cext_card.add(
            row_of(
                labeled("Coupling iterations", self.cext_iters),
                labeled("Coupling tolerance", self.cext_tol),
            )
        )
        self.cext_card.add(labeled("Coupling acceleration", self.cext_accel))
        self.column.addWidget(self.cext_card)

        expert = Card(
            "Expert overrides",
            "Check a setting to write an explicit override. Inapplicable sections are dimmed but remain inspectable.",
        )
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter settings…")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Override / setting", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setMinimumHeight(330)
        self._populate_expert()
        expert.add(self.filter)
        expert.add(self.tree)
        self.column.addWidget(expert)
        self.filter.textChanged.connect(self._filter_expert)
        self.conc_solver.currentIndexChanged.connect(self._applicability)
        self.closure.currentIndexChanged.connect(self._update_kappa_visibility)
        self.preset.activated.connect(self._apply_preset)
        self.finish()

    def _populate_expert(self):
        self.tree.clear()
        seen = set()
        for section_name, section in SETTINGS_SECTIONS.items():
            if section_name in {"kirchhoff", "concentration"}:
                continue
            parent = QTreeWidgetItem([section_name.replace("_", " ").title(), ""])
            parent.setData(0, Qt.UserRole, section_name)
            self.tree.addTopLevelItem(parent)
            for name, value in section.defaults.items():
                if (section_name, name) in seen:
                    continue
                seen.add((section_name, name))
                shown = (
                    json.dumps(_json_safe(value), separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else str(_json_safe(value))
                )
                child = QTreeWidgetItem([name, shown])
                child.setFlags(
                    child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable
                )
                child.setCheckState(0, Qt.Unchecked)
                child.setData(0, Qt.UserRole, (section_name, name))
                parent.addChild(child)
        self.tree.collapseAll()

    def _filter_expert(self, text):
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            any_visible = False
            for j in range(parent.childCount()):
                child = parent.child(j)
                visible = (
                    not needle
                    or needle in child.text(0).lower()
                    or needle in parent.text(0).lower()
                )
                child.setHidden(not visible)
                any_visible |= visible
            parent.setHidden(not any_visible)
            if needle and any_visible:
                parent.setExpanded(True)

    def _applicability(self):
        if self._lattice_mode and self.conc_solver.currentData() not in {
            "network_ext",
            "network",
            "network_ext_hybrid_bg",
        }:
            _set_combo(self.conc_solver, "network_ext")
            _set_combo(self.flow_solver, "spsolve")
        network_solver = self.conc_solver.currentData() in {
            "network_ext",
            "network_ext_hybrid_bg",
            "network",
        }
        if network_solver:
            _set_combo(self.closure, "wellmixed")
        self.closure.setEnabled(not network_solver)
        self.closure.setToolTip(
            "General-network oxygen currently uses the well-mixed lumen closure."
            if network_solver
            else ""
        )
        self._update_kappa_visibility()
        use_cext = "ext" in str(self.conc_solver.currentData())
        self.cext_card.setEnabled(use_cext)
        self.cext_gl_order.setEnabled(use_cext)
        self.cext_gl_order.setToolTip(
            "Gauss–Legendre points per vessel used by the coupled external field."
            if use_cext
            else "Not used when vessel–vessel oxygen coupling is disabled."
        )
        if self._lattice_mode:
            self.solver_hint.set_message(
                "Network oxygen and sparse-direct flow are required for lattices",
                "info",
            )
        else:
            self.solver_hint.setVisible(False)
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            active = parent.data(0, Qt.UserRole) != "cext" or use_cext
            parent.setForeground(0, QColor("#273944" if active else "#9ca6ac"))

    def _update_kappa_visibility(self, *_):
        visible = self.closure.isEnabled() and self.closure.currentData() == "custom_kappa"
        self.kappa_row.setVisible(visible)
        self.kappa.setEnabled(visible)

    def set_lattice_mode(self, enabled: bool) -> None:
        self._lattice_mode = bool(enabled)
        if self._lattice_mode:
            if self.conc_solver.currentData() not in {
                "network_ext",
                "network_ext_hybrid_bg",
                "network",
            }:
                _set_combo(self.conc_solver, "network_ext")
            _set_combo(self.flow_solver, "spsolve")
            self.conc_solver.setToolTip(
                "Lattices require one of the general-network oxygen solvers."
            )
            self.flow_solver.setToolTip(
                "Locked to Sparse direct (spsolve) for the generated lattice."
            )
        else:
            self.conc_solver.setToolTip("")
            self.flow_solver.setToolTip("")
        self.conc_solver.setEnabled(True)
        self.flow_solver.setEnabled(not self._lattice_mode)
        self._applicability()

    def _apply_preset(self):
        mode = self.preset.currentData()
        if mode == "fast":
            _set_combo(self.conc_solver, "topdown")
            _set_combo(self.closure, "wellmixed")
            _set_combo(self.finite_radius, "none")
            self.tissue_gl_order.setValue(3)
            self.cext_gl_order.setValue(3)
            self.axial_steps.setValue(3)
            self.cext_iters.setValue(1)
            self.cext_grid.setValue(128)
        elif mode == "paper":
            _set_combo(self.conc_solver, "topdown_ext_hybrid_bg")
            _set_combo(self.closure, "graetz")
            _set_combo(self.finite_radius, "both")
            self.tissue_gl_order.setValue(5)
            self.cext_gl_order.setValue(5)
            self.axial_steps.setValue(5)
            self.cext_iters.setValue(5)
            self.cext_grid.setValue(256)
            self.lambda_bins.setValue(5)
            self.window.setValue(6)
        elif mode == "accurate":
            _set_combo(self.conc_solver, "topdown_ext_hybrid_bg")
            _set_combo(self.closure, "graetz")
            _set_combo(self.finite_radius, "both")
            self.tissue_gl_order.setValue(7)
            self.cext_gl_order.setValue(7)
            self.axial_steps.setValue(8)
            self.cext_iters.setValue(10)
            self.cext_grid.setValue(384)
            self.lambda_bins.setValue(8)
            self.window.setValue(8)
        if self._lattice_mode:
            if self.conc_solver.currentData() not in {
                "network_ext",
                "network_ext_hybrid_bg",
                "network",
            }:
                _set_combo(self.conc_solver, "network_ext")
            _set_combo(self.flow_solver, "spsolve")
        self._applicability()

    def load(self, config):
        sim = config.get("simulation", {})
        gui = config.get("gui", {})
        _set_combo(self.preset, gui.get("solver_preset", "custom"))
        solver_value = sim.get("concentration_solver", "network_ext")
        # Keep old treecode projects loadable without exposing the unvalidated
        # choice in the current interface.
        if solver_value == "topdown_ext_treecode":
            solver_value = "topdown_ext_hybrid_bg"
        _set_combo(self.conc_solver, solver_value)
        _set_combo(
            self.flow_solver, _value(config, "hemodynamics", "kirchhoff_solver", "tree")
        )
        _set_combo(
            self.closure, _value(config, "oxygen", "lumen_wall_closure", "graetz")
        )
        self.kappa.setText(str(config.get("gui", {}).get("custom_kappa", "")))
        finite_radius = _value(config, "oxygen", "finite_radius_o2_terms", "both")
        finite_radius = {"source": "monopole", "target": "dipole"}.get(
            str(finite_radius).lower(), finite_radius
        )
        _set_combo(self.finite_radius, finite_radius)
        legacy_order = int(_value(config, "oxygen", "gl_order", 5))
        self.tissue_gl_order.setValue(legacy_order)
        self.cext_gl_order.setValue(
            int(_value(config, "oxygen", "gl_order_cext", legacy_order))
        )
        self.axial_steps.setValue(int(_value(config, "oxygen", "axial_blood_steps", 5)))
        self.hct_iterations.setValue(
            int(_value(config, "hematocrit", "flow_iterations", 2))
        )
        self.hct_tol.setValue(float(_value(config, "hematocrit", "hdtol", 0.001)))
        _set_combo(self.backend, _value(config, "tissue", "accel_mode", "gpu"))
        _set_combo(self.cext_backend, _value(config, "cext", "accel_mode", "gpu"))
        _set_combo(self.precision, _value(config, "cext", "float_dtype", "float32"))
        _set_combo(self.cext_mode, _value(config, "cext", "hybrid_bg_mode", "fft"))
        self.cext_grid.setValue(int(_value(config, "cext", "hybrid_bg_grid", 256)))
        self.lambda_bins.setValue(
            int(_value(config, "cext", "hybrid_bg_lambda_bins", 5))
        )
        self.window.setValue(float(_value(config, "cext", "window_factor", 6)))
        self.cext_iters.setValue(
            int(_value(config, "cext", "vess_coupling_max_iter", 1))
        )
        self.cext_tol.setValue(float(_value(config, "cext", "vess_coupling_tol", 1e-3)))
        _set_combo(
            self.cext_accel, _value(config, "cext", "vess_coupling_accel", "anderson")
        )
        self._load_overrides(config.get("settings", {}))
        self._applicability()

    def _load_overrides(self, settings):
        controlled = {
            "ROOT_PRESSURE",
            "TERMINAL_PRESSURE",
            "KIRCHHOFF_SOLVER",
            "KIRCHHOFF_BC_MODE",
            "HEMATOCRIT_MODEL",
            "HEMATOCRIT_FLOW_ITERATIONS",
            "HEMATOCRIT_HDTOL",
            "HD_DISCHARGE",
            "CONCENTRATION_SOLVER",
            "CONCENTRATION_INLET_BY_FLUID",
            "CONC_MAX_FOR_NORMALIZATION",
            "SOLUTE_DIFFUSIVITY",
            "VMAX_MM",
            "K_M_MM",
            "O2_CAP_PER_HCT",
            "LUMEN_WALL_CLOSURE",
            "FINITE_RADIUS_O2_TERMS",
            "GL_ORDER",
            "GL_ORDER_CEXT",
            "AXIAL_BLOOD_STEPS",
            "TISSUE_ACCEL_MODE",
            "TISSUE_CACHE_FLOAT_DTYPE",
            "CEXT_ACCEL_MODE",
            "CEXT_FLOAT_DTYPE",
            "CEXT_HYBRID_BG_MODE",
            "CEXT_HYBRID_BG_GRID",
            "CEXT_HYBRID_BG_LAMBDA_BINS",
            "CEXT_WINDOW_FACTOR",
            "CEXT_VESS_COUPLING_MAX_ITER",
            "CEXT_VESS_COUPLING_TOL",
            "CEXT_VESS_COUPLING_ACCEL",
        }
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            section_name = parent.data(0, Qt.UserRole)
            supplied = settings.get(section_name, {})
            section = SETTINGS_SECTIONS.get(section_name)
            normalized = {}
            if section:
                for key, value in supplied.items():
                    const = key.upper()
                    if const not in section.defaults:
                        const = section.aliases.get(
                            key.lower().replace("-", "_"), const
                        )
                    normalized[const] = value
            for j in range(parent.childCount()):
                child = parent.child(j)
                _, name = child.data(0, Qt.UserRole)
                if name in normalized and name not in controlled:
                    value = normalized[name]
                    child.setText(
                        1,
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else str(value),
                    )
                    child.setCheckState(0, Qt.Checked)
                else:
                    child.setCheckState(0, Qt.Unchecked)

    def write(self, config):
        config.setdefault("gui", {})["solver_preset"] = self.preset.currentData()
        config.setdefault("gui", {})["custom_kappa"] = self.kappa.text().strip()
        config.setdefault("simulation", {})["concentration_solver"] = (
            self.conc_solver.currentData()
        )
        settings = config.setdefault("settings", {})
        settings.setdefault("hemodynamics", {})["kirchhoff_solver"] = (
            self.flow_solver.currentData()
        )
        settings.setdefault("oxygen", {}).update(
            {
                "lumen_wall_closure": self.closure.currentData(),
                "finite_radius_o2_terms": self.finite_radius.currentData(),
                "gl_order": self.tissue_gl_order.value(),
                "gl_order_cext": self.cext_gl_order.value(),
                "axial_blood_steps": self.axial_steps.value(),
            }
        )
        settings.setdefault("hematocrit", {}).update(
            {
                "flow_iterations": self.hct_iterations.value(),
                "hdtol": self.hct_tol.value(),
            }
        )
        settings.setdefault("tissue", {})["accel_mode"] = self.backend.currentData()
        settings["tissue"]["cache_float_dtype"] = self.precision.currentData()
        settings.setdefault("cext", {}).update(
            {
                "accel_mode": self.cext_backend.currentData(),
                "float_dtype": self.precision.currentData(),
                "hybrid_bg_mode": self.cext_mode.currentData(),
                "hybrid_bg_grid": self.cext_grid.value(),
                "hybrid_bg_lambda_bins": self.lambda_bins.value(),
                "window_factor": self.window.value(),
                "vess_coupling_max_iter": self.cext_iters.value(),
                "vess_coupling_tol": self.cext_tol.value(),
                "vess_coupling_accel": self.cext_accel.currentData(),
            }
        )
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) != Qt.Checked:
                    continue
                section, name = child.data(0, Qt.UserRole)
                settings.setdefault(section, {})[name] = _parse_jsonish(child.text(1))


class OutputsPage(Page):
    preview_changed = Signal()

    SWEEP_PATHS = [
        ("Inlet flow", "simulation.qin_target_ul_min"),
        ("Terminal count", "network.target_terminal_count"),
        ("Diffusivity", "settings.oxygen.solute_diffusivity"),
        ("Vmax", "settings.oxygen.vmax_mm"),
        ("Km", "settings.oxygen.k_m_mm"),
        ("Discharge hematocrit", "settings.hematocrit.hd_discharge"),
        ("Inlet concentration", "settings.oxygen.conc_max_for_normalization"),
        ("FFT grid", "settings.cext.hybrid_bg_grid"),
        ("Cₑₓₜ iterations", "settings.cext.vess_coupling_max_iter"),
    ]
    SWEEP_UNITS = {
        "simulation.qin_target_ul_min": ("µL/min", ["µL/min", "cm³/s", "mL/min", "m³/s"]),
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
        self.sample_controls = QStackedWidget()
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
        self.save_network.setChecked(True)
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
        output.add(self.out_dir)
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
        self.job_count.setStyleSheet("font-weight:800;color:#285b6c;")
        row.addWidget(self.job_count)
        sweep.add(self.sweep_rows_panel)
        sweep.add(footer)
        self.column.addWidget(sweep)
        self.sample_mode.currentIndexChanged.connect(self._sampling_changed)
        self.sample_mode.currentIndexChanged.connect(lambda *_: self.preview_changed.emit())
        self.sample_points.valueChanged.connect(lambda *_: self.preview_changed.emit())
        self.sample_file.changed.connect(lambda *_: self.preview_changed.emit())
        self.combined_sweep_csv.toggled.connect(self._combined_sweep_output_changed)
        self.summary_csv.toggled.connect(self._summary_output_changed)
        for field in (self.grid_x, self.grid_y, self.grid_z):
            field.valueChanged.connect(self._update_grid_total)
            field.valueChanged.connect(lambda *_: self.preview_changed.emit())
        self._update_grid_total()
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
        self.sample_controls.setCurrentIndex({"random": 0, "grid": 1, "file": 2}.get(mode, 0))
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
            available = [path for _label, path in self.SWEEP_PATHS if path not in used_paths]
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
        self.sweep_rows_layout.insertWidget(self.sweep_rows_layout.count() - 1, row_widget)
        combo.currentIndexChanged.connect(lambda *_: self._sweep_parameter_changed(record))
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

    def _set_sweep_unit_options(self, record, path: str, selected: str | None = None) -> None:
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
            format(float(value), ".12g") if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
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
        sim["distance_sample_count"] = 0 if sim["sample_mode"] == "file" else self.sample_points.value()
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


class QueuePage(Page):
    add_requested = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(
            "Run queue",
            "",
            parent,
        )
        control = Card("Queue controls")
        toolbar = QWidget()
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        self.add_btn = QPushButton("Add current setup")
        self.add_btn.setProperty("queueRole", "neutral")
        self.add_btn.setIcon(_queue_action_icon("add"))
        self.add_run_btn = QPushButton("Add and run")
        self.add_run_btn.setProperty("queueRole", "run")
        self.add_run_btn.setIcon(_queue_action_icon("add_run"))
        self.run_selected_btn = QPushButton("Run selected")
        self.run_selected_btn.setProperty("queueRole", "run")
        self.run_selected_btn.setIcon(_queue_action_icon("play_circle"))
        self.run_selected_btn.setEnabled(False)
        self.run_all_btn = QPushButton("Run all")
        self.run_all_btn.setProperty("queueRole", "run")
        self.run_all_btn.setIcon(_queue_action_icon("fast_forward"))
        self.run_all_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel active")
        self.cancel_btn.setProperty("queueRole", "stop")
        self.cancel_btn.setIcon(_queue_action_icon("stop"))
        self.cancel_btn.setEnabled(False)
        for button in [
            self.add_btn,
            self.add_run_btn,
            self.run_selected_btn,
            self.run_all_btn,
            self.cancel_btn,
        ]:
            button.setIconSize(QSize(22, 18))
            row.addWidget(button)
        row.addStretch()
        control.add(toolbar)
        self.column.addWidget(control)

        jobs = Card("Simulations")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Run", "Timestamp", "Runtime", "Status", "Results"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(64)
        header.setCascadingSectionResizes(False)
        header.setStretchLastSection(True)
        # Dense default proportions for the primary queue view.  Keep every
        # column interactive so users can still tailor the table to a study.
        # The status column is deliberately the largest: it carries active
        # progress, while the results action needs a dependable readable width.
        for column, width in enumerate((270, 155, 88, 545, 255)):
            self.table.setColumnWidth(column, width)
        self.table.setMinimumHeight(270)
        jobs.add(self.table)
        small = QWidget()
        small_row = QHBoxLayout(small)
        small_row.setContentsMargins(0, 0, 0, 0)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.setProperty("secondary", True)
        self.remove_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear finished")
        self.clear_btn.setProperty("secondary", True)
        self.clear_btn.setEnabled(False)
        self.open_sweep_csv_btn = QPushButton("Open sweep CSV")
        self.open_sweep_csv_btn.setProperty("secondary", True)
        self.open_sweep_csv_btn.setEnabled(False)
        small_row.addWidget(self.remove_btn)
        small_row.addWidget(self.clear_btn)
        small_row.addWidget(self.open_sweep_csv_btn)
        small_row.addStretch()
        jobs.add(small)
        self.column.addWidget(jobs)

        self.details_btn = QPushButton("+  Solver details")
        self.details_btn.setProperty("secondary", True)
        self.details_btn.setCheckable(True)
        self.details_btn.setChecked(False)
        self.details_btn.setToolTip("Show the selected simulation's diagnostic output.")
        self.column.addWidget(self.details_btn, 0, Qt.AlignLeft)
        log = Card("Diagnostics")
        self.log = FocusPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setMinimumHeight(220)
        log.add(self.log)
        self.diagnostics_panel = log
        self.diagnostics_panel.setVisible(False)
        self.column.addWidget(self.diagnostics_panel)
        self.finish()
        self.runner: JobRunner | None = None
        self.runtime_timer = QTimer(self)
        self.runtime_timer.setInterval(1000)
        self.runtime_timer.timeout.connect(self.refresh)
        self.add_btn.clicked.connect(lambda: self.add_requested.emit(False))
        self.add_run_btn.clicked.connect(lambda: self.add_requested.emit(True))
        self.run_selected_btn.clicked.connect(self._run_selected)
        self.run_all_btn.clicked.connect(lambda: self.runner and self.runner.run())
        self.cancel_btn.clicked.connect(
            lambda: self.runner and self.runner.cancel_current()
        )
        self.remove_btn.clicked.connect(self._remove)
        self.clear_btn.clicked.connect(
            lambda: self.runner and self.runner.clear_finished()
        )
        self.open_sweep_csv_btn.clicked.connect(self._open_sweep_csv)
        self.details_btn.toggled.connect(self._toggle_diagnostics)
        self.table.itemSelectionChanged.connect(self._show_selected_log)
        self.table.itemSelectionChanged.connect(self._update_action_states)

    def set_runner(self, runner: JobRunner):
        if self.runner:
            try:
                self.runner.jobs_changed.disconnect(self.refresh)
            except Exception:
                pass
        self.runner = runner
        runner.jobs_changed.connect(self.refresh)
        runner.log_line.connect(self._append_log)
        runner.running_changed.connect(self._running_changed)
        self.refresh()

    def selected_ids(self):
        return [
            self.table.item(row, 0).data(Qt.UserRole)
            for row in sorted({i.row() for i in self.table.selectedIndexes()})
        ]

    def refresh(self):
        if not self.runner:
            return
        selected = set(self.selected_ids())
        self.table.setRowCount(len(self.runner.jobs))
        tones = {
            "Completed": Tokens.SUCCESS,
            "Failed": Tokens.DANGER,
            "Running": Tokens.TEXT,
            "Queued": Tokens.AMBER,
            "Cancelled": Tokens.TEXT_3,
        }
        for row, job in enumerate(self.runner.jobs):
            values = [job.name, self._timestamp_text(job), self._runtime_text(job)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, job.id if col == 0 else None)
                if col == 1:
                    item.setToolTip(str(getattr(job, "created_at", "") or ""))
                self.table.setItem(row, col, item)

            status_cell = QWidget()
            status_cell.setProperty("jobStatus", job.status)
            status_layout = QVBoxLayout(status_cell)
            status_layout.setContentsMargins(7, 4, 7, 4)
            status_layout.setSpacing(4)
            status_text = job.status
            if job.status == "Running":
                stage = str(job.stage or "Running")
                status_text = f"{stage}  │  {int(job.progress or 0)}%"
            status_label = QLabel(status_text)
            status_label.setStyleSheet(
                f"color:{tones.get(job.status, Tokens.TEXT_2)};font-weight:600;"
            )
            status_layout.addWidget(status_label)
            if job.status == "Running":
                progress = QProgressBar()
                progress.setRange(0, 100)
                progress.setValue(int(job.progress or 0))
                progress.setTextVisible(False)
                progress.setFixedHeight(5)
                progress.setProperty("activeProgress", True)
                status_layout.addWidget(progress)
            self.table.setCellWidget(row, 3, status_cell)

            results = QPushButton("Open results folder")
            results.setProperty("secondary", True)
            results.setToolTip(str(job.output_dir))
            results.clicked.connect(
                lambda _checked=False, path=str(job.output_dir): open_folder(path)
            )
            self.table.setCellWidget(row, 4, results)
            self.table.setRowHeight(row, 46 if job.status == "Running" else 38)
            if job.id in selected:
                self.table.selectRow(row)
        if self.runner.running and not self.runtime_timer.isActive():
            self.runtime_timer.start()
        elif not self.runner.running:
            self.runtime_timer.stop()
        self._update_action_states()

    @staticmethod
    def _timestamp_text(job) -> str:
        raw = str(getattr(job, "created_at", "") or "")
        try:
            return datetime.fromisoformat(raw).astimezone().strftime("%Y-%m-%d  %H:%M")
        except (TypeError, ValueError):
            return raw[:16].replace("T", " ") or "—"

    @staticmethod
    def _runtime_text(job) -> str:
        started = getattr(job, "started_at", None)
        if not started:
            return "—"
        try:
            start = datetime.fromisoformat(str(started))
            finished = getattr(job, "finished_at", None)
            end = datetime.fromisoformat(str(finished)) if finished else datetime.now(timezone.utc)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            seconds = max(0, int((end - start).total_seconds()))
        except (TypeError, ValueError):
            return "—"
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

    def _update_action_states(self):
        running = bool(self.runner and self.runner.running)
        selected = bool(self.runner and self.selected_ids())
        queued = bool(
            self.runner and any(job.status == "Queued" for job in self.runner.jobs)
        )
        selected_ids = set(self.selected_ids()) if self.runner else set()
        active_id = getattr(getattr(self.runner, "current", None), "id", None)
        removable = bool(selected_ids) and active_id not in selected_ids
        finished = bool(
            self.runner
            and any(
                job.status in {"Completed", "Failed", "Cancelled"}
                for job in self.runner.jobs
            )
        )
        self.add_btn.setEnabled(not running)
        self.add_run_btn.setEnabled(not running)
        self.run_selected_btn.setEnabled(not running and selected)
        self.run_all_btn.setEnabled(not running and queued)
        self.cancel_btn.setEnabled(running)
        self.remove_btn.setEnabled(removable)
        self.clear_btn.setEnabled(finished)
        selected_job = next(
            (
                job
                for job in (self.runner.jobs if self.runner else [])
                if job.id in selected_ids and job.combined_csv_path
            ),
            None,
        )
        self.open_sweep_csv_btn.setEnabled(
            bool(selected_job and Path(str(selected_job.combined_csv_path)).is_file())
        )

    def _run_selected(self):
        selected = self.selected_ids()
        if self.runner and selected:
            self.runner.run(selected)

    def _remove(self):
        if not self.runner:
            return
        try:
            self.runner.remove(self.selected_ids())
        except Exception as exc:
            QMessageBox.warning(self, "Cannot remove job", str(exc))

    def _open_sweep_csv(self):
        if not self.runner:
            return
        selected = set(self.selected_ids())
        job = next(
            (
                item
                for item in self.runner.jobs
                if item.id in selected and item.combined_csv_path
            ),
            None,
        )
        if job is not None:
            try:
                open_path(job.combined_csv_path)
            except Exception as exc:
                QMessageBox.warning(self, "Cannot open sweep CSV", str(exc))

    def _append_log(self, job_id, line):
        if not self.selected_ids() or job_id in self.selected_ids():
            self.log.appendPlainText(line)

    def _show_selected_log(self):
        if not self.runner:
            return
        ids = self.selected_ids()
        job = next((j for j in self.runner.jobs if j.id in ids), None)
        if job and job.log_path and Path(job.log_path).exists():
            try:
                text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")
                self.log.setPlainText(text[-250_000:])
                self.log.verticalScrollBar().setValue(
                    self.log.verticalScrollBar().maximum()
                )
            except Exception:
                pass

    def _running_changed(self, running):
        if running:
            self.runtime_timer.start()
        else:
            self.runtime_timer.stop()
        self.refresh()

    def _toggle_diagnostics(self, visible: bool) -> None:
        self.details_btn.setText(
            "−  Solver details" if visible else "+  Solver details"
        )
        self.diagnostics_panel.setVisible(visible)
        if visible:
            self._show_selected_log()


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


class WindowResizeHandle(QWidget):
    """Narrow frameless-window edge that delegates resizing to the compositor."""

    def __init__(self, window: QMainWindow, edge, *, horizontal: bool):
        super().__init__(window)
        self.host_window = window
        self.edge = edge
        self.horizontal = horizontal
        if horizontal:
            self.setFixedHeight(4)
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setFixedWidth(4)
            self.setCursor(Qt.SizeHorCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            edges = self.edge
            if self.horizontal:
                if event.position().x() <= 10:
                    edges |= Qt.LeftEdge
                elif event.position().x() >= self.width() - 10:
                    edges |= Qt.RightEdge
            handle = self.host_window.windowHandle()
            if handle is not None and handle.startSystemResize(edges):
                event.accept()
                return
        super().mousePressEvent(event)


class WindowTitleBar(QFrame):
    """Dark client-side chrome with native compositor move behavior."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.host_window = window
        self.setObjectName("windowTitleBar")
        self.setFixedHeight(30)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)

        self.minimize_button = self._control("−", "Minimize")
        self.maximize_button = self._control("□", "Maximize")
        self.close_button = self._control("×", "Close", close=True)
        self.minimize_button.clicked.connect(window.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.close_button.clicked.connect(window.close)
        row.addWidget(self.minimize_button)
        row.addWidget(self.maximize_button)
        row.addWidget(self.close_button)

    @staticmethod
    def _control(text: str, accessible_name: str, *, close: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setFixedSize(44, 30)
        button.setFocusPolicy(Qt.NoFocus)
        button.setProperty("windowControl", "close" if close else "standard")
        button.setAccessibleName(accessible_name)
        return button

    def toggle_maximized(self) -> None:
        if self.host_window.isMaximized():
            self.host_window.showNormal()
        else:
            self.host_window.showMaximized()
        self.host_window._sync_window_chrome()

    def sync_state(self) -> None:
        maximized = self.host_window.isMaximized()
        self.maximize_button.setText("❐" if maximized else "□")
        self.maximize_button.setAccessibleName("Restore" if maximized else "Maximize")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            handle = self.host_window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    PAGE_NAMES = [
        "Project",
        "Domain",
        "Network",
        "Physics",
        "Solver",
        "Outputs",
        "Run",
        "Results",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self._initializing = True
        self.setWindowTitle("CASCADE O2 Simulation Studio")
        self.resize(1580, 940)
        self.setMinimumSize(1180, 720)
        self.config = default_project()
        self.project_path: Path | None = None
        self.project_dir = _default_project_directory()
        self.hardware = hardware_info()
        self.runner = JobRunner(self.project_dir, self)
        self._preview_process: QProcess | None = None
        self._preview_seed_signature = ""
        self._preview_seed_response: dict[str, Any] = {}
        self._preview_pending_signature = ""
        self._preview_failed_signature = ""
        self._preview_failure_message = ""
        self._preview_output_buffer = ""
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(180)
        self._status_timer.timeout.connect(self._refresh_status)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._build_ui()
        self._build_menu()
        self._load_pages()
        self._initializing = False
        self._refresh_status()

    def _build_ui(self):
        self.title_bar = WindowTitleBar(self)
        self.app_menu = QMenuBar()
        self.app_menu.setObjectName("appMenu")
        chrome = QFrame()
        chrome.setObjectName("windowChrome")
        chrome_layout = QVBoxLayout(chrome)
        chrome_layout.setContentsMargins(0, 0, 0, 0)
        chrome_layout.setSpacing(0)
        chrome_layout.addWidget(self.title_bar)
        chrome_layout.addWidget(self.app_menu)

        root = FlowBackdrop()
        frame = QGridLayout(root)
        frame.setContentsMargins(0, 0, 0, 0)
        frame.setSpacing(0)
        top_resize = WindowResizeHandle(self, Qt.TopEdge, horizontal=True)
        bottom_resize = WindowResizeHandle(self, Qt.BottomEdge, horizontal=True)
        left_resize = WindowResizeHandle(self, Qt.LeftEdge, horizontal=False)
        right_resize = WindowResizeHandle(self, Qt.RightEdge, horizontal=False)
        self.resize_handles = (
            top_resize,
            bottom_resize,
            left_resize,
            right_resize,
        )
        frame.addWidget(top_resize, 0, 0, 1, 3)
        frame.addWidget(left_resize, 1, 0)
        frame.addWidget(right_resize, 1, 2)
        frame.addWidget(bottom_resize, 2, 0, 1, 3)
        surface = QWidget()
        surface.setObjectName("windowSurface")
        frame.addWidget(surface, 1, 1)
        shell = QVBoxLayout(surface)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(chrome)
        work_area = QWidget()
        shell.addWidget(work_area, 1)
        self.backdrop = root
        outer = QHBoxLayout(work_area)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(216)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 20, 14, 14)
        brand = QLabel("CASCADE")
        brand.setObjectName("brand")
        sub = QLabel("O₂ SIMULATION STUDIO")
        sub.setObjectName("brandSub")
        side.addWidget(brand)
        side.addWidget(sub)
        side.addSpacing(18)
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        self.nav.setIconSize(QSize(28, 28))
        stage_icons = (
            "project",
            "domain",
            "network",
            "physics",
            "solver",
            "outputs",
            "run",
            "results",
        )
        for icon_name, name in zip(stage_icons, self.PAGE_NAMES):
            item = QListWidgetItem(_workflow_icon(icon_name), name)
            item.setSizeHint(QSize(0, 54))
            self.nav.addItem(item)
        side.addWidget(self.nav, 1)
        outer.addWidget(sidebar)

        content = QWidget()
        content_col = QVBoxLayout(content)
        content_col.setContentsMargins(0, 0, 0, 0)
        content_col.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(16, 8, 16, 8)
        self.project_label = QLabel("Unsaved project")
        self.project_label.setStyleSheet(
            f"font-weight:650;color:{Tokens.TEXT_2};"
        )
        self.validation_pill = StatusPill("Checking…")
        self.validation_pill.setToolTip("Click to review setup issues and warnings.")
        self.validation_pill.clicked.connect(self._show_validation_details)
        save = QPushButton("Save project")
        save.setProperty("secondary", True)
        save.clicked.connect(self.save)
        top_row.addWidget(self.project_label)
        top_row.addStretch()
        top_row.addWidget(self.validation_pill)
        top_row.addWidget(save)
        content_col.addWidget(top)

        self.pages = [
            OverviewPage(self.hardware),
            DomainPage(),
            VesselsPage(),
            PhysicsPage(),
            SolverPage(),
            OutputsPage(),
            QueuePage(),
            AnalysisPage(),
        ]

        # The model is the primary workspace.  Setup and result controls live in
        # a narrow contextual inspector rather than competing with the viewport.
        self.workspace_stack = QStackedWidget()
        instrument = QWidget()
        instrument_layout = QHBoxLayout(instrument)
        instrument_layout.setContentsMargins(0, 0, 0, 0)
        instrument_layout.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal)
        self.instrument_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        self.preview = CasePreview()
        splitter.addWidget(self.preview)
        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setMinimumWidth(470)
        inspector.setMaximumWidth(1000)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        self.pages_stack = QStackedWidget()
        self._inspector_page_index = {}
        for page_index in (0, 1, 2, 3, 4, 5, 7):
            self._inspector_page_index[page_index] = self.pages_stack.addWidget(
                self.pages[page_index]
            )
        inspector_layout.addWidget(self.pages_stack)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 720])
        instrument_layout.addWidget(splitter)
        self.workspace_stack.addWidget(instrument)
        self.workspace_stack.addWidget(self.pages[6])
        content_col.addWidget(self.workspace_stack, 1)

        bottom = QFrame()
        bottom.setObjectName("solverStatus")
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setContentsMargins(16, 7, 16, 7)
        bottom_row.setSpacing(18)

        def status_group(key: str, value: str) -> tuple[QWidget, QLabel]:
            group = QWidget()
            column = QVBoxLayout(group)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(1)
            key_label = QLabel(key)
            key_label.setObjectName("statusKey")
            value_label = QLabel(value)
            value_label.setObjectName("statusValue")
            column.addWidget(key_label)
            column.addWidget(value_label)
            return group, value_label

        solve_group, self.status_solver = status_group("ACTIVE RUN", "Idle")
        bottom_row.addWidget(solve_group)
        bottom_row.addStretch()
        self.back_btn = QPushButton("Back")
        self.back_btn.setProperty("secondary", True)
        self.next_btn = QPushButton("Next")
        bottom_row.addWidget(self.back_btn)
        bottom_row.addWidget(self.next_btn)
        content_col.addWidget(bottom)
        outer.addWidget(content, 1)
        self.setCentralWidget(root)
        self.nav.currentRowChanged.connect(self._page_changed)
        self.nav.setCurrentRow(0)
        self.back_btn.clicked.connect(
            lambda: self.nav.setCurrentRow(max(0, self.nav.currentRow() - 1))
        )
        self.next_btn.clicked.connect(
            lambda: self.nav.setCurrentRow(
                min(len(self.pages) - 1, self.nav.currentRow() + 1)
            )
        )
        queue: QueuePage = self.pages[6]
        analysis: AnalysisPage = self.pages[7]
        queue.set_runner(self.runner)
        analysis.set_runner(self.runner)
        analysis.render_requested.connect(self._render_analysis)
        self.preview.view_settings_changed.connect(self._viewer_settings_changed)
        self.preview.result_fields_loaded.connect(analysis.set_render_fields)
        self.preview.selection_changed.connect(analysis.set_selection)
        self.runner.running_changed.connect(self._preview_running_changed)
        self.runner.jobs_changed.connect(self._update_solver_status)
        queue.add_requested.connect(self._enqueue)
        self.pages[2].open_physics_requested.connect(
            lambda: self.nav.setCurrentRow(3)
        )
        self.pages[2].source.currentIndexChanged.connect(self._network_source_changed)
        self.pages[2].topology.currentIndexChanged.connect(self._sync_inlet_condition_count)
        self.pages[2].inlet_count.valueChanged.connect(self._sync_inlet_condition_count)
        self.pages[2].auto_roots.toggled.connect(self._sync_inlet_condition_count)
        self.pages[2].roots.textChanged.connect(self._sync_inlet_condition_count)
        self.pages[3].changed.connect(self._schedule_status_refresh)
        self.pages[5].preview_changed.connect(self._schedule_output_preview_refresh)
        self._wire_live_validation()

    def _build_menu(self):
        file_menu = self.app_menu.addMenu("File")
        actions = [
            ("New project", self.new),
            ("Open project…", self.open),
            ("Save", self.save),
            ("Save as…", self.save_as),
        ]
        for text, slot in actions:
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        run_menu = self.app_menu.addMenu("Run")
        add = QAction("Add current setup to queue", self)
        add.triggered.connect(lambda: self._enqueue(False))
        run_menu.addAction(add)
        run_all = QAction("Run queued simulations", self)
        run_all.triggered.connect(lambda: self.runner.run())
        run_menu.addAction(run_all)

    def _wire_live_validation(self):
        for page in self.pages[:6]:
            for widget in page.findChildren(QComboBox):
                widget.currentIndexChanged.connect(self._schedule_status_refresh)
            for spin_type in (QSpinBox, QDoubleSpinBox):
                for widget in page.findChildren(spin_type):
                    widget.valueChanged.connect(self._schedule_status_refresh)
            for widget in page.findChildren(QCheckBox):
                widget.toggled.connect(self._schedule_status_refresh)
            for widget in page.findChildren(QLineEdit):
                widget.textChanged.connect(self._schedule_status_refresh)
            for widget in page.findChildren(QPlainTextEdit):
                widget.textChanged.connect(self._schedule_status_refresh)

    def _schedule_status_refresh(self, *_):
        if not self._initializing:
            self._status_timer.start()
            # Outputs edits do not alter geometry. Avoid re-entering the WSLg
            # canvas while QTableWidget is changing its cell widgets.
            source = self.sender()
            outputs = self.pages[5] if len(self.pages) > 5 else None
            if outputs is None or source is None or not outputs.isAncestorOf(source):
                self._preview_timer.start()

    def _schedule_output_preview_refresh(self, *_):
        if not self._initializing and self.nav.currentRow() == 5:
            self._preview_timer.start()

    def _load_pages(self):
        for page in self.pages[:6]:
            page.load(self.config)
        self.pages[7].load(self.config)
        self.preview.load_view_settings(self.config)
        self._network_source_changed()
        self._sync_inlet_condition_count()
        self._update_project_label()

    def _update_project_label(self) -> None:
        if self.project_path is None:
            self.project_label.setText("Unsaved project")
            self.project_label.setToolTip("")
            self.project_label.setAccessibleDescription("Unsaved project")
            return

        display_path = self.project_path
        while display_path.suffix:
            display_path = display_path.with_suffix("")
        saved_at = datetime.fromtimestamp(self.project_path.stat().st_mtime).astimezone()
        saved_text = saved_at.strftime("%b %-d, %Y, %-I:%M %p")
        label = f"Project {display_path.name}: Last saved {saved_text}"
        self.project_label.setText(label)
        self.project_label.setToolTip("")
        self.project_label.setAccessibleDescription(label)

    def _network_source_changed(self, *_):
        vessels: VesselsPage = self.pages[2]
        solvers: SolverPage = self.pages[4]
        is_lattice = vessels.source.currentData() == "lattice"
        solvers.set_lattice_mode(is_lattice)
        self._sync_inlet_condition_count()

    def _sync_inlet_condition_count(self, *_):
        if not getattr(self, "pages", None):
            return
        vessels: VesselsPage = self.pages[2]
        physics: PhysicsPage = self.pages[3]
        physics.set_inlet_count(vessels.configured_inlet_count())

    def _collect(self, show_error=True):
        updated = deepcopy(self.config)
        try:
            self._sync_inlet_condition_count()
            for page in self.pages[:6]:
                page.write(updated)
            self.pages[7].write(updated)
            updated.setdefault("gui", {})["viewer"] = self.preview.view_settings()
            if updated.get("gui", {}).get("network_source") == "lattice":
                if updated.setdefault("simulation", {}).get("concentration_solver") not in {
                    "network_ext",
                    "network",
                    "network_ext_hybrid_bg",
                }:
                    updated["simulation"]["concentration_solver"] = "network_ext"
                updated.setdefault("settings", {}).setdefault("hemodynamics", {})[
                    "kirchhoff_solver"
                ] = "spsolve"
        except Exception as exc:
            if show_error:
                QMessageBox.warning(self, "Check the current setup", str(exc))
            return None
        self.config = updated
        return updated

    def _page_changed(self, index):
        if index == 6:
            self.workspace_stack.setCurrentIndex(1)
        else:
            self.workspace_stack.setCurrentIndex(0)
            self.pages_stack.setCurrentIndex(self._inspector_page_index[index])
        self.back_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.pages) - 1)
        self.next_btn.setText(
            f"Continue to {self.PAGE_NAMES[index + 1]}"
            if index < len(self.pages) - 1
            else "Workflow complete"
        )
        if index == 6:
            self.preview.release()
        elif index == 7:
            self.pages[7]._request_render()
        else:
            self._preview_timer.start()
        if not self._initializing:
            self._refresh_status()

    def _refresh_preview(self):
        index = self.nav.currentRow()
        if index < 0 or index in {6, 7}:
            return
        config = self._collect(show_error=False)
        if config is not None:
            self.preview.show_case(
                config,
                include_network=index >= 2,
                include_tissue=index == 5,
            )
            source = config.get("gui", {}).get("network_source")
            if index >= 2 and source in {"svv_generated", "uploaded"}:
                signature = self._case_preview_signature(config)
                response = self._preview_seed_response
                geometry = response.get("geometry_path")
                if (
                    signature == self._preview_seed_signature
                    and geometry
                    and Path(str(geometry)).exists()
                ):
                    self.preview.show_seed(
                        config,
                        str(geometry),
                        response,
                        include_tissue=index == 5,
                    )
                elif signature == self._preview_failed_signature:
                    self.preview.status.setText(self._preview_failure_message)
                else:
                    self._start_preview_seed(config, signature)

    def _render_analysis(self, manifest_path, options):
        if self.nav.currentRow() == 7:
            self.preview.show_result(manifest_path, options)

    def _viewer_settings_changed(self):
        if self._initializing:
            return
        index = self.nav.currentRow()
        if index == 7:
            self.pages[7]._request_render()
        elif index not in {6}:
            self._preview_timer.start()

    def _preview_running_changed(self, running):
        self.backdrop.set_animation_enabled(running)
        self._update_solver_status()
        if running:
            self.preview.release()
        elif self.nav.currentRow() not in {6, 7}:
            self._preview_timer.start()
        elif self.nav.currentRow() == 7:
            self.pages[7]._request_render()

    def _update_solver_status(self):
        """Show only the global state that remains useful across pages."""
        job = self.runner.current
        if job is None:
            self.status_solver.setText("Idle")
        else:
            stage = str(job.stage or "Running")
            if len(stage) > 32:
                stage = stage[:31] + "…"
            self.status_solver.setText(f"{stage}  │  {int(job.progress or 0)}%")

    @staticmethod
    def _case_preview_signature(config):
        simulation = config.get("simulation", {})
        settings = config.get("settings", {})
        gui = config.get("gui", {})
        relevant = {
            "preview_version": 4,
            "domain": deepcopy(config.get("domain", {})),
            "network": deepcopy(config.get("network", {})),
            "growth": deepcopy(config.get("growth", {})),
            "simulation": {
                key: simulation.get(key)
                for key in (
                    "build_fluid",
                    "fluid",
                    "qin_target_ul_min",
                    "flow_source",
                    "inlet_conditions",
                )
            },
            "settings": {
                "hemodynamics": deepcopy(settings.get("hemodynamics", {})),
                "hematocrit": deepcopy(settings.get("hematocrit", {})),
            },
            "gui": {
                "network_source": gui.get("network_source"),
                "sweeps": [
                    deepcopy(sweep)
                    for sweep in gui.get("sweeps", [])
                    if sweep.get("path")
                    == "network.target_terminal_count"
                ],
            },
        }
        if gui.get("network_source") == "svv_generated":
            relevant["network"].pop("input_path", None)
        payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _start_preview_seed(self, config, signature):
        if self._preview_process is not None:
            return
        if signature == self._preview_failed_signature:
            return
        preview_dir = self.project_dir / ".cascade_gui" / "previews" / signature[:16]
        response_path = preview_dir / "response.json"
        if response_path.exists():
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if Path(str(response.get("geometry_path", ""))).exists() and Path(
                    str(response.get("seed_path", ""))
                ).exists():
                    self._preview_seed_signature = signature
                    self._preview_seed_response = response
                    self._preview_failed_signature = ""
                    self._preview_failure_message = ""
                    self.pages[2].set_radius_summary(response)
                    self._refresh_preview()
                    return
            except Exception:
                pass
        preview_dir.mkdir(parents=True, exist_ok=True)
        request = preview_dir / "request.json"
        preview_config = deepcopy(config)
        if preview_config.get("gui", {}).get("network_source") == "uploaded":
            raw_path = preview_config.get("network", {}).get("input_path")
            if raw_path:
                path = Path(str(raw_path)).expanduser()
                if not path.is_absolute():
                    preview_config["network"]["input_path"] = str(
                        (self.project_dir / path).resolve()
                    )
        request.write_text(json.dumps(preview_config, indent=2), encoding="utf-8")
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        process.readyReadStandardOutput.connect(self._preview_seed_progress)
        process.finished.connect(
            lambda code, status, sig=signature, folder=preview_dir: self._preview_seed_finished(
                code, sig, folder
            )
        )
        self._preview_process = process
        self._preview_pending_signature = signature
        self._preview_output_buffer = ""
        source = config.get("gui", {}).get("network_source")
        if source == "svv_generated":
            self.pages[2].set_radius_summary(calculating=True)
        uploaded_domain = str(config.get("domain", {}).get("type", "cube")) not in {
            "cube",
            "box",
            "sphere",
        }
        self.preview.status.setText(
            "Preparing the uploaded domain for preview…"
            if uploaded_domain
            else (
                "Growing reusable preview seed  │  up to about 1,000 vessel segments total…"
                if source == "svv_generated"
                else "Loading a bounded network preview…"
            )
        )
        process.start(
            sys.executable,
            [
                "-m",
                "cascade.gui.preview_worker",
                "--request",
                str(request),
                "--output",
                str(preview_dir),
            ],
        )

    def _cancel_preview_seed(self) -> None:
        process = self._preview_process
        self._preview_process = None
        self._preview_pending_signature = ""
        self._preview_output_buffer = ""
        if process is None:
            return
        try:
            process.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        if process.state() != QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1500)
        process.deleteLater()

    def _preview_seed_progress(self) -> None:
        process = self._preview_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if not chunk:
            return
        self._preview_output_buffer += chunk
        normalized = self._preview_output_buffer[-4_000:].replace("\r", "\n")
        percentages = re.findall(r"Adding vessels:\s*(\d+)%", normalized)
        if percentages:
            self.preview.status.setText(
                f"Growing reusable preview seed  │  {percentages[-1]}%"
            )
        elif "Preview domain ready:" in normalized:
            self.preview.status.setText(
                "Uploaded domain prepared  │  growing reusable preview seed…"
            )
        elif "repair" in normalized.lower() or "tetra" in normalized.lower():
            self.preview.status.setText(
                "Repairing and volume-meshing the uploaded domain…"
            )

    def _preview_seed_finished(self, exit_code, signature, preview_dir):
        process = self._preview_process
        output = self._preview_output_buffer
        if process is not None:
            output += bytes(process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            )
            process.deleteLater()
        self._preview_process = None
        self._preview_pending_signature = ""
        self._preview_output_buffer = ""
        log_path = Path(preview_dir) / "preview.log"
        try:
            log_path.write_text(output, encoding="utf-8")
        except OSError:
            pass
        response_path = Path(preview_dir) / "response.json"
        if int(exit_code) == 0 and response_path.exists():
            try:
                self._preview_seed_response = json.loads(
                    response_path.read_text(encoding="utf-8")
                )
                self._preview_seed_signature = signature
                self._preview_failed_signature = ""
                self._preview_failure_message = ""
                self.pages[2].set_radius_summary(self._preview_seed_response)
            except Exception:
                self.preview.status.setText("preview seed unreadable")
        else:
            self._preview_failed_signature = signature
            summary = self._preview_failure_summary(output)
            self._preview_failure_message = f"Preview seed failed  │  {summary}"
            self.preview.status.setText(self._preview_failure_message)
            self.pages[2].radius_summary.setText(
                f"Radii could not be calculated. {summary}"
            )
        if self.nav.currentRow() not in {6, 7}:
            self._refresh_preview()

    @staticmethod
    def _preview_failure_summary(output: str) -> str:
        text = str(output or "")
        lowered = text.lower()
        if "tetgen" in lowered or "tetrahedraliz" in lowered:
            if "after automatic manifold repair" in lowered:
                return "The uploaded domain could not be volume-meshed after automatic repair."
            return "The uploaded domain could not be volume-meshed."
        meaningful = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and "unknown exception" not in line.lower()
            and not line.lstrip().startswith("Traceback")
        ]
        for line in reversed(meaningful):
            if line.startswith(("ValueError:", "RuntimeError:")):
                return line.split(":", 1)[1].strip()
        return meaningful[-1][-180:] if meaningful else "The seed worker stopped unexpectedly."

    def _refresh_status(self):
        config = self._collect(show_error=False)
        if config is None:
            self.validation_pill.set_status("Needs attention", "danger")
            self.validation_pill.setToolTip(
                "The current page contains a value that cannot be read. Click for details."
            )
            return
        report = validate_project(config)
        self._validation_report = report
        estimate = estimate_resources(config, self.hardware)
        if report.errors:
            self.validation_pill.set_status(
                f"{len(report.errors)} issue{'s' if len(report.errors) != 1 else ''}",
                "danger",
            )
        elif report.warnings:
            self.validation_pill.set_status(
                f"Ready  │  {len(report.warnings)} warning{'s' if len(report.warnings) != 1 else ''}",
                "warning",
            )
        else:
            self.validation_pill.set_status("Ready to queue", "success")
        details = self._validation_details(report)
        self.validation_pill.setToolTip(details)
        self.validation_pill.setAccessibleDescription(details)
        self._update_solver_status()
        self.pages[5].set_resource(estimate)

    @staticmethod
    def _validation_details(report) -> str:
        sections = []
        for title, messages in (
            ("Issues", report.errors),
            ("Warnings", report.warnings),
        ):
            if messages:
                sections.append(title + ":\n" + "\n".join(f"• {m}" for m in messages))
        return "\n\n".join(sections) or "Fully specified."

    def _show_validation_details(self) -> None:
        config = self._collect(show_error=False)
        if config is None:
            QMessageBox.warning(
                self,
                "Setup needs attention",
                "A value on the current page cannot be read. Review highlighted inputs or try queueing to reveal the exact field error.",
            )
            return
        report = validate_project(config)
        text = self._validation_details(report)
        if report.errors:
            QMessageBox.warning(self, "Setup issues", text)
        elif report.warnings:
            QMessageBox.information(self, "Setup warnings", text)
        else:
            QMessageBox.information(self, "Setup status", text)

    def _enqueue(self, run_now=False):
        config = self._collect()
        if config is None:
            return
        if config.get("gui", {}).get("network_source") == "svv_generated":
            signature = self._case_preview_signature(config)
            seed_path = self._preview_seed_response.get("seed_path")
            seed_ready = not (
                signature != self._preview_seed_signature
                or not seed_path
                or not Path(str(seed_path)).exists()
            )
            if seed_ready:
                config.setdefault("network", {})["input_path"] = str(seed_path)
                config.setdefault("gui", {})["preview_seed"] = {
                    "signature": signature,
                    "path": str(seed_path),
                }
            else:
                # Do not make a production run wait for a cosmetic preview.
                # The worker grows directly from the same roots and random seed.
                self._cancel_preview_seed()
                config.setdefault("network", {}).pop("input_path", None)
                config.setdefault("gui", {}).pop("preview_seed", None)
            config.setdefault("growth", {})["enabled"] = True
            config["growth"]["resume_from_checkpoint"] = False
        report = validate_project(config)
        if report.errors:
            QMessageBox.warning(
                self, "Cannot queue this setup", "\n\n".join(report.errors)
            )
            return
        if report.warnings:
            answer = QMessageBox.question(
                self,
                "Queue with warnings?",
                "\n\n".join(report.warnings) + "\n\nAdd the run anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.project_dir.mkdir(parents=True, exist_ok=True)
        if self.project_path is None:
            self.project_path = self.project_dir / "project.cascade.json"
        save_project(self.project_path, config)
        records = create_jobs(config, self.project_dir)
        self.runner.add(records)
        self._update_project_label()
        self.nav.setCurrentRow(6)
        if run_now:
            self.runner.run([job.id for job in records])

    def new(self):
        if self.runner.running:
            QMessageBox.warning(
                self,
                "Simulation running",
                "Cancel the active simulation before changing projects.",
            )
            return
        self.config = default_project()
        self.project_path = None
        self.project_dir = _default_project_directory()
        self.runner.replace_project(self.project_dir)
        self._load_pages()
        self.nav.setCurrentRow(0)
        self._refresh_status()

    def open(self):
        if self.runner.running:
            QMessageBox.warning(
                self,
                "Simulation running",
                "Cancel the active simulation before changing projects.",
            )
            return
        selected = choose_native_path(
            self,
            mode="file",
            caption="Open CASCADE project",
            start=str(self.project_dir),
            file_filter="CASCADE project (*.json *.cascade.json);;All JSON (*.json)",
        )
        if not selected:
            return
        try:
            self.config = load_project(selected)
            self.project_path = Path(selected).resolve()
            self.project_dir = self.project_path.parent
            self.runner.replace_project(self.project_dir)
            self._load_pages()
            self._refresh_status()
        except Exception as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))

    def save(self):
        if self.project_path is None:
            return self.save_as()
        config = self._collect()
        if config is None:
            return
        try:
            save_project(self.project_path, config)
            self._update_project_label()
            self.statusBar().showMessage("Project saved", 2500)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))

    def save_as(self):
        selected = choose_native_path(
            self,
            mode="save",
            caption="Save CASCADE project",
            start=str(self.project_dir / "project.cascade.json"),
            file_filter="CASCADE project (*.cascade.json);;JSON (*.json)",
        )
        if not selected:
            return
        self.project_path = Path(selected).resolve()
        self.project_dir = self.project_path.parent
        try:
            self.runner.replace_project(self.project_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot move project", str(exc))
            return
        self.save()

    def _sync_window_chrome(self):
        self.title_bar.sync_state()
        show_resize_handles = not (self.isMaximized() or self.isFullScreen())
        for handle in self.resize_handles:
            handle.setVisible(show_resize_handles)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange and hasattr(self, "title_bar"):
            self._sync_window_chrome()

    def closeEvent(self, event):
        cancel_native_pickers()
        self._cancel_preview_seed()
        if self.runner.running:
            answer = QMessageBox.question(
                self,
                "Simulation is running",
                "Cancel the active worker and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.cancel_pending()
            self.runner.cancel_current()
        event.accept()


def _parse_jsonish(text: str) -> Any:
    value = text.strip()
    try:
        return json.loads(value)
    except Exception:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value


def _json_safe(value):
    try:
        import numpy as np

        if value in (np.float32, np.float64, np.int32, np.int64):
            return np.dtype(value).name
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("CASCADE O2 Simulation Studio")
    app.setOrganizationName("CASCADE")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
