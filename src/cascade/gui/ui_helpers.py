"""Shared form controls, dialogs, and drawn icons for CASCADE Studio."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from PySide6.QtWidgets import QMessageBox as QtMessageBox

from cascade.runtime.paths import project_directory

from .theme import Tokens, stylesheet
from .widgets import (
    NumberInput,
    Select,
)


def _default_project_directory() -> Path:
    return project_directory()


class QMessageBox:
    """CASCADE-styled replacement for Qt's native message dialogs.

    Native Windows message boxes retain a light title bar even when the
    application is dark.  Using a frameless Qt dialog keeps every alert and
    confirmation visually part of the studio.
    """

    Yes = QtMessageBox.Yes
    No = QtMessageBox.No

    @staticmethod
    def _show(
        parent, icon, title: str, text: str, buttons=QtMessageBox.Ok, default=None
    ):
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
        selected = [
            default
            or (QtMessageBox.No if buttons & QtMessageBox.No else QtMessageBox.Ok)
        ]
        for label, answer in choices:
            button = QPushButton(label)
            button.setDefault(answer == selected[0])
            button.clicked.connect(
                lambda _checked=False, choice=answer: (
                    selected.__setitem__(0, choice),
                    dialog.accept(),
                )
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
            painter.drawLine(QPointF(center - 3.5, 9.0), QPointF(center + 3.5, 9.0))
            painter.drawLine(QPointF(center, 5.5), QPointF(center, 12.5))
        if kind == "add_run":
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(
                QPolygonF((QPointF(12.0, 4.5), QPointF(19.0, 9.0), QPointF(12.0, 13.5)))
            )
        elif kind == "play_circle":
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(9.0, 9.0), 7.0, 7.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygonF((QPointF(7.5, 5.5), QPointF(13.0, 9.0), QPointF(7.5, 12.5)))
            )
        elif kind == "fast_forward":
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(
                QPolygonF((QPointF(2.0, 4.5), QPointF(9.0, 9.0), QPointF(2.0, 13.5)))
            )
            painter.drawPolygon(
                QPolygonF((QPointF(9.0, 4.5), QPointF(16.0, 9.0), QPointF(9.0, 13.5)))
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
                (
                    QPointF(4.0, 7.0),
                    QPointF(10.0, 3.5),
                    QPointF(16.0, 7.0),
                    QPointF(16.0, 14.0),
                    QPointF(10.0, 17.5),
                    QPointF(4.0, 14.0),
                )
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
            for point in (
                QPointF(5.0, 5.0),
                QPointF(10.0, 10.0),
                QPointF(15.0, 5.0),
                QPointF(15.0, 15.0),
            ):
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
            painter.drawPolygon(
                QPolygonF((QPointF(8.0, 6.5), QPointF(14.0, 10.0), QPointF(8.0, 13.5)))
            )
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


__all__ = (
    "_default_project_directory",
    "QMessageBox",
    "_combo",
    "_spin",
    "_double",
    "_optional_float",
    "_concentration_unit",
    "_queue_action_icon",
    "_workflow_icon",
    "_set_combo",
    "_value",
)
