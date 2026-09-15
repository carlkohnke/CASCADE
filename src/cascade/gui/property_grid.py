"""Compact cell-based property inspector widgets for CASCADE Studio pages."""

from __future__ import annotations

import re

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


_LABEL_WIDTH = 270
_UNIT_WIDTH = 128
_UNIT_PATTERN = re.compile(
    r"^(?P<label>.+?)\s*\((?P<unit>cm|mm|m|cm²/s|m²/s|mmHg|Pa|cP|g/cm³|"
    r"µL/min|mL/min|m³/s|mol/m³|mol/m³/s|mmHg/s|%|count|points|iterations)\)$"
)


_GRID_STYLE = """
QFrame#propertyGridSection {
    background: #08090c;
    border: 1px solid #343740;
    border-radius: 1px;
}
QWidget#propertyGridHeader {
    background: #090a0d;
    border: 0;
    border-bottom: 1px solid #41444e;
}
QLabel#propertyGridTitle {
    color: #f2f2f5;
    font-weight: 700;
    padding: 9px 12px;
}
QFrame#propertyGridRow {
    background: #101217;
    border: 0;
    border-bottom: 1px solid #2b2e36;
    border-radius: 0;
}
QFrame#propertyGridRow[selected="true"] {
    background: #1b1422;
}
QFrame#propertyGridRow[selected="true"] {
    border-left: 2px solid #a966d6;
}
QLabel#propertyGridLabel {
    color: #ffffff;
    background: transparent;
    border: 0;
    border-right: 1px solid #2b2e36;
    padding: 7px 11px;
}
QLabel#propertyGridUnit {
    color: #ffffff;
    background: #0c0e12;
    border: 0;
    border-left: 1px solid #2b2e36;
    padding: 0 10px;
}
QFrame#propertyGridRow QLineEdit,
QFrame#propertyGridRow QAbstractSpinBox,
QFrame#propertyGridRow QComboBox,
QFrame#propertyGridRow QPlainTextEdit,
QFrame#propertyGridRow QTextEdit,
QFrame#propertyGridRow QCheckBox {
    color: #ffffff;
    background: #0c0e12;
    border: 0;
    border-radius: 0;
    padding: 7px 10px;
    min-height: 24px;
}
QFrame#propertyGridRow QLineEdit:focus,
QFrame#propertyGridRow QAbstractSpinBox:focus,
QFrame#propertyGridRow QComboBox:focus,
QFrame#propertyGridRow QPlainTextEdit:focus,
QFrame#propertyGridRow QTextEdit:focus {
    background: #191120;
    border: 1px solid #a966d6;
}
QFrame#propertyGridRow QWidget#propertyGridUnitValue QComboBox {
    border-left: 1px solid #2b2e36;
}
QFrame#propertyGridRow QLineEdit:disabled,
QFrame#propertyGridRow QAbstractSpinBox:disabled,
QFrame#propertyGridRow QComboBox:disabled,
QFrame#propertyGridRow QPlainTextEdit:disabled,
QFrame#propertyGridRow QTextEdit:disabled,
QFrame#propertyGridRow QCheckBox:disabled {
    color: #777984;
    background: #0a0b0e;
}
QFrame#propertyGridRow QLabel#propertyGridLabel:disabled,
QFrame#propertyGridRow QLabel#propertyGridUnit:disabled {
    color: #777984;
    background: #0a0b0e;
}
QFrame#propertyGridInfo {
    background: #0c0e12;
    border: 0;
    border-bottom: 1px solid #2b2e36;
}
QFrame#propertyGridInfo QLabel {
    color: #a9a6b0;
    padding: 8px 11px;
}
QWidget#propertyGridGroup {
    background: transparent;
}
"""


def _refresh_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class PropertyGridRow(QFrame):
    """One aligned label/value/units row with full-cell editors."""

    def __init__(
        self,
        label: str,
        widget: QWidget,
        help_text: str = "",
        unit: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("propertyGridRow")
        self.setProperty("selected", False)
        self.setMinimumHeight(42)
        self.source_widget = widget
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Ignored, widget.sizePolicy().verticalPolicy())
        for editor in widget.findChildren(QWidget):
            if isinstance(editor, QAbstractButton):
                editor.setMinimumWidth(editor.sizeHint().width())
            else:
                editor.setMinimumWidth(0)

        parsed = _UNIT_PATTERN.match(label.strip())
        if parsed and not unit:
            label = parsed.group("label")
            unit = parsed.group("unit")

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        grid.setColumnMinimumWidth(0, _LABEL_WIDTH)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, _UNIT_WIDTH)

        label_widget = QLabel(label)
        label_widget.setObjectName("propertyGridLabel")
        label_widget.setWordWrap(True)
        label_widget.setMinimumWidth(_LABEL_WIDTH)
        label_widget.setMaximumWidth(_LABEL_WIDTH)
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(label_widget, 0, 0)
        self.label_widget = label_widget

        if _is_unit_value(widget):
            widget.setObjectName("propertyGridUnitValue")
            widget.layout().setContentsMargins(0, 0, 0, 0)
            widget.layout().setSpacing(0)
            widget.combo.setFixedWidth(_UNIT_WIDTH)
            grid.addWidget(widget, 0, 1, 1, 2)
        elif unit:
            grid.addWidget(widget, 0, 1)
            unit_widget = QLabel(unit)
            unit_widget.setObjectName("propertyGridUnit")
            unit_widget.setMinimumWidth(_UNIT_WIDTH)
            unit_widget.setMaximumWidth(_UNIT_WIDTH)
            unit_widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(unit_widget, 0, 2)
        else:
            grid.addWidget(widget, 0, 1, 1, 2)

        if help_text:
            self.setToolTip(help_text)
            label_widget.setToolTip(help_text)

        self._install_selection_filter(widget)

    def _install_selection_filter(self, widget: QWidget) -> None:
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in {QEvent.FocusIn, QEvent.MouseButtonPress}:
            self.select()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        if event.position().x() < _LABEL_WIDTH:
            self.clear_selection()
        super().mousePressEvent(event)

    def select(self) -> None:
        self.clear_selection()
        self.setProperty("selected", True)
        _refresh_style(self)

    def clear_selection(self) -> None:
        for row in self.window().findChildren(PropertyGridRow):
            if row.property("selected"):
                row.setProperty("selected", False)
                _refresh_style(row)


class PropertyGridInfo(QFrame):
    def __init__(self, widget: QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("propertyGridInfo")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(widget)

    def mousePressEvent(self, event):
        for row in self.window().findChildren(PropertyGridRow):
            if row.property("selected"):
                row.setProperty("selected", False)
                _refresh_style(row)
        super().mousePressEvent(event)


class CompactStack(QStackedWidget):
    """A conditional panel whose height follows only its visible page."""

    def sizeHint(self):
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self):
        current = self.currentWidget()
        return (
            current.minimumSizeHint()
            if current is not None
            else super().minimumSizeHint()
        )

    def sync_height(self) -> None:
        current = self.currentWidget()
        if current is None:
            return
        if current.layout() is not None:
            current.layout().activate()
        height = max(current.sizeHint().height(), current.minimumSizeHint().height())
        self.setFixedHeight(max(height, 0))
        self.updateGeometry()

    def setCurrentIndex(self, index):
        super().setCurrentIndex(index)
        self.sync_height()


class PropertyGridGroup(QWidget):
    def __init__(self, widgets, parent=None):
        super().__init__(parent)
        self.setObjectName("propertyGridGroup")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for widget in widgets:
            layout.addWidget(_as_property_widget(widget))


class Card(QFrame):
    """A section header followed by a contiguous stack of property rows."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("propertyGridSection")
        self.setStyleSheet(_GRID_STYLE)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        heading_widget = QWidget()
        heading_widget.setObjectName("propertyGridHeader")
        self.heading_row = QHBoxLayout(heading_widget)
        self.heading_row.setContentsMargins(0, 0, 8, 0)
        self.heading_row.setSpacing(6)
        title_widget = QLabel(title)
        title_widget.setObjectName("propertyGridTitle")
        if subtitle:
            title_widget.setToolTip(subtitle)
            self.setToolTip(subtitle)
        self.heading_row.addWidget(title_widget)
        self.heading_row.addStretch(1)
        self.layout.addWidget(heading_widget)
        heading_widget.setVisible(bool(title))

    def add(self, widget: QWidget) -> None:
        if self._flatten_checkbox_panel(widget):
            return
        self.layout.addWidget(_as_property_widget(widget))

    def _flatten_checkbox_panel(self, widget: QWidget) -> bool:
        layout = widget.layout()
        if not isinstance(layout, QGridLayout) or layout.count() == 0:
            return False
        controls = [layout.itemAt(i).widget() for i in range(layout.count())]
        if not all(isinstance(control, QCheckBox) for control in controls):
            return False
        for control in controls:
            label = control.text()
            control.setText("Enabled")
            self.layout.addWidget(PropertyGridRow(label, control))
        widget.setParent(None)
        return True

    def mousePressEvent(self, event):
        for row in self.window().findChildren(PropertyGridRow):
            if row.property("selected"):
                row.setProperty("selected", False)
                _refresh_style(row)
        super().mousePressEvent(event)


def _is_unit_value(widget: QWidget) -> bool:
    return (
        hasattr(widget, "spin")
        and hasattr(widget, "combo")
        and isinstance(widget.combo, QComboBox)
    )


def _as_property_widget(widget: QWidget) -> QWidget:
    if isinstance(widget, (PropertyGridRow, PropertyGridGroup, PropertyGridInfo)):
        return widget
    if isinstance(widget, QCheckBox):
        label = widget.text()
        widget.setText("Enabled")
        return PropertyGridRow(label, widget)
    return PropertyGridInfo(widget)


def labeled(
    text: str,
    widget: QWidget,
    help_text: str = "",
    *,
    important: bool = False,
) -> PropertyGridRow:
    del important
    return PropertyGridRow(text, widget, help_text)


def row_of(*widgets: QWidget) -> QWidget:
    if widgets and all(
        isinstance(widget, (QAbstractButton, QPushButton))
        and not isinstance(widget, QCheckBox)
        for widget in widgets
    ):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        return panel
    return PropertyGridGroup(widgets)


__all__ = ("Card", "CompactStack", "PropertyGridRow", "labeled", "row_of")
