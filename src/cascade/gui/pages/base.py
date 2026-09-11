"""Base class for scrollable CASCADE GUI pages."""

from __future__ import annotations

from PySide6.QtCore import (
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.widgets import title_label
from typing import Any


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
        """Populate page controls from ``config``; subclasses override as needed."""

    def write(self, config: dict[str, Any]) -> None:
        """Store page controls in ``config``; subclasses override as needed."""


__all__ = ("Page",)
