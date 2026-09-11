"""Base class for scrollable CASCADE GUI pages."""

from __future__ import annotations

from cascade.gui._common import (
    Any,
    QFrame,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
    title_label,
)

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




__all__ = ('Page',)
