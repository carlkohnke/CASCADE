"""Project overview page."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.model import human_bytes
from cascade.gui.theme import Tokens
from cascade.gui.widgets import Card as LegacyCard

from cascade.gui.pages.base import (
    Page,
)


class OverviewPage(Page):
    def __init__(self, hardware, parent=None):
        super().__init__(
            "Project overview",
            "",
            parent,
        )
        intro = Card("")
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Perfused construct oxygen sweep")
        intro.add(labeled("Project name", self.name, important=True))
        self.column.addWidget(intro)

        hw = LegacyCard("System specifications")
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

    def write(self, config):
        config.setdefault("gui", {})["project_name"] = (
            self.name.text().strip() or "Untitled CASCADE project"
        )


__all__ = ("OverviewPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, labeled, row_of
