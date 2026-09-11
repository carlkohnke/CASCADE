"""CASCADE desktop GUI entry point."""

from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication
from cascade.gui.ui_helpers import (
    APP_STYLE,
    QMessageBox,
    _combo,
    _concentration_unit,
    _default_project_directory,
    _double,
    _optional_float,
    _queue_action_icon,
    _set_combo,
    _spin,
    _value,
    _workflow_icon,
)

from cascade.gui.helpers import (
    _json_safe,
    _parse_jsonish,
)

from cascade.gui.window import (
    MainWindow,
    WindowResizeHandle,
    WindowTitleBar,
)

from cascade.gui.pages.base import (
    Page,
)

from cascade.gui.pages.overview import (
    OverviewPage,
)

from cascade.gui.pages.domain import (
    DomainPage,
)

from cascade.gui.pages.vessels import (
    VesselsPage,
)

from cascade.gui.pages.physics import (
    PhysicsPage,
)

from cascade.gui.pages.solver import (
    SolverPage,
)

from cascade.gui.pages.outputs import (
    OutputsPage,
)

from cascade.gui.pages.queue import (
    QueuePage,
)

from cascade.gui.pages.analysis import (
    AnalysisPage,
)


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
    "Page",
    "OverviewPage",
    "DomainPage",
    "VesselsPage",
    "PhysicsPage",
    "SolverPage",
    "OutputsPage",
    "QueuePage",
    "AnalysisPage",
    "WindowResizeHandle",
    "WindowTitleBar",
    "MainWindow",
    "_parse_jsonish",
    "_json_safe",
    "main",
)
