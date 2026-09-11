"""Lightweight in-window case and result preview.

The canvas deliberately uses Qt painting instead of an embedded VTK render window.
That keeps preview memory small, avoids WSL compositor issues, and allows the main
application to release all geometry before a simulation starts.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QOffscreenSurface,
    QOpenGLContext,
    QSurfaceFormat,
    QIntValidator,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cascade.vessels.lattice import channel_count, generate_lattice
from cascade.utils.resources import resolve_domain_path

from ..widgets import ChoiceComboBox, IconButton


COLORMAPS = ("viridis", "magma", "plasma", "inferno", "coolwarm", "jet", "gray")

_MAP_STOPS = {
    "viridis": ((68, 1, 84), (49, 104, 142), (53, 183, 121), (253, 231, 37)),
    "magma": ((0, 0, 4), (91, 22, 126), (221, 73, 104), (252, 253, 191)),
    "plasma": ((13, 8, 135), (156, 23, 158), (237, 121, 83), (240, 249, 33)),
    "inferno": ((0, 0, 4), (87, 16, 110), (220, 81, 57), (252, 255, 164)),
    "coolwarm": ((59, 76, 192), (141, 176, 254), (244, 152, 122), (180, 4, 38)),
    "jet": ((0, 0, 128), (0, 220, 255), (255, 235, 0), (128, 0, 0)),
    "gray": ((30, 34, 42), (105, 112, 125), (185, 190, 199), (250, 250, 250)),
}




__all__ = ()
