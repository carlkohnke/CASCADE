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
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cascade.lattice import channel_count, generate_lattice
from cascade.resources import resolve_domain_path

from .widgets import ChoiceComboBox, IconButton


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


class FlowBackdrop(QWidget):
    """Dark application surface with restrained animated data-flow edges."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("root")
        self._phase = 0.0
        self._reduced_motion = os.environ.get("CASCADE_REDUCED_MOTION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._edges = [_FlowEdge(side, self) for side in range(4)]
        self._timer = QTimer(self)
        self._timer.setInterval(850)
        self._timer.timeout.connect(self._advance)
        for edge in self._edges:
            edge.hide()

    def set_animation_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled) and not self._reduced_motion
        if enabled and not self._timer.isActive():
            for edge in self._edges:
                edge.show()
            self._timer.start()
        elif not enabled:
            self._timer.stop()
            for edge in self._edges:
                edge.hide()

    def _advance(self) -> None:
        if not self.window().isActiveWindow():
            return
        self._phase = (self._phase + 0.72) % 1000.0
        for edge in self._edges:
            edge.phase = self._phase
            edge.raise_()
            edge.update()

    def resizeEvent(self, event) -> None:
        width = 7
        self._edges[0].setGeometry(0, 0, self.width(), width)
        self._edges[1].setGeometry(self.width() - width, 0, width, self.height())
        self._edges[2].setGeometry(0, self.height() - width, self.width(), width)
        self._edges[3].setGeometry(0, 0, width, self.height())
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#08090c"))
        gradient.setColorAt(0.62, QColor("#09090d"))
        gradient.setColorAt(1.0, QColor("#0b090d"))
        painter.fillRect(self.rect(), gradient)


class _FlowEdge(QWidget):
    """Tiny animated strip; avoids repainting the full application surface."""

    def __init__(self, side: int, parent: QWidget):
        super().__init__(parent)
        self.side = side
        self.phase = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Keep the animated strips opaque.  Transparent child widgets force Qt
        # to repaint the full window behind every frame, which is needlessly
        # expensive under WSL's compositor.
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08090c"))
        painter.setRenderHint(QPainter.Antialiasing)
        horizontal = self.side in {0, 2}
        coordinate = self.height() * 0.5 if horizontal else self.width() * 0.5
        for color, width, offset, pattern in (
            (QColor(106, 56, 194, 112), 1.3, 0.0, [1.0, 15.0]),
            (QColor(200, 60, 152, 78), 1.7, 23.0, [1.0, 23.0]),
            (QColor(241, 106, 60, 60), 1.0, 51.0, [2.0, 31.0]),
        ):
            pen = QPen(color, width)
            pen.setDashPattern(pattern)
            direction = -1.0 if self.side in {0, 1} else 1.0
            pen.setDashOffset(direction * (self.phase + offset))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            if horizontal:
                painter.drawLine(0, int(coordinate), self.width(), int(coordinate))
            else:
                painter.drawLine(int(coordinate), 0, int(coordinate), self.height())


class GeometryCanvas(QWidget):
    """Small-memory mouse-navigable 3D line and point renderer."""

    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(300, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.domain_lines = np.empty((0, 2, 3), dtype=np.float32)
        self.domain_triangles = np.empty((0, 3, 3), dtype=np.float32)
        self.domain_mode = "wireframe"
        self.vessel_starts = np.empty((0, 3), dtype=np.float32)
        self.vessel_ends = np.empty((0, 3), dtype=np.float32)
        self.vessel_values: np.ndarray | None = None
        self.vessel_radii: np.ndarray | None = None
        self.vessel_alpha = np.empty((0,), dtype=np.float32)
        self.inlet_points = np.empty((0, 3), dtype=np.float32)
        self.outlet_points = np.empty((0, 3), dtype=np.float32)
        self.tissue_points = np.empty((0, 3), dtype=np.float32)
        self.tissue_values: np.ndarray | None = None
        self.tissue_alpha = np.empty((0,), dtype=np.float32)
        self.colormap = "plasma"
        self.vessel_range: tuple[float | None, float | None] = (None, None)
        self.tissue_range: tuple[float | None, float | None] = (None, None)
        self.vessel_scale = "linear"
        self.tissue_scale = "linear"
        self.vessel_label = "Vessels"
        self.tissue_label = "Tissue"
        self.vessel_opacity = 1.0
        self.tissue_opacity = 0.38
        self._rotation = self._home_rotation()
        self._zoom = 0.82
        self._last_mouse = QPoint()
        self._press_mouse = QPoint()
        self._selected_vessel = -1
        self._selected_tissue = -1
        self._center = np.zeros(3, dtype=float)
        self._span = 1.0
        self.message = "no case defined"

    def clear(self, message: str = "preview paused") -> None:
        self.set_geometry(message=message)

    def set_geometry(
        self,
        *,
        domain_lines: np.ndarray | None = None,
        domain_triangles: np.ndarray | None = None,
        vessel_starts: np.ndarray | None = None,
        vessel_ends: np.ndarray | None = None,
        vessel_values: np.ndarray | None = None,
        vessel_radii: np.ndarray | None = None,
        vessel_alpha: np.ndarray | None = None,
        inlet_points: np.ndarray | None = None,
        outlet_points: np.ndarray | None = None,
        tissue_points: np.ndarray | None = None,
        tissue_values: np.ndarray | None = None,
        tissue_alpha: np.ndarray | None = None,
        message: str = "",
    ) -> None:
        self.domain_lines = _line_array(domain_lines)
        self.domain_triangles = _triangle_array(domain_triangles)
        self.vessel_starts = _point_array(vessel_starts)
        self.vessel_ends = _point_array(vessel_ends)
        n = min(len(self.vessel_starts), len(self.vessel_ends))
        self.vessel_starts = self.vessel_starts[:n]
        self.vessel_ends = self.vessel_ends[:n]
        self.vessel_values = _values(vessel_values, n)
        self.vessel_radii = _values(vessel_radii, n)
        if vessel_alpha is None:
            self.vessel_alpha = np.ones(n, dtype=np.float32)
        else:
            self.vessel_alpha = np.clip(_values(vessel_alpha, n, fill=1.0), 0.05, 1.0)
        self.inlet_points = _point_array(inlet_points)
        self.outlet_points = _point_array(outlet_points)
        self.tissue_points = _point_array(tissue_points)
        self.tissue_values = _values(tissue_values, len(self.tissue_points))
        self.tissue_alpha = np.clip(
            _values(tissue_alpha, len(self.tissue_points), fill=1.0), 0.0, 1.0
        )
        self._selected_vessel = -1
        self._selected_tissue = -1
        self.message = message
        self._fit_bounds()
        self.update()

    def home(self) -> None:
        self._rotation = self._home_rotation()
        self._zoom = 0.82
        self.update()

    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
            dtype=float,
        )

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
            dtype=float,
        )

    @classmethod
    def _home_rotation(cls) -> np.ndarray:
        return cls._rotation_x(0.38) @ cls._rotation_y(-0.68)

    def _orbit(self, horizontal: float, vertical: float) -> None:
        """Orbit around the current screen axes without Euler-angle poles."""
        self._rotation = (
            self._rotation_x(vertical)
            @ self._rotation_y(horizontal)
            @ self._rotation
        )

    def set_domain_mode(self, mode: str) -> None:
        value = str(mode).lower()
        self.domain_mode = value if value in {"none", "surface", "wireframe"} else "wireframe"
        self.update()

    def _fit_bounds(self) -> None:
        arrays = []
        if self.domain_lines.size:
            arrays.append(self.domain_lines.reshape(-1, 3))
        elif self.domain_triangles.size:
            arrays.append(self.domain_triangles.reshape(-1, 3))
        if self.vessel_starts.size:
            arrays.extend((self.vessel_starts, self.vessel_ends))
        if self.inlet_points.size:
            arrays.append(self.inlet_points)
        if self.outlet_points.size:
            arrays.append(self.outlet_points)
        if self.tissue_points.size:
            arrays.append(self.tissue_points)
        if not arrays:
            self._center = np.zeros(3)
            self._span = 1.0
            return
        points = np.concatenate(arrays, axis=0)
        lo, hi = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
        self._center = 0.5 * (lo + hi)
        self._span = max(float(np.max(hi - lo)), 1.0e-9)

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(points):
            return np.empty((0, 2)), np.empty((0,))
        p = (np.asarray(points, dtype=float) - self._center) / self._span
        rotated = p @ self._rotation.T
        x, y, depth = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        perspective = 1.0 / np.clip(1.55 - 0.42 * depth, 0.75, 2.2)
        scale = min(self.width(), self.height()) * self._zoom
        screen = np.column_stack(
            (self.width() * 0.5 + x * scale * perspective,
             self.height() * 0.5 - y * scale * perspective)
        )
        return screen, depth

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0.0, QColor("#08090d"))
        bg.setColorAt(1.0, QColor("#040506"))
        painter.fillRect(self.rect(), bg)
        self._draw_grid(painter)

        vessel_norm = vessel_limits = None
        tissue_norm = tissue_limits = None
        triangles = np.empty((0, 3, 2))
        triangle_light = np.empty((0,))
        domain_segments = np.empty((0, 2, 2))
        tissue_screen = np.empty((0, 2))
        vessel_starts = np.empty((0, 2))
        vessel_ends = np.empty((0, 2))
        radius_widths = np.empty((0,))
        scene_depths: list[np.ndarray] = []
        scene_kinds: list[np.ndarray] = []
        scene_indices: list[np.ndarray] = []

        # QPainter has no depth buffer.  Build one ordered scene instead of
        # painting the entire domain before the entire vessel network: front
        # domain edges/faces must be able to cover vessels, while back ones
        # remain behind them.
        def add_scene_items(depths: np.ndarray, kind: int) -> None:
            scene_depths.append(depths)
            scene_kinds.append(np.full(len(depths), kind, dtype=np.int8))
            scene_indices.append(np.arange(len(depths), dtype=int))

        if self.domain_mode == "surface" and self.domain_triangles.size:
            flat, depth = self._project(self.domain_triangles.reshape(-1, 3))
            triangles = flat.reshape(-1, 3, 2)
            triangle_depth = depth.reshape(-1, 3).mean(axis=1)
            depth_span = float(np.ptp(triangle_depth))
            light = (
                (triangle_depth - triangle_depth.min()) / depth_span
                if depth_span > 1.0e-12
                else np.full(len(triangles), 0.5)
            )
            triangle_light = light
            add_scene_items(triangle_depth, 0)
        elif self.domain_mode == "wireframe" and self.domain_lines.size:
            flat, depth = self._project(self.domain_lines.reshape(-1, 3))
            domain_segments = flat.reshape(-1, 2, 2)
            add_scene_items(depth.reshape(-1, 2).mean(axis=1), 1)

        if self.tissue_points.size:
            ids = np.arange(len(self.tissue_points), dtype=int)
            tissue_screen, depth = self._project(self.tissue_points[ids])
            vals = self.tissue_values[ids] if self.tissue_values is not None else None
            tissue_norm, tissue_limits = _normalize_with_scale(
                vals, *self.tissue_range, self.tissue_scale
            )
            add_scene_items(depth, 2)

        if self.vessel_starts.size:
            vessel_starts, ds = self._project(self.vessel_starts)
            vessel_ends, de = self._project(self.vessel_ends)
            vessel_norm, vessel_limits = _normalize_with_scale(
                self.vessel_values, *self.vessel_range, self.vessel_scale
            )
            radius_widths = self._radius_widths()
            add_scene_items(0.5 * (ds + de), 3)

        if scene_depths:
            depths = np.concatenate(scene_depths)
            kinds = np.concatenate(scene_kinds)
            indices = np.concatenate(scene_indices)
            for scene_item in np.argsort(depths, kind="stable"):
                kind, j = int(kinds[scene_item]), int(indices[scene_item])
                if kind == 0:
                    amount = float(triangle_light[j])
                    painter.setBrush(QColor(
                        int(104 + 35 * amount), int(48 + 22 * amount),
                        int(164 + 45 * amount), int(34 + 18 * amount),
                    ))
                    painter.setPen(QPen(QColor(205, 126, 244, 68), 0.65))
                    painter.drawPolygon(QPolygonF([QPointF(*point) for point in triangles[j]]))
                elif kind == 1:
                    painter.setPen(QPen(QColor(190, 100, 238, 210), 1.3))
                    painter.drawLine(
                        QPointF(*domain_segments[j, 0]), QPointF(*domain_segments[j, 1])
                    )
                elif kind == 2:
                    color = (
                        QColor(91, 221, 238)
                        if tissue_norm is None
                        else _map_color(self.colormap, tissue_norm[j])
                    )
                    opacity = self.tissue_opacity * float(self.tissue_alpha[j])
                    color.setAlphaF(min(max(opacity, 0.0), 1.0))
                    painter.setPen(QPen(color, 2.2))
                    painter.drawPoint(QPointF(*tissue_screen[j]))
                else:
                    color = (
                        QColor(185, 181, 190)
                        if vessel_norm is None
                        else _map_color(self.colormap, vessel_norm[j])
                    )
                    alpha = float(self.vessel_alpha[j]) * self.vessel_opacity
                    color.setAlphaF(min(max(alpha, 0.04), 1.0))
                    painter.setPen(
                        QPen(color, float(radius_widths[j]), Qt.SolidLine, Qt.RoundCap)
                    )
                    painter.drawLine(QPointF(*vessel_starts[j]), QPointF(*vessel_ends[j]))

        if 0 <= self._selected_tissue < len(self.tissue_points):
            selected, _ = self._project(self.tissue_points[self._selected_tissue : self._selected_tissue + 1])
            painter.setPen(QPen(QColor(246, 184, 74, 235), 7.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoint(QPointF(*selected[0]))
        if 0 <= self._selected_vessel < len(vessel_starts):
            j = self._selected_vessel
            painter.setPen(QPen(QColor(246, 184, 74, 225), 3.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(*vessel_starts[j]), QPointF(*vessel_ends[j]))

        self._draw_boundary_points(
            painter, self.inlet_points, QColor("#6FE3BC"), "IN"
        )
        self._draw_boundary_points(
            painter, self.outlet_points, QColor("#FF9D78"), "OUT"
        )
        if vessel_norm is not None and vessel_limits is not None:
            self._draw_scalar_legend(
                painter,
                self.vessel_label,
                vessel_limits,
                side="left",
                scale=self.vessel_scale,
            )
        if tissue_norm is not None and tissue_limits is not None:
            self._draw_scalar_legend(
                painter,
                self.tissue_label,
                tissue_limits,
                side="right",
                scale=self.tissue_scale,
            )

        if (
            not self.domain_lines.size
            and not self.domain_triangles.size
            and not self.vessel_starts.size
            and not self.tissue_points.size
        ):
            self._draw_vascular_motif(painter)
            painter.setPen(QColor("#B9BBC3"))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message)

        if self._has_geometry():
            self._draw_scale_bar(painter)

    def _draw_scalar_legend(
        self,
        painter: QPainter,
        label: str,
        limits: tuple[float, float],
        *,
        side: str,
        scale: str,
    ) -> None:
        margin, top = 13.0, 13.0
        # Five labelled ticks make the scale useful at a glance. Keep both
        # legends apart on narrow preview panes rather than letting labels
        # collide in the middle.
        width = max(180.0, min(340.0, (float(self.width()) - 2.0 * margin - 40.0) / 2.0))
        height = 12.0
        left = margin if side == "left" else float(self.width()) - width - margin
        painter.save()
        font = painter.font()
        font.setPixelSize(13)
        painter.setFont(font)
        panel = QRectF(left - 7.0, top - 6.0, width + 14.0, 58.0)
        painter.fillRect(panel, QColor(7, 8, 11, 218))
        painter.setPen(QColor(224, 226, 232, 235))
        painter.drawText(QRectF(left, top, width, 14), Qt.AlignLeft, label)
        gradient = QLinearGradient(left, top + 20.0, left + width, top + 20.0)
        stops = _MAP_STOPS.get(self.colormap, _MAP_STOPS["viridis"])
        for index, color in enumerate(stops):
            gradient.setColorAt(index / max(len(stops) - 1, 1), QColor(*color))
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        bar_top = top + 18.0
        painter.drawRect(QRectF(left, bar_top, width, height))
        tick_values = _legend_tick_values(limits, scale)
        tick_font = painter.font()
        tick_font.setPixelSize(9)
        painter.setFont(tick_font)
        painter.setPen(QPen(QColor(205, 208, 216, 225), 0.8))
        metrics = painter.fontMetrics()
        for index, value in enumerate(tick_values):
            fraction = index / max(len(tick_values) - 1, 1)
            x = left + width * fraction
            painter.drawLine(QPointF(x, bar_top + height), QPointF(x, bar_top + height + 4.0))
            tick_text = _scientific(value)
            text_width = float(metrics.horizontalAdvance(tick_text))
            text_left = min(max(x - text_width / 2.0, left), left + width - text_width)
            painter.drawText(
                QRectF(text_left, bar_top + height + 6.0, text_width, 12.0),
                Qt.AlignHCenter,
                tick_text,
            )
        if scale == "log":
            painter.setPen(QColor(201, 169, 234, 230))
            painter.drawText(QRectF(left, top, width, 14), Qt.AlignRight, "LOG")
        painter.restore()

    def _has_geometry(self) -> bool:
        return bool(
            self.domain_lines.size
            or self.domain_triangles.size
            or self.vessel_starts.size
            or self.tissue_points.size
        )

    def _scale_bar_spec(self) -> tuple[float, str]:
        """Return a readable center-plane scale for the current zoom."""
        screen_scale = max(min(self.width(), self.height()) * self._zoom, 1.0)
        cm_per_pixel = max(self._span, 1.0e-12) * 1.55 / screen_scale
        target_cm = 105.0 * cm_per_pixel
        exponent = math.floor(math.log10(max(target_cm, 1.0e-15)))
        magnitude = 10.0**exponent
        candidates = np.asarray([1.0, 2.0, 5.0, 10.0]) * magnitude
        value_cm = float(candidates[np.argmin(np.abs(np.log(candidates / target_cm)))])
        pixels = value_cm / cm_per_pixel
        if value_cm >= 1.0:
            value, unit = value_cm, "cm"
        elif value_cm >= 0.01:
            value, unit = value_cm * 10.0, "mm"
        else:
            value, unit = value_cm * 10_000.0, "µm"
        label = f"{value:g} {unit}"
        return float(pixels), label

    def _draw_scale_bar(self, painter: QPainter) -> None:
        pixels, label = self._scale_bar_spec()
        right = float(self.width() - 18)
        left = max(18.0, right - pixels)
        baseline = float(self.height() - 18)
        painter.save()
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QPen(QColor(224, 226, 232, 220), 1.4, Qt.SolidLine, Qt.SquareCap))
        painter.drawLine(QPointF(left, baseline), QPointF(right, baseline))
        painter.drawLine(QPointF(left, baseline - 4), QPointF(left, baseline + 4))
        painter.drawLine(QPointF(right, baseline - 4), QPointF(right, baseline + 4))
        painter.drawText(
            QRectF(left - 20, baseline - 22, (right - left) + 40, 16),
            Qt.AlignCenter,
            label,
        )
        painter.restore()

    def _radius_widths(self) -> np.ndarray:
        n = len(self.vessel_starts)
        if self.vessel_radii is None or not n:
            return np.full(n, 1.7, dtype=float)
        radii = np.asarray(self.vessel_radii, dtype=float)
        positive = radii[np.isfinite(radii) & (radii > 0)]
        if not len(positive):
            return np.full(n, 1.7, dtype=float)
        reference = max(float(np.nanmax(positive)), 1e-12)
        ratio = np.clip(np.nan_to_num(radii / reference, nan=0.0), 0.0, 1.0)
        screen_scale = max(min(self.width(), self.height()) * self._zoom, 1.0)
        physical_diameter = (
            2.0
            * np.maximum(radii, 0.0)
            * screen_scale
            / max(self._span, 1e-12)
        )
        widths = 0.75 + physical_diameter + 0.55 * np.sqrt(ratio)
        return np.clip(np.nan_to_num(widths, nan=0.8), 0.8, 12.0)

    def _draw_boundary_points(
        self, painter: QPainter, points: np.ndarray, color: QColor, label: str
    ) -> None:
        if not len(points):
            return
        screen, _depth = self._project(points)
        painter.save()
        for index, point in enumerate(screen):
            fill = QColor(color)
            fill.setAlpha(230)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor(8, 9, 13, 230), 1.5))
            painter.drawEllipse(QPointF(*point), 5.2, 5.2)
            painter.setPen(QPen(color, 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(*point), 8.0, 8.0)
            painter.drawText(
                QPointF(point[0] + 10.0, point[1] - 7.0), f"{label}{index + 1}"
            )
        painter.restore()

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(139, 136, 148, 28), 1))
        gap = 40
        for x in range(0, self.width(), gap):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), gap):
            painter.drawLine(0, y, self.width(), y)

    def _draw_vascular_motif(self, painter: QPainter) -> None:
        """Quiet organic branching for the empty scientific viewport."""
        w, h = float(self.width()), float(self.height())
        painter.save()
        painter.setPen(QPen(QColor(106, 56, 194, 34), 1.0, Qt.SolidLine, Qt.RoundCap))
        paths = (
            ((.08, .78), (.24, .70), (.31, .55), (.45, .50)),
            ((.30, .57), (.38, .43), (.52, .40), (.64, .25)),
            ((.43, .50), (.54, .57), (.67, .55), (.80, .67)),
            ((.52, .40), (.60, .31), (.70, .30), (.83, .18)),
            ((.65, .55), (.73, .47), (.80, .46), (.91, .36)),
        )
        for start, control_a, control_b, end in paths:
            path = QPainterPath(QPointF(start[0] * w, start[1] * h))
            path.cubicTo(
                QPointF(control_a[0] * w, control_a[1] * h),
                QPointF(control_b[0] * w, control_b[1] * h),
                QPointF(end[0] * w, end[1] * h),
            )
            painter.drawPath(path)
        painter.restore()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._last_mouse = event.position().toPoint()
            self._press_mouse = self._last_mouse
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            point = event.position().toPoint()
            delta = point - self._last_mouse
            self._last_mouse = point
            self._orbit(delta.x() * 0.009, delta.y() * 0.009)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.unsetCursor()
        if (
            event.button() == Qt.LeftButton
            and (event.position().toPoint() - self._press_mouse).manhattanLength() <= 5
        ):
            self._pick(event.position())

    def _pick(self, point: QPointF) -> None:
        """Select the closest rendered scientific object within nine pixels."""
        click = np.array([point.x(), point.y()], dtype=float)
        best_kind, best_index, best_distance = "", -1, 9.0
        if self.vessel_starts.size:
            starts, _ = self._project(self.vessel_starts)
            ends, _ = self._project(self.vessel_ends)
            delta = ends - starts
            denom = np.sum(delta * delta, axis=1)
            along = np.sum((click - starts) * delta, axis=1) / np.maximum(denom, 1e-12)
            projection = starts + np.clip(along, 0.0, 1.0)[:, None] * delta
            distances = np.linalg.norm(projection - click, axis=1)
            index = int(np.argmin(distances))
            if float(distances[index]) < best_distance:
                best_kind, best_index, best_distance = "vessel", index, float(distances[index])
        if self.tissue_points.size:
            ids = np.arange(len(self.tissue_points), dtype=int)
            screen, _ = self._project(self.tissue_points[ids])
            distances = np.linalg.norm(screen - click, axis=1)
            local_index = int(np.argmin(distances))
            if float(distances[local_index]) < best_distance:
                best_kind = "tissue"
                best_index = int(ids[local_index])
        self._selected_vessel = best_index if best_kind == "vessel" else -1
        self._selected_tissue = best_index if best_kind == "tissue" else -1
        self.update()
        self.selection_changed.emit(
            {"kind": best_kind, "index": best_index} if best_kind else {}
        )

    def wheelEvent(self, event) -> None:
        self._zoom = float(np.clip(self._zoom * np.exp(event.angleDelta().y() / 1100.0), 0.18, 4.0))
        self.update()


def _quadrature_order(config: dict[str, Any]) -> int:
    oxygen = config.get("settings", {}).get("oxygen", {})
    return max(
        1,
        int(
            oxygen.get(
                "gl_order_cext",
                oxygen.get("GL_ORDER_CEXT", oxygen.get("gl_order", oxygen.get("GL_ORDER", 1))),
            )
        ),
    )


def _preview_tree_count(config: dict[str, Any], inlet_points=None) -> int:
    network = config.get("network", {})
    source = config.get("gui", {}).get("network_source", "svv_generated")
    simple = network.get("simple", {})
    if source in {"lattice", "simple"} or simple.get("mode") in {
        "lattice",
        "onechannel",
        "multichannel",
        "snake",
    }:
        return 0
    roots = network.get("roots") or ([network["root"]] if network.get("root") else [])
    if roots:
        return len(roots)
    inlet_count = len(inlet_points) if inlet_points is not None else 0
    return max(inlet_count, 1)


def _count_status(vessels: int, trees: int, quadrature: int, tissue: int) -> str:
    tree_label = "tree" if int(trees) == 1 else "trees"
    return (
        f"{int(vessels):,} vessels  │  {int(trees):,} {tree_label}  │  "
        f"{int(quadrature):,} quadrature nodes  │  {int(tissue):,} tissue points"
    )


def _home_icon() -> QIcon:
    """A crisp drawn home icon that remains legible without an icon font."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#E6E8EE"), 1.9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    painter.drawPolyline(
        QPolygonF([QPointF(4.0, 11.0), QPointF(12.0, 4.2), QPointF(20.0, 11.0)])
    )
    painter.drawLine(QPointF(6.5, 10.0), QPointF(6.5, 19.0))
    painter.drawLine(QPointF(17.5, 10.0), QPointF(17.5, 19.0))
    painter.drawLine(QPointF(6.5, 19.0), QPointF(17.5, 19.0))
    painter.drawLine(QPointF(10.0, 19.0), QPointF(10.0, 14.0))
    painter.drawLine(QPointF(14.0, 19.0), QPointF(14.0, 14.0))
    painter.drawLine(QPointF(10.0, 14.0), QPointF(14.0, 14.0))
    painter.end()
    return QIcon(pixmap)


class CasePreview(QFrame):
    """Persistent preview panel shared by setup and analysis pages."""

    result_fields_loaded = Signal(list, list)
    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("previewPanel")
        self.setMinimumWidth(520)
        self._result_cache: dict[str, Any] = {}
        self._visible_result_indices = np.empty((0,), dtype=int)
        self._visible_tissue_indices = np.empty((0,), dtype=int)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.title = QLabel("LIVE CASE PREVIEW")
        self.title.setObjectName("previewTitle")
        header.addWidget(self.title)
        header.addStretch()
        home = IconButton()
        home.setObjectName("previewHome")
        home.setIcon(_home_icon())
        home.setIconSize(QSize(20, 20))
        home.setFixedSize(34, 30)
        home.setToolTip("Reset view")
        home.setAccessibleName("Reset view")
        home.clicked.connect(self.home)
        header.addWidget(home)
        layout.addLayout(header)
        self.canvas = GeometryCanvas(self)
        self.canvas.selection_changed.connect(self._canvas_selection)
        layout.addWidget(self.canvas, 1)
        view_controls = QHBoxLayout()
        view_controls.setSpacing(8)
        domain_label = QLabel("STYLE")
        domain_label.setObjectName("eyebrow")
        domain_label.setAccessibleName("Domain display style")
        self.domain_view = ChoiceComboBox(self)
        self.domain_view.addItem("Wireframe", "wireframe")
        self.domain_view.addItem("Surface", "surface")
        self.domain_view.addItem("None", "none")
        # "Wireframe" plus the arrow needs a little more than the 94 px
        # compact width; 104 px stays compact without clipping the label.
        self.domain_view.setFixedWidth(104)
        self.domain_view.setToolTip("Switch the domain between sectioned wireframe and translucent surface views.")
        self.domain_view.currentIndexChanged.connect(
            lambda: self.canvas.set_domain_mode(self.domain_view.currentData())
        )
        view_controls.addWidget(domain_label)
        view_controls.addWidget(self.domain_view)
        view_controls.addStretch()
        layout.addLayout(view_controls)
        self.status = QLabel("Domain preview")
        self.status.setObjectName("previewStatus")
        self.status.setWordWrap(True)
        self.status.setVisible(False)
        # Keep a stable footprint for the scientific-count strip even when
        # Project/Domain intentionally hide it.  Otherwise the canvas grows
        # and contracts as the user advances through the workflow.
        self._status_slot = QWidget(self)
        self._status_slot.setMinimumHeight(28)
        status_layout = QVBoxLayout(self._status_slot)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status)
        layout.addWidget(self._status_slot)

    def home(self) -> None:
        self.canvas.home()

    def release(self) -> None:
        self._result_cache.clear()
        self._visible_result_indices = np.empty((0,), dtype=int)
        self._visible_tissue_indices = np.empty((0,), dtype=int)
        self.canvas.clear()
        self.status.setText("preview memory released")

    def _canvas_selection(self, selection: dict[str, Any]) -> None:
        if not selection or not self._result_cache:
            self.selection_changed.emit({})
            return
        kind = selection.get("kind")
        shown_index = int(selection.get("index", -1))
        if kind == "vessel" and 0 <= shown_index < len(self._visible_result_indices):
            index = int(self._visible_result_indices[shown_index])
            arrays = self._result_cache.get("vessel_arrays", {})
        elif kind == "tissue" and 0 <= shown_index < len(self._visible_tissue_indices):
            index = int(self._visible_tissue_indices[shown_index])
            arrays = self._result_cache.get("tissue_arrays", {})
        else:
            self.selection_changed.emit({})
            return
        values = {
            name: np.asarray(values).reshape(-1)[index]
            for name, values in arrays.items()
            if np.asarray(values).size > index
        }
        self.selection_changed.emit(
            {
                "kind": kind,
                "index": index,
                "id": values.get("segment_id", values.get("local_segment_id", index)),
                "values": values,
            }
        )

    def show_case(
        self,
        config: dict[str, Any],
        *,
        include_network: bool,
        include_tissue: bool = False,
    ) -> None:
        self._visible_result_indices = np.empty((0,), dtype=int)
        self._visible_tissue_indices = np.empty((0,), dtype=int)
        domain_config = config.get("domain", {})
        domain_lines = domain_wireframe(domain_config)
        domain_triangles = domain_surface_triangles(domain_config)
        starts = ends = radii = alpha = inlet_points = outlet_points = None
        tissue_points = tissue_alpha = None
        status = "Domain ready"
        if include_network:
            try:
                (
                    starts,
                    ends,
                    radii,
                    inlet_points,
                    outlet_points,
                    alpha,
                    detail,
                ) = network_geometry(config)
                starts, ends, visible_radii, alpha = limit_near_inlets(
                    starts,
                    ends,
                    inlet_points,
                    limit=5000,
                    values=radii,
                    alpha=alpha,
                )
                radii = visible_radii if radii is not None else None
                status = detail
            except Exception as exc:
                status = f"Preview unavailable  │  {exc}"
        if include_tissue:
            try:
                tissue_points, tissue_alpha, requested = preview_tissue_geometry(
                    config, inlet_points, limit=10_000
                )
                status += (
                    f"  │  {len(tissue_points):,}/{requested:,} tissue points"
                    if requested > len(tissue_points)
                    else f"  │  {len(tissue_points):,} tissue points"
                )
            except Exception as exc:
                status += f"  │  tissue preview unavailable: {exc}"
        if include_network and "unavailable" not in status.lower():
            vessel_count = len(starts) if starts is not None else 0
            tree_count = _preview_tree_count(config, inlet_points)
            tissue_count = len(tissue_points) if tissue_points is not None else 0
            status = _count_status(
                vessel_count,
                tree_count,
                vessel_count * _quadrature_order(config),
                tissue_count,
            )
        self.canvas.tissue_opacity = 0.52
        self.canvas.set_geometry(
            domain_lines=domain_lines,
            domain_triangles=domain_triangles,
            vessel_starts=starts,
            vessel_ends=ends,
            vessel_radii=radii,
            vessel_alpha=alpha,
            inlet_points=inlet_points,
            outlet_points=outlet_points,
            tissue_points=tissue_points,
            tissue_alpha=tissue_alpha,
        )
        self.status.setText(status)
        self.status.setVisible(include_network or include_tissue)

    def show_seed(
        self,
        config: dict[str, Any],
        geometry_path: str,
        response: dict[str, Any],
        *,
        include_tissue: bool = False,
    ) -> None:
        with np.load(geometry_path) as data:
            starts = np.asarray(data["starts"], dtype=np.float32)
            ends = np.asarray(data["ends"], dtype=np.float32)
            alpha = np.asarray(data["alpha"], dtype=np.float32)
            radii = (
                np.asarray(data["radii"], dtype=np.float32)
                if "radii" in data.files
                else None
            )
        roots = config.get("network", {}).get("roots") or []
        inlets = np.asarray([root.get("start", [0, 0, 0]) for root in roots], dtype=float)
        tissue_points = tissue_alpha = None
        requested = 0
        if include_tissue:
            tissue_points, tissue_alpha, requested = preview_tissue_geometry(
                config, inlets, limit=10_000
            )
        self.canvas.tissue_opacity = 0.52
        self.canvas.set_geometry(
            domain_lines=domain_wireframe(config.get("domain", {})),
            domain_triangles=domain_surface_triangles(config.get("domain", {})),
            vessel_starts=starts,
            vessel_ends=ends,
            vessel_radii=radii,
            vessel_alpha=alpha,
            inlet_points=inlets,
            tissue_points=tissue_points,
            tissue_alpha=tissue_alpha,
        )
        segments = response.get("segments", [])
        shown = response.get("shown_segments", segments)
        vessel_count = sum(int(value) for value in shown)
        self.status.setText(
            _count_status(
                vessel_count,
                len(segments),
                vessel_count * _quadrature_order(config),
                len(tissue_points) if tissue_points is not None else 0,
            )
        )
        self.status.setVisible(True)

    def show_result(self, manifest_path: str, options: dict[str, Any]) -> None:
        if not manifest_path:
            self.canvas.clear("no result selected")
            self.status.setText("no result selected")
            self.status.setVisible(True)
            return
        try:
            cache = self._load_result(manifest_path)
            vessel_field = str(options.get("vessel_field") or "")
            tissue_field = str(options.get("tissue_field") or "")
            limit_value = options.get("vessel_limit", 5000)
            if str(limit_value).lower() == "none":
                limit_value = 0
            logical_ids = cache["vessel_arrays"].get("global_segment_id")
            visible_logical_ids = np.empty((0,), dtype=np.int64)
            if logical_ids is not None and len(logical_ids) == len(cache["starts"]):
                logical_ids = np.asarray(logical_ids, dtype=np.int64)
                unique_ids, first_ids = np.unique(logical_ids, return_index=True)
                _, reverse_first_ids = np.unique(logical_ids[::-1], return_index=True)
                last_ids = len(logical_ids) - 1 - reverse_first_ids
                limit = len(unique_ids) if limit_value == "all" else int(limit_value)
                _, _, selected_logical_ids, _alpha = limit_near_inlets(
                    cache["starts"][first_ids],
                    cache["ends"][last_ids],
                    cache["inlets"],
                    limit=limit,
                    values=unique_ids,
                )
                visible_indices = np.flatnonzero(
                    np.isin(logical_ids, np.asarray(selected_logical_ids, dtype=np.int64))
                )
                visible_logical_ids = np.unique(logical_ids[visible_indices]).astype(
                    np.int64
                )
                starts = cache["starts"][visible_indices]
                ends = cache["ends"][visible_indices]
                shown_vessels = int(np.unique(logical_ids[visible_indices]).size)
                total_vessels = int(unique_ids.size)
            else:
                limit = len(cache["starts"]) if limit_value == "all" else int(limit_value)
                starts, ends, visible_indices, _alpha = limit_near_inlets(
                    cache["starts"],
                    cache["ends"],
                    cache["inlets"],
                    limit=limit,
                    values=np.arange(len(cache["starts"]), dtype=int),
                )
                shown_vessels = len(starts)
                total_vessels = len(cache["starts"])
            self._visible_result_indices = np.asarray(visible_indices, dtype=int)
            values, vessel_label = _display_field_values(
                "vessel", vessel_field, cache, options
            )
            values = values[self._visible_result_indices] if values is not None else None
            tissue_mode = str(options.get("tissue_mode", "near"))
            tissue_indices, tissue_alpha = select_tissue_points(
                cache["tissue_points"],
                cache["inlets"],
                mode=tissue_mode,
                limit=10_000,
            )
            self._visible_tissue_indices = tissue_indices
            visible_tissue_points = cache["tissue_points"][tissue_indices]
            tissue_values, tissue_label = _display_field_values(
                "tissue", tissue_field, cache, options
            )
            tissue_values = tissue_values[tissue_indices] if tissue_values is not None else None
            radius_source = next(
                (
                    values
                    for name, values in cache["vessel_arrays"].items()
                    if "radius" in name.lower()
                ),
                None,
            )
            visible_radii = (
                np.asarray(radius_source).reshape(-1)[self._visible_result_indices]
                if radius_source is not None
                else None
            )
            self.canvas.colormap = str(options.get("colormap", "plasma"))
            self.canvas.vessel_opacity = float(options.get("vessel_opacity", 1.0))
            self.canvas.tissue_opacity = float(options.get("tissue_opacity", 0.35))
            self.canvas.vessel_range = tuple(options.get("vessel_range", (None, None)))
            self.canvas.tissue_range = tuple(options.get("tissue_range", (None, None)))
            self.canvas.vessel_scale = str(options.get("vessel_scale", "linear"))
            self.canvas.tissue_scale = str(options.get("tissue_scale", "linear"))
            self.canvas.vessel_label = _layer_field_label("Vessels", vessel_label)
            self.canvas.tissue_label = _layer_field_label("Tissue", tissue_label)
            self.canvas.set_geometry(
                domain_lines=cache["domain_lines"],
                domain_triangles=cache["domain_triangles"],
                vessel_starts=starts,
                vessel_ends=ends,
                vessel_values=values,
                vessel_radii=visible_radii,
                tissue_points=visible_tissue_points,
                tissue_values=tissue_values,
                tissue_alpha=tissue_alpha,
            )
            shown_tissue = len(visible_tissue_points)
            tree_values = cache["vessel_arrays"].get("tree_id")
            if tree_values is not None and len(tree_values) == len(cache["starts"]):
                visible_trees = np.asarray(tree_values)[self._visible_result_indices]
                tree_count = int(np.unique(visible_trees[visible_trees >= 0]).size)
            else:
                tree_count = int(cache.get("network", {}).get("tree_count", 0) or 0)
            quadrature_by_segment = cache.get("quadrature_by_segment", {})
            if len(visible_logical_ids) and quadrature_by_segment:
                quadrature_count = sum(
                    int(quadrature_by_segment.get(int(segment_id), 0))
                    for segment_id in visible_logical_ids
                )
            else:
                quadrature_count = shown_vessels * _quadrature_order(
                    cache.get("settings", {})
                )
            self.status.setText(
                _count_status(
                    shown_vessels,
                    tree_count,
                    quadrature_count,
                    shown_tissue,
                )
            )
            self.status.setVisible(True)
            self.result_fields_loaded.emit(
                list(cache["vessel_arrays"]), list(cache["tissue_arrays"])
            )
        except Exception as exc:
            self.canvas.clear(str(exc))
            self.status.setText(f"Could not load result  │  {exc}")
            self.status.setVisible(True)

    def _load_result(self, manifest_path: str) -> dict[str, Any]:
        path = str(Path(manifest_path).resolve())
        if self._result_cache.get("manifest") == path:
            return self._result_cache
        import pyvista as pv

        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        outputs = manifest.get("outputs", {})

        def read(key):
            raw = outputs.get(key)
            if not raw:
                return None
            target = Path(raw)
            if not target.is_absolute():
                target = Path(path).parent / target
            return pv.read(target) if target.exists() else None

        vessels = read("vessels_vtp")
        tissue = read("oxygen_points_vtp")
        domain = read("domain_boundary_vtp")
        configured_domain = manifest.get("settings", {}).get("domain", {})
        domain_kind = str(
            configured_domain.get("type", configured_domain.get("kind", ""))
        ).lower()
        if domain_kind in {"cube", "box", "rectangular", "rectangular_box"}:
            domain_lines = domain_wireframe(configured_domain)
            domain_triangles = domain_surface_triangles(configured_domain)
        else:
            domain_lines = _mesh_wireframe(domain)
            domain_triangles = _mesh_triangles(domain)
        starts, ends, arrays, inlets = _polyline_data(vessels)
        tissue_points = (
            np.asarray(tissue.points, dtype=np.float32)
            if tissue is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        tissue_arrays = _numeric_point_arrays(tissue, len(tissue_points))
        quadrature_by_segment: dict[int, int] = {}
        if vessels is not None:
            raw_ids = getattr(vessels, "point_data", {}).get("global_segment_id")
            raw_quadrature = getattr(vessels, "point_data", {}).get(
                "solver_quadrature_node"
            )
            if raw_ids is not None and raw_quadrature is not None:
                raw_ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
                raw_quadrature = np.asarray(raw_quadrature).reshape(-1) != 0
                if len(raw_ids) == len(raw_quadrature):
                    for segment_id in np.unique(raw_ids[raw_quadrature]):
                        quadrature_by_segment[int(segment_id)] = int(
                            np.count_nonzero(raw_quadrature & (raw_ids == segment_id))
                        )
        cache = {
            "manifest": path,
            "starts": starts,
            "ends": ends,
            "inlets": inlets,
            "vessel_arrays": arrays,
            "tissue_points": tissue_points,
            "tissue_arrays": tissue_arrays,
            "domain_lines": domain_lines,
            "domain_triangles": domain_triangles,
            "settings": manifest.get("settings", {}),
            "network": manifest.get("network", {}),
            "quadrature_by_segment": quadrature_by_segment,
        }
        self._result_cache = cache
        return cache


def domain_wireframe(domain: dict[str, Any]) -> np.ndarray:
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    if kind == "sphere":
        radius = float(domain.get("radius", 0.5))
        center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
        lines = []
        theta = np.linspace(0, 2 * np.pi, 49)
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 7)[1:-1]:
            ring = center + radius * np.column_stack(
                (np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.full_like(theta, np.sin(phi)))
            )
            lines.extend(np.stack((ring[:-1], ring[1:]), axis=1))
        phi = np.linspace(-np.pi / 2, np.pi / 2, 25)
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            arc = center + radius * np.column_stack(
                (np.cos(phi) * np.cos(angle), np.cos(phi) * np.sin(angle), np.sin(phi))
            )
            lines.extend(np.stack((arc[:-1], arc[1:]), axis=1))
        return np.asarray(lines, dtype=np.float32)
    if kind == "file" and domain.get("path"):
        try:
            path = resolve_domain_path(domain["path"])
            if path is None:
                raise FileNotFoundError(domain["path"])
            stat = path.stat()
            return _cached_domain_wireframe(
                str(path), int(stat.st_mtime_ns), int(stat.st_size)
            ).copy()
        except Exception:
            return np.empty((0, 2, 3), dtype=np.float32)
    dims = np.array(
        [
            domain.get("x_length", domain.get("side_length", 1.0)),
            domain.get("y_length", domain.get("side_length", 1.0)),
            domain.get("z_length", domain.get("side_length", 1.0)),
        ],
        dtype=float,
    )
    center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
    corners = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=float,
    ) * dims + center
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners[i] != corners[j]) == 1]
    return np.asarray([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)


def domain_surface_triangles(domain: dict[str, Any]) -> np.ndarray:
    """Return a bounded triangle surface suitable for the lightweight canvas."""
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    if kind == "file" and domain.get("path"):
        try:
            path = resolve_domain_path(domain["path"])
            if path is None:
                raise FileNotFoundError(domain["path"])
            stat = path.stat()
            return _cached_domain_triangles(
                str(path), int(stat.st_mtime_ns), int(stat.st_size)
            ).copy()
        except Exception:
            return np.empty((0, 3, 3), dtype=np.float32)
    if kind == "sphere":
        radius = float(domain.get("radius", 0.5))
        center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
        longitude = np.linspace(0.0, 2.0 * np.pi, 33)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 19)
        points = center + radius * np.asarray(
            [
                [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]
                for phi in latitude
                for theta in longitude
            ],
            dtype=float,
        )
        triangles = []
        width = len(longitude)
        for row in range(len(latitude) - 1):
            for column in range(len(longitude) - 1):
                a = row * width + column
                b = a + 1
                c = a + width
                d = c + 1
                triangles.extend((points[[a, c, b]], points[[b, c, d]]))
        return np.asarray(triangles, dtype=np.float32)

    dims, center = _domain_dimensions(domain)
    corners = np.asarray(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=float,
    ) * dims + center
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    triangles = []
    for a, b, c, d in quads:
        triangles.extend((corners[[a, b, c]], corners[[a, c, d]]))
    return np.asarray(triangles, dtype=np.float32)


def network_geometry(config: dict[str, Any]):
    network = config.get("network", {})
    source = config.get("gui", {}).get("network_source", "svv_generated")
    simple = network.get("simple", {})
    if source == "lattice" or simple.get("mode") == "lattice":
        cells = int(simple.get("cells", 4))
        lattice_type = str(simple.get("lattice_type", "cubic"))
        estimated = channel_count(cells, lattice_type) * max(int(simple.get("subdivisions", 1)), 1)
        preview_cells = cells
        simplified = False
        if estimated > 120_000:
            preview_cells = min(cells, 12)
            if lattice_type == "octet" and preview_cells % 2:
                preview_cells -= 1
            simplified = True
        dims, center = _domain_dimensions(config.get("domain", {}))
        inside = _analytic_inside(config.get("domain", {}))
        lattice = generate_lattice(
            preview_cells,
            tuple(dims),
            float(simple.get("radius_cm", 0.0005)),
            lattice_type=lattice_type,
            inlet_points_cm=simple.get("inlet_points_cm"),
            outlet_points_cm=simple.get("outlet_points_cm"),
            radius_expression=simple.get("radius_expression"),
            subdivisions=int(simple.get("subdivisions", 1)),
            center_cm=tuple(center),
            node_inside=inside,
        )
        inlet_connections = int(lattice.get("inlet_connection_count", 0))
        outlet_connections = int(lattice.get("outlet_connection_count", 0))
        detail = (
            f"{len(lattice['segment_starts_cm']):,} vessels  │  "
            f"{len(lattice['inlet_nodes'])} inlet{'s' if len(lattice['inlet_nodes']) != 1 else ''}  │  "
            f"{len(lattice['outlet_nodes'])} outlet{'s' if len(lattice['outlet_nodes']) != 1 else ''}"
        )
        if inlet_connections or outlet_connections:
            detail += f"  │  {inlet_connections + outlet_connections} boundary connections"
        if simplified:
            detail += f"  │  preview simplified from ~{estimated:,}"
        return (
            lattice["segment_starts_cm"],
            lattice["segment_ends_cm"],
            lattice["segment_radii_cm"],
            lattice["inlet_points_cm"],
            lattice["outlet_points_cm"],
            None,
            detail,
        )
    if source == "simple" or network.get("mode") == "simple":
        starts, ends = _simple_geometry(config)
        radii = np.full(len(starts), float(simple.get("radius_cm", 0.015)))
        return starts, ends, radii, starts[:1], ends[-1:], None, f"{len(starts):,} channels shown"
    if source == "uploaded" and network.get("input_path"):
        starts, ends = _load_uploaded_geometry(Path(str(network["input_path"])))
        return starts, ends, None, starts[:1], ends[-1:], None, f"{len(starts):,} uploaded vessels"
    roots = network.get("roots") or ([network["root"]] if network.get("root") else [])
    starts, ends, alpha, inlets = _svv_placeholder_geometry(config, roots)
    return starts, ends, None, inlets, None, alpha, "Preparing the exact hydraulic SVV seed…"


def _svv_placeholder_geometry(config, roots):
    """Immediate low-cost branching cue while the exact seed worker runs."""
    side = float(config.get("domain", {}).get("side_length", 1.0))
    all_starts, all_ends, all_alpha, inlets = [], [], [], []
    for root_index, root in enumerate(roots or [{}]):
        start = np.asarray(root.get("start", [0.0, 0.0, 0.0]), dtype=float)
        direction = np.asarray(root.get("direction", [1.0, 0.0, 0.0]), dtype=float)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm else np.array([1.0, 0.0, 0.0])
        inlets.append(start)
        frontier = [(start, direction)]
        length = 0.18 * side
        for depth in range(5):
            next_frontier = []
            for branch_index, (origin, heading) in enumerate(frontier):
                endpoint = origin + heading * length
                all_starts.append(origin)
                all_ends.append(endpoint)
                all_alpha.append(max(0.22, 1.0 - 0.18 * depth))
                if depth == 4:
                    continue
                axis = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(heading, axis))) > 0.82:
                    axis = np.array([0.0, 1.0, 0.0])
                lateral = np.cross(heading, axis)
                lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
                twist = np.cross(heading, lateral)
                sign = -1.0 if (branch_index + root_index + depth) % 2 else 1.0
                for branch_sign in (-1.0, 1.0):
                    child = (
                        0.84 * heading
                        + 0.48 * branch_sign * lateral
                        + 0.13 * sign * twist
                    )
                    child /= max(float(np.linalg.norm(child)), 1e-12)
                    next_frontier.append((endpoint, child))
            frontier = next_frontier
            length *= 0.60
    return (
        np.asarray(all_starts, dtype=np.float32),
        np.asarray(all_ends, dtype=np.float32),
        np.asarray(all_alpha, dtype=np.float32),
        np.asarray(inlets, dtype=np.float32),
    )


def limit_near_inlets(
    starts: np.ndarray,
    ends: np.ndarray,
    inlet_points: np.ndarray | list,
    *,
    limit: int,
    values: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
):
    starts, ends = _point_array(starts), _point_array(ends)
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    vals = _values(values, n)
    alphas = _values(alpha, n, fill=1.0)
    if n <= max(int(limit), 0):
        return starts, ends, vals if values is not None else alphas, alphas
    inlets = _point_array(inlet_points)
    if not len(inlets):
        inlets = starts[:1]
    midpoint = 0.5 * (starts + ends)
    distances = np.linalg.norm(midpoint[:, None, :] - inlets[None, :, :], axis=2)
    owner = np.argmin(distances, axis=1)
    chosen: list[int] = []
    quota = max(int(limit) // len(inlets), 1)
    for inlet in range(len(inlets)):
        ids = np.flatnonzero(owner == inlet)
        ids = ids[np.argsort(distances[ids, inlet], kind="stable")]
        chosen.extend(ids[:quota].tolist())
    if len(chosen) < int(limit):
        remaining = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=int), assume_unique=False)
        nearest = np.min(distances[remaining], axis=1)
        chosen.extend(remaining[np.argsort(nearest, kind="stable")[: int(limit) - len(chosen)]].tolist())
    ids = np.asarray(chosen[: int(limit)], dtype=int)
    third = vals[ids] if values is not None else alphas[ids]
    return starts[ids], ends[ids], third, alphas[ids]


def select_tissue_points(
    points: np.ndarray,
    inlet_points: np.ndarray | list,
    *,
    mode: str,
    limit: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose result/preview points without changing the underlying dataset."""
    points = _point_array(points)
    count = len(points)
    if count == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=np.float32)
    mode = str(mode or "near").lower()
    limit = max(int(limit), 0)
    if mode == "none":
        return np.empty((0,), dtype=int), np.empty((0,), dtype=np.float32)
    if mode == "all" or count <= limit:
        return np.arange(count, dtype=int), np.ones(count, dtype=np.float32)
    if mode == "random":
        ids = np.random.default_rng(42).choice(count, size=limit, replace=False)
        ids.sort()
        return ids, np.ones(len(ids), dtype=np.float32)

    inlets = _point_array(inlet_points)
    if not len(inlets):
        inlets = np.mean(points, axis=0, keepdims=True)
    nearest = np.full(count, np.inf, dtype=float)
    for inlet in inlets:
        nearest = np.minimum(nearest, np.linalg.norm(points - inlet, axis=1))
    ids = np.argpartition(nearest, limit - 1)[:limit]
    ids = ids[np.argsort(nearest[ids], kind="stable")]
    selected_distance = nearest[ids]
    span = float(np.ptp(selected_distance))
    if span <= 1.0e-12:
        alpha = np.ones(len(ids), dtype=np.float32)
    else:
        fade = 1.0 - (selected_distance - selected_distance.min()) / span
        alpha = np.asarray(np.sqrt(np.clip(fade, 0.0, 1.0)), dtype=np.float32)
    return ids, alpha


def preview_tissue_geometry(
    config: dict[str, Any],
    inlet_points: np.ndarray | list | None,
    *,
    limit: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a bounded visual sample of the requested tissue-point layout."""
    simulation = config.get("simulation", {})
    mode = str(simulation.get("sample_mode", "random"))
    if mode == "grid":
        raw_shape = list(simulation.get("tissue_grid", {}).get("shape", [64, 64, 64]))
        raw_shape = (raw_shape + [64, 64, 64])[:3]
        shape = np.maximum(np.asarray(raw_shape, dtype=int), 2)
        requested = int(np.prod(shape, dtype=np.int64))
    else:
        requested = max(int(simulation.get("distance_sample_count", 10_000)), 0)
        shape = None
    if requested == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            0,
        )

    # Evaluate enough candidates to make "nearest" representative while
    # keeping setup previews small even if a production grid has billions of points.
    candidate_count = min(requested, max(limit * 6, limit))
    points = _sample_preview_domain_points(
        config.get("domain", {}),
        mode=mode,
        count=candidate_count,
        shape=shape,
        random_seed=int(config.get("domain", {}).get("random_seed", 42)),
    )
    selection_mode = "near" if requested > limit else "all"
    ids, alpha = select_tissue_points(
        points, inlet_points if inlet_points is not None else [], mode=selection_mode, limit=limit
    )
    return points[ids], alpha, requested


def _sample_preview_domain_points(
    domain: dict[str, Any],
    *,
    mode: str,
    count: int,
    shape: np.ndarray | None,
    random_seed: int,
) -> np.ndarray:
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    dims, center = _domain_dimensions(domain)
    surface = None
    if kind == "file" and domain.get("path"):
        path = resolve_domain_path(domain["path"])
        if path is None:
            raise FileNotFoundError(domain["path"])
        stat = path.stat()
        surface = _cached_domain_surface(str(path), int(stat.st_mtime_ns), int(stat.st_size))
        bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
        center = bounds.mean(axis=1)
        dims = bounds[:, 1] - bounds[:, 0]

    already_inside = False
    if mode == "grid":
        grid_shape = _bounded_grid_shape(shape, count)
        axes = [
            np.linspace(center[axis] - 0.5 * dims[axis], center[axis] + 0.5 * dims[axis], int(grid_shape[axis]))
            for axis in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([values.reshape(-1) for values in mesh])
    elif kind == "sphere":
        rng = np.random.default_rng(random_seed)
        direction = rng.normal(size=(count, 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-12)
        radius = float(domain.get("radius", dims[0] / 2.0))
        points = center + direction * (rng.random(count) ** (1.0 / 3.0) * radius)[:, None]
    elif surface is not None:
        rng = np.random.default_rng(random_seed)
        accepted = []
        remaining = count
        for _attempt in range(8):
            if remaining <= 0:
                break
            batch_count = min(max(remaining * 6, 10_000), 500_000)
            candidates = center + (rng.random((batch_count, 3)) - 0.5) * dims
            enclosed = _points_inside_surface(candidates, surface)
            if len(enclosed):
                accepted.append(enclosed[:remaining])
                remaining -= min(len(enclosed), remaining)
        points = (
            np.concatenate(accepted, axis=0)
            if accepted
            else np.empty((0, 3), dtype=float)
        )
        already_inside = True
    else:
        rng = np.random.default_rng(random_seed)
        points = center + (rng.random((count, 3)) - 0.5) * dims

    inside = _analytic_inside(domain)
    if inside is not None:
        points = points[np.asarray(inside(points), dtype=bool)]
    elif surface is not None and len(points) and not already_inside:
        points = _points_inside_surface(points, surface)
    return np.asarray(points, dtype=np.float32)


def _points_inside_surface(points: np.ndarray, surface) -> np.ndarray:
    import pyvista as pv

    selected = pv.PolyData(points).select_enclosed_points(
        surface, tolerance=1.0e-6, check_surface=False
    )
    mask = np.asarray(selected["SelectedPoints"], dtype=bool)
    return np.asarray(points)[mask]


def _bounded_grid_shape(shape: np.ndarray | None, maximum: int) -> np.ndarray:
    result = np.maximum(
        np.asarray(shape if shape is not None else [2, 2, 2], dtype=int), 2
    )
    product = int(np.prod(result, dtype=np.int64))
    if product <= maximum:
        return result
    scale = (float(maximum) / float(product)) ** (1.0 / 3.0)
    result = np.maximum(np.floor(result * scale).astype(int), 2)
    while int(np.prod(result, dtype=np.int64)) > maximum:
        axis = int(np.argmax(result))
        result[axis] = max(int(result[axis]) - 1, 2)
    return result


def _domain_dimensions(domain):
    side = float(domain.get("side_length", 1.0))
    if str(domain.get("type", "cube")) == "sphere":
        side = 2.0 * float(domain.get("radius", side / 2.0))
    dims = np.asarray([domain.get("x_length", side), domain.get("y_length", side), domain.get("z_length", side)], dtype=float)
    center = np.asarray(domain.get("center", [0, 0, 0]), dtype=float)
    return dims, center


def _analytic_inside(domain):
    kind = str(domain.get("type", "cube")).lower()
    dims, center = _domain_dimensions(domain)
    if kind == "sphere":
        radius = float(domain.get("radius", dims[0] / 2.0))
        return lambda points: np.linalg.norm(np.asarray(points) - center, axis=1) <= radius * (1.0 + 1e-10)
    if kind in {"cube", "box"}:
        return lambda points: np.all(np.abs(np.asarray(points) - center) <= 0.5 * dims + 1e-10, axis=1)
    return None


def _simple_geometry(config):
    simple = config.get("network", {}).get("simple", {})
    dims, center = _domain_dimensions(config.get("domain", {}))
    mode = str(simple.get("mode", "onechannel"))
    axis = 0 if str(simple.get("axis", "x")) == "x" else 1
    lo, hi = center - 0.48 * dims, center + 0.48 * dims
    if mode == "multichannel":
        offsets = [float(v) for v in simple.get("y_offsets_cm", [-0.2, 0.0, 0.2])]
        starts, ends = [], []
        for offset in offsets:
            a, b = center.copy(), center.copy()
            a[axis], b[axis] = lo[axis], hi[axis]
            a[1 - axis] += offset
            b[1 - axis] += offset
            starts.append(a); ends.append(b)
        return np.asarray(starts), np.asarray(ends)
    if mode == "snake":
        # Use the solver's channel generator rather than a decorative sine
        # wave.  The setup preview must be the exact path that is exported.
        from cascade.simple import _snake_channel_points

        z = -dims[2] / 2.0 + float(simple.get("z_from_bottom_cm", dims[2] * 0.5))
        points = _snake_channel_points(
            z,
            int(simple.get("snake_arc_segments", 5)),
            int(simple.get("snake_straight_segments", 5)),
        )
        return points[:-1], points[1:]
    a, b = center.copy(), center.copy()
    a[axis], b[axis] = lo[axis], hi[axis]
    return a.reshape(1, 3), b.reshape(1, 3)


def _load_uploaded_geometry(path: Path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "starts" in data and "ends" in data:
                return np.asarray(data["starts"]), np.asarray(data["ends"])
            if "data" in data:
                raw = np.asarray(data["data"])
                return raw[:, :3], raw[:, 3:6]
    if path.suffix.lower() in {".vtk", ".vtp", ".vtu", ".ply"}:
        import pyvista as pv

        starts, ends, _, _ = _polyline_data(pv.read(path))
        return starts, ends
    raise ValueError("Build the preview seed to inspect this network format")


def _mesh_wireframe(mesh) -> np.ndarray:
    if mesh is None:
        return np.empty((0, 2, 3), dtype=np.float32)
    try:
        surface = mesh.extract_surface().triangulate().clean()
        if surface.n_cells > 5_000:
            reduction = 1.0 - 4_000.0 / float(surface.n_cells)
            surface = surface.decimate_pro(
                reduction, preserve_topology=True
            ).triangulate().clean()
        bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
        center = bounds.mean(axis=1)
        contour_segments = []
        for axis in range(3):
            normal = np.zeros(3, dtype=float)
            normal[axis] = 1.0
            # Nine coherent orthogonal contours show the full domain extent
            # without reverting to the old every-triangle visual noise.
            for fraction in (0.20, 0.50, 0.80):
                origin = center.copy()
                origin[axis] = (
                    bounds[axis, 0]
                    + fraction * (bounds[axis, 1] - bounds[axis, 0])
                )
                sliced = surface.slice(normal=normal, origin=origin)
                segments = _mesh_line_segments(sliced)
                if len(segments):
                    contour_segments.append(segments)

        feature_mesh = surface.extract_feature_edges(
            feature_angle=85.0,
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=True,
            manifold_edges=False,
        )
        features = _mesh_line_segments(feature_mesh)
        if len(features) > 600:
            # Sampling individual edge fragments creates broken, dotted
            # contours.  Prefer the coherent section curves on very intricate
            # surfaces instead.
            features = np.empty((0, 2, 3), dtype=np.float32)
        if len(features):
            contour_segments.append(features)
        if not contour_segments:
            return np.empty((0, 2, 3), dtype=np.float32)
        lines = np.concatenate(contour_segments, axis=0)
        return np.asarray(lines, dtype=np.float32)
    except Exception:
        return np.empty((0, 2, 3), dtype=np.float32)


def _mesh_triangles(mesh, *, maximum: int = 1_800) -> np.ndarray:
    if mesh is None:
        return np.empty((0, 3, 3), dtype=np.float32)
    try:
        surface = mesh.extract_surface().triangulate().clean()
        if surface.n_cells > maximum:
            reduction = 1.0 - float(maximum) / float(surface.n_cells)
            surface = surface.decimate_pro(
                reduction, preserve_topology=True
            ).triangulate().clean()
        faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
        faces = faces[faces[:, 0] == 3, 1:4]
        if len(faces) > maximum:
            faces = faces[:maximum]
        return np.asarray(surface.points[faces], dtype=np.float32)
    except Exception:
        return np.empty((0, 3, 3), dtype=np.float32)


@lru_cache(maxsize=6)
def _cached_domain_wireframe(
    path: str, _modified_ns: int, _file_size: int
) -> np.ndarray:
    return _mesh_wireframe(
        _cached_domain_surface(path, _modified_ns, _file_size)
    )


@lru_cache(maxsize=6)
def _cached_domain_triangles(
    path: str, _modified_ns: int, _file_size: int
) -> np.ndarray:
    return _mesh_triangles(
        _cached_domain_surface(path, _modified_ns, _file_size)
    )


@lru_cache(maxsize=6)
def _cached_domain_surface(path: str, _modified_ns: int, _file_size: int):
    import pyvista as pv

    return pv.read(path).extract_surface().triangulate().clean()


def _mesh_line_segments(mesh) -> np.ndarray:
    if mesh is None or not getattr(mesh, "n_points", 0):
        return np.empty((0, 2, 3), dtype=np.float32)
    points = np.asarray(mesh.points, dtype=np.float32)
    raw = np.asarray(getattr(mesh, "lines", []), dtype=np.int64)
    segments = []
    cursor = 0
    while cursor < len(raw):
        count = int(raw[cursor])
        ids = raw[cursor + 1 : cursor + 1 + count]
        if len(ids) >= 2:
            segments.extend((int(a), int(b)) for a, b in zip(ids[:-1], ids[1:]))
        cursor += count + 1
    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32)
    edge_ids = np.asarray(segments, dtype=np.int64)
    return points[edge_ids]


def _polyline_data(mesh):
    if mesh is None or not getattr(mesh, "n_points", 0):
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy(), {}, empty.copy()
    points = np.asarray(mesh.points, dtype=np.float32)
    raw = np.asarray(getattr(mesh, "lines", []), dtype=np.int64)
    first, last = [], []
    cell_first = []
    cursor = 0
    while cursor < len(raw):
        count = int(raw[cursor])
        ids = raw[cursor + 1 : cursor + 1 + count]
        if len(ids) >= 2:
            cell_first.append(int(ids[0]))
            for left, right in zip(ids[:-1], ids[1:]):
                first.append(int(left))
                last.append(int(right))
        cursor += count + 1
    if not first:
        n = len(points) // 2
        first, last = list(range(0, 2 * n, 2)), list(range(1, 2 * n, 2))
    first_arr, last_arr = np.asarray(first), np.asarray(last)
    arrays = {}
    for name, raw_values in getattr(mesh, "point_data", {}).items():
        values = np.asarray(raw_values)
        if values.ndim == 1 and np.issubdtype(values.dtype, np.number) and len(values) == len(points):
            arrays[str(name)] = (values[first_arr].astype(float) + values[last_arr].astype(float)) * 0.5
    raw_local_ids = getattr(mesh, "point_data", {}).get("local_segment_id")
    if raw_local_ids is not None and cell_first:
        raw_local_ids = np.asarray(raw_local_ids)
        inlet_ids = [point_id for point_id in cell_first if raw_local_ids[point_id] == 0]
        inlets = points[np.asarray(inlet_ids, dtype=int)] if inlet_ids else points[np.asarray(cell_first[:1], dtype=int)]
    else:
        inlets = points[np.asarray(cell_first[:1], dtype=int)] if cell_first else points[first_arr[:1]]
    return points[first_arr], points[last_arr], arrays, inlets


def _numeric_point_arrays(mesh, n):
    arrays = {}
    if mesh is None:
        return arrays
    for name, raw in getattr(mesh, "point_data", {}).items():
        values = np.asarray(raw)
        if values.ndim == 1 and len(values) == n and np.issubdtype(values.dtype, np.number):
            arrays[str(name)] = values.astype(float, copy=False)
    return arrays


def _line_array(value):
    if value is None:
        return np.empty((0, 2, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1, 2, 3) if array.size else np.empty((0, 2, 3), dtype=np.float32)


def _triangle_array(value):
    if value is None:
        return np.empty((0, 3, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return (
        array.reshape(-1, 3, 3)
        if array.size
        else np.empty((0, 3, 3), dtype=np.float32)
    )


def _point_array(value):
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1, 3) if array.size else np.empty((0, 3), dtype=np.float32)


def _values(value, n, fill=np.nan):
    if value is None:
        return None if np.isnan(fill) else np.full(n, fill, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(array) >= n:
        return array[:n]
    return np.pad(array, (0, n - len(array)), constant_values=fill)


def _normalize_with_scale(
    values: np.ndarray | None,
    requested_min: float | None = None,
    requested_max: float | None = None,
    scale: str = "linear",
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """Normalize a scalar field while keeping its displayed limits explicit."""
    if values is None or not len(values):
        return None, None
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 0.5), None
    log_scale = str(scale).lower() == "log"
    valid = finite & (values > 0.0) if log_scale else finite
    if not valid.any():
        return np.full(len(values), 0.5), None
    source = values[valid]
    auto_lo, auto_hi = np.nanpercentile(source, [2, 98])
    lo = float(requested_min) if requested_min is not None and math.isfinite(requested_min) else float(auto_lo)
    hi = float(requested_max) if requested_max is not None and math.isfinite(requested_max) else float(auto_hi)
    if log_scale and lo <= 0.0:
        # A zero linear bound cannot be represented on a log legend.  Use the
        # smallest available positive scalar instead of producing an invalid
        # scale or silently hiding the field.
        lo = float(np.nanmin(source))
    if hi <= lo:
        hi = float(np.nanmax(source))
    if hi <= lo:
        return np.full(len(values), 0.5), (lo, hi)
    if log_scale:
        logged = np.zeros(len(values), dtype=float)
        logged[valid] = np.log10(values[valid])
        norm = np.zeros(len(values), dtype=float)
        norm[valid] = (logged[valid] - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    else:
        norm = (values - lo) / (hi - lo)
    return np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0), (lo, hi)


def _scientific(value: float) -> str:
    return f"{float(value):.2e}"


def _display_field_values(kind: str, field: str, cache: dict[str, Any], options: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    """Return user-facing scalar values; never expose VTK/solver implementation names."""
    arrays = cache["vessel_arrays"] if kind == "vessel" else cache["tissue_arrays"]
    source_map = {
        "flow": "flow_ul_min", "pressure": "pressure_pa",
        "bulk_concentration": "concentration", "wall_concentration": "wall_oxygen",
        "radius": "radius_cm", "length": "length_cm", "hematocrit": "discharge_hematocrit",
        "tissue_concentration": "local_concentration", "viability": "viability", "distance": "dnc_cm",
    }
    labels = {
        "flow": "Flow Rate", "pressure": "Fluid pressure", "bulk_concentration": "Bulk concentration",
        "wall_concentration": "Wall concentration", "radius": "Radius", "length": "Length",
        "hematocrit": "Discharge hematocrit", "tissue_concentration": "Tissue concentration",
        "viability": "Viable tissue", "distance": "Distance to nearest vessel",
    }
    raw = arrays.get(source_map.get(field, field))
    if raw is None:
        return None, labels.get(field, str(field).replace("_", " ").title())
    values = np.asarray(raw, dtype=float).reshape(-1)
    if field == "flow":
        if options.get("flow_unit") == "cm3_s":
            values = values / 60_000.0
            unit = "cm³/s"
        else:
            unit = "μL/min"
    elif field == "pressure":
        values = values / 133.322387415
        unit = "mmHg"
    elif field in {"radius", "length", "distance"}:
        values = values * 10_000.0
        unit = "μm"
    elif field in {"bulk_concentration", "wall_concentration", "tissue_concentration"}:
        if options.get("concentration_unit") == "mmhg":
            values = values / 0.001408
            unit = "mmHg equivalent"
        else:
            unit = "mol/m³"
    else:
        unit = ""
    if options.get("normalize_fields") and field in {"flow", "pressure", "bulk_concentration", "wall_concentration", "tissue_concentration"}:
        settings = cache.get("settings", {})
        if field == "flow":
            denominator = float(settings.get("simulation", {}).get("qin_target_ul_min", 1.0))
            if options.get("flow_unit") == "cm3_s":
                denominator /= 60_000.0
        elif field == "pressure":
            denominator = float(settings.get("settings", {}).get("hemodynamics", {}).get("root_pressure", 1.0)) / 133.322387415
        else:
            denominator = float(settings.get("settings", {}).get("oxygen", {}).get("conc_max_for_normalization", 1.0))
            if options.get("concentration_unit") == "mmhg":
                denominator /= 0.001408
        if math.isfinite(denominator) and denominator != 0.0:
            values = values / denominator
        unit = "normalized"
    label = labels.get(field, str(field).replace("_", " ").title())
    return values, f"{label} ({unit})" if unit else label


def _layer_field_label(layer: str, field: str) -> str:
    return f"{layer}: {field}" if field else layer


def _legend_tick_values(
    limits: tuple[float, float], scale: str, count: int = 5
) -> np.ndarray:
    """Return evenly spaced legend labels in the active display space."""
    lo, hi = (float(limits[0]), float(limits[1]))
    if count < 2 or not math.isfinite(lo) or not math.isfinite(hi):
        return np.asarray([lo, hi], dtype=float)
    if str(scale).lower() == "log" and lo > 0.0 and hi > 0.0:
        return np.geomspace(lo, hi, count)
    return np.linspace(lo, hi, count)


def _map_color(name, value):
    stops = _MAP_STOPS.get(name, _MAP_STOPS["viridis"])
    value = float(np.clip(value, 0.0, 1.0))
    scaled = value * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    frac = scaled - index
    rgb = tuple(int(round(stops[index][i] * (1 - frac) + stops[index + 1][i] * frac)) for i in range(3))
    return QColor(*rgb)
