"""Qt/OpenGL vessel and tissue preview canvases."""

from __future__ import annotations

from cascade.gui.visualization.fields import (
    _legend_tick_values,
    _line_array,
    _map_color,
    _normalize_with_scale,
    _point_array,
    _scientific,
    _triangle_array,
    _values,
)

from cascade.gui.visualization.common import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPoint,
    QPointF,
    QPolygonF,
    QRectF,
    QSizePolicy,
    QTimer,
    QWidget,
    Qt,
    Signal,
    _MAP_STOPS,
    math,
    np,
    os,
)

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

    renderer_backend = "software-qpaint"
    vessel_preview_limit = 5_000
    tissue_preview_limit = 10_000

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
            tissue_screen, _depth = self._project(self.tissue_points)
            vals = self.tissue_values if self.tissue_values is not None else None
            tissue_norm, tissue_limits = _normalize_with_scale(
                vals, *self.tissue_range, self.tissue_scale
            )

            # Tissue is a translucent context layer, so draw it behind the
            # vessel/domain scene in a bounded number of QPainter calls. The
            # old path depth-sorted and painted every point independently,
            # making ordinary 10k previews needlessly sluggish while orbiting.
            self._draw_tissue_points(painter, tissue_screen, tissue_norm)

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

    def _draw_tissue_points(
        self,
        painter: QPainter,
        screen_points: np.ndarray,
        normalized_values: np.ndarray | None,
    ) -> None:
        """Render a large translucent point cloud with bounded painter calls."""
        if not len(screen_points):
            return
        color_bins = 24 if normalized_values is not None else 1
        alpha_bins = 6
        if normalized_values is None:
            color_index = np.zeros(len(screen_points), dtype=np.int16)
        else:
            normalized = np.nan_to_num(
                np.asarray(normalized_values, dtype=float), nan=0.0, posinf=1.0, neginf=0.0
            )
            color_index = np.rint(np.clip(normalized, 0.0, 1.0) * (color_bins - 1)).astype(
                np.int16
            )
        opacity = np.clip(
            self.tissue_opacity * np.asarray(self.tissue_alpha, dtype=float), 0.0, 1.0
        )
        alpha_index = np.rint(opacity * (alpha_bins - 1)).astype(np.int16)
        style_index = color_index * alpha_bins + alpha_index
        painter.save()
        for style in np.unique(style_index):
            ids = np.flatnonzero(style_index == style)
            color_slot = int(style) // alpha_bins
            alpha_slot = int(style) % alpha_bins
            color = (
                QColor(91, 221, 238)
                if normalized_values is None
                else _map_color(self.colormap, color_slot / max(color_bins - 1, 1))
            )
            color.setAlphaF(alpha_slot / (alpha_bins - 1))
            painter.setPen(QPen(color, 2.0))
            painter.drawPoints(
                QPolygonF([QPointF(*screen_points[index]) for index in ids])
            )
        painter.restore()

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




__all__ = ('FlowBackdrop', '_FlowEdge', 'GeometryCanvas')
