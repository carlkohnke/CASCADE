"""Retained OpenGL renderer for the CASCADE scientific preview.

Geometry is uploaded only when the case/result changes. Camera interaction then
updates a few shader uniforms and submits a bounded number of GPU draw calls.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPen,
    QSurfaceFormat,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy, QWidget

from .visualization.common import _MAP_STOPS
from .visualization.fields import (
    _legend_tick_values,
    _line_array,
    _normalize_with_scale,
    _point_array,
    _scientific,
    _triangle_array,
    _values,
)
from .shaders import load_shader_source


GL_BLEND = 0x0BE2
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_FLOAT = 0x1406
GL_LEQUAL = 0x0203
GL_LINES = 0x0001
GL_MULTISAMPLE = 0x809D
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_POINTS = 0x0000
GL_PROGRAM_POINT_SIZE = 0x8642
GL_SRC_ALPHA = 0x0302
GL_TRIANGLES = 0x0004


_VESSEL_VERTEX = load_shader_source("vessel.vert")
_VESSEL_FRAGMENT = load_shader_source("vessel.frag")
_POINT_VERTEX = load_shader_source("point.vert")
_POINT_FRAGMENT = load_shader_source("point.frag")
_SOLID_VERTEX = load_shader_source("solid.vert")
_SOLID_FRAGMENT = load_shader_source("solid.frag")


def _color_lut(name: str, size: int = 256) -> np.ndarray:
    stops = np.asarray(_MAP_STOPS.get(name, _MAP_STOPS["viridis"]), dtype=np.float32)
    positions = np.linspace(0.0, 1.0, len(stops))
    samples = np.linspace(0.0, 1.0, size)
    table = np.column_stack(
        [np.interp(samples, positions, stops[:, channel]) for channel in range(3)]
    )
    return table / 255.0


def _rgba_values(
    values: np.ndarray | None,
    count: int,
    *,
    colormap: str,
    limits: tuple[float | None, float | None],
    scale: str,
    alpha: np.ndarray,
    opacity: float,
    default_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[float, float] | None]:
    normalized, actual_limits = _normalize_with_scale(values, *limits, scale)
    if normalized is None:
        rgb = np.broadcast_to(
            np.asarray(default_rgb, dtype=np.float32) / 255.0, (count, 3)
        )
    else:
        indexes = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = _color_lut(colormap)[indexes]
    rgba = np.empty((count, 4), dtype=np.float32)
    rgba[:, :3] = rgb
    rgba[:, 3] = np.clip(np.asarray(alpha, dtype=np.float32) * float(opacity), 0.0, 1.0)
    return rgba, actual_limits


class OpenGLGeometryCanvas(QOpenGLWidget):
    """GPU-resident line/point renderer with a constant draw-call budget."""

    selection_changed = Signal(object)
    renderer_backend = "opengl-instanced"
    vessel_preview_limit = 50_000

    def __init__(self, parent: QWidget | None = None):
        surface = QSurfaceFormat()
        surface.setRenderableType(QSurfaceFormat.OpenGL)
        surface.setVersion(3, 3)
        surface.setProfile(QSurfaceFormat.CoreProfile)
        surface.setDepthBufferSize(24)
        surface.setStencilBufferSize(8)
        surface.setSamples(4)
        super().__init__(parent)
        self.setFormat(surface)
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
        self._programs: dict[str, QOpenGLShaderProgram] = {}
        self._buffers: dict[str, QOpenGLBuffer] = {}
        self._vaos: dict[str, QOpenGLVertexArrayObject] = {}
        self._counts: dict[str, int] = {}
        self._gpu_dirty = True
        self._selection_dirty = True
        self._vessel_limits: tuple[float, float] | None = None
        self._tissue_limits: tuple[float, float] | None = None
        self.renderer_info = "OpenGL context pending"
        self.hardware_accelerated = False
        self.vessel_preview_limit = 50_000
        self.tissue_preview_limit = 50_000

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

    def clear(self, message: str = "preview paused") -> None:
        self.set_geometry(message=message)

    def release_gpu_memory(self, message: str = "preview memory released") -> None:
        """Synchronously release scene buffers before a simulation starts."""
        self.set_geometry(message=message)
        if self.isValid():
            self.makeCurrent()
            for buffer in self._buffers.values():
                buffer.destroy()
            for vao in self._vaos.values():
                vao.destroy()
            self._buffers.clear()
            self._vaos.clear()
            self._counts.clear()
            self.doneCurrent()

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
        count = min(len(self.vessel_starts), len(self.vessel_ends))
        self.vessel_starts = self.vessel_starts[:count]
        self.vessel_ends = self.vessel_ends[:count]
        self.vessel_values = _values(vessel_values, count)
        self.vessel_radii = _values(vessel_radii, count)
        self.vessel_alpha = np.clip(_values(vessel_alpha, count, fill=1.0), 0.05, 1.0)
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
        self._gpu_dirty = True
        self._selection_dirty = True
        self.update()

    def set_domain_mode(self, mode: str) -> None:
        value = str(mode).lower()
        self.domain_mode = (
            value if value in {"none", "surface", "wireframe"} else "wireframe"
        )
        self.update()

    def home(self) -> None:
        self._rotation = self._home_rotation()
        self._zoom = 0.82
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
        low, high = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
        self._center = 0.5 * (low + high)
        self._span = max(float(np.max(high - low)), 1.0e-9)

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(points):
            return np.empty((0, 2)), np.empty((0,))
        normalized = (np.asarray(points, dtype=float) - self._center) / self._span
        rotated = normalized @ self._rotation.T
        x, y, depth = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        perspective = 1.0 / np.clip(1.55 - 0.42 * depth, 0.75, 2.2)
        scale = min(self.width(), self.height()) * self._zoom
        screen = np.column_stack(
            (
                self.width() * 0.5 + x * scale * perspective,
                self.height() * 0.5 - y * scale * perspective,
            )
        )
        return screen, depth

    def initializeGL(self) -> None:
        functions = self.context().extraFunctions()
        functions.initializeOpenGLFunctions()
        self._programs = {
            "vessel": self._make_program(_VESSEL_VERTEX, _VESSEL_FRAGMENT),
            "point": self._make_program(_POINT_VERTEX, _POINT_FRAGMENT),
            "solid": self._make_program(_SOLID_VERTEX, _SOLID_FRAGMENT),
        }
        self.context().aboutToBeDestroyed.connect(self._cleanup)
        functions.glEnable(GL_DEPTH_TEST)
        functions.glDepthFunc(GL_LEQUAL)
        functions.glEnable(GL_BLEND)
        functions.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        functions.glEnable(GL_MULTISAMPLE)
        functions.glEnable(GL_PROGRAM_POINT_SIZE)
        raw_renderer = functions.glGetString(0x1F01)
        self.renderer_info = str(raw_renderer or "unknown OpenGL renderer")
        software_tokens = ("llvmpipe", "softpipe", "swiftshader", "software")
        self.hardware_accelerated = not any(
            token in self.renderer_info.lower() for token in software_tokens
        )
        self.vessel_preview_limit = 250_000 if self.hardware_accelerated else 50_000
        self.tissue_preview_limit = 1_000_000 if self.hardware_accelerated else 50_000
        self._gpu_dirty = True

    @staticmethod
    def _make_program(vertex: str, fragment: str) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram()
        if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex):
            raise RuntimeError(f"OpenGL vertex shader failed: {program.log()}")
        if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment):
            raise RuntimeError(f"OpenGL fragment shader failed: {program.log()}")
        if not program.link():
            raise RuntimeError(f"OpenGL shader link failed: {program.log()}")
        return program

    def _upload_buffer(
        self,
        name: str,
        values: np.ndarray,
        attributes: tuple[tuple[int, int, int], ...],
        *,
        stride_floats: int,
        instances: bool = False,
    ) -> None:
        functions = self.context().extraFunctions()
        vao = self._vaos.get(name)
        buffer = self._buffers.get(name)
        if not len(values):
            if buffer is not None:
                buffer.destroy()
                self._buffers.pop(name, None)
            if vao is not None:
                vao.destroy()
                self._vaos.pop(name, None)
            return
        if vao is None:
            vao = QOpenGLVertexArrayObject(self)
            if not vao.create():
                raise RuntimeError(f"Could not create OpenGL vertex array for {name}")
            self._vaos[name] = vao
        if buffer is None:
            buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            if not buffer.create():
                raise RuntimeError(f"Could not create OpenGL vertex buffer for {name}")
            buffer.setUsagePattern(QOpenGLBuffer.DynamicDraw)
            self._buffers[name] = buffer
        contiguous = np.ascontiguousarray(values, dtype=np.float32)
        program_name = (
            "vessel"
            if name in {"vessels", "selection"}
            else "point"
            if name in {"tissue", "boundary"}
            else "solid"
        )
        program = self._programs[program_name]
        program.bind()
        vao.bind()
        if not buffer.bind():
            raise RuntimeError(f"Could not bind OpenGL vertex buffer for {name}")
        payload = contiguous.tobytes()
        buffer.allocate(payload, len(payload))
        stride = stride_floats * np.dtype(np.float32).itemsize
        for location, size, offset_floats in attributes:
            program.enableAttributeArray(location)
            program.setAttributeBuffer(
                location,
                GL_FLOAT,
                offset_floats * np.dtype(np.float32).itemsize,
                size,
                stride,
            )
            functions.glVertexAttribDivisor(location, 1 if instances else 0)
        buffer.release()
        vao.release()
        program.release()

    def _upload_geometry(self) -> None:
        vessel_count = len(self.vessel_starts)
        vessel_colors, self._vessel_limits = _rgba_values(
            self.vessel_values,
            vessel_count,
            colormap=self.colormap,
            limits=self.vessel_range,
            scale=self.vessel_scale,
            alpha=self.vessel_alpha,
            opacity=self.vessel_opacity,
            default_rgb=(185, 181, 190),
        )
        radii = (
            np.asarray(self.vessel_radii, dtype=np.float32)
            if self.vessel_radii is not None
            else np.full(vessel_count, -1.0, dtype=np.float32)
        )
        vessel_data = np.column_stack(
            (self.vessel_starts, self.vessel_ends, vessel_colors, radii)
        ).astype(np.float32, copy=False)
        self._upload_buffer(
            "vessels",
            vessel_data,
            ((0, 3, 0), (1, 3, 3), (2, 4, 6), (3, 1, 10)),
            stride_floats=11,
            instances=True,
        )
        self._counts["vessels"] = vessel_count

        tissue_count = len(self.tissue_points)
        tissue_colors, self._tissue_limits = _rgba_values(
            self.tissue_values,
            tissue_count,
            colormap=self.colormap,
            limits=self.tissue_range,
            scale=self.tissue_scale,
            alpha=self.tissue_alpha,
            opacity=self.tissue_opacity,
            default_rgb=(91, 221, 238),
        )
        tissue_data = np.column_stack((self.tissue_points, tissue_colors)).astype(
            np.float32, copy=False
        )
        self._upload_buffer(
            "tissue",
            tissue_data,
            ((0, 3, 0), (1, 4, 3)),
            stride_floats=7,
        )
        self._counts["tissue"] = tissue_count

        boundary_points = np.concatenate(
            (self.inlet_points, self.outlet_points), axis=0
        )
        inlet_colors = np.tile(
            np.asarray((0.435, 0.890, 0.737, 0.95)), (len(self.inlet_points), 1)
        )
        outlet_colors = np.tile(
            np.asarray((1.0, 0.616, 0.471, 0.95)), (len(self.outlet_points), 1)
        )
        boundary_colors = np.concatenate((inlet_colors, outlet_colors), axis=0)
        boundary_data = np.column_stack((boundary_points, boundary_colors)).astype(
            np.float32, copy=False
        )
        self._upload_buffer(
            "boundary",
            boundary_data,
            ((0, 3, 0), (1, 4, 3)),
            stride_floats=7,
        )
        self._counts["boundary"] = len(boundary_points)

        domain_lines = self.domain_lines.reshape(-1, 3)
        self._upload_buffer("domain_lines", domain_lines, ((0, 3, 0),), stride_floats=3)
        self._counts["domain_lines"] = len(domain_lines)
        domain_triangles = self.domain_triangles.reshape(-1, 3)
        self._upload_buffer(
            "domain_triangles", domain_triangles, ((0, 3, 0),), stride_floats=3
        )
        self._counts["domain_triangles"] = len(domain_triangles)
        self._gpu_dirty = False
        self._selection_dirty = True

    def _upload_selection(self) -> None:
        if 0 <= self._selected_vessel < len(self.vessel_starts):
            index = self._selected_vessel
            radius = (
                float(self.vessel_radii[index])
                if self.vessel_radii is not None
                else -1.0
            )
            values = np.asarray(
                [
                    [
                        *self.vessel_starts[index],
                        *self.vessel_ends[index],
                        0.965,
                        0.722,
                        0.290,
                        1.0,
                        radius,
                    ]
                ],
                dtype=np.float32,
            )
        else:
            values = np.empty((0, 11), dtype=np.float32)
        self._upload_buffer(
            "selection",
            values,
            ((0, 3, 0), (1, 3, 3), (2, 4, 6), (3, 1, 10)),
            stride_floats=11,
            instances=True,
        )
        self._counts["selection"] = len(values)
        if 0 <= self._selected_tissue < len(self.tissue_points):
            tissue = np.asarray(
                [
                    [
                        *self.tissue_points[self._selected_tissue],
                        0.965,
                        0.722,
                        0.290,
                        1.0,
                    ]
                ],
                dtype=np.float32,
            )
        else:
            tissue = np.empty((0, 7), dtype=np.float32)
        self._upload_buffer(
            "selected_tissue",
            tissue,
            ((0, 3, 0), (1, 4, 3)),
            stride_floats=7,
        )
        self._counts["selected_tissue"] = len(tissue)
        self._selection_dirty = False

    def _set_common_uniforms(self, program: QOpenGLShaderProgram) -> None:
        width = max(float(self.width() * self.devicePixelRatioF()), 1.0)
        height = max(float(self.height() * self.devicePixelRatioF()), 1.0)
        minimum = min(width, height)
        self._set_uniform(program, "u_center", *map(float, self._center))
        for index, name in enumerate(("u_rotation_0", "u_rotation_1", "u_rotation_2")):
            self._set_uniform(program, name, *map(float, self._rotation[index]))
        self._set_uniform(program, "u_span", float(self._span))
        self._set_uniform(
            program,
            "u_xy_scale",
            2.0 * minimum * self._zoom / width,
            2.0 * minimum * self._zoom / height,
        )

    @staticmethod
    def _set_uniform(program: QOpenGLShaderProgram, name: str, *values: Any) -> None:
        location = program.uniformLocation(name)
        if location >= 0:
            if len(values) == 1 and isinstance(values[0], (float, np.floating)):
                program.setUniformValue1f(location, float(values[0]))
            else:
                program.setUniformValue(location, *values)

    def _draw_points(self, name: str, size: float) -> None:
        count = self._counts.get(name, 0)
        if not count:
            return
        program = self._programs["point"]
        program.bind()
        self._set_common_uniforms(program)
        self._set_uniform(
            program, "u_point_size", float(size * self.devicePixelRatioF())
        )
        self._vaos[name].bind()
        self.context().functions().glDrawArrays(GL_POINTS, 0, count)
        self._vaos[name].release()
        program.release()

    def _draw_vessels(self, name: str, width_boost: float = 0.0) -> None:
        count = self._counts.get(name, 0)
        if not count:
            return
        functions = self.context().extraFunctions()
        program = self._programs["vessel"]
        program.bind()
        self._set_common_uniforms(program)
        width = max(float(self.width() * self.devicePixelRatioF()), 1.0)
        height = max(float(self.height() * self.devicePixelRatioF()), 1.0)
        positive = (
            np.asarray(self.vessel_radii, dtype=float)
            if self.vessel_radii is not None
            else np.empty((0,))
        )
        positive = positive[np.isfinite(positive) & (positive > 0)]
        self._set_uniform(program, "u_pixel_scale", 2.0 / width, 2.0 / height)
        self._set_uniform(program, "u_min_viewport", min(width, height))
        self._set_uniform(program, "u_zoom", float(self._zoom))
        self._set_uniform(
            program, "u_max_radius", float(np.max(positive)) if len(positive) else 0.0
        )
        self._set_uniform(
            program, "u_width_boost", float(width_boost * self.devicePixelRatioF())
        )
        self._vaos[name].bind()
        functions.glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count)
        self._vaos[name].release()
        program.release()

    def _draw_solid(
        self, name: str, mode: int, color: QColor, width: float = 1.0
    ) -> None:
        count = self._counts.get(name, 0)
        if not count:
            return
        functions = self.context().extraFunctions()
        program = self._programs["solid"]
        program.bind()
        self._set_common_uniforms(program)
        self._set_uniform(
            program,
            "u_color",
            color.redF(),
            color.greenF(),
            color.blueF(),
            color.alphaF(),
        )
        functions.glLineWidth(float(width * self.devicePixelRatioF()))
        self._vaos[name].bind()
        self.context().functions().glDrawArrays(mode, 0, count)
        self._vaos[name].release()
        program.release()

    def paintGL(self) -> None:
        functions = self.context().extraFunctions()
        functions.glClearColor(0.018, 0.020, 0.028, 1.0)
        functions.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._gpu_dirty:
            self._upload_geometry()
        if self._selection_dirty:
            self._upload_selection()
        functions.glDepthMask(False)
        self._draw_points("tissue", 2.2)
        if self.domain_mode == "surface":
            self._draw_solid("domain_triangles", GL_TRIANGLES, QColor(139, 68, 198, 44))
        functions.glDepthMask(True)
        if self.domain_mode == "wireframe":
            self._draw_solid("domain_lines", GL_LINES, QColor(190, 100, 238, 210), 1.3)
        self._draw_vessels("vessels")
        self._draw_points("boundary", 10.0)
        self._draw_vessels("selection", 3.0)
        self._draw_points("selected_tissue", 7.0)
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.vessel_values is not None and self._vessel_limits is not None:
            self._draw_scalar_legend(
                painter,
                self.vessel_label,
                self._vessel_limits,
                side="left",
                scale=self.vessel_scale,
            )
        if self.tissue_values is not None and self._tissue_limits is not None:
            self._draw_scalar_legend(
                painter,
                self.tissue_label,
                self._tissue_limits,
                side="right",
                scale=self.tissue_scale,
            )
        self._draw_boundary_labels(painter)
        if self._has_geometry():
            self._draw_scale_bar(painter)
        else:
            painter.setPen(QColor("#B9BBC3"))
            painter.drawText(
                self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message
            )
        painter.end()

    def _draw_boundary_labels(self, painter: QPainter) -> None:
        for points, label, color in (
            (self.inlet_points, "IN", QColor("#6FE3BC")),
            (self.outlet_points, "OUT", QColor("#FF9D78")),
        ):
            screen, _ = self._project(points)
            painter.setPen(color)
            for index, point in enumerate(screen):
                painter.drawText(
                    QPointF(point[0] + 8.0, point[1] - 6.0), f"{label}{index + 1}"
                )

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
        width = max(
            180.0, min(340.0, (float(self.width()) - 2.0 * margin - 40.0) / 2.0)
        )
        height = 12.0
        left = margin if side == "left" else float(self.width()) - width - margin
        font = painter.font()
        font.setPixelSize(13)
        painter.setFont(font)
        painter.fillRect(
            QRectF(left - 7.0, top - 6.0, width + 14.0, 58.0), QColor(7, 8, 11, 218)
        )
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
        tick_font = painter.font()
        tick_font.setPixelSize(9)
        painter.setFont(tick_font)
        painter.setPen(QPen(QColor(205, 208, 216, 225), 0.8))
        metrics = painter.fontMetrics()
        for index, value in enumerate(_legend_tick_values(limits, scale)):
            fraction = index / 4.0
            x = left + width * fraction
            painter.drawLine(
                QPointF(x, bar_top + height), QPointF(x, bar_top + height + 4.0)
            )
            text = _scientific(value)
            text_width = float(metrics.horizontalAdvance(text))
            text_left = min(max(x - text_width / 2.0, left), left + width - text_width)
            painter.drawText(
                QRectF(text_left, bar_top + height + 6.0, text_width, 12.0),
                Qt.AlignHCenter,
                text,
            )
        if scale == "log":
            painter.setPen(QColor(201, 169, 234, 230))
            painter.drawText(QRectF(left, top, width, 14), Qt.AlignRight, "LOG")

    def _has_geometry(self) -> bool:
        return bool(
            self.domain_lines.size
            or self.domain_triangles.size
            or self.vessel_starts.size
            or self.tissue_points.size
        )

    def _scale_bar_spec(self) -> tuple[float, str]:
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
        return float(pixels), f"{value:g} {unit}"

    def _draw_scale_bar(self, painter: QPainter) -> None:
        pixels, label = self._scale_bar_spec()
        right = float(self.width() - 18)
        left = max(18.0, right - pixels)
        baseline = float(self.height() - 18)
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(
            QPen(QColor(224, 226, 232, 220), 1.4, Qt.SolidLine, Qt.SquareCap)
        )
        painter.drawLine(QPointF(left, baseline), QPointF(right, baseline))
        painter.drawLine(QPointF(left, baseline - 4), QPointF(left, baseline + 4))
        painter.drawLine(QPointF(right, baseline - 4), QPointF(right, baseline + 4))
        painter.drawText(
            QRectF(left - 20, baseline - 22, (right - left) + 40, 16),
            Qt.AlignCenter,
            label,
        )

    def _radius_widths(self) -> np.ndarray:
        count = len(self.vessel_starts)
        if self.vessel_radii is None or not count:
            return np.full(count, 1.7, dtype=float)
        radii = np.asarray(self.vessel_radii, dtype=float)
        positive = radii[np.isfinite(radii) & (radii > 0)]
        if not len(positive):
            return np.full(count, 1.7, dtype=float)
        reference = max(float(np.nanmax(positive)), 1e-12)
        ratio = np.clip(np.nan_to_num(radii / reference, nan=0.0), 0.0, 1.0)
        screen_scale = max(min(self.width(), self.height()) * self._zoom, 1.0)
        physical = 2.0 * np.maximum(radii, 0.0) * screen_scale / max(self._span, 1e-12)
        return np.clip(
            np.nan_to_num(0.75 + physical + 0.55 * np.sqrt(ratio), nan=0.8), 0.8, 12.0
        )

    def _orbit(self, horizontal: float, vertical: float) -> None:
        self._rotation = (
            self._rotation_x(vertical) @ self._rotation_y(horizontal) @ self._rotation
        )

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

    @staticmethod
    def _nearest_point_index(
        screen: np.ndarray, click: np.ndarray
    ) -> tuple[int, float]:
        best_index, best_distance = -1, float("inf")
        chunk = 100_000
        for start in range(0, len(screen), chunk):
            distances = np.linalg.norm(screen[start : start + chunk] - click, axis=1)
            local = int(np.argmin(distances))
            distance = float(distances[local])
            if distance < best_distance:
                best_index, best_distance = start + local, distance
        return best_index, best_distance

    def _pick(self, point: QPointF) -> None:
        click = np.asarray([point.x(), point.y()], dtype=float)
        best_kind, best_index, best_distance = "", -1, 9.0
        if self.vessel_starts.size:
            starts, _ = self._project(self.vessel_starts)
            ends, _ = self._project(self.vessel_ends)
            delta = ends - starts
            denominator = np.sum(delta * delta, axis=1)
            along = np.sum((click - starts) * delta, axis=1) / np.maximum(
                denominator, 1e-12
            )
            projection = starts + np.clip(along, 0.0, 1.0)[:, None] * delta
            index, distance = self._nearest_point_index(projection, click)
            if distance < best_distance:
                best_kind, best_index, best_distance = "vessel", index, distance
        if self.tissue_points.size:
            screen, _ = self._project(self.tissue_points)
            index, distance = self._nearest_point_index(screen, click)
            if distance < best_distance:
                best_kind, best_index = "tissue", index
        self._selected_vessel = best_index if best_kind == "vessel" else -1
        self._selected_tissue = best_index if best_kind == "tissue" else -1
        self._selection_dirty = True
        self.update()
        self.selection_changed.emit(
            {"kind": best_kind, "index": best_index} if best_kind else {}
        )

    def wheelEvent(self, event) -> None:
        self._zoom = float(
            np.clip(self._zoom * np.exp(event.angleDelta().y() / 1100.0), 0.18, 4.0)
        )
        self.update()

    def _cleanup(self) -> None:
        if not self.context() or not self.context().isValid():
            return
        self.makeCurrent()
        for buffer in self._buffers.values():
            buffer.destroy()
        for vao in self._vaos.values():
            vao.destroy()
        self._buffers.clear()
        self._vaos.clear()
        self._programs.clear()
        self.doneCurrent()

    def closeEvent(self, event) -> None:
        self._cleanup()
        super().closeEvent(event)
