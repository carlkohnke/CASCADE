"""Preview counts, status text, icons, and backend selection."""

from __future__ import annotations

from cascade.gui.visualization.common import (
    Any,
    QApplication,
    QColor,
    QIcon,
    QOffscreenSurface,
    QOpenGLContext,
    QPainter,
    QPen,
    QPixmap,
    QPointF,
    QPolygonF,
    QSurfaceFormat,
    QWidget,
    Qt,
    math,
    os,
)

from cascade.gui.visualization.canvas import (
    GeometryCanvas,
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


def _count_status(
    shown_vessels: int,
    total_vessels: int,
    trees: int,
    shown_tissue: int,
    total_tissue: int,
) -> str:
    tree_label = "tree" if int(trees) == 1 else "trees"
    return (
        f"{int(shown_vessels):,} / {int(total_vessels):,} vessels shown  │  "
        f"{int(trees):,} {tree_label}  │  "
        f"{int(shown_tissue):,} / {int(total_tissue):,} tissue points shown"
    )


def _seed_count_status(
    shown_vessels: int,
    total_vessels: int,
    trees: int,
    shown_tissue: int,
    total_tissue: int,
    response: dict[str, Any] | None = None,
) -> str:
    """Use the same concise shown/actual summary for generated seed previews."""
    return _count_status(
        shown_vessels,
        total_vessels,
        trees,
        shown_tissue,
        total_tissue,
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


def _settings_icon() -> QIcon:
    """Draw a compact gear without depending on a platform icon theme."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#E6E8EE"), 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    center = QPointF(12.0, 12.0)
    painter.drawEllipse(center, 3.0, 3.0)
    painter.drawEllipse(center, 6.2, 6.2)
    for index in range(8):
        angle = index * math.pi / 4.0
        inner = QPointF(12.0 + 6.2 * math.cos(angle), 12.0 + 6.2 * math.sin(angle))
        outer = QPointF(12.0 + 8.2 * math.cos(angle), 12.0 + 8.2 * math.sin(angle))
        painter.drawLine(inner, outer)
    painter.end()
    return QIcon(pixmap)


def _create_geometry_canvas(parent: QWidget) -> QWidget:
    """Select the retained GPU renderer with a deterministic software fallback."""
    requested = os.environ.get("CASCADE_RENDER_BACKEND", "auto").strip().lower()
    platform = QApplication.platformName().lower()
    use_gpu = requested in {"auto", "gpu", "opengl"} and platform not in {
        "offscreen",
        "minimal",
    }
    if requested in {"gpu", "opengl"} and platform in {"offscreen", "minimal"}:
        raise RuntimeError(
            f"CASCADE cannot create the required OpenGL preview on Qt platform {platform!r}"
        )
    if use_gpu and _opengl_33_available():
        try:
            from .gpu_preview import OpenGLGeometryCanvas

            return OpenGLGeometryCanvas(parent)
        except (ImportError, RuntimeError):
            if requested in {"gpu", "opengl"}:
                raise
    elif use_gpu and requested in {"gpu", "opengl"}:
        raise RuntimeError("CASCADE requires an OpenGL 3.3 context for the GPU preview")
    return GeometryCanvas(parent)


def _opengl_33_available() -> bool:
    """Preflight the context version so auto mode can fall back before layout."""
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.OpenGL)
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface = QOffscreenSurface()
    surface.setFormat(surface_format)
    surface.create()
    context = QOpenGLContext()
    context.setFormat(surface_format)
    if not surface.isValid() or not context.create() or not context.isValid():
        return False
    if not context.makeCurrent(surface):
        return False
    actual = context.format()
    supported = (actual.majorVersion(), actual.minorVersion()) >= (3, 3)
    context.doneCurrent()
    return supported




__all__ = ('_preview_tree_count', '_count_status', '_seed_count_status', '_home_icon', '_settings_icon', '_create_geometry_canvas', '_opengl_33_available')
