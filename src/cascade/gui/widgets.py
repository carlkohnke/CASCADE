"""Reusable CASCADE Studio controls, path pickers, status displays, and native dialogs."""

from __future__ import annotations

import base64
import html
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from PySide6.QtCore import QEvent, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .theme import Tokens


_ACTIVE_NATIVE_PICKERS: set[subprocess.Popen] = set()
_CANCELLED_NATIVE_PICKERS: set[subprocess.Popen] = set()
_NATIVE_PICKER_SOURCE = Path(__file__).with_name("native") / "CascadePickerNative.cs"


def _windows_powershell_executable() -> str:
    """Locate Windows PowerShell even when WSL omits Windows paths from PATH."""
    discovered = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if discovered:
        return discovered
    for candidate in (
        Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"),
        Path("/mnt/C/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Windows PowerShell was not found in PATH or under the mounted Windows directory."
    )


class ChoiceComboBox(QComboBox):
    """Combo box whose list stays inside the main window.

    WSLg can paint a native Qt popup at a different scale/position than its mouse
    hit-testing surface. Keeping the list in the same window avoids that entire
    compositor path.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._choice_frame: QFrame | None = None
        self._choice_list: QListWidget | None = None

    def _ensure_choice_list(self) -> None:
        host = self.window()
        if self._choice_frame is not None and self._choice_frame.parent() is host:
            return
        if self._choice_frame is not None:
            self._choice_frame.deleteLater()

        frame = QFrame(host)
        frame.setObjectName("choiceOverlay")
        frame.setStyleSheet(
            f"QFrame#choiceOverlay {{background:{Tokens.SURFACE_2};border:1px solid {Tokens.BORDER_ACTIVE};"
            "border-radius:7px;}"
            f"QListWidget {{background:{Tokens.SURFACE_2};color:{Tokens.TEXT};border:0;outline:0;"
            "padding:3px;}"
            "QListWidget::item {min-height:28px;padding:4px 8px;}"
            f"QListWidget::item:hover {{background:{Tokens.HOVER};color:{Tokens.TEXT};}}"
            f"QListWidget::item:selected {{background:{Tokens.VIOLET};color:white;}}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(1, 1, 1, 1)
        choices = QListWidget(frame)
        choices.setMouseTracking(True)
        choices.viewport().setMouseTracking(True)
        choices.setAutoScroll(False)
        choices.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        choices.viewport().installEventFilter(self)
        choices.itemEntered.connect(choices.setCurrentItem)
        choices.itemClicked.connect(self._choose_item)
        choices.itemActivated.connect(self._choose_item)
        layout.addWidget(choices)
        self._choice_frame = frame
        self._choice_list = choices

    def showPopup(self) -> None:
        if not self.isEnabled() or self.count() == 0:
            return
        self._ensure_choice_list()
        if self._choice_frame is None or self._choice_list is None:
            raise RuntimeError("Choice-list popup initialization failed.")

        choices = self._choice_list
        choices.clear()
        choices.addItems([self.itemText(i) for i in range(self.count())])
        choices.setCurrentRow(self.currentIndex())
        choices.ensurePolished()

        host = self.window()
        text_width = max(
            (
                choices.fontMetrics().horizontalAdvance(self.itemText(i))
                for i in range(self.count())
            ),
            default=0,
        )
        width = min(
            max(self.width(), text_width + 34, 190), max(host.width() - 12, 190)
        )
        visible_rows = min(self.count(), 8)
        rows_height = sum(choices.sizeHintForRow(i) for i in range(visible_rows))
        list_margins = choices.contentsMargins()
        layout_margins = self._choice_frame.layout().contentsMargins()
        height = (
            rows_height
            + list_margins.top()
            + list_margins.bottom()
            + layout_margins.top()
            + layout_margins.bottom()
            + 2 * self._choice_frame.frameWidth()
        )
        choices.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff if self.count() <= 8 else Qt.ScrollBarAsNeeded
        )
        below = self.mapTo(host, QPoint(0, self.height()))
        above_y = self.mapTo(host, QPoint(0, 0)).y() - height
        x = max(0, min(below.x(), host.width() - width))
        y = below.y() if below.y() + height <= host.height() else max(0, above_y)
        self._choice_frame.setGeometry(x, y, width, height)
        self._choice_frame.raise_()
        self._choice_frame.show()
        choices.scrollToItem(choices.currentItem())
        choices.setFocus(Qt.PopupFocusReason)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _choose_item(self, item) -> None:
        if self._choice_list is None:
            return
        row = self._choice_list.row(item)
        if row < 0:
            return
        self.setCurrentIndex(row)
        self.activated.emit(row)
        self.hidePopup()

    def eventFilter(self, watched, event) -> bool:
        """Dismiss the in-window list on an outside click or Escape."""
        frame = self._choice_frame
        if frame is not None and frame.isVisible():
            if (
                self._choice_list is not None
                and watched is self._choice_list.viewport()
                and event.type() in (QEvent.MouseMove, QEvent.HoverMove)
            ):
                item = self._choice_list.itemAt(event.position().toPoint())
                if item is not None:
                    self._choice_list.setCurrentItem(item)
            if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
                self.hidePopup()
                return True
            if event.type() == QEvent.MouseButtonPress and isinstance(watched, QWidget):
                in_list = watched is frame or frame.isAncestorOf(watched)
                in_combo = watched is self or self.isAncestorOf(watched)
                if not in_list and not in_combo:
                    self.hidePopup()
            if event.type() in (QEvent.WindowDeactivate, QEvent.ApplicationDeactivate):
                self.hidePopup()
        return super().eventFilter(watched, event)

    def hidePopup(self) -> None:
        if self._choice_frame is not None:
            self._choice_frame.hide()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)


class Select(ChoiceComboBox):
    """CASCADE select primitive."""


class NumberInput(QDoubleSpinBox):
    """Numerical input that displays meaningful digits without zero padding."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        significant_digits: int = 6,
    ):
        self._significant_digits = max(1, int(significant_digits))
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignRight)
        self.setProperty("numericField", True)

    def setSignificantDigits(self, digits: int) -> None:
        self._significant_digits = max(1, int(digits))
        self.lineEdit().setText(self.textFromValue(self.value()))

    def textFromValue(self, value: float) -> str:
        if not math.isfinite(value):
            return super().textFromValue(value)
        if value == 0.0:
            return "0"
        text = format(float(value), f".{self._significant_digits}g")
        if "e" in text:
            mantissa, exponent = text.split("e", 1)
            text = f"{mantissa}e{int(exponent)}"
        return text


class Toggle(QCheckBox):
    """CASCADE boolean input primitive."""


class Button(QPushButton):
    """Flat CASCADE action button with semantic variants."""

    def __init__(self, text: str = "", variant: str = "primary", parent=None):
        super().__init__(text, parent)
        if variant == "secondary":
            self.setProperty("secondary", True)
        elif variant == "danger":
            self.setProperty("danger", True)


class IconButton(Button):
    """Compact button primitive for viewport tools."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, "secondary", parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


class FocusPlainTextEdit(QPlainTextEdit):
    """Multiline editor where Tab follows the form instead of inserting a tab."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setTabChangesFocus(True)


class Panel(QFrame):
    """Neutral application surface primitive."""


class SectionHeader(QLabel):
    """Consistent inspector section heading."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("cardTitle")


class InfoTip(QLabel):
    """Small, hover-only access to contextual help."""

    def __init__(
        self,
        text: str,
        accessible_name: str = "More information",
        parent: QWidget | None = None,
    ):
        super().__init__("i", parent)
        self.setObjectName("infoTip")
        self.setToolTip(text)
        self.setAccessibleName(accessible_name)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(15, 15)

    def _show_tip(self) -> None:
        window = self.window()
        anchor = self.mapToGlobal(QPoint(self.width() + 5, -3))
        screen = QGuiApplication.screenAt(anchor) or window.screen()
        screen_bounds = screen.availableGeometry() if screen is not None else QRect()
        window_bounds = window.frameGeometry()
        # Prefer the application window, but do not make a tiny/minimized
        # window produce a tooltip that is narrower than a readable sentence.
        bounds = window_bounds if window_bounds.width() >= 260 else screen_bounds
        if not bounds.isValid():
            bounds = screen_bounds
        max_wrap_width = min(360, max(180, bounds.width() - 28))
        natural_rect = self.fontMetrics().boundingRect(self.toolTip())
        # Use a compact bubble for short help.  Reserving the full maximum
        # width here was what made an otherwise small tooltip jump far from
        # the information icon near the right edge of a panel.
        content_width = min(
            max_wrap_width,
            max(150, natural_rect.width() + 8),
        )
        text = html.escape(self.toolTip()).replace("\n", "<br>")
        tip = f"<div style='width:{content_width}px; white-space:normal;'>{text}</div>"
        text_rect = self.fontMetrics().boundingRect(
            QRect(0, 0, content_width, 2000), Qt.TextWordWrap, self.toolTip()
        )
        tip_width = min(content_width + 18, max(120, bounds.width() - 8))
        tip_height = text_rect.height() + 18
        x = anchor.x()
        if x + tip_width > bounds.right() - 3:
            x = self.mapToGlobal(QPoint(-tip_width - 5, -3)).x()
        x = min(max(x, bounds.left() + 4), bounds.right() - tip_width - 3)
        y = anchor.y()
        if y + tip_height > bounds.bottom() - 3:
            y = self.mapToGlobal(QPoint(self.width() + 5, -tip_height - 4)).y()
        y = min(max(y, bounds.top() + 4), bounds.bottom() - tip_height - 3)
        QToolTip.showText(QPoint(x, y), tip, self)

    def enterEvent(self, event) -> None:
        """Show help immediately, aligned tightly with the information icon."""
        self._show_tip()
        super().enterEvent(event)

    def event(self, event) -> bool:
        # Consume Qt's delayed tooltip event so it cannot later move the help
        # bubble back to the platform's default lower-right position.
        if event.type() == QEvent.ToolTip:
            self._show_tip()
            return True
        return super().event(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)


class InspectorSection(Panel):
    """A quiet parameter group, not a floating dashboard card."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 10, 0, 12)
        self.layout.setSpacing(7)
        heading = SectionHeader(title)
        heading_row = QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)
        heading_row.setSpacing(5)
        heading_row.addWidget(heading)
        if subtitle:
            heading_row.addWidget(InfoTip(subtitle, f"About {title}"))
        heading_row.addStretch()
        self.heading_row = heading_row
        self.layout.addLayout(heading_row)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.layout.addWidget(widget, stretch)
        return widget


class Card(InspectorSection):
    """Compatibility name used by the existing scientific forms."""


class Banner(QFrame):
    def __init__(
        self, text: str = "", tone: str = "info", parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setObjectName("banner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 4, 4, 4)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.set_tone(tone)

    def set_message(self, text: str, tone: str = "info") -> None:
        self.label.setText(text)
        self.set_tone(tone)
        self.setVisible(bool(text))

    def set_tone(self, tone: str) -> None:
        colors = {
            "info": (Tokens.TEXT_2, Tokens.PURPLE),
            "success": ("#7AD3B7", "#285245"),
            "warning": ("#F2C878", "#5E4925"),
            "danger": ("#F1A1AB", "#60303A"),
        }
        fg, border = colors.get(tone, colors["info"])
        self.setStyleSheet(
            f"QFrame#banner {{background:transparent;border:none;border-left:2px solid {border};border-radius:0;}}"
            f"QFrame#banner QLabel {{color:{fg};font-weight:550;background:transparent;}}"
        )


class PathPicker(QWidget):
    changed = Signal(str)

    def __init__(
        self,
        mode: str = "file",
        caption: str = "Choose file",
        file_filter: str = "All files (*)",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode, self.caption, self.file_filter = mode, caption, file_filter
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Choose a path…")
        self.edit.textChanged.connect(self.changed)
        button = QPushButton("Browse")
        button.setProperty("secondary", True)
        button.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value or "")

    def _browse(self) -> None:
        start = self.edit.text().strip() or str(Path.home())
        selected = choose_native_path(
            self,
            mode=self.mode,
            caption=self.caption,
            start=start,
            file_filter=self.file_filter,
        )
        if selected:
            self.edit.setText(selected)


def choose_native_path(
    parent: QWidget | None,
    *,
    mode: str,
    caption: str,
    start: str,
    file_filter: str = "All files (*)",
) -> str:
    """Open the host-native chooser and return a path usable by CASCADE.

    A Qt application running in WSL considers Linux to be its native desktop.
    On that platform we deliberately bridge to Windows Forms so users get the
    normal Windows Explorer search, locations, and navigation experience.  The
    selected Windows or WSL UNC path is converted back to a Linux path.
    """
    if _running_in_wsl():
        try:
            return _windows_native_picker(
                parent=parent,
                mode=mode,
                caption=caption,
                start=start,
                file_filter=file_filter,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            if parent is not None and parent.window().isVisible():
                QMessageBox.critical(
                    parent,
                    "Windows file picker unavailable",
                    "CASCADE could not open the Windows file picker. The Linux/Qt "
                    "fallback was not opened because CASCADE is running under WSL.\n\n"
                    f"{exc}",
                )
            return ""
    else:
        options = QFileDialog.Options()
    if mode == "directory":
        return QFileDialog.getExistingDirectory(parent, caption, start, options=options)
    if mode == "save":
        selected, _ = QFileDialog.getSaveFileName(
            parent, caption, start, file_filter, options=options
        )
    else:
        selected, _ = QFileDialog.getOpenFileName(
            parent, caption, start, file_filter, options=options
        )
    return selected


def _running_in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    for marker in (Path("/proc/sys/kernel/osrelease"), Path("/proc/version")):
        try:
            if "microsoft" in marker.read_text(encoding="utf-8").lower():
                return True
        except OSError:
            continue
    return False


def _windows_native_picker(
    *,
    mode: str,
    caption: str,
    start: str,
    file_filter: str,
    parent: QWidget | None = None,
) -> str:
    initial = Path(start).expanduser()
    if initial.is_file() or (mode == "save" and initial.suffix):
        initial = initial.parent
    initial_windows = _to_windows_path(str(initial))
    title = _ps_quote(caption)
    initial_ps = _ps_quote(initial_windows)
    owner_setup = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$owner = New-Object System.Windows.Forms.Form
$owner.Text = 'CASCADE Studio file selection'
$owner.ShowInTaskbar = $false
$owner.TopMost = $true
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Size = New-Object System.Drawing.Size(2, 2)
$owner.Opacity = 0.01
$owner.Show()
$owner.Activate()
"""
    owner_cleanup = """
$owner.Close()
$owner.Dispose()
"""

    if mode == "directory":
        native_source = base64.b64encode(
            _NATIVE_PICKER_SOURCE.read_bytes()
        ).decode("ascii")
        script = (
            owner_setup
            + f"""
$nativeSource = [System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String('{native_source}')
)
Add-Type -TypeDefinition $nativeSource -Language CSharp
$selected = [CascadePickerNative]::PickFolder(
    $owner.Handle,
    '{title}',
    '{initial_ps}'
)
if (-not [String]::IsNullOrWhiteSpace($selected)) {{
    [Console]::Write($selected)
}}
"""
            + owner_cleanup
        )
    else:
        dialog_class = "SaveFileDialog" if mode == "save" else "OpenFileDialog"
        filter_ps = _ps_quote(_qt_filter_to_windows(file_filter))
        extra = (
            "$dialog.OverwritePrompt = $true\n$dialog.CheckPathExists = $true"
            if mode == "save"
            else "$dialog.Multiselect = $false\n$dialog.CheckFileExists = $true"
        )
        script = (
            owner_setup
            + f"""
$dialog = New-Object System.Windows.Forms.{dialog_class}
$dialog.Title = '{title}'
$dialog.InitialDirectory = '{initial_ps}'
$dialog.Filter = '{filter_ps}'
$dialog.RestoreDirectory = $true
{extra}
$result = $dialog.ShowDialog($owner)
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::Write($dialog.FileName)
}}
"""
            + owner_cleanup
        )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    process = subprocess.Popen(
        [
            _windows_powershell_executable(),
            "-NoProfile",
            "-STA",
            "-EncodedCommand",
            encoded,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _ACTIVE_NATIVE_PICKERS.add(process)
    try:
        app = QApplication.instance()
        while process.poll() is None:
            if app is not None:
                app.processEvents()
            if parent is not None and not parent.window().isVisible():
                process.terminate()
                break
            time.sleep(0.025)
        stdout, stderr = process.communicate(timeout=2)
        if process.returncode and process in _CANCELLED_NATIVE_PICKERS:
            return ""
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                output=stdout,
                stderr=stderr,
            )
        selected = stdout.decode("utf-8", errors="strict").strip().lstrip("\ufeff")
        return _to_wsl_path(selected) if selected else ""
    finally:
        _ACTIVE_NATIVE_PICKERS.discard(process)
        _CANCELLED_NATIVE_PICKERS.discard(process)


def cancel_native_pickers() -> None:
    """Terminate native chooser bridges when CASCADE exits."""
    for process in tuple(_ACTIVE_NATIVE_PICKERS):
        if process.poll() is None:
            _CANCELLED_NATIVE_PICKERS.add(process)
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        _ACTIVE_NATIVE_PICKERS.discard(process)


def _to_windows_path(path: str) -> str:
    result = subprocess.run(
        ["wslpath", "-w", path], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _windows_temp_directory() -> str:
    result = subprocess.run(
        [
            _windows_powershell_executable(),
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
            "[Console]::Write([IO.Path]::GetTempPath())",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="strict").strip().lstrip("\ufeff")


def _to_wsl_path(path: str) -> str:
    result = subprocess.run(
        ["wslpath", "-u", path], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _ps_quote(value: str) -> str:
    return str(value).replace("'", "''")


def _qt_filter_to_windows(value: str) -> str:
    parts = []
    for raw_group in str(value or "All files (*)").split(";;"):
        group = raw_group.strip()
        if group.endswith(")") and "(" in group:
            label, patterns = group.rsplit("(", 1)
            patterns = patterns[:-1].strip().replace(" ", ";")
            if patterns == "*":
                patterns = "*.*"
            parts.extend((label.strip(), patterns))
        else:
            parts.extend((group or "All files", "*.*"))
    return "|".join(parts)


class UnitValue(QWidget):
    valueChanged = Signal(float)
    unitChanged = Signal(str)

    def __init__(
        self,
        units: list[str],
        value: float = 0.0,
        *,
        minimum: float = 0.0,
        maximum: float = 1e12,
        decimals: int = 6,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.spin = NumberInput()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(value)
        self.spin.setKeyboardTracking(False)
        self.combo = ChoiceComboBox()
        self.combo.addItems(units)
        self.combo.setMinimumWidth(92)
        row.addWidget(self.spin, 1)
        row.addWidget(self.combo)
        self.spin.valueChanged.connect(self.valueChanged)
        self.combo.currentTextChanged.connect(self.unitChanged)

    def value(self) -> float:
        return self.spin.value()

    def unit(self) -> str:
        return self.combo.currentText()

    def setValue(self, value: float) -> None:
        self.spin.setValue(float(value))

    def setUnit(self, unit: str) -> None:
        index = self.combo.findText(unit)
        if index >= 0:
            self.combo.setCurrentIndex(index)


class StatusPill(QLabel):
    clicked = Signal()

    def __init__(
        self, text: str = "Ready", tone: str = "neutral", parent: QWidget | None = None
    ):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Setup status and issues")
        self.set_tone(tone)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_status(self, text: str, tone: str) -> None:
        self.setText(text)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        colors = {
            "neutral": (Tokens.SURFACE_2, Tokens.TEXT_2),
            "info": ("#21172C", "#C9A9EA"),
            "success": ("#10231D", "#7AD3B7"),
            "warning": ("#2A2114", "#F1C875"),
            "danger": ("#2D191F", "#F1A1AB"),
        }
        bg, fg = colors.get(tone, colors["neutral"])
        self.setStyleSheet(
            f"background:{bg};color:{fg};border:1px solid {Tokens.BORDER};border-radius:4px;padding:3px 8px;font-size:10px;font-weight:700;"
        )


class StatusIndicator(StatusPill):
    """Semantic status primitive retained alongside the legacy name."""


def labeled(
    label: str, widget: QWidget, helper: str = "", *, important: bool = False
) -> QWidget:
    return ParameterRow(label, widget, helper, important=important)


class ParameterRow(QWidget):
    """Label, control, and optional validation/help text kept as one unit."""

    def __init__(
        self,
        label: str,
        widget: QWidget,
        helper: str = "",
        *,
        important: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._scalar = isinstance(widget, (QAbstractSpinBox, UnitValue))
        self._widget = widget
        self._widget_maximum_width = widget.maximumWidth()
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(5)
        self._grid.setVerticalSpacing(3)
        self._text = QLabel(label)
        self._text.setObjectName("fieldLabelImportant" if important else "fieldLabel")
        self._label_row = QHBoxLayout()
        self._label_row.setContentsMargins(0, 0, 0, 0)
        self._label_row.setSpacing(4)
        self._label_row.addWidget(self._text, 1)
        if helper:
            self._label_row.addWidget(InfoTip(helper, f"About {label}"))
        self._label_host = QWidget()
        self._label_host.setLayout(self._label_row)
        self._apply_layout(stacked=not self._scalar)

    def _apply_layout(self, *, stacked: bool) -> None:
        self._grid.removeWidget(self._label_host)
        self._grid.removeWidget(self._widget)
        self._grid.setColumnStretch(0, 0)
        self._grid.setColumnStretch(1, 0)
        self._grid.setColumnStretch(2, 0)
        if stacked:
            self._label_host.setMinimumWidth(0)
            self._label_host.setMaximumWidth(16_777_215)
            self._label_host.setFixedHeight(16)
            self._text.setWordWrap(False)
            self._text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._label_row.setAlignment(self._text, Qt.AlignLeft | Qt.AlignVCenter)
            self._widget.setMaximumWidth(self._widget_maximum_width)
            self._grid.addWidget(self._label_host, 0, 0)
            self._grid.addWidget(self._widget, 1, 0)
            self._grid.setColumnStretch(0, 1)
        else:
            self._label_host.setMinimumHeight(0)
            self._label_host.setMaximumHeight(16_777_215)
            self._label_host.setFixedWidth(108)
            self._text.setMinimumWidth(0)
            self._text.setWordWrap(True)
            self._text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self._text.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._label_row.setAlignment(self._text, Qt.AlignRight | Qt.AlignVCenter)
            self._widget.setMaximumWidth(
                300 if isinstance(self._widget, UnitValue) else 220
            )
            self._grid.addWidget(self._label_host, 0, 0)
            self._grid.addWidget(self._widget, 0, 1, Qt.AlignLeft | Qt.AlignVCenter)
            self._grid.setColumnStretch(2, 1)

    def use_stacked_layout(self) -> None:
        """Place the label above its control in a dense multi-field row."""
        self._apply_layout(stacked=True)


def row_of(*widgets: QWidget, proportions: tuple[int, ...] | None = None) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    parameter_rows = [widget for widget in widgets if isinstance(widget, ParameterRow)]
    if len(parameter_rows) > 1:
        for parameter in parameter_rows:
            parameter.use_stacked_layout()
    for index, widget in enumerate(widgets):
        row.addWidget(
            widget,
            proportions[index] if proportions else 1,
            Qt.AlignTop,
        )
    return container


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("eyebrow")
    return label


def title_label(title: str, subtitle: str) -> QWidget:
    area = QWidget()
    layout = QHBoxLayout(area)
    layout.setContentsMargins(0, 0, 0, 2)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    layout.addWidget(heading)
    if subtitle:
        layout.addWidget(InfoTip(subtitle, f"About {title}"))
    layout.addStretch()
    return area
