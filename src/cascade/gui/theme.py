"""Central CASCADE visual tokens and Qt stylesheet.

The palette is deliberately neutral-first. Plasma colors are reserved for
selection, transport, progress, and the primary workflow action.
"""

from __future__ import annotations

from pathlib import Path


class Tokens:
    APP = "#08090C"
    VIEWPORT = "#050608"
    SURFACE_1 = "#0D0F14"
    SURFACE_2 = "#12151B"
    HOVER = "#171A21"
    BORDER = "#202229"
    BORDER_ACTIVE = "#343740"
    TEXT = "#EBECF0"
    TEXT_2 = "#B9BBC3"
    TEXT_3 = "#A2A5AE"
    VIOLET = "#6A38C2"
    PURPLE = "#8E3FC7"
    MAGENTA = "#C83C98"
    ORANGE = "#F16A3C"
    AMBER = "#F6B84A"
    SUCCESS = "#61C7A4"
    DANGER = "#D45B6A"
    RADIUS = 4
    SPACE_1 = 4
    SPACE_2 = 8
    SPACE_3 = 12
    SPACE_4 = 16
    SPACE_5 = 24


def stylesheet() -> str:
    t = Tokens
    checkmark = (Path(__file__).parent / "assets" / "checkmark.svg").as_posix()
    disabled_checkmark = (
        Path(__file__).parent / "assets" / "checkmark-disabled.svg"
    ).as_posix()
    return f"""
QMainWindow, QWidget#root {{ background: {t.APP}; color: {t.TEXT}; }}
QWidget#root {{ border:1px solid {t.BORDER}; }}
QWidget#windowSurface {{ background:{t.APP}; border:none; }}
QWidget {{
  font-family: "Inter", "Geist", "IBM Plex Sans", "Segoe UI Variable", "Segoe UI", sans-serif;
  font-size: 13px; color: {t.TEXT};
}}
QMenuBar {{ background:{t.APP}; color:{t.TEXT_2}; border-bottom:1px solid {t.BORDER}; padding:2px 6px; }}
QMenuBar::item {{ padding:4px 8px; background:transparent; }}
QMenuBar::item:selected {{ background:{t.HOVER}; color:{t.TEXT}; }}
QMenu {{ background:{t.SURFACE_2}; color:{t.TEXT}; border:1px solid {t.BORDER_ACTIVE}; padding:5px; }}
QMenu::item {{ padding:5px 10px; border-radius:3px; }}
QMenu::item:selected {{ background:#54299D; color:#FFFFFF; }}
QFrame#windowChrome {{ background:{t.APP}; border-bottom:1px solid {t.BORDER}; }}
QFrame#windowTitleBar {{ background:#090A0D; border:none; }}
QPushButton[windowControl] {{
  background:transparent; color:{t.TEXT_2}; border:none; border-radius:0;
  padding:0; font-family:"Segoe UI Symbol", "Segoe UI", sans-serif;
  font-size:14px; font-weight:500;
}}
QPushButton[windowControl="standard"]:hover {{ background:#242832; color:{t.TEXT}; }}
QPushButton[windowControl="close"]:hover {{ background:#C42B3A; color:white; }}

QFrame#sidebar {{ background:{t.SURFACE_1}; border-right:1px solid {t.BORDER}; }}
QLabel#brand {{ color:{t.TEXT}; font-size:29px; font-weight:800; letter-spacing:2px; }}
QLabel#brandSub {{ color:{t.TEXT_2}; font-size:11px; font-weight:700; letter-spacing:.8px; }}
QListWidget#navigation {{ background:transparent; border:none; color:{t.TEXT_3}; outline:none; font-size:16px; }}
QListWidget#navigation::item {{ padding:10px 10px; margin:3px 0; border-radius:4px; border-left:2px solid transparent; }}
QListWidget#navigation::item:selected {{ background:{t.HOVER}; color:{t.TEXT}; font-weight:650; border-left:2px solid {t.PURPLE}; }}
QListWidget#navigation::item:hover {{ background:{t.SURFACE_2}; color:{t.TEXT_2}; }}

QFrame#topbar {{ background:{t.APP}; border-bottom:1px solid {t.BORDER}; }}
QFrame#inspector {{ background:{t.SURFACE_1}; border-left:1px solid {t.BORDER}; }}
QFrame#solverStatus {{ background:{t.SURFACE_1}; border-top:1px solid {t.BORDER}; }}
QSplitter::handle {{ background:{t.BORDER}; width:1px; }}
QSplitter::handle:hover {{ background:{t.PURPLE}; }}
QLabel#statusKey {{ color:{t.TEXT_3}; font-size:10px; font-weight:700; letter-spacing:1px; }}
QLabel#statusValue {{ color:{t.TEXT_2}; font-family:"JetBrains Mono", "Geist Mono", "Consolas", monospace; font-size:12px; }}

QFrame#card {{ background:transparent; border:none; border-top:1px solid {t.BORDER}; border-radius:0; }}
QFrame#computeTile {{ background:{t.SURFACE_2}; border:1px solid {t.BORDER}; border-radius:4px; }}
QFrame#radiusSizing {{ background:{t.SURFACE_2}; border:1px solid {t.BORDER_ACTIVE}; border-radius:4px; }}
QFrame#previewPanel {{ background:{t.VIEWPORT}; border:none; }}
QFrame#viewerSettingsPanel {{
  background:{t.SURFACE_1}; border:1px solid {t.BORDER_ACTIVE}; border-radius:5px;
}}
QLabel#previewTitle {{ font-size:14px; font-weight:700; color:{t.MAGENTA}; letter-spacing:.5px; }}
QLabel#previewStatus {{ background:rgba(13,15,20,220); border:1px solid {t.BORDER}; border-radius:4px; padding:5px 7px; color:{t.TEXT_2}; font-size:11px; }}
QLabel#pageTitle {{ font-size:16px; font-weight:680; color:{t.TEXT}; }}
QLabel#cardTitle {{ font-size:13px; font-weight:700; color:{t.TEXT}; letter-spacing:.3px; }}
QLabel#muted, QLabel#fieldHelp {{ color:{t.TEXT_3}; font-size:11px; }}
QLabel#fieldLabel {{ color:{t.TEXT_2}; font-weight:550; }}
QLabel#fieldLabelImportant {{ color:{t.TEXT}; font-weight:700; }}
QLabel#eyebrow {{ color:{t.MAGENTA}; font-size:9px; font-weight:800; letter-spacing:1.5px; }}
QLabel#infoTip {{
  background:{t.SURFACE_2}; color:{t.TEXT_3}; border:1px solid {t.BORDER_ACTIVE};
  border-radius:7px; font-family:"Georgia", serif; font-size:10px; font-weight:700;
}}
QLabel#infoTip:hover {{ color:{t.TEXT}; border-color:{t.PURPLE}; background:{t.HOVER}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
  background:{t.SURFACE_2}; color:{t.TEXT}; border:1px solid {t.BORDER_ACTIVE};
  border-radius:{t.RADIUS}px; padding:4px 6px; selection-background-color:{t.PURPLE};
}}
QSpinBox, QDoubleSpinBox {{
  font-family:"JetBrains Mono", "Geist Mono", "Consolas", monospace;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
  border:1px solid {t.PURPLE}; background:{t.HOVER};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
  background:{t.SURFACE_1}; color:{t.TEXT_3}; border-color:{t.BORDER};
}}
QComboBox::drop-down {{ border:0; width:22px; }}

QPushButton {{
  background:{t.VIOLET}; color:white; border:1px solid {t.PURPLE};
  border-radius:{t.RADIUS}px; padding:6px 11px; font-weight:650;
}}
QPushButton:hover {{ background:#7543C9; border-color:#9868D7; }}
QPushButton:pressed {{ background:#54289F; }}
QPushButton:focus {{ border:1px solid {t.AMBER}; }}
QPushButton:disabled {{ background:{t.SURFACE_2}; color:{t.TEXT_3}; border-color:{t.BORDER}; }}
QPushButton[secondary="true"] {{ background:{t.SURFACE_2}; color:{t.TEXT_2}; border:1px solid {t.BORDER_ACTIVE}; }}
QPushButton[secondary="true"]:hover {{ background:{t.HOVER}; color:{t.TEXT}; border-color:{t.PURPLE}; }}
QPushButton[secondary="true"]:disabled {{
  background:#090A0D; color:#777B86; border-color:#292C34; font-weight:550;
}}
QPushButton[danger="true"] {{ background:#3A1D24; color:#F2B3BB; border-color:#66303A; }}
QPushButton[danger="true"]:hover {{ background:#4A252E; border-color:{t.DANGER}; }}
QPushButton[accentOutline="true"] {{
  background:#15101D; color:#D7BDEA; border:1px solid {t.PURPLE};
}}
QPushButton[accentOutline="true"]:hover {{
  background:#21172C; color:{t.TEXT}; border-color:#9868D7;
}}
QPushButton[queueRole="neutral"] {{ background:#242933; color:#F1F2F5; border:1px solid #555C69; }}
QPushButton[queueRole="neutral"]:hover {{ background:#303641; border-color:#777F8D; }}
QPushButton[queueRole="run"] {{ background:#173C2D; color:#D7F8EA; border:1px solid #3C8A68; }}
QPushButton[queueRole="run"]:hover {{ background:#20513D; border-color:{t.SUCCESS}; }}
QPushButton[queueRole="run"]:pressed {{ background:#102E22; }}
QPushButton[queueRole="stop"] {{ background:#3A1D24; color:#F2B3BB; border:1px solid #66303A; }}
QPushButton[queueRole="stop"]:hover {{ background:#4A252E; border-color:{t.DANGER}; }}
QPushButton[queueRole="neutral"]:disabled,
QPushButton[queueRole="run"]:disabled,
QPushButton[queueRole="stop"]:disabled {{
  background:#090A0D; color:#777B86; border-color:#292C34; font-weight:550;
}}
QPushButton[queueRole="warning"] {{
  background:#4A3618; color:#FFD166; border-color:#8B682A; font-weight:700;
}}
QPushButton[queueRole="warning"]:hover {{
  background:#5A421D; color:#FFE099; border-color:#C79535;
}}

QCheckBox {{ spacing:7px; color:{t.TEXT_2}; }}
QCheckBox::indicator {{ width:14px; height:14px; border:1px solid {t.BORDER_ACTIVE}; border-radius:3px; background:{t.SURFACE_2}; }}
QCheckBox::indicator:checked {{ background:{t.PURPLE}; border-color:{t.MAGENTA}; image:url({checkmark}); }}
QCheckBox::indicator:disabled {{ background:{t.SURFACE_1}; border-color:{t.BORDER}; }}
QCheckBox::indicator:checked:disabled {{ background:{t.SURFACE_2}; border-color:{t.BORDER}; image:url({disabled_checkmark}); }}
QHeaderView {{ background:{t.APP}; color:{t.TEXT_2}; }}
QHeaderView::section {{ background:{t.SURFACE_2}; color:{t.TEXT_2}; padding:6px; border:none; border-bottom:1px solid {t.BORDER_ACTIVE}; font-weight:650; }}
QTableCornerButton::section {{ background:{t.SURFACE_2}; border:none; border-bottom:1px solid {t.BORDER_ACTIVE}; }}
QTableWidget, QTreeWidget {{ background:{t.APP}; color:{t.TEXT_2}; border:1px solid {t.BORDER}; border-radius:4px; gridline-color:{t.BORDER}; alternate-background-color:{t.SURFACE_1}; selection-background-color:{t.VIOLET}; }}
QProgressBar {{ border:none; border-radius:3px; background:{t.SURFACE_2}; color:{t.TEXT}; text-align:center; height:7px; }}
QProgressBar::chunk {{ background:{t.BORDER_ACTIVE}; border-radius:3px; }}
QProgressBar[activeProgress="true"]::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {t.VIOLET},stop:.52 {t.MAGENTA},stop:1 {t.ORANGE}); }}
QTabWidget::pane {{ border:none; }}
QTabBar::tab {{ padding:7px 10px; color:{t.TEXT_3}; }}
QTabBar::tab:selected {{ color:{t.TEXT}; font-weight:650; border-bottom:2px solid {t.PURPLE}; }}
QScrollArea, QWidget#page {{ background:transparent; border:0; }}
QScrollBar:vertical {{ background:{t.SURFACE_1}; width:8px; margin:0; }}
QScrollBar::handle:vertical {{ background:{t.BORDER_ACTIVE}; min-height:28px; border-radius:4px; }}
QScrollBar::handle:vertical:hover {{ background:{t.TEXT_3}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QToolTip {{ background:{t.SURFACE_2}; color:{t.TEXT}; border:1px solid {t.BORDER_ACTIVE}; padding:5px; }}
QFileDialog, QDialog {{ background:{t.SURFACE_1}; color:{t.TEXT}; }}
QFileDialog QListView, QFileDialog QTreeView {{ background:{t.APP}; color:{t.TEXT}; border:1px solid {t.BORDER_ACTIVE}; selection-background-color:{t.VIOLET}; }}
QFileDialog QHeaderView, QFileDialog QHeaderView::section {{ background:{t.SURFACE_2}; color:{t.TEXT_2}; }}
QFileDialog QToolButton {{ background:{t.SURFACE_2}; color:{t.TEXT}; border:1px solid {t.BORDER}; border-radius:4px; padding:5px; }}
"""
