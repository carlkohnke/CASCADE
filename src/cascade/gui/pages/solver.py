"""Numerical solver configuration page.

Controls are grouped by flow, vessel transport, external field, tissue, and
acceleration policy, with applicability rules reflected in the enabled widgets.
"""

from __future__ import annotations

import json
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
)
from cascade.configuration.settings import SETTINGS_SECTIONS
from cascade.gui.widgets import (
    Banner,
    Card,
    labeled,
    row_of,
)
from cascade.gui.ui_helpers import (
    _combo,
    _double,
    _set_combo,
    _spin,
    _value,
)

from cascade.gui.helpers import (
    _json_safe,
    _parse_jsonish,
)

from cascade.gui.pages.base import (
    Page,
)


class SolverPage(Page):
    def __init__(self, parent=None):
        super().__init__(
            "Solver decisions",
            "",
            parent,
        )
        choose = Card("Solver path")
        self._lattice_mode = False
        self.conc_solver = _combo(
            [
                ("Network direct solver (best for large domains)", "network_ext"),
                (
                    "Network FFT solver (best for primitive-shape domains)",
                    "network_ext_hybrid_bg",
                ),
                ("Top-down direct solver (for tree structures only)", "topdown_ext"),
                (
                    "Top-down FFT solver (for tree structures only)",
                    "topdown_ext_hybrid_bg",
                ),
                (
                    "Network, no vessel-vessel coupling (faster but less accurate)",
                    "network",
                ),
                (
                    "Top-down, no vessel-vessel coupling (faster but less accurate)",
                    "topdown",
                ),
            ]
        )
        self.conc_solver.setMinimumWidth(520)
        self.flow_solver = _combo(
            [
                ("Tree-specialized", "tree"),
                ("Automatic sparse", "auto"),
                ("Sparse direct (spsolve)", "spsolve"),
                ("Conjugate gradient", "cg"),
                ("GMRES + ILU", "gmres"),
            ]
        )
        choose.add(
            row_of(
                labeled("Vessel oxygen solver", self.conc_solver, important=True),
                labeled("Flow solver", self.flow_solver, important=True),
            )
        )
        self.solver_hint = Banner()
        self.solver_hint.setVisible(False)
        choose.add(self.solver_hint)
        self.column.addWidget(choose)

        primary = Card("Important numerical choices")
        self.closure = _combo(
            [
                ("Resolved intralumen radial transport (recommended)", "graetz"),
                ("Well mixed lumen", "wellmixed"),
                ("Custom wall exchange expression", "custom_kappa"),
            ]
        )
        self.finite_radius = _combo(
            [
                ("Both monopole and dipole terms", "both"),
                ("Monopole term only", "monopole"),
                ("Dipole term only", "dipole"),
                ("Neither (centerline approximation)", "none"),
            ]
        )
        self.kappa = QLineEdit()
        self.kappa.setPlaceholderText("e.g. 0.12 or 0.12 * (1 + x/L)")
        self.tissue_gl_order = _spin(5, 1, 32)
        # Retain the old attribute for extensions that addressed the original
        # single control directly.
        self.gl_order = self.tissue_gl_order
        self.cext_gl_order = _spin(1, 1, 32)
        self.axial_steps = _spin(5, 1, 100)
        self.hct_stop = _combo(
            [
                ("Fixed number of iterations", "iterations"),
                ("Convergence tolerance", "tolerance"),
            ]
        )
        self.hct_iterations = _spin(2, 0, 1000)
        self.hct_tol = _double(0.001, 0, 1, 8, 0.0001)
        primary.add(
            row_of(
                labeled("Lumen-to-wall closure", self.closure, important=True),
                labeled("Finite-radius Green's terms", self.finite_radius),
            )
        )
        self.kappa_row = labeled(
            "Custom κ for q = κ(C − Cₑₓₜ)",
            self.kappa,
            "Constant or position-dependent wall exchange coefficient.",
        )
        self.kappa_row.setVisible(False)
        primary.add(self.kappa_row)
        primary.add(
            row_of(
                labeled("Tissue quadrature", self.tissue_gl_order),
                labeled("External-field quadrature", self.cext_gl_order),
                labeled("Axial blood steps", self.axial_steps),
            )
        )
        self.hct_iterations_row = labeled(
            "Hct / flow iterations", self.hct_iterations
        )
        self.hct_tol_row = labeled("Hct convergence tolerance", self.hct_tol)
        primary.add(
            row_of(
                labeled("Hct / flow stopping rule", self.hct_stop, important=True),
                self.hct_iterations_row,
                self.hct_tol_row,
            )
        )
        self.column.addWidget(primary)

        self.cext_card = Card("External concentration / tissue coupling")
        self.backend = _combo([("Automatic", "auto"), ("GPU", "gpu"), ("CPU", "cpu")])
        self.cext_backend = _combo(
            [("Automatic", "auto"), ("GPU", "gpu"), ("CPU", "cpu")]
        )
        self.precision = _combo(
            [
                ("Float32 accelerator work arrays", "float32"),
                ("Float64 accelerator work arrays", "float64"),
            ]
        )
        self.cext_mode = _combo(
            [
                ("FFT background only", "fft"),
                ("Local screened interactions", "local_only_nlambda"),
                ("FFT background + local correction", "hybrid"),
            ]
        )
        self.cext_grid = _spin(256, 16, 1024, 16)
        self.lambda_bins = _spin(5, 1, 64)
        self.window = _double(6.0, 0.1, 100, 3, 0.5)
        self.cext_stop = _combo(
            [
                ("Fixed number of iterations", "iterations"),
                ("Convergence tolerance", "tolerance"),
            ]
        )
        self.cext_iters = _spin(1, 1, 10000)
        self.cext_tol = _double(1e-3, 0, 1e9, 10, 1e-4)
        self.cext_accel = _combo(
            [("Anderson", "anderson"), ("Aitken", "aitken"), ("None", "none")]
        )
        self.cext_card.add(
            row_of(
                labeled("Tissue backend", self.backend, important=True),
                labeled("Vessel network backend", self.cext_backend, important=True),
            )
        )
        self.cext_card.add(
            labeled(
                "Accelerator working precision",
                self.precision,
                "Precision for accelerated tissue calculations.",
            )
        )
        self.cext_method_row = labeled(
            "Vessel-vessel coupling method", self.cext_mode
        )
        self.window_row = labeled("Interaction window λ", self.window)
        self.cext_card.add(row_of(self.cext_method_row, self.window_row))
        self.cext_grid_row = labeled(
            "FFT grid per axis", self.cext_grid, important=True
        )
        self.lambda_bins_row = labeled("Screening bins", self.lambda_bins)
        self.cext_card.add(
            row_of(self.cext_grid_row, self.lambda_bins_row)
        )
        self.cext_iterations_row = labeled("Coupling iterations", self.cext_iters)
        self.cext_tol_row = labeled("Coupling convergence tolerance", self.cext_tol)
        self.cext_card.add(
            row_of(
                labeled("Coupling stopping rule", self.cext_stop, important=True),
                self.cext_iterations_row,
                self.cext_tol_row,
            )
        )
        self.cext_card.add(labeled("Coupling acceleration", self.cext_accel))
        self.column.addWidget(self.cext_card)

        expert = Card(
            "Expert overrides",
            "Check a setting to write an explicit override. Inapplicable sections are dimmed but remain inspectable.",
        )
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter settings…")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Override / setting", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setMinimumHeight(330)
        self._populate_expert()
        expert.add(self.filter)
        expert.add(self.tree)
        self.column.addWidget(expert)
        self.filter.textChanged.connect(self._filter_expert)
        self.conc_solver.currentIndexChanged.connect(self._applicability)
        self.closure.currentIndexChanged.connect(self._update_kappa_visibility)
        self.hct_stop.currentIndexChanged.connect(self._update_stop_controls)
        self.cext_stop.currentIndexChanged.connect(self._update_stop_controls)
        self.cext_mode.currentIndexChanged.connect(
            self._update_cext_method_controls
        )
        self._update_stop_controls()
        self._update_cext_method_controls()
        self.finish()

    def _update_stop_controls(self, *_):
        hct_fixed = self.hct_stop.currentData() == "iterations"
        self.hct_iterations_row.setVisible(hct_fixed)
        self.hct_tol_row.setVisible(not hct_fixed)
        cext_fixed = self.cext_stop.currentData() == "iterations"
        self.cext_iterations_row.setVisible(cext_fixed)
        self.cext_tol_row.setVisible(not cext_fixed)

    def _update_cext_method_controls(self, *_):
        """Show only controls used by the selected vessel-coupling algorithm."""
        solver = str(self.conc_solver.currentData() or "")
        hybrid_path = solver.endswith("_hybrid_bg")
        mode = str(self.cext_mode.currentData() or "fft")
        self.cext_method_row.setVisible(hybrid_path)
        self.cext_grid_row.setVisible(hybrid_path and mode in {"fft", "hybrid"})
        self.lambda_bins_row.setVisible(hybrid_path)
        self.window_row.setVisible(not hybrid_path or mode in {"local_only_nlambda", "hybrid"})

    def _populate_expert(self):
        self.tree.clear()
        seen = set()
        for section_name, section in SETTINGS_SECTIONS.items():
            if section_name in {"kirchhoff", "concentration"}:
                continue
            parent = QTreeWidgetItem([section_name.replace("_", " ").title(), ""])
            parent.setData(0, Qt.UserRole, section_name)
            self.tree.addTopLevelItem(parent)
            for name, value in section.defaults.items():
                if (section_name, name) in seen:
                    continue
                seen.add((section_name, name))
                shown = (
                    json.dumps(_json_safe(value), separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else str(_json_safe(value))
                )
                child = QTreeWidgetItem([name, shown])
                child.setFlags(
                    child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable
                )
                child.setCheckState(0, Qt.Unchecked)
                child.setData(0, Qt.UserRole, (section_name, name))
                parent.addChild(child)
        self.tree.collapseAll()

    def _filter_expert(self, text):
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            any_visible = False
            for j in range(parent.childCount()):
                child = parent.child(j)
                visible = (
                    not needle
                    or needle in child.text(0).lower()
                    or needle in parent.text(0).lower()
                )
                child.setHidden(not visible)
                any_visible |= visible
            parent.setHidden(not any_visible)
            if needle and any_visible:
                parent.setExpanded(True)

    def _applicability(self):
        if self._lattice_mode and self.conc_solver.currentData() not in {
            "network_ext",
            "network",
            "network_ext_hybrid_bg",
        }:
            _set_combo(self.conc_solver, "network_ext")
            _set_combo(self.flow_solver, "spsolve")
        wellmixed_only = self.conc_solver.currentData() in {
            "network_ext_hybrid_bg",
            "network",
        }
        if wellmixed_only:
            _set_combo(self.closure, "wellmixed")
        self.closure.setEnabled(not wellmixed_only)
        self.closure.setToolTip(
            "This FFT or uncoupled network path currently uses the well-mixed lumen closure."
            if wellmixed_only
            else ""
        )
        self._update_kappa_visibility()
        use_cext = "ext" in str(self.conc_solver.currentData())
        self.cext_card.setEnabled(use_cext)
        self._update_cext_method_controls()
        self.cext_gl_order.setEnabled(use_cext)
        self.cext_gl_order.setToolTip(
            "Gauss–Legendre points per vessel used by the coupled external field."
            if use_cext
            else "Not used when vessel–vessel oxygen coupling is disabled."
        )
        if self._lattice_mode:
            self.solver_hint.set_message(
                "Network oxygen and sparse-direct flow are required for lattices",
                "info",
            )
        else:
            self.solver_hint.setVisible(False)
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            active = parent.data(0, Qt.UserRole) != "cext" or use_cext
            parent.setForeground(0, QColor("#273944" if active else "#9ca6ac"))

    def _update_kappa_visibility(self, *_):
        visible = (
            self.closure.isEnabled() and self.closure.currentData() == "custom_kappa"
        )
        self.kappa_row.setVisible(visible)
        self.kappa.setEnabled(visible)

    def set_lattice_mode(self, enabled: bool) -> None:
        self._lattice_mode = bool(enabled)
        if self._lattice_mode:
            if self.conc_solver.currentData() not in {
                "network_ext",
                "network_ext_hybrid_bg",
                "network",
            }:
                _set_combo(self.conc_solver, "network_ext")
            _set_combo(self.flow_solver, "spsolve")
            self.conc_solver.setToolTip(
                "Lattices require one of the general-network oxygen solvers."
            )
            self.flow_solver.setToolTip(
                "Locked to Sparse direct (spsolve) for the generated lattice."
            )
        else:
            self.conc_solver.setToolTip("")
            self.flow_solver.setToolTip("")
        self.conc_solver.setEnabled(True)
        self.flow_solver.setEnabled(not self._lattice_mode)
        self._applicability()

    def load(self, config):
        sim = config.get("simulation", {})
        gui = config.get("gui", {})
        solver_value = sim.get("concentration_solver", "network_ext")
        # Keep old treecode projects loadable without exposing the unvalidated
        # choice in the current interface.
        if solver_value == "topdown_ext_treecode":
            solver_value = "topdown_ext_hybrid_bg"
        _set_combo(self.conc_solver, solver_value)
        _set_combo(
            self.flow_solver, _value(config, "hemodynamics", "kirchhoff_solver", "spsolve")
        )
        _set_combo(
            self.closure, _value(config, "oxygen", "lumen_wall_closure", "graetz")
        )
        self.kappa.setText(str(config.get("gui", {}).get("custom_kappa", "")))
        finite_radius = _value(config, "oxygen", "finite_radius_o2_terms", "both")
        finite_radius = {"source": "monopole", "target": "dipole"}.get(
            str(finite_radius).lower(), finite_radius
        )
        _set_combo(self.finite_radius, finite_radius)
        legacy_order = int(_value(config, "oxygen", "gl_order", 5))
        self.tissue_gl_order.setValue(legacy_order)
        self.cext_gl_order.setValue(
            int(_value(config, "oxygen", "gl_order_cext", legacy_order))
        )
        self.axial_steps.setValue(int(_value(config, "oxygen", "axial_blood_steps", 5)))
        self.hct_iterations.setValue(
            int(_value(config, "hematocrit", "flow_iterations", 2))
        )
        self.hct_tol.setValue(
            max(0.0, float(_value(config, "hematocrit", "hdtol", 0.001)))
        )
        _set_combo(self.hct_stop, gui.get("hct_stop_mode", "iterations"))
        _set_combo(self.backend, _value(config, "tissue", "accel_mode", "gpu"))
        _set_combo(self.cext_backend, _value(config, "cext", "accel_mode", "gpu"))
        _set_combo(self.precision, _value(config, "cext", "float_dtype", "float32"))
        _set_combo(self.cext_mode, _value(config, "cext", "hybrid_bg_mode", "fft"))
        self.cext_grid.setValue(int(_value(config, "cext", "hybrid_bg_grid", 256)))
        self.lambda_bins.setValue(
            int(_value(config, "cext", "hybrid_bg_lambda_bins", 5))
        )
        self.window.setValue(float(_value(config, "cext", "window_factor", 6)))
        self.cext_iters.setValue(
            int(_value(config, "cext", "vess_coupling_max_iter", 1))
        )
        self.cext_tol.setValue(
            max(0.0, float(_value(config, "cext", "vess_coupling_tol", 1e-3)))
        )
        _set_combo(self.cext_stop, gui.get("cext_stop_mode", "iterations"))
        _set_combo(
            self.cext_accel, _value(config, "cext", "vess_coupling_accel", "anderson")
        )
        self._load_overrides(config.get("settings", {}))
        self._update_stop_controls()
        self._applicability()

    def _load_overrides(self, settings):
        controlled = {
            "ROOT_PRESSURE",
            "TERMINAL_PRESSURE",
            "KIRCHHOFF_SOLVER",
            "KIRCHHOFF_BC_MODE",
            "HEMATOCRIT_MODEL",
            "HEMATOCRIT_FLOW_ITERATIONS",
            "HEMATOCRIT_HDTOL",
            "HEMATOCRIT_QTOL_NL_MIN",
            "HD_DISCHARGE",
            "CONCENTRATION_SOLVER",
            "CONCENTRATION_INLET_BY_FLUID",
            "CONC_MAX_FOR_NORMALIZATION",
            "SOLUTE_DIFFUSIVITY",
            "VMAX_MM",
            "K_M_MM",
            "O2_CAP_PER_HCT",
            "LUMEN_WALL_CLOSURE",
            "FINITE_RADIUS_O2_TERMS",
            "GL_ORDER",
            "GL_ORDER_CEXT",
            "AXIAL_BLOOD_STEPS",
            "TISSUE_ACCEL_MODE",
            "TISSUE_CACHE_FLOAT_DTYPE",
            "CEXT_ACCEL_MODE",
            "CEXT_FLOAT_DTYPE",
            "CEXT_HYBRID_BG_MODE",
            "CEXT_HYBRID_BG_GRID",
            "CEXT_HYBRID_BG_LAMBDA_BINS",
            "CEXT_WINDOW_FACTOR",
            "CEXT_VESS_COUPLING_MAX_ITER",
            "CEXT_VESS_COUPLING_TOL",
            "CEXT_VESS_COUPLING_ACCEL",
        }
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            section_name = parent.data(0, Qt.UserRole)
            supplied = settings.get(section_name, {})
            section = SETTINGS_SECTIONS.get(section_name)
            normalized = {}
            if section:
                for key, value in supplied.items():
                    const = key.upper()
                    if const not in section.defaults:
                        const = section.aliases.get(
                            key.lower().replace("-", "_"), const
                        )
                    normalized[const] = value
            for j in range(parent.childCount()):
                child = parent.child(j)
                _, name = child.data(0, Qt.UserRole)
                if name in normalized and name not in controlled:
                    value = normalized[name]
                    child.setText(
                        1,
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else str(value),
                    )
                    child.setCheckState(0, Qt.Checked)
                else:
                    child.setCheckState(0, Qt.Unchecked)

    def write(self, config):
        gui = config.setdefault("gui", {})
        gui.pop("solver_preset", None)
        gui["custom_kappa"] = self.kappa.text().strip()
        gui["hct_stop_mode"] = self.hct_stop.currentData()
        gui["cext_stop_mode"] = self.cext_stop.currentData()
        config.setdefault("simulation", {})["concentration_solver"] = (
            self.conc_solver.currentData()
        )
        settings = config.setdefault("settings", {})
        settings.setdefault("hemodynamics", {})["kirchhoff_solver"] = (
            self.flow_solver.currentData()
        )
        settings.setdefault("oxygen", {}).update(
            {
                "lumen_wall_closure": self.closure.currentData(),
                "finite_radius_o2_terms": self.finite_radius.currentData(),
                "gl_order": self.tissue_gl_order.value(),
                "gl_order_cext": self.cext_gl_order.value(),
                "axial_blood_steps": self.axial_steps.value(),
            }
        )
        settings.setdefault("hematocrit", {}).update(
            {
                "flow_iterations": (
                    self.hct_iterations.value()
                    if self.hct_stop.currentData() == "iterations"
                    else 1000
                ),
                "hdtol": (
                    -1.0
                    if self.hct_stop.currentData() == "iterations"
                    else self.hct_tol.value()
                ),
                "qtol_nl_min": (
                    -1.0
                    if self.hct_stop.currentData() == "iterations"
                    else 1.0e30
                ),
            }
        )
        settings.setdefault("tissue", {})["accel_mode"] = self.backend.currentData()
        settings["tissue"]["cache_float_dtype"] = self.precision.currentData()
        settings.setdefault("cext", {}).update(
            {
                "accel_mode": self.cext_backend.currentData(),
                "float_dtype": self.precision.currentData(),
                "hybrid_bg_mode": self.cext_mode.currentData(),
                "hybrid_bg_grid": self.cext_grid.value(),
                "hybrid_bg_lambda_bins": self.lambda_bins.value(),
                "window_factor": self.window.value(),
                "vess_coupling_max_iter": (
                    self.cext_iters.value()
                    if self.cext_stop.currentData() == "iterations"
                    else 1000
                ),
                "vess_coupling_tol": (
                    -1.0
                    if self.cext_stop.currentData() == "iterations"
                    else self.cext_tol.value()
                ),
                "vess_coupling_accel": self.cext_accel.currentData(),
            }
        )
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) != Qt.Checked:
                    continue
                section, name = child.data(0, Qt.UserRole)
                settings.setdefault(section, {})[name] = _parse_jsonish(child.text(1))


__all__ = ("SolverPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, labeled, row_of
