"""Flow and oxygen physics configuration page.

The page maps physical boundary conditions, fluid properties, inlet values,
hematocrit, and tissue-consumption choices to validated configuration fields.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.model import (
    ALPHA_MMHG,
    flow_from_ul_min,
    flow_to_ul_min,
    oxygen_from_concentration,
    oxygen_to_concentration,
    pressure_from_pa,
    pressure_to_pa,
)
from cascade.gui.widgets import (
    Banner,
    Card,
    UnitValue,
    labeled,
    row_of,
)
from cascade.gui.ui_helpers import (
    _combo,
    _concentration_unit,
    _double,
    _set_combo,
    _value,
)

from cascade.gui.pages.base import (
    Page,
)


class PhysicsPage(Page):
    def __init__(self, parent=None):
        super().__init__(
            "Physical setup",
            "",
            parent,
        )
        bc = Card("Boundary conditions")
        self.bc_mode = _combo(
            [
                ("Inlet flow + outlet pressure", "flow_pressure"),
                ("Inlet pressure + outlet pressure", "pressure_pressure"),
            ]
        )
        self.flow = UnitValue(
            ["µL/min", "mL/min", "m³/s"], 100.0, minimum=0, maximum=1e12, decimals=7
        )
        self.inlet_pressure = UnitValue(
            ["mmHg", "Pa"], 500.0, minimum=-1e9, maximum=1e9, decimals=5
        )
        self.outlet_pressure = UnitValue(
            ["mmHg", "Pa"], 300.0, minimum=-1e9, maximum=1e9, decimals=5
        )
        bc.add(labeled("Boundary-condition mode", self.bc_mode, important=True))
        self.inlet_pressure_row = labeled(
            "Nominal inlet pressure",
            self.inlet_pressure,
            "Prescribed in pressure-pressure mode or used to rebuild an svVascularize network according to pressure drop.",
        )
        bc.add(
            row_of(
                labeled("Shared inlet flow", self.flow, important=True),
                labeled(
                    "Outlet / reference pressure", self.outlet_pressure, important=True
                ),
                self.inlet_pressure_row,
            )
        )
        self.rebuild_network = QCheckBox(
            "Rebuild vessel network according to pressure drop"
        )
        self.rebuild_network.toggled.connect(self._update_bc)
        bc.add(self.rebuild_network)
        self.column.addWidget(bc)

        self.inlet_card = Card(
            "Inlet-specific conditions",
            "Use different boundary conditions for each tree in a forest.",
        )
        self.use_inlet_conditions = QCheckBox(
            "Set conditions separately for each inlet"
        )
        self.inlet_selector = _combo([])
        self.inlet_condition_stack = CompactStack()
        self._inlet_widgets: list[dict[str, UnitValue]] = []
        self._inlet_count = 1
        self.inlet_card.add(self.use_inlet_conditions)
        self.inlet_card.add(labeled("Editing", self.inlet_selector))
        self.inlet_card.add(self.inlet_condition_stack)
        self.column.addWidget(self.inlet_card)
        self.inlet_selector.currentIndexChanged.connect(
            self.inlet_condition_stack.setCurrentIndex
        )
        self.use_inlet_conditions.toggled.connect(self.inlet_condition_stack.setEnabled)
        self.use_inlet_conditions.toggled.connect(self.inlet_selector.setEnabled)
        self.use_inlet_conditions.toggled.connect(lambda *_: self.changed.emit())
        self.inlet_card.setVisible(False)

        oxy = Card("Diffusion and consumption")
        # CASCADE currently exposes the validated Michaelis-Menten Vmax/Km
        # contract; custom consumption laws remain disabled in the interface.
        self.diffusivity = UnitValue(
            ["cm²/s", "m²/s"], 2.41e-5, minimum=0, maximum=1e6, decimals=10
        )
        self.vmax = UnitValue(
            ["mol/m³/s", "mmHg/s"], 0.001, minimum=0, maximum=1e6, decimals=8
        )
        self.km = UnitValue(
            ["mol/m³", "mmHg"], 0.005, minimum=0, maximum=1e6, decimals=8
        )
        self.inlet_o2 = UnitValue(
            ["mmHg", "mol/m³"], 100.0, minimum=0, maximum=1e9, decimals=6
        )
        self.viability_enabled = QCheckBox("Classify viable tissue")
        self.viability = UnitValue(
            ["mmHg", "mol/m³"], 1.0, minimum=0, maximum=1e9, decimals=6
        )
        self.viability_enabled.setChecked(True)
        self.viability_enabled.toggled.connect(self.viability.setEnabled)
        oxy.add(
            row_of(
                labeled("Tissue diffusivity D", self.diffusivity, important=True),
                labeled("Inlet concentration C<sub>0</sub>", self.inlet_o2, important=True),
            )
        )
        oxy.add(
            row_of(
                labeled("Maximum uptake rate V<sub>max</sub>", self.vmax, important=True),
                labeled("Michaelis-Menten constant K<sub>M</sub>", self.km, important=True),
            )
        )
        oxy.add(
            row_of(
                self.viability_enabled,
                labeled(
                    "Viability threshold C<sup>*</sup>",
                    self.viability,
                    "Optional; used in tissue point exports.",
                ),
            )
        )
        self.column.addWidget(oxy)

        blood = Card("Fluid model")
        self.fluid = _combo(
            [
                ("Blood", "blood"),
                ("Water / cell media", "water"),
                ("Custom constant-viscosity fluid", "custom"),
            ]
        )
        self.hematocrit_model = _combo(
            [
                ("Pries–Secomb phase separation", "pries_secomb"),
                ("Uniform discharge hematocrit", "uniform_tube"),
            ]
        )
        self.hematocrit = _double(0.42, 0, 0.95, 4, 0.01)
        self.hb_capacity = _double(20.3, 0, 1000, 4, 0.1)
        self.vessel_diffusivity = UnitValue(
            ["cm²/s", "m²/s"], 2.41e-5, minimum=0, maximum=1e6, decimals=10
        )
        self.hematocrit_model_display = CompactStack()
        self.hematocrit_display = CompactStack()
        self.hb_capacity_display = CompactStack()
        for stack, control in (
            (self.hematocrit_model_display, self.hematocrit_model),
            (self.hematocrit_display, self.hematocrit),
            (self.hb_capacity_display, self.hb_capacity),
        ):
            unavailable = QLineEdit("N/a")
            unavailable.setEnabled(False)
            stack.addWidget(control)
            stack.addWidget(unavailable)
        blood.add(
            row_of(
                labeled("Perfusate", self.fluid, important=True),
                labeled(
                    "Vessel fluid diffusivity D<sub>v</sub>",
                    self.vessel_diffusivity,
                    important=True,
                ),
                labeled("Hematocrit model", self.hematocrit_model_display),
            )
        )
        blood.add(
            row_of(
                labeled(
                    "Discharge hematocrit", self.hematocrit_display, important=True
                ),
                labeled(
                    "Hemoglobin O₂ capacity per Hct",
                    self.hb_capacity_display,
                    "Oxygen-carrying capacity normalized by hematocrit.",
                ),
            )
        )
        self.custom_density = _double(1.0, 1e-6, 100.0, 6, 0.01)
        self.custom_viscosity = _double(1.0, 1e-6, 1e6, 6, 0.1)
        self.custom_fluid_controls = row_of(
            labeled("Density (g/cm³)", self.custom_density, important=True),
            labeled("Dynamic viscosity (cP)", self.custom_viscosity, important=True),
        )
        blood.add(self.custom_fluid_controls)
        self.preset_fluid_properties = labeled(
            "Density and viscosity",
            QLineEdit("Derived by the selected blood/media model"),
            "Blood uses radius-dependent rheology; water/media uses fixed preset properties.",
        )
        self.preset_fluid_properties.findChild(QLineEdit).setEnabled(False)
        blood.add(self.preset_fluid_properties)
        self.column.addWidget(blood)
        self.fluid.currentIndexChanged.connect(self._update_blood_controls)
        self._bind_units(self.flow, flow_to_ul_min, flow_from_ul_min)
        self._bind_units(self.inlet_pressure, pressure_to_pa, pressure_from_pa)
        self._bind_units(self.outlet_pressure, pressure_to_pa, pressure_from_pa)
        self._bind_units(
            self.inlet_o2, oxygen_to_concentration, oxygen_from_concentration
        )
        self._bind_units(
            self.viability, oxygen_to_concentration, oxygen_from_concentration
        )
        self._bind_units(self.vmax, oxygen_to_concentration, oxygen_from_concentration)
        self._bind_units(self.km, oxygen_to_concentration, oxygen_from_concentration)
        self._bind_units(
            self.diffusivity,
            lambda value, unit: value * (1e4 if unit == "m²/s" else 1.0),
            lambda value, unit: value / (1e4 if unit == "m²/s" else 1.0),
        )
        self._bind_units(
            self.vessel_diffusivity,
            lambda value, unit: value * (1e4 if unit == "m²/s" else 1.0),
            lambda value, unit: value / (1e4 if unit == "m²/s" else 1.0),
        )
        self.set_inlet_count(1)
        self.bc_mode.currentIndexChanged.connect(self._update_bc)
        self._update_bc()
        self._update_blood_controls()
        self.finish()

    def _update_blood_controls(self, *_):
        """Blood-specific parameters do not apply to water or cell media."""
        is_blood = self.fluid.currentData() == "blood"
        is_custom = self.fluid.currentData() == "custom"
        for widget in (self.hematocrit_model, self.hematocrit, self.hb_capacity):
            widget.setEnabled(is_blood)
        for display in (
            self.hematocrit_model_display,
            self.hematocrit_display,
            self.hb_capacity_display,
        ):
            display.setCurrentIndex(0 if is_blood else 1)
        self.custom_fluid_controls.setVisible(is_custom)
        self.preset_fluid_properties.setVisible(not is_custom)

    @staticmethod
    def _bind_units(widget, to_base, from_base):
        widget.setProperty("previousUnit", widget.unit())

        def changed(new_unit):
            old_unit = widget.property("previousUnit") or new_unit
            base_value = to_base(widget.value(), str(old_unit))
            widget.spin.blockSignals(True)
            widget.setValue(from_base(base_value, new_unit))
            widget.spin.blockSignals(False)
            widget.setProperty("previousUnit", new_unit)

        widget.unitChanged.connect(changed)

    def _update_bc(self):
        mode = self.bc_mode.currentData()
        pressure_only = mode == "pressure_pressure"
        self.flow.setEnabled(not pressure_only)
        self.inlet_pressure_row.setEnabled(
            pressure_only or self.rebuild_network.isChecked()
        )
        for fields in self._inlet_widgets:
            fields["flow"].setEnabled(not pressure_only)

    def set_network_source(self, source: str) -> None:
        self.rebuild_network.setVisible(source == "svv_generated")

    def _make_inlet_condition_panel(self, index: int) -> dict[str, UnitValue]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(9)
        fields = {
            "flow": UnitValue(
                ["µL/min", "mL/min", "m³/s"],
                self.flow.value(),
                minimum=0,
                maximum=1e12,
                decimals=7,
            ),
            "inlet_pressure": UnitValue(
                ["mmHg", "Pa"],
                self.inlet_pressure.value(),
                minimum=-1e9,
                maximum=1e9,
                decimals=5,
            ),
            "outlet_pressure": UnitValue(
                ["mmHg", "Pa"],
                self.outlet_pressure.value(),
                minimum=-1e9,
                maximum=1e9,
                decimals=5,
            ),
            "inlet_o2": UnitValue(
                ["mmHg", "mol/m³"],
                self.inlet_o2.value(),
                minimum=0,
                maximum=1e9,
                decimals=6,
            ),
        }
        fields["flow"].setUnit(self.flow.unit())
        fields["inlet_pressure"].setUnit(self.inlet_pressure.unit())
        fields["outlet_pressure"].setUnit(self.outlet_pressure.unit())
        fields["inlet_o2"].setUnit(self.inlet_o2.unit())
        self._bind_units(fields["flow"], flow_to_ul_min, flow_from_ul_min)
        self._bind_units(fields["inlet_pressure"], pressure_to_pa, pressure_from_pa)
        self._bind_units(fields["outlet_pressure"], pressure_to_pa, pressure_from_pa)
        self._bind_units(
            fields["inlet_o2"], oxygen_to_concentration, oxygen_from_concentration
        )
        layout.addWidget(labeled("Inlet flow", fields["flow"], important=True))
        layout.addWidget(
            labeled("Inlet pressure", fields["inlet_pressure"], important=True)
        )
        layout.addWidget(
            labeled(
                "Outlet / reference pressure", fields["outlet_pressure"], important=True
            )
        )
        layout.addWidget(
            labeled("Inlet concentration", fields["inlet_o2"], important=True)
        )
        for widget in fields.values():
            widget.valueChanged.connect(lambda *_: self.changed.emit())
            widget.unitChanged.connect(lambda *_: self.changed.emit())
        fields["panel"] = panel
        return fields

    def _canonical_inlet_conditions(self) -> list[dict[str, float]]:
        conditions = []
        for fields in self._inlet_widgets:
            conditions.append(
                {
                    "flow_ul_min": flow_to_ul_min(
                        fields["flow"].value(), fields["flow"].unit()
                    ),
                    "inlet_pressure_pa": pressure_to_pa(
                        fields["inlet_pressure"].value(),
                        fields["inlet_pressure"].unit(),
                    ),
                    "outlet_pressure_pa": pressure_to_pa(
                        fields["outlet_pressure"].value(),
                        fields["outlet_pressure"].unit(),
                    ),
                    "inlet_concentration_mmol_l": oxygen_to_concentration(
                        fields["inlet_o2"].value(), fields["inlet_o2"].unit()
                    ),
                }
            )
        return conditions

    def set_inlet_count(self, count: int) -> None:
        count = max(1, int(count))
        if count == self._inlet_count and len(self._inlet_widgets) == count:
            self.inlet_card.setVisible(count > 1)
            return
        preserved = self._canonical_inlet_conditions() if self._inlet_widgets else []
        while self.inlet_condition_stack.count():
            widget = self.inlet_condition_stack.widget(0)
            self.inlet_condition_stack.removeWidget(widget)
            widget.deleteLater()
        self._inlet_widgets = []
        self.inlet_selector.blockSignals(True)
        self.inlet_selector.clear()
        for index in range(count):
            fields = self._make_inlet_condition_panel(index)
            fields["flow"].setEnabled(
                self.bc_mode.currentData() != "pressure_pressure"
            )
            self._inlet_widgets.append(fields)
            self.inlet_condition_stack.addWidget(fields["panel"])
            self.inlet_selector.addItem(f"Inlet {index + 1}", index)
            if index < len(preserved):
                self._set_inlet_condition(fields, preserved[index])
        self.inlet_selector.blockSignals(False)
        self.inlet_selector.setCurrentIndex(0)
        self.inlet_condition_stack.setCurrentIndex(0)
        self._inlet_count = count
        self.inlet_card.setVisible(count > 1)
        enabled = self.use_inlet_conditions.isChecked()
        self.inlet_selector.setEnabled(enabled)
        self.inlet_condition_stack.setEnabled(enabled)

    @staticmethod
    def _set_inlet_condition(fields, condition) -> None:
        fields["flow"].setValue(
            flow_from_ul_min(
                float(condition.get("flow_ul_min", 100.0)), fields["flow"].unit()
            )
        )
        fields["inlet_pressure"].setValue(
            pressure_from_pa(
                float(condition.get("inlet_pressure_pa", 66661.0)),
                fields["inlet_pressure"].unit(),
            )
        )
        fields["outlet_pressure"].setValue(
            pressure_from_pa(
                float(condition.get("outlet_pressure_pa", 40000.0)),
                fields["outlet_pressure"].unit(),
            )
        )
        fields["inlet_o2"].setValue(
            oxygen_from_concentration(
                float(condition.get("inlet_concentration_mmol_l", 0.14)),
                fields["inlet_o2"].unit(),
            )
        )

    def load(self, config):
        gui = config.get("gui", {})
        self.rebuild_network.setChecked(
            bool(gui.get("rebuild_svv_for_pressure_drop", False))
        )
        bc = gui.get("boundary_conditions", {})
        mode = bc.get("mode", "flow_pressure")
        settings = config.get("settings", {})
        hemo = settings.get("hemodynamics", settings.get("kirchhoff", {}))
        runtime_mode = config.get("simulation", {}).get("kirchhoff_bc_mode")
        if runtime_mode is None:
            runtime_mode = hemo.get(
                "kirchhoff_bc_mode",
                hemo.get(
                    "KIRCHHOFF_BC_MODE",
                    settings.get("kirchhoff", {}).get("bc_mode"),
                ),
            )
        runtime_mode = (
            str(runtime_mode).strip().lower().replace("-", "_")
            if runtime_mode is not None
            else None
        )
        if runtime_mode in {"pressure_pressure", "fixed_pressure_drop", "dirichlet"}:
            mode = "pressure_pressure"
        elif runtime_mode in {
            "legacy_equal_terminal_flow",
            "equal_terminal_flow",
            "equal_terminal_flows",
        }:
            mode = "legacy_equal_flow"
        elif runtime_mode in {"terminal_pressure", "mixed", "pressure_terminals"}:
            mode = "flow_pressure"
        _set_combo(self.bc_mode, mode)
        self._update_bc()
        pressure_unit = bc.get("pressure_unit", "mmHg")
        flow_unit = bc.get("flow_unit", "µL/min")
        self.flow.setUnit(flow_unit)
        self.flow.setValue(
            flow_from_ul_min(
                float(config.get("simulation", {}).get("qin_target_ul_min", 100.0)),
                flow_unit,
            )
        )
        root_pa = float(_value(config, "hemodynamics", "root_pressure", 66661.0))
        terminal_pa = float(
            _value(config, "hemodynamics", "terminal_pressure", 40000.0)
        )
        self.inlet_pressure.setUnit(pressure_unit)
        self.inlet_pressure.setValue(pressure_from_pa(root_pa, pressure_unit))
        self.outlet_pressure.setUnit(pressure_unit)
        self.outlet_pressure.setValue(pressure_from_pa(terminal_pa, pressure_unit))
        oxy_unit = _concentration_unit(gui.get("oxygen_input_unit", "mmHg"))
        self.inlet_o2.setUnit(oxy_unit)
        inlet_map = _value(
            config,
            "oxygen",
            "concentration_inlet_by_fluid",
            {"blood": 0.14, "water": 0.2211, "cell media": 0.2211, "media": 0.2211},
        )
        fluid = config.get("simulation", {}).get("fluid", "blood")
        if not isinstance(fluid, str) or self.fluid.findData(fluid) < 0:
            raise ValueError(
                f"CASCADE Studio cannot edit unsupported fluid {fluid!r}. "
                "Use Blood, Water / cell media, or Custom constant-viscosity fluid."
            )
        inlet_c = (
            float(inlet_map.get(fluid, 0.14))
            if isinstance(inlet_map, dict)
            else float(inlet_map)
        )
        self.inlet_o2.setValue(oxygen_from_concentration(inlet_c, oxy_unit))
        diff = float(_value(config, "oxygen", "solute_diffusivity", 2.41e-5))
        diff_unit = gui.get("diffusivity_unit", "cm²/s")
        self.diffusivity.setUnit(diff_unit)
        self.diffusivity.setValue(diff / 1e4 if diff_unit == "m²/s" else diff)
        vessel_diff = _value(config, "oxygen", "lumen_diffusivity_cm2_s", None)
        if vessel_diff is None:
            vessel_diff = 3.2e-5 if fluid == "water" else 2.41e-5
        vessel_diff_unit = gui.get("vessel_diffusivity_unit", "cm²/s")
        self.vessel_diffusivity.setUnit(vessel_diff_unit)
        self.vessel_diffusivity.setValue(
            float(vessel_diff) / 1e4
            if vessel_diff_unit == "m²/s"
            else float(vessel_diff)
        )
        vmax_unit = _concentration_unit(gui.get("vmax_unit", "mol/m³/s"), rate=True)
        km_unit = _concentration_unit(gui.get("km_unit", "mol/m³"))
        self.vmax.setUnit(vmax_unit)
        self.km.setUnit(km_unit)
        self.vmax.setValue(
            oxygen_from_concentration(
                float(_value(config, "oxygen", "vmax_mm", 0.001)), self.vmax.unit()
            )
        )
        self.km.setValue(
            oxygen_from_concentration(
                float(_value(config, "oxygen", "k_m_mm", 0.005)), self.km.unit()
            )
        )
        _set_combo(self.fluid, fluid)
        hematocrit_model = _value(config, "hematocrit", "model", "pries_secomb")
        if str(hematocrit_model).strip().lower() == "constant":
            hematocrit_model = "uniform_tube"
        _set_combo(
            self.hematocrit_model, hematocrit_model
        )
        self.hematocrit.setValue(
            float(_value(config, "hematocrit", "hd_discharge", 0.42))
        )
        self.hb_capacity.setValue(
            float(_value(config, "oxygen", "o2_cap_per_hct", 20.3))
        )
        self.custom_density.setValue(
            float(_value(config, "hemodynamics", "custom_fluid_density_g_cm3", 1.0))
        )
        self.custom_viscosity.setValue(
            float(_value(config, "hemodynamics", "custom_fluid_dynamic_viscosity_cp", 1.0))
        )
        threshold = config.get("simulation", {}).get("viability_threshold")
        self.viability_enabled.setChecked(threshold is not None)
        self.viability.setUnit(_concentration_unit(gui.get("viability_unit", "mmHg")))
        self.viability.setValue(
            oxygen_from_concentration(
                float(threshold or ALPHA_MMHG), self.viability.unit()
            )
        )
        conditions = list(
            config.get("simulation", {}).get("inlet_conditions", []) or []
        )
        self.set_inlet_count(max(1, len(conditions)))
        self.use_inlet_conditions.setChecked(bool(conditions))
        for fields, condition in zip(self._inlet_widgets, conditions):
            self._set_inlet_condition(fields, condition)
        enabled = self.use_inlet_conditions.isChecked()
        self.inlet_selector.setEnabled(enabled)
        self.inlet_condition_stack.setEnabled(enabled)
        self._update_blood_controls()

    def write(self, config):
        gui = config.setdefault("gui", {})
        gui["rebuild_svv_for_pressure_drop"] = self.rebuild_network.isChecked()
        bc = gui.setdefault("boundary_conditions", {})
        bc.update(
            {
                "mode": self.bc_mode.currentData(),
                "pressure_unit": self.outlet_pressure.unit(),
                "flow_unit": self.flow.unit(),
            }
        )
        gui["oxygen_input_unit"] = self.inlet_o2.unit()
        gui["vmax_unit"] = self.vmax.unit()
        gui["km_unit"] = self.km.unit()
        gui["viability_unit"] = self.viability.unit()
        gui["diffusivity_unit"] = self.diffusivity.unit()
        gui["vessel_diffusivity_unit"] = self.vessel_diffusivity.unit()
        sim = config.setdefault("simulation", {})
        sim["qin_target_ul_min"] = flow_to_ul_min(self.flow.value(), self.flow.unit())
        if self.use_inlet_conditions.isChecked() and self._inlet_count > 1:
            sim["inlet_conditions"] = self._canonical_inlet_conditions()
            sim["flow_source"] = "per_inlet"
        else:
            sim.pop("inlet_conditions", None)
            if sim.get("flow_source") == "per_inlet":
                sim["flow_source"] = "per_tree"
        sim["fluid"] = self.fluid.currentData()
        sim["build_fluid"] = self.fluid.currentData()
        sim["viability_threshold"] = (
            oxygen_to_concentration(self.viability.value(), self.viability.unit())
            if self.viability_enabled.isChecked()
            else None
        )
        settings = config.setdefault("settings", {})
        hemo = settings.setdefault("hemodynamics", {})
        mode = self.bc_mode.currentData()
        hemo.update(
            {
                "root_pressure": pressure_to_pa(
                    self.inlet_pressure.value(), self.inlet_pressure.unit()
                ),
                "terminal_pressure": pressure_to_pa(
                    self.outlet_pressure.value(), self.outlet_pressure.unit()
                ),
                "kirchhoff_bc_mode": (
                    "legacy_equal_terminal_flow"
                    if mode == "legacy_equal_flow"
                    else "pressure_pressure"
                    if mode == "pressure_pressure"
                    else "terminal_pressure"
                ),
                "custom_fluid_density_g_cm3": self.custom_density.value(),
                "custom_fluid_dynamic_viscosity_cp": self.custom_viscosity.value(),
            }
        )
        oxygen = settings.setdefault("oxygen", {})
        inlet = oxygen_to_concentration(self.inlet_o2.value(), self.inlet_o2.unit())
        oxygen.update(
            {
                "concentration_inlet_by_fluid": {
                    "blood": inlet,
                    "water": inlet,
                    "cell media": inlet,
                    "media": inlet,
                    "custom": inlet,
                },
                "conc_max_for_normalization": inlet,
                "solute_diffusivity": self.diffusivity.value()
                * (1e4 if self.diffusivity.unit() == "m²/s" else 1.0),
                "lumen_diffusivity_cm2_s": self.vessel_diffusivity.value()
                * (1e4 if self.vessel_diffusivity.unit() == "m²/s" else 1.0),
                "vmax_mm": oxygen_to_concentration(self.vmax.value(), self.vmax.unit()),
                "k_m_mm": oxygen_to_concentration(self.km.value(), self.km.unit()),
                "o2_cap_per_hct": self.hb_capacity.value(),
            }
        )
        settings.setdefault("hematocrit", {}).update(
            {
                "model": self.hematocrit_model.currentData(),
                "hd_discharge": self.hematocrit.value(),
            }
        )


__all__ = ("PhysicsPage",)

# Constructors resolve these shared layout helpers at runtime.
from cascade.gui.property_grid import Card, CompactStack, labeled, row_of
