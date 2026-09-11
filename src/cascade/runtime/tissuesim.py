# Legacy symbol names are bound by the temporary TissueSim compatibility facade.
# ruff: noqa: F821
"""Compatibility facade for the modular CASCADE solver.

New code should import from ``cascade.flow`` or ``cascade.concentration``.
This module preserves the pre-refactor runtime surface while callers migrate.
"""

from __future__ import annotations

from types import ModuleType as _ModuleType
import sys as _sys

from cascade.configuration import _legacy_state as _state
import cascade.concentration.tissue.geometry as _implementation_0
import cascade.exporting.schema as _implementation_1
import cascade.diagnostics.runtime as _implementation_2
import cascade.domain.builders as _implementation_3
import cascade.vessels.generation.legacy as _implementation_4
import cascade.domain.sampling as _implementation_5
import cascade.domain.visualization as _implementation_6
import cascade.flow.topology as _implementation_7
import cascade.flow.kirchhoff as _implementation_8
import cascade.flow.rheology as _implementation_9
import cascade.flow.hematocrit as _implementation_10
import cascade.concentration.vessel.topdown as _implementation_11
import cascade.concentration.vessel.dispatch as _implementation_12
import cascade.concentration.vessel.greens as _implementation_13
import cascade.concentration.vessel.graetz as _implementation_14
import cascade.concentration.external_field.backend as _implementation_15
import cascade.concentration.external_field.frozen as _implementation_16
import cascade.concentration.external_field.state as _implementation_17
import cascade.concentration.external_field.geometry as _implementation_18
import cascade.concentration.external_field.direct as _implementation_19
import cascade.concentration.external_field.hybrid_geometry as _implementation_20
import cascade.concentration.external_field.hybrid_deposit as _implementation_21
import cascade.concentration.external_field.hybrid_corrections as _implementation_22
import cascade.concentration.external_field.hybrid_solver as _implementation_23
import cascade.concentration.external_field.treecode as _implementation_24
import cascade.concentration.external_field.diagnostics as _implementation_25
import cascade.concentration.external_field.acceleration as _implementation_26
import cascade.concentration.external_field.coupling_steps as _implementation_27
import cascade.concentration.external_field.topdown_solver as _implementation_28
import cascade.concentration.external_field.treecode_solver as _implementation_29
import cascade.concentration.external_field.hybrid_coupled_solver as _implementation_30
import cascade.concentration.tissue.cache as _implementation_31
import cascade.concentration.tissue.gpu as _implementation_32
import cascade.concentration.tissue.streaming as _implementation_33
import cascade.concentration.tissue.greens as _implementation_34
import cascade.concentration.tissue.metrics as _implementation_35
import cascade.exporting.quadrature as _implementation_36
import cascade.exporting.visualization as _implementation_37
import cascade.exporting.statistics as _implementation_38
import cascade.exporting.tables as _implementation_39
import cascade.workflows.tree_cache as _implementation_40
import cascade.workflows.tree_solver as _implementation_41
import cascade.workflows.summaries as _implementation_42
import cascade.commands.legacy_arguments as _implementation_43
import cascade.commands.legacy_runner as _implementation_44

_IMPLEMENTATIONS = (_implementation_0, _implementation_1, _implementation_2, _implementation_3, _implementation_4, _implementation_5, _implementation_6, _implementation_7, _implementation_8, _implementation_9, _implementation_10, _implementation_11, _implementation_12, _implementation_13, _implementation_14, _implementation_15, _implementation_16, _implementation_17, _implementation_18, _implementation_19, _implementation_20, _implementation_21, _implementation_22, _implementation_23, _implementation_24, _implementation_25, _implementation_26, _implementation_27, _implementation_28, _implementation_29, _implementation_30, _implementation_31, _implementation_32, _implementation_33, _implementation_34, _implementation_35, _implementation_36, _implementation_37, _implementation_38, _implementation_39, _implementation_40, _implementation_41, _implementation_42, _implementation_43, _implementation_44)

for _name, _value in vars(_state).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_EXPORTED_SYMBOLS: set[str] = set()
for _implementation in _IMPLEMENTATIONS:
    for _name in _implementation.__all__:
        _value = getattr(_implementation, _name)
        globals()[_name] = _value
        _EXPORTED_SYMBOLS.add(_name)

_bound_symbols = {name: globals()[name] for name in _EXPORTED_SYMBOLS}
for _implementation in (_state, *_IMPLEMENTATIONS):
    _implementation.__dict__.update(_bound_symbols)

_STATE_NAMES = frozenset(name for name in vars(_state) if not name.startswith("__"))
_bound_state = {name: getattr(_state, name) for name in _STATE_NAMES}
for _implementation in _IMPLEMENTATIONS:
    _implementation.__dict__.update(_bound_state)


class _TissueSimCompatibilityModule(_ModuleType):
    def __getattribute__(self, name: str):
        if name not in {"_STATE_NAMES", "_state"}:
            state_names = _ModuleType.__getattribute__(self, "_STATE_NAMES")
            if name in state_names:
                state = _ModuleType.__getattribute__(self, "_state")
                return getattr(state, name)
        return _ModuleType.__getattribute__(self, name)

    def __setattr__(self, name: str, value) -> None:
        if name in _STATE_NAMES:
            setattr(_state, name, value)
            for implementation in _IMPLEMENTATIONS:
                implementation.__dict__[name] = value
        if name in _EXPORTED_SYMBOLS:
            for implementation in (_state, *_IMPLEMENTATIONS):
                implementation.__dict__[name] = value
        _ModuleType.__setattr__(self, name, value)


_sys.modules[__name__].__class__ = _TissueSimCompatibilityModule


if __name__ == "__main__":
    args = parse_args()
    if args.profile_out:
        _run_main_with_profile(args.profile_out)
    elif args.line_profile_out:
        _run_main_with_line_profile(args.line_profile_out)
    else:
        main()
