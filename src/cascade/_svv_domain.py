"""Backward-compatible aliases for domain objects saved before CASCADE 0.1.

The implementation now belongs to :mod:`cascade.domain.svv`.  This private
module exists only so older DMN files, pickles, and private imports can resolve
their historical ``cascade._svv_domain.*`` module names.
"""

from __future__ import annotations

from importlib import import_module
import sys


_MODULES = (
    "core",
    "core.a_matrix",
    "core.h_matrix",
    "core.m_matrix",
    "core.n_matrix",
    "io",
    "io.dmn",
    "io.read",
    "kernel",
    "kernel.coordinate_system",
    "kernel.cost",
    "kernel.kernel",
    "routines",
    "routines.allocate",
    "routines.boolean",
    "routines.c_allocate",
    "routines.c_sample",
    "routines.discretize",
    "routines.tetgen_worker",
    "routines.tetrahedralize",
    "solver",
    "solver.solver",
    "domain",
    "patch",
    "weingarten",
)

# Mark this compatibility module as package-like and pre-register the old
# submodule names.  Unpickling an older Domain or Patch then resolves to the
# exact current class instead of constructing a second compatibility class.
__path__ = []
for _suffix in _MODULES:
    _module = import_module(f"cascade.domain.svv.{_suffix}")
    sys.modules[f"{__name__}.{_suffix}"] = _module
    if "." not in _suffix:
        globals()[_suffix] = _module

from cascade.domain.svv.domain import Domain
from cascade.domain.svv.patch import Patch

__all__ = ["Domain", "Patch"]
