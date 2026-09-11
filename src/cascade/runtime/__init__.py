"""Runtime access to the unified TissueSim namespace.

Scientific APIs live in their named subsystem packages. TissueSim remains
lazy so importing a low-level solver never initializes the full engine.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["tissuesim"]


def __getattr__(name: str):
    if name != "tissuesim":
        raise AttributeError(name)
    module = import_module("cascade.runtime.tissuesim")
    globals()[name] = module
    return module
