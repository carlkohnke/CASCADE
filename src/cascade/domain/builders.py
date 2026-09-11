"""Domain construction entry points.

These constructors are directly usable by command-line, GUI, and library code.
"""

from __future__ import annotations

from numbers import Number
import random as _py_random

import numpy as np
import pyvista as pv

from cascade.domain.svv import Domain


def build_domain_from_pyvista(mesh, *, random_seed: int = 42) -> Domain:
    if mesh is None:
        raise ValueError("mesh is required.")
    domain_mesh = mesh.copy(deep=True) if hasattr(mesh, "copy") else mesh
    domain = Domain(domain_mesh)
    domain.random_seed = int(random_seed)
    np.random.seed(int(domain.random_seed))
    _py_random.seed(int(domain.random_seed))
    domain.create()
    domain.solve()
    domain.build()
    domain.set_random_generator()
    return domain


def build_domain(
    side_length: float = 1.0, *, mesh=None, random_seed: int = 42
) -> Domain:
    if mesh is None and not isinstance(side_length, Number):
        mesh = side_length
    if mesh is not None:
        return build_domain_from_pyvista(mesh, random_seed=random_seed)
    side = float(side_length)
    return build_domain_from_pyvista(
        pv.Cube(x_length=side, y_length=side, z_length=side),
        random_seed=random_seed,
    )


__all__ = ["build_domain_from_pyvista", "build_domain"]
