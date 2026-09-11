"""Gauss-Legendre quadrature helpers for exported fields.

The common orders are stored here rather than installed into mutable runtime
state as an import side effect.
"""

from __future__ import annotations

import numpy as np

_GL5_NODES = np.array(
    [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459],
    dtype=float,
)
_GL5_WEIGHTS = np.array(
    [0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850],
    dtype=float,
)
_GL9_NODES = np.array(
    [
        -0.9681602395076261,
        -0.8360311073266358,
        -0.6133714327005904,
        -0.3242534234038089,
        0.0,
        0.3242534234038089,
        0.6133714327005904,
        0.8360311073266358,
        0.9681602395076261,
    ],
    dtype=float,
)
_GL9_WEIGHTS = np.array(
    [
        0.0812743883615744,
        0.1806481606948574,
        0.2606106964029354,
        0.3123470770400029,
        0.3302393550012598,
        0.3123470770400029,
        0.2606106964029354,
        0.1806481606948574,
        0.0812743883615744,
    ],
    dtype=float,
)

def _get_gl_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order <= 0:
        raise ValueError(f"Unsupported GL order: {order}. Use a positive integer.")
    if order == 5:
        return _GL5_NODES, _GL5_WEIGHTS
    if order == 9:
        return _GL9_NODES, _GL9_WEIGHTS
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(order)
    return nodes.astype(float), weights.astype(float)


__all__ = ['_get_gl_nodes_weights']
