"""Reusable O(E+V) scheduling for pressure-directed vascular graphs."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _levels(up, down, offsets, edges, indegree):
    n = len(indegree)
    queue = np.empty(n, np.int64)
    depth = np.zeros(n, np.int64)
    head, tail = 0, 0
    for node in range(n):
        if indegree[node] == 0:
            queue[tail] = node
            tail += 1
    while head < tail:
        node = queue[head]
        head += 1
        for idx in range(offsets[node], offsets[node + 1]):
            child = down[edges[idx]]
            depth[child] = max(depth[child], depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue[tail] = child
                tail += 1
    return depth, tail == n


def graph_schedule(up, down, q):
    """Return node and edge level batches, or None for a directed cycle.

    Positive-resistance pressure-driven flows strictly decrease pressure along
    every active edge, so even anatomically loopy networks are directed DAGs.
    User-supplied circulating flows are detected rather than misclassified.
    """
    n = int(max(up.max(), down.max()) + 1) if len(up) else 0
    active = np.flatnonzero(q > 1e-30)
    counts = np.bincount(up[active], minlength=n)
    edges = active[np.argsort(up[active], kind="stable")]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    degree = np.bincount(down[active], minlength=n)
    depth, dag = _levels(up, down, offsets, edges, degree)
    if not dag:
        return None
    nodes = np.argsort(depth, kind="stable").astype(np.int32)
    node_offsets = np.concatenate(([0], np.cumsum(np.bincount(depth))))
    # Include zero-flow edges for complete output; they have no dependency.
    edges = np.argsort(depth[up], kind="stable").astype(np.int32)
    edge_offsets = np.concatenate(
        ([0], np.cumsum(np.bincount(depth[up], minlength=len(node_offsets) - 1)))
    )
    return nodes, node_offsets, edges, edge_offsets
