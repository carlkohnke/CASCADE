from __future__ import annotations

import os
from typing import Optional

import numpy as np

import svv.forest.forest as _forest_module
from svv.forest.forest import Forest as _BaseForest
from svv.tree.data.data import TreeData, TreeParameters
from svv.tree.data.units import UnitSystem
import svv.tree.tree as _tree_module
from svv.tree.tree import Tree as _BaseTree

from ._svv_domain.domain import Domain
from ._svv_branch_bifurcation import add_vessel as _gfm_add_vessel
from ._svv_branch_root import set_root as _gfm_set_root
from ._svv_forest_compat import ForestCompatibilityMixin
from ._svv_tree_compat import (
    TreeCompatibilityMixin,
    _normalize_float_dtype,
    _normalize_int_dtype,
)


DEFAULT_TREE_PREALLOCATION_STEP = int(4e6)

_tree_module.set_root = _gfm_set_root
_tree_module.add_vessel = _gfm_add_vessel


def _resolve_tree_data_dtype(value: object | None) -> np.dtype:
    dtype = _normalize_float_dtype(value, np.float64)
    if dtype != np.dtype(np.float64):
        # Public svv/svv-accelerated is reliable for float64. Float32 support
        # lives in the local patched svv and is intentionally not part of this
        # public-svv adapter pass.
        return np.dtype(np.float64)
    return dtype


class Tree(TreeCompatibilityMixin, _BaseTree):
    """GFM compatibility wrapper around the installed/public svv Tree."""

    def __init__(
        self,
        *,
        parameters: Optional[TreeParameters] = None,
        unit_system: Optional[UnitSystem] = None,
        data_dtype: Optional[object] = None,
        index_dtype: Optional[object] = None,
        preallocation_step: int = DEFAULT_TREE_PREALLOCATION_STEP,
    ):
        requested_data_dtype = data_dtype
        if requested_data_dtype is None:
            requested_data_dtype = os.environ.get("SVV_TREE_DATA_DTYPE", "float64")
        requested_index_dtype = index_dtype
        if requested_index_dtype is None:
            requested_index_dtype = os.environ.get("SVV_TREE_INDEX_DTYPE", "int64")

        resolved_data_dtype = _resolve_tree_data_dtype(requested_data_dtype)
        resolved_index_dtype = _normalize_int_dtype(requested_index_dtype, np.int64)

        super().__init__(
            parameters=parameters,
            unit_system=unit_system,
            preallocation_step=int(preallocation_step),
        )

        self.data_dtype = resolved_data_dtype
        self.index_dtype = resolved_index_dtype
        if not hasattr(self, "_idx_cache"):
            self._idx_cache = []
        if not hasattr(self, "_col21_cache"):
            self._col21_cache = []

        self.data = TreeData.from_array(np.asarray(self.data, dtype=self.data_dtype))
        self.preallocate = TreeData.from_array(np.asarray(self.preallocate, dtype=self.data_dtype))
        self.preallocate_midpoints = np.asarray(self.preallocate_midpoints, dtype=self.data_dtype)
        if getattr(self, "connectivity", None) is not None:
            self.connectivity = np.asarray(self.connectivity, dtype=self.index_dtype)


_forest_module.Tree = Tree


class Forest(ForestCompatibilityMixin, _BaseForest):
    """GFM compatibility wrapper around the installed/public svv Forest."""


__all__ = ["Domain", "Tree", "Forest"]
