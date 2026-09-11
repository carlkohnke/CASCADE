from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyvista as pv
from svv.tree.collision.tree_collision import tree_collision
from svv.tree.data.data import TreeMap
from svv.tree.utils.TreeManager import KDTreeManager, USearchTree

from .connectivity import repair_trees, validate_trees
from cascade.configuration.models import RunConfig, RootConfig
from cascade.configuration.runtime import RuntimeConfiguration
from cascade.domain.grid import sample_grid_points
from .generation.compatibility import branch_bifurcation as cascade_bifurcation
from .simple import build_simple_network
from .generation.svv_adapter import Domain, Forest, Tree
from cascade.utils.resources import resolve_domain_path, resolve_path




__all__ = ()
