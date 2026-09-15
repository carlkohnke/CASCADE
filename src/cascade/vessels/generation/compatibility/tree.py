"""Provide incremental growth and persistence behavior required by SVV tree objects.

The mixin manages capacity, repeated branch insertion, equal-terminal growth,
serialization, and restoration of the spatial indices used during construction.
"""

from __future__ import annotations

import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter
from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm, trange

from svv.tree.data.data import TreeData, TreeMap
from svv.tree.utils.TreeManager import KDTreeManager, USearchTree

from .branch_bifurcation import add_vessel
from .branch_root import set_root


def _normalize_float_dtype(value, default=np.float64):
    if value is None:
        return np.dtype(default)
    if isinstance(value, np.dtype):
        return value
    if isinstance(value, type) and value in (np.float32, np.float64):
        return np.dtype(value)
    s = str(value).strip().lower()
    if s in ("float32", "f32", "32"):
        return np.dtype(np.float32)
    if s in ("float64", "f64", "64"):
        return np.dtype(np.float64)
    try:
        dt = np.dtype(value)
        return dt if dt.kind == "f" else np.dtype(default)
    except Exception:
        return np.dtype(default)


def _normalize_int_dtype(value, default=np.int64):
    if value is None:
        return np.dtype(default)
    if isinstance(value, np.dtype):
        return value
    if isinstance(value, type) and value in (np.int32, np.int64):
        return np.dtype(value)
    s = str(value).strip().lower()
    if s in ("int32", "i32", "32"):
        return np.dtype(np.int32)
    if s in ("int64", "i64", "64"):
        return np.dtype(np.int64)
    try:
        dt = np.dtype(value)
        return dt if dt.kind == "i" else np.dtype(default)
    except Exception:
        return np.dtype(default)


class TreeCompatibilityMixin:
    def ensure_preallocation(self, required_rows: int) -> None:
        try:
            required = int(required_rows)
        except Exception:
            return
        if required <= 0:
            return

        current = int(getattr(self.preallocate, "shape", (0,))[0]) or 0
        if required <= current:
            return

        new_size = max(required, max(1, current * 2))
        data_dtype = _normalize_float_dtype(getattr(self, "data_dtype", None), np.float64)

        new_preallocate = TreeData((new_size, self.preallocate.shape[1]), dtype=data_dtype)
        if current:
            new_preallocate[:current, :] = self.preallocate[:current, :]
        self.preallocate = new_preallocate

        new_midpoints = np.zeros((new_size, 3), dtype=data_dtype)
        old_midpoints = int(getattr(self.preallocate_midpoints, "shape", (0,))[0]) or 0
        if old_midpoints:
            new_midpoints[:old_midpoints, :] = self.preallocate_midpoints[:old_midpoints, :]
        self.preallocate_midpoints = new_midpoints
        self.preallocation_step = int(new_size)

        try:
            n_rows = int(getattr(self.data, "shape", (0,))[0]) or 0
        except Exception:
            n_rows = 0
        if n_rows:
            self.data = self.preallocate[:n_rows, :]

        midpoints = getattr(self, "midpoints", None)
        if isinstance(midpoints, np.ndarray):
            if midpoints.ndim == 2 and midpoints.shape[1] == 3:
                self.midpoints = self.preallocate_midpoints[:midpoints.shape[0], :]
            elif midpoints.ndim == 1 and midpoints.shape[0] == 3:
                self.midpoints = self.preallocate_midpoints[0, :]

    def set_root(self, *args, **kwargs):
        """
        Set the root point of the tree.
        """
        if len(args) == 0:
            pass
        elif len(args) == 1:
            kwargs['start'] = args[0]
        elif len(args) == 2:
            kwargs['start'] = args[0]
            kwargs['direction'] = args[1]
            self.clamped_root = True
        else:
            raise ValueError("Too many arguments.")
        inplace = kwargs.pop('inplace', True)
        if not isinstance(kwargs.get('direction', None), type(None)):
            self.clamped_root = True
        if self.physical_clearance > 0.0:
            kwargs['interior_range'] = [-1.0, 0.0-self.domain_clearance]
        root, root_map = set_root(self, **kwargs)
        if inplace:
            self.data = root
            self.preallocate[0, :] = root
            self.connectivity = np.nan_to_num(root[:, 15:18], nan=-1.0).astype(self.index_dtype).reshape(1, 3)
            self.preallocate_midpoints[0, :] = (root[:, 0:3] + root[:, 3:6]) / 2
            self.midpoints = self.preallocate_midpoints[0, :]
            self.vessel_map.update(root_map)
            self.vessel_map_copy = deepcopy(self.vessel_map)
            self.n_terminals = 1
            self.kdtm = KDTreeManager(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3))
            self.hnsw_tree = USearchTree(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3).astype(np.float32))
            self.hnsw_tree_id = id(self.hnsw_tree)
            self.probability = np.array(self.domain.mesh.cell_data['probability'])
            self.max_distal_node = 1
            self.tree_scale = np.pi * root[0, 21]**self.parameters.radius_exponent*root[0, 20]**self.parameters.length_exponent
            self.segment_count = 1
        else:
            connectivity = np.nan_to_num(root[:, 15:18], nan=-1.0).astype(self.index_dtype).reshape(1, 3)
            kdtm = KDTreeManager(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3))
            hnsw_tree = USearchTree(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3).astype(np.float32))
            hnsw_tree_id = id(hnsw_tree)
            probability = np.array(self.domain.mesh.cell_data['probability'])
            tree_scale = np.pi * root[0, 21] ** self.parameters.radius_exponent * root[
                0, 20] ** self.parameters.length_exponent
            return root, root_map, connectivity, kdtm, hnsw_tree, hnsw_tree_id, probability, tree_scale

    def add(self, inplace=True, **kwargs):
        if isinstance(self.data, np.ndarray) and self.data.dtype == np.float32:
            raise RuntimeError(
                "CCO growth requires float64 tree data. "
                "Load or build float64 trees for CCO, or use terminal-only equal-bifurcation mode."
            )
        # A preview/checkpoint file may contain only enough spare rows for the
        # seed that was saved.  Resume growth must expand that buffer before
        # appending two child segments; otherwise a valid seed fails exactly at
        # its original capacity boundary.
        self.ensure_preallocation(int(self.segment_count) + 2)
        all_start = perf_counter()
        decay_probability = kwargs.pop('decay_probability', 0.9)
        new_data, added_vessels, new_vessel_map, history, lines, nonconvex_outside, new_inds, mesh_cell, connectivity, change_i, change_j, new_tmp_data, old_tmp_data = add_vessel(self, **kwargs)
        start = perf_counter()
        if not self.convex:
            if nonconvex_outside:
                self.nonconvex_count = 0
            else:
                self.nonconvex_count += 1
            if self.nonconvex_count > self.parameters.max_nonconvex_count:
                self.convex = True
        if inplace:
            start_chunk_4_0 = perf_counter()
            self.preallocate[self.segment_count,:] = added_vessels[0]
            self.preallocate[self.segment_count+1,:] = added_vessels[1]
            end_chunk_4_0 = perf_counter()
            self.times['chunk_4_0'].append(end_chunk_4_0 - start_chunk_4_0)
            start_chunk_4_1 = perf_counter()
            change_i = np.array(change_i, dtype=int)
            change_j = np.array(change_j, dtype=int)
            self.preallocate[change_i, change_j] = np.array(new_tmp_data)
            end_chunk_4_1 = perf_counter()
            self.times['chunk_4_1'].append(end_chunk_4_1 - start_chunk_4_1)
            start_chunk_4_2 = perf_counter()
            self.data = self.preallocate[:self.segment_count+2,:]
            new_data = self.data
            end_chunk_4_2 = perf_counter()
            self.times['chunk_4_2'].append(end_chunk_4_2 - start_chunk_4_2)
            start_chunk_4_3 = perf_counter()
            for key in new_vessel_map.keys():
                if key in self.vessel_map.keys():
                    self.vessel_map[key]['upstream'].extend(new_vessel_map[key]['upstream'])
                    self.vessel_map[key]['downstream'].extend(new_vessel_map[key]['downstream'])
                else:
                    self.vessel_map[key] = deepcopy(new_vessel_map[key])
            end_chunk_4_3 = perf_counter()
            self.times['chunk_4_3'].append(end_chunk_4_3 - start_chunk_4_3)
            self.n_terminals += 1
            self.segment_count += 2
            self.connectivity = connectivity #self.connectivity_copy.view()
            self.max_distal_node += 2
            if mesh_cell >= 0:
                self.probability[mesh_cell] *= decay_probability
                self.probability = self.probability / self.probability.sum()
                self.domain.cumulative_probability = np.cumsum(self.probability)
            self.hnsw_tree.replace(((new_data[new_inds[0], 0:3] + new_data[new_inds[0], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[0]]))
            self.hnsw_tree.add_items(((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[1]]))
            self.hnsw_tree.add_items(((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[2]]))
            self.preallocate_midpoints[new_inds, :] = (new_data[new_inds, 0:3] + new_data[new_inds, 3:6]) / 2
            self.tree_scale = self.new_tree_scale
            end = perf_counter()
            self.times['chunk_4'].append(end - start)
            self.times['all'].append(end - all_start)
            return None
        else:
            end = perf_counter()
            self.times['chunk_4'].append(end - start)
            self.times['all'].append(end - all_start)
            return change_i, change_j, new_tmp_data, old_tmp_data, new_vessel_map, connectivity, new_inds, mesh_cell, added_vessels

    def n_add(self, n, **kwargs):
        n = int(n)
        if n <= 0:
            return None
        params = getattr(self, "parameters", None)
        use_equal_terminal = False
        n_equal = None
        if params is not None:
            n_equal = getattr(params, "n_equal_bifurcations", None)
            if n_equal is not None and int(n_equal) >= 0 and bool(getattr(params, "equal_bifurcation_terminal_only", False)):
                use_equal_terminal = True

        if not use_equal_terminal:
            for i in trange(n, desc='Adding vessels', unit='vessel', leave=False):
                self.add(**kwargs)
            return None

        n_equal = int(n_equal)
        if self.n_terminals < n_equal:
            if isinstance(self.data, np.ndarray) and self.data.dtype == np.float32:
                raise RuntimeError(
                    "Equal-terminal mode would fall back to CCO to reach n_equal_bifurcations. "
                    "For float32 trees, set n_equal_bifurcations <= current terminal count or use float64."
                )
            normal = min(n, max(0, n_equal - int(self.n_terminals)))
            for i in trange(normal, desc='Adding vessels', unit='vessel', leave=False):
                self.add(**kwargs)
            n -= normal
            if n <= 0:
                return None

        # terminal-only equal-bifurcation growth
        self._n_add_equal_terminal(n)
        return None

    def _n_add_equal_terminal(self, n: int) -> None:
        params = getattr(self, "parameters", None)
        if params is None:
            return None

        batch_size = max(1, int(getattr(params, "equal_terminal_batch_size", 1)))
        n_candidates = max(1, int(getattr(params, "equal_terminal_n_candidates", 1)))
        alpha_min = float(getattr(params, "equal_terminal_alpha_min_deg", 5.0))
        alpha_mode = float(getattr(params, "equal_terminal_alpha_mode_deg", 37.5))
        alpha_max = float(getattr(params, "equal_terminal_alpha_max_deg", 85.0))
        psi_step = float(getattr(params, "equal_terminal_psi_step_deg", 0.0))
        psi_max = float(getattr(params, "equal_terminal_psi_max_deg", 0.0))
        domain_margin = float(getattr(params, "equal_terminal_domain_margin", 0.0))
        check_midpoint = bool(getattr(params, "equal_terminal_check_midpoint", True))
        domain_workers = max(1, int(getattr(params, "equal_terminal_domain_workers", 1)))
        domain_chunk = int(getattr(params, "equal_terminal_domain_chunk", 0))
        domain_min_points = int(getattr(params, "equal_terminal_domain_parallel_min_points", 0))
        length_fixed = getattr(params, "equal_terminal_length", None)
        length_mode = str(getattr(params, "equal_terminal_length_mode", "power") or "power")
        length_log10_intercept = float(getattr(params, "equal_terminal_total_length_log10_intercept", 0.0))
        length_log10_slope = float(getattr(params, "equal_terminal_total_length_log10_slope", 0.0))
        length_scale = float(getattr(params, "equal_terminal_length_scale", 0.01))
        length_power = float(getattr(params, "equal_terminal_length_power", -0.33))
        length_min = float(getattr(params, "equal_terminal_length_min", 0.0))
        length_max = float(getattr(params, "equal_terminal_length_max", np.inf))
        length_shrink = float(getattr(params, "equal_terminal_length_shrink", 0.7))
        density_k = int(getattr(params, "equal_terminal_density_k", 3))
        density_alpha = float(getattr(params, "equal_terminal_density_alpha", -0.17))
        density_beta = float(getattr(params, "equal_terminal_density_beta", 0.6))
        density_epsilon = float(getattr(params, "equal_terminal_density_epsilon", 0.99))
        gen_f_k = int(getattr(params, "equal_terminal_gen_f_k", 3))
        gen_f_mix_w = float(getattr(params, "equal_terminal_gen_f_mix_w", 0.2))
        gen_f_mix_mu1_ln = float(getattr(params, "equal_terminal_gen_f_mix_mu1_ln", -1.5))
        gen_f_mix_sig1_ln = float(getattr(params, "equal_terminal_gen_f_mix_sig1_ln", 0.65))
        gen_f_mix_mu2_ln = float(getattr(params, "equal_terminal_gen_f_mix_mu2_ln", -0.95))
        gen_f_mix_sig2_ln = float(getattr(params, "equal_terminal_gen_f_mix_sig2_ln", 0.34))
        report_timings = bool(getattr(params, "equal_terminal_report_timings", False))
        report_every = max(1, int(getattr(params, "equal_terminal_report_every", 1)))
        defer_hnsw = bool(getattr(params, "equal_terminal_defer_hnsw_updates", False))

        min_radius_floor = float(getattr(params, "equal_bifurcation_radius_floor", 0.0) or 0.0)
        min_length_floor = float(getattr(params, "equal_bifurcation_length_floor", 0.0) or 0.0)

        if getattr(self, "domain", None) is None:
            raise RuntimeError("Equal-terminal growth requires an attached Domain.")

        rng = getattr(getattr(self, "domain", None), "random_generator", None)
        if rng is None:
            rng = np.random.default_rng()

        if not hasattr(self, "_equal_terminal_blacklist"):
            self._equal_terminal_blacklist = set()
        blacklist = self._equal_terminal_blacklist

        remaining = int(n)
        progress = tqdm(total=remaining, desc="Adding vessels", unit="vessel", leave=False)
        t_domain = 0.0
        t_pick = 0.0
        t_add = 0.0

        # breadth-first generations
        current_gen = self._equal_terminal_list_terminals(exclude=blacklist)
        next_gen: list[int] = []
        gen_f_cache = None

        while remaining > 0 and current_gen:
            if length_mode == "gen_f_mix2" and gen_f_cache is None:
                gen_f_cache = self._equal_terminal_build_gen_f_cache(current_gen, density_k=gen_f_k)
            batch = current_gen[:batch_size]
            current_gen = current_gen[batch_size:]
            if remaining < len(batch):
                batch = batch[:remaining]

            t0 = perf_counter()
            if length_mode in ("lp_rp_density", "gen_f_mix2"):
                base_mode = "segments_law"
            else:
                base_mode = length_mode
            base_length = self._equal_terminal_compute_length(
                len(batch),
                length_fixed=length_fixed,
                length_mode=base_mode,
                length_log10_intercept=length_log10_intercept,
                length_log10_slope=length_log10_slope,
                length_scale=length_scale,
                length_power=length_power,
                length_min=length_min,
                length_max=length_max,
                length_floor=min_length_floor,
            )
            if length_mode == "lp_rp_density":
                base_length = self._equal_terminal_compute_length_model(
                    np.asarray(batch, dtype=int),
                    base_length=base_length,
                    alpha=density_alpha,
                    beta=density_beta,
                    epsilon=density_epsilon,
                    density_k=density_k,
                    length_min=length_min,
                    length_max=length_max,
                    length_floor=min_length_floor,
                )
            elif length_mode == "gen_f_mix2":
                base_length = self._equal_terminal_compute_length_gen_f_mix2(
                    np.asarray(batch, dtype=int),
                    base_length=base_length,
                    gen_f_cache=gen_f_cache,
                    mix_w=gen_f_mix_w,
                    mix_mu1_ln=gen_f_mix_mu1_ln,
                    mix_sig1_ln=gen_f_mix_sig1_ln,
                    mix_mu2_ln=gen_f_mix_mu2_ln,
                    mix_sig2_ln=gen_f_mix_sig2_ln,
                    length_min=length_min,
                    length_max=length_max,
                    length_floor=min_length_floor,
                    rng=rng,
                )

            ok, dirs1, dirs2, lengths_used, t_dom = self._equal_terminal_select(
                np.asarray(batch, dtype=int),
                base_length,
                n_candidates=n_candidates,
                alpha_min=alpha_min,
                alpha_mode=alpha_mode,
                alpha_max=alpha_max,
                psi_step=psi_step,
                psi_max=psi_max,
                domain_margin=domain_margin,
                check_midpoint=check_midpoint,
                domain_workers=domain_workers,
                domain_chunk=domain_chunk,
                domain_min_points=domain_min_points,
                length_shrink=length_shrink,
                rng=rng,
            )
            t_pick += perf_counter() - t0
            t_domain += t_dom

            if ok.any():
                add_idx = np.asarray(batch, dtype=int)[ok]
                add_dirs1 = dirs1[ok]
                add_dirs2 = dirs2[ok]
                add_lengths = lengths_used[ok]
                t1 = perf_counter()
                new_children = self._equal_terminal_commit(
                    add_idx,
                    add_dirs1,
                    add_dirs2,
                    add_lengths,
                    min_radius_floor=min_radius_floor,
                    defer_hnsw=defer_hnsw,
                )
                t_add += perf_counter() - t1
                remaining -= int(add_idx.size)
                progress.update(int(add_idx.size))
                next_gen.extend(new_children)

            # blacklisted terminals
            if (~ok).any():
                fail_idx = np.asarray(batch, dtype=int)[~ok]
                blacklist.update(fail_idx.tolist())

            # update progress postfix
            if (progress.n % report_every) == 0:
                pct = 100.0 * (len(blacklist) / max(1, int(self.n_terminals)))
                progress.set_postfix(blacklisted=f"{pct:.2f}%")

            if not current_gen:
                current_gen = next_gen
                next_gen = []
                gen_f_cache = None

        progress.close()

        if defer_hnsw and getattr(self, "segment_count", 0) > 0:
            # rebuild HNSW index in one shot for consistency
            midpoints = self.preallocate_midpoints[:self.segment_count, :]
            self.hnsw_tree = USearchTree(midpoints.astype(np.float32))
            self.hnsw_tree_id = id(self.hnsw_tree)

        if report_timings:
            print(
                f"[equal-terminal] domain={t_domain:.3f}s pick={t_pick:.3f}s add={t_add:.3f}s "
                f"blacklisted={len(blacklist)}"
            )
        return None

    def _equal_terminal_list_terminals(self, *, exclude: Optional[set[int]] = None) -> list[int]:
        seg_count = int(getattr(self, "segment_count", 0))
        if seg_count <= 0:
            return []
        data = np.asarray(self.data[:seg_count])
        left = data[:, 15]
        right = data[:, 16]
        is_terminal = np.isnan(left) & np.isnan(right)
        idx = np.flatnonzero(is_terminal).tolist()
        if exclude:
            idx = [i for i in idx if i not in exclude]
        return idx

    def _equal_terminal_compute_length(
        self,
        n_new: int,
        *,
        length_fixed: Optional[float],
        length_mode: str,
        length_log10_intercept: float,
        length_log10_slope: float,
        length_scale: float,
        length_power: float,
        length_min: float,
        length_max: float,
        length_floor: float,
    ) -> float:
        if n_new <= 0:
            return max(length_min, length_floor)
        if length_fixed is not None and np.isfinite(length_fixed):
            L = float(length_fixed)
        elif length_mode == "segments_law":
            seg_count = int(getattr(self, "segment_count", 0))
            data = np.asarray(self.data[:seg_count])
            lengths = np.asarray(data[:, 20], dtype=float)
            bad = ~np.isfinite(lengths) | (lengths <= 0.0)
            if np.any(bad):
                starts = data[:, 0:3]
                ends = data[:, 3:6]
                lengths[bad] = np.linalg.norm(ends[bad] - starts[bad], axis=1)
            total_current = float(np.nansum(lengths))
            total_segments = max(1, seg_count + 2 * n_new)
            target_total = 10.0 ** (length_log10_intercept + length_log10_slope * math.log10(total_segments))
            L = (target_total - total_current) / max(1, 2 * n_new)
        else:
            target_terminals = max(1, int(getattr(self, "n_terminals", 0)) + n_new)
            L = float(length_scale) * (target_terminals ** float(length_power))

        if not np.isfinite(L):
            L = length_min
        L = max(float(length_floor), float(length_min), float(L))
        if np.isfinite(length_max):
            L = min(L, float(length_max))
        return float(L)

    def _equal_terminal_compute_length_model(
        self,
        parent_idx: np.ndarray,
        *,
        base_length: float,
        alpha: float,
        beta: float,
        epsilon: float,
        density_k: int,
        length_min: float,
        length_max: float,
        length_floor: float,
    ) -> np.ndarray:
        parent_idx = np.asarray(parent_idx, dtype=int)
        if parent_idx.size == 0:
            return np.full((0,), float(base_length), dtype=float)
        seg_count = int(getattr(self, "segment_count", 0))
        if seg_count <= 0:
            return np.full((parent_idx.size,), float(base_length), dtype=float)

        data = np.asarray(self.data[:seg_count])
        parent_end = data[parent_idx, 3:6]
        lengths = np.asarray(data[:, 20], dtype=float)
        bad = ~np.isfinite(lengths) | (lengths <= 0.0)
        if np.any(bad):
            starts = np.asarray(data[:, 0:3], dtype=float)
            ends = np.asarray(data[:, 3:6], dtype=float)
            lengths[bad] = np.linalg.norm(ends[bad] - starts[bad], axis=1)
        Lp = lengths[parent_idx]
        Rp = np.asarray(data[parent_idx, 21], dtype=float)

        k = max(1, int(density_k))
        k_eff = min(k, max(0, seg_count - 1))
        if k_eff <= 0:
            lengths = np.full((parent_idx.size,), float(base_length), dtype=float)
        else:
            distal = np.asarray(data[:seg_count, 3:6], dtype=float)
            tree = cKDTree(distal)
            dists, _ = tree.query(parent_end, k=k_eff + 1)
            if dists.ndim == 1:
                dists = dists[:, None]
            density = np.mean(dists[:, 1 : k_eff + 1], axis=1)

            valid = (
                np.isfinite(Lp)
                & np.isfinite(Rp)
                & np.isfinite(density)
                & (Lp > 0.0)
                & (Rp > 0.0)
                & (density > 0.0)
            )
            lengths = np.full((parent_idx.size,), float(base_length), dtype=float)
            if np.any(valid):
                raw = (Lp[valid] ** float(alpha)) * (Rp[valid] ** float(beta)) * (density[valid] ** float(epsilon))
                raw_mean = float(np.mean(raw)) if raw.size else float("nan")
                if np.isfinite(raw_mean) and raw_mean > 0.0 and np.isfinite(base_length):
                    scale = float(base_length) / raw_mean
                    lengths[valid] = raw * scale

        if not np.isfinite(length_max):
            length_max = np.inf
        lengths = np.maximum(lengths, float(length_floor))
        lengths = np.maximum(lengths, float(length_min))
        lengths = np.minimum(lengths, float(length_max))
        return lengths

    def _equal_terminal_build_gen_f_cache(
        self,
        gen_idx: list[int],
        *,
        density_k: int,
    ) -> dict:
        gen_idx_arr = np.asarray(gen_idx, dtype=int)
        out = {
            "pos": np.empty((0,), dtype=int),
            "D": np.empty((0,), dtype=float),
            "k": int(density_k),
        }
        if gen_idx_arr.size == 0:
            return out

        seg_count = int(getattr(self, "segment_count", 0))
        if seg_count <= 0:
            return out

        data = np.asarray(self.data[:seg_count])
        distal = np.asarray(data[gen_idx_arr, 3:6], dtype=float)
        valid = np.all(np.isfinite(distal), axis=1)
        d_mean = np.full((gen_idx_arr.size,), np.nan, dtype=float)

        k = max(1, int(density_k))
        if np.count_nonzero(valid) > k:
            pts = distal[valid]
            kd = cKDTree(pts)
            dists, _ = kd.query(pts, k=k + 1)
            if dists.ndim == 1:
                dists = dists[:, None]
            d_mean[valid] = np.mean(dists[:, 1 : k + 1], axis=1)

        pos = np.full((seg_count,), -1, dtype=int)
        pos[gen_idx_arr] = np.arange(gen_idx_arr.size, dtype=int)
        out["pos"] = pos
        out["D"] = d_mean
        return out

    @staticmethod
    def _equal_terminal_sample_gen_f_mix2(
        n: int,
        *,
        mix_w: float,
        mix_mu1_ln: float,
        mix_sig1_ln: float,
        mix_mu2_ln: float,
        mix_sig2_ln: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0,), dtype=float)
        w = float(mix_w)
        if not np.isfinite(w):
            w = 0.5
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        choose1 = rng.random(n) < w
        out = np.empty((n,), dtype=float)
        n1 = int(np.count_nonzero(choose1))
        n2 = n - n1
        if n1 > 0:
            out[choose1] = np.exp(rng.normal(float(mix_mu1_ln), float(mix_sig1_ln), size=n1))
        if n2 > 0:
            out[~choose1] = np.exp(rng.normal(float(mix_mu2_ln), float(mix_sig2_ln), size=n2))
        return out

    def _equal_terminal_compute_length_gen_f_mix2(
        self,
        parent_idx: np.ndarray,
        *,
        base_length: float,
        gen_f_cache: Optional[dict],
        mix_w: float,
        mix_mu1_ln: float,
        mix_sig1_ln: float,
        mix_mu2_ln: float,
        mix_sig2_ln: float,
        length_min: float,
        length_max: float,
        length_floor: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        parent_idx = np.asarray(parent_idx, dtype=int)
        if parent_idx.size == 0:
            return np.full((0,), float(base_length), dtype=float)

        lengths = np.full((parent_idx.size,), float(base_length), dtype=float)
        if not gen_f_cache or "pos" not in gen_f_cache or "D" not in gen_f_cache:
            return lengths

        pos = np.asarray(gen_f_cache.get("pos"), dtype=int)
        d_map = np.asarray(gen_f_cache.get("D"), dtype=float)
        if pos.size == 0 or d_map.size == 0:
            return lengths

        parent_pos = pos[parent_idx]
        valid = parent_pos >= 0
        if not np.any(valid):
            return lengths
        D = np.full_like(parent_pos, np.nan, dtype=float)
        D[valid] = d_map[parent_pos[valid]]
        valid &= np.isfinite(D) & (D > 0.0)
        if not np.any(valid):
            return lengths

        f = self._equal_terminal_sample_gen_f_mix2(
            int(np.count_nonzero(valid)),
            mix_w=mix_w,
            mix_mu1_ln=mix_mu1_ln,
            mix_sig1_ln=mix_sig1_ln,
            mix_mu2_ln=mix_mu2_ln,
            mix_sig2_ln=mix_sig2_ln,
            rng=rng,
        )
        lengths[valid] = D[valid] * f

        if not np.isfinite(length_max):
            length_max = np.inf
        lengths = np.maximum(lengths, float(length_floor))
        lengths = np.maximum(lengths, float(length_min))
        lengths = np.minimum(lengths, float(length_max))
        return lengths

    def _equal_terminal_domain_eval(
        self,
        points: np.ndarray,
        *,
        workers: int,
        chunk: int,
        min_points: int,
    ) -> np.ndarray:
        if points.size == 0:
            return np.empty((0,), dtype=float)
        domain = getattr(self, "domain", None)
        if domain is None:
            return np.full((points.shape[0],), np.inf, dtype=float)
        points = np.asarray(points, dtype=float)
        n = points.shape[0]
        if workers <= 1 or (min_points > 0 and n < min_points):
            return np.asarray(domain(points)).reshape(-1)
        if chunk <= 0:
            chunk = max(1, int(math.ceil(n / workers)))
        chunk = max(1, int(chunk))
        out = np.empty((n,), dtype=float)
        slices = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(domain, points[s:e]): (s, e) for s, e in slices}
            for fut, (s, e) in futures.items():
                out[s:e] = np.asarray(fut.result()).reshape(-1)
        return out

    def _equal_terminal_select(
        self,
        parent_idx: np.ndarray,
        base_length: Union[np.ndarray, float],
        *,
        n_candidates: int,
        alpha_min: float,
        alpha_mode: float,
        alpha_max: float,
        psi_step: float,
        psi_max: float,
        domain_margin: float,
        check_midpoint: bool,
        domain_workers: int,
        domain_chunk: int,
        domain_min_points: int,
        length_shrink: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        parent_idx = np.asarray(parent_idx, dtype=int)
        P = parent_idx.size
        dirs1 = np.full((P, 3), np.nan, dtype=float)
        dirs2 = np.full((P, 3), np.nan, dtype=float)
        if np.ndim(base_length) == 0:
            base_lengths = np.full((P,), float(base_length), dtype=float)
        else:
            base_lengths = np.asarray(base_length, dtype=float)
            if base_lengths.shape[0] != P:
                raise ValueError("base_length array must match parent_idx size.")
        lengths_used = base_lengths.copy()
        ok = np.zeros((P,), dtype=bool)

        t_domain = 0.0

        def _attempt(length_val: Union[np.ndarray, float], target_mask: np.ndarray):
            nonlocal ok, dirs1, dirs2, lengths_used, t_domain
            if not np.any(target_mask):
                return
            idx = np.flatnonzero(target_mask)
            data = np.asarray(self.data[: self.segment_count])
            parent_end = data[parent_idx[idx], 3:6]
            parent_start = data[parent_idx[idx], 0:3]
            w = parent_end - parent_start
            w_norm = np.linalg.norm(w, axis=1)
            valid = w_norm > 1e-12
            if not np.any(valid):
                return
            w = w[valid] / w_norm[valid][:, None]
            parent_end = parent_end[valid]
            idx = idx[valid]

            # Orthonormal basis
            ref = np.tile(np.array([0.0, 0.0, 1.0]), (w.shape[0], 1))
            swap = np.abs(w[:, 2]) > 0.9
            ref[swap] = np.array([0.0, 1.0, 0.0])
            u = np.cross(ref, w)
            u_norm = np.linalg.norm(u, axis=1)
            u_norm[u_norm <= 1e-12] = 1.0
            u = u / u_norm[:, None]
            v = np.cross(w, u)

            C = max(1, int(n_candidates))
            alpha = rng.triangular(alpha_min, alpha_mode, alpha_max, size=(w.shape[0], C))
            alpha = np.deg2rad(alpha)
            phi = rng.uniform(0.0, math.pi, size=(w.shape[0], C))
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)
            p_vec = cos_p[..., None] * u[:, None, :] + sin_p[..., None] * v[:, None, :]
            d1 = w[:, None, :] * cos_a[..., None] + p_vec * sin_a[..., None]
            d2 = w[:, None, :] * cos_a[..., None] - p_vec * sin_a[..., None]

            # normalize directions
            d1 = d1 / np.linalg.norm(d1, axis=2, keepdims=True)
            d2 = d2 / np.linalg.norm(d2, axis=2, keepdims=True)

            # build psi list
            psi_list = [0.0]
            if psi_step > 0 and psi_max > 0:
                steps = int(math.floor(psi_max / psi_step))
                for k in range(1, steps + 1):
                    psi_list.append(math.radians(k * psi_step))
                    psi_list.append(math.radians(-k * psi_step))

            chosen = np.zeros((w.shape[0],), dtype=bool)
            chosen_d1 = np.full((w.shape[0], 3), np.nan, dtype=float)
            chosen_d2 = np.full((w.shape[0], 3), np.nan, dtype=float)

            for psi in psi_list:
                remaining = ~chosen
                if not np.any(remaining):
                    break
                d1_use = d1[remaining]
                d2_use = d2[remaining]
                axis = p_vec[remaining]
                if abs(psi) > 1e-12:
                    cos_t = math.cos(psi)
                    sin_t = math.sin(psi)
                    d1_use = (
                        d1_use * cos_t
                        + np.cross(axis, d1_use) * sin_t
                        + axis * np.sum(axis * d1_use, axis=2, keepdims=True) * (1.0 - cos_t)
                    )
                    d2_use = (
                        d2_use * cos_t
                        + np.cross(axis, d2_use) * sin_t
                        + axis * np.sum(axis * d2_use, axis=2, keepdims=True) * (1.0 - cos_t)
                    )
                    d1_use = d1_use / np.linalg.norm(d1_use, axis=2, keepdims=True)
                    d2_use = d2_use / np.linalg.norm(d2_use, axis=2, keepdims=True)

                start = parent_end[remaining][:, None, :]
                if np.ndim(length_val) == 0:
                    L = float(length_val)
                    end1 = start + L * d1_use
                    end2 = start + L * d2_use
                else:
                    L = np.asarray(length_val, dtype=float)[idx][remaining][:, None, None]
                    end1 = start + L * d1_use
                    end2 = start + L * d2_use
                points = [end1.reshape(-1, 3), end2.reshape(-1, 3)]
                if check_midpoint:
                    mid1 = start + 0.5 * L * d1_use
                    mid2 = start + 0.5 * L * d2_use
                    points.extend([mid1.reshape(-1, 3), mid2.reshape(-1, 3)])
                pts = np.vstack(points)

                t0 = perf_counter()
                vals = self._equal_terminal_domain_eval(
                    pts,
                    workers=domain_workers,
                    chunk=domain_chunk,
                    min_points=domain_min_points,
                )
                t_domain += perf_counter() - t0
                inside = vals <= -float(domain_margin)
                total = d1_use.shape[0] * d1_use.shape[1]
                cursor = 0
                ok1 = inside[cursor: cursor + total].reshape(d1_use.shape[0], d1_use.shape[1])
                cursor += total
                ok2 = inside[cursor: cursor + total].reshape(d1_use.shape[0], d1_use.shape[1])
                cursor += total
                valid = ok1 & ok2
                if check_midpoint:
                    ok3 = inside[cursor: cursor + total].reshape(d1_use.shape[0], d1_use.shape[1])
                    cursor += total
                    ok4 = inside[cursor: cursor + total].reshape(d1_use.shape[0], d1_use.shape[1])
                    valid &= ok3 & ok4

                any_valid = valid.any(axis=1)
                if np.any(any_valid):
                    first_idx = valid.argmax(axis=1)
                    rows = np.flatnonzero(any_valid)
                    cols = first_idx[any_valid]
                    remaining_idx = np.flatnonzero(remaining)
                    sel = remaining_idx[any_valid]
                    chosen[sel] = True
                    chosen_d1[sel] = d1_use[rows, cols]
                    chosen_d2[sel] = d2_use[rows, cols]

            ok[idx] = chosen
            dirs1[idx] = chosen_d1
            dirs2[idx] = chosen_d2
            if np.ndim(length_val) == 0:
                lengths_used[idx] = float(length_val)
            else:
                lengths_used[idx] = np.asarray(length_val, dtype=float)[idx]

        # first attempt with base length
        _attempt(base_lengths, np.ones((P,), dtype=bool))
        # shrink and retry for failures
        if length_shrink and length_shrink > 0.0 and length_shrink < 1.0:
            pending = ~ok
            if np.any(pending):
                _attempt(base_lengths * float(length_shrink), pending)
                lengths_used[pending] = base_lengths[pending] * float(length_shrink)

        return ok, dirs1, dirs2, lengths_used, t_domain

    def _equal_terminal_commit(
        self,
        parent_idx: np.ndarray,
        dirs1: np.ndarray,
        dirs2: np.ndarray,
        lengths: np.ndarray,
        *,
        min_radius_floor: float,
        defer_hnsw: bool,
    ) -> list[int]:
        parent_idx = np.asarray(parent_idx, dtype=int)
        if parent_idx.size == 0:
            return []

        data = self.preallocate
        seg_count = int(self.segment_count)
        n_new = parent_idx.size * 2
        needed = seg_count + n_new
        if needed > data.shape[0]:
            grow = max(self.preallocation_step, n_new + 1)
            new_size = data.shape[0] + grow
            new_data = TreeData((new_size, 31), dtype=self.data_dtype)
            new_data[:seg_count, :] = data[:seg_count, :]
            self.preallocate = new_data
            data = self.preallocate
            new_mid = np.zeros((new_size, 3), dtype=self.data_dtype)
            new_mid[:seg_count, :] = self.preallocate_midpoints[:seg_count, :]
            self.preallocate_midpoints = new_mid

        child1_idx = seg_count + 2 * np.arange(parent_idx.size)
        child2_idx = child1_idx + 1

        parent_end = data[parent_idx, 3:6]
        parent_radius = data[parent_idx, 21]
        parent_depth = data[parent_idx, 26]
        parent_scale = data[parent_idx, 28]
        parent_scale = np.where(np.isfinite(parent_scale), parent_scale, 1.0)
        parent_depth = np.where(np.isfinite(parent_depth), parent_depth, 0.0)
        parent_distal_node = data[parent_idx, 19]

        gamma = float(getattr(self.parameters, "murray_exponent", 3.0))
        ratio = 2.0 ** (-1.0 / gamma)
        child_radius = parent_radius * ratio
        if min_radius_floor and np.isfinite(min_radius_floor):
            child_radius = np.maximum(child_radius, float(min_radius_floor))

        lengths = np.asarray(lengths, dtype=float).reshape(-1)
        lengths = np.where(lengths > 0, lengths, 1e-6)

        child1_end = parent_end + dirs1 * lengths[:, None]
        child2_end = parent_end + dirs2 * lengths[:, None]

        w1 = dirs1 / np.linalg.norm(dirs1, axis=1, keepdims=True)
        w2 = dirs2 / np.linalg.norm(dirs2, axis=1, keepdims=True)
        u1, v1 = self._equal_terminal_basis(w1)
        u2, v2 = self._equal_terminal_basis(w2)

        # new node ids
        start_node = int(getattr(self, "max_distal_node", int(np.nanmax(data[:seg_count, 19]) if seg_count else 0)))
        new_nodes = np.arange(start_node + 1, start_node + 1 + n_new)
        self.max_distal_node = int(start_node + n_new)

        # fill child 1
        data[child1_idx, 0:3] = parent_end
        data[child1_idx, 3:6] = child1_end
        data[child1_idx, 6:9] = u1
        data[child1_idx, 9:12] = v1
        data[child1_idx, 12:15] = w1
        data[child1_idx, 15:17] = np.nan
        data[child1_idx, 17] = parent_idx
        data[child1_idx, 18] = parent_distal_node
        data[child1_idx, 19] = new_nodes[0::2]
        data[child1_idx, 20] = lengths
        data[child1_idx, 21] = child_radius
        data[child1_idx, 22] = np.nan
        data[child1_idx, 23] = np.nan
        data[child1_idx, 24] = np.nan
        data[child1_idx, 25] = np.nan
        data[child1_idx, 26] = parent_depth + 1
        data[child1_idx, 27] = lengths
        data[child1_idx, 28] = parent_scale * ratio

        # fill child 2
        data[child2_idx, 0:3] = parent_end
        data[child2_idx, 3:6] = child2_end
        data[child2_idx, 6:9] = u2
        data[child2_idx, 9:12] = v2
        data[child2_idx, 12:15] = w2
        data[child2_idx, 15:17] = np.nan
        data[child2_idx, 17] = parent_idx
        data[child2_idx, 18] = parent_distal_node
        data[child2_idx, 19] = new_nodes[1::2]
        data[child2_idx, 20] = lengths
        data[child2_idx, 21] = child_radius
        data[child2_idx, 22] = np.nan
        data[child2_idx, 23] = np.nan
        data[child2_idx, 24] = np.nan
        data[child2_idx, 25] = np.nan
        data[child2_idx, 26] = parent_depth + 1
        data[child2_idx, 27] = lengths
        data[child2_idx, 28] = parent_scale * ratio

        # update parent children + bifurcation ratios
        data[parent_idx, 15] = child1_idx
        data[parent_idx, 16] = child2_idx
        data[parent_idx, 23] = ratio
        data[parent_idx, 24] = ratio

        # update connectivity
        if getattr(self, "connectivity", None) is None or self.connectivity.shape[0] < needed:
            conn = np.full((max(needed, 1), 3), -1, dtype=self.index_dtype)
            if getattr(self, "connectivity", None) is not None:
                conn[: self.connectivity.shape[0], :] = self.connectivity
            self.connectivity = conn
        self.connectivity[parent_idx, 0] = child1_idx
        self.connectivity[parent_idx, 1] = child2_idx
        self.connectivity[parent_idx, 2] = np.nan_to_num(data[parent_idx, 17], nan=-1.0).astype(self.index_dtype)
        self.connectivity[child1_idx, :] = np.vstack([np.full(child1_idx.shape, -1, dtype=self.index_dtype),
                                                      np.full(child1_idx.shape, -1, dtype=self.index_dtype),
                                                      parent_idx]).T
        self.connectivity[child2_idx, :] = np.vstack([np.full(child2_idx.shape, -1, dtype=self.index_dtype),
                                                      np.full(child2_idx.shape, -1, dtype=self.index_dtype),
                                                      parent_idx]).T

        # update vessel map
        if not isinstance(self.vessel_map, dict):
            self.vessel_map = TreeMap()
        for i, p in enumerate(parent_idx):
            p = int(p)
            if p not in self.vessel_map:
                self.vessel_map[p] = {"upstream": [], "downstream": []}
            upstream = list(self.vessel_map[p]["upstream"])
            child_upstream = upstream + [p]
            c1 = int(child1_idx[i])
            c2 = int(child2_idx[i])
            self.vessel_map[p]["downstream"].extend([c1, c2])
            self.vessel_map[c1] = {"upstream": list(child_upstream), "downstream": []}
            self.vessel_map[c2] = {"upstream": list(child_upstream), "downstream": []}

        # update midpoints + hnsw
        mid1 = 0.5 * (parent_end + child1_end)
        mid2 = 0.5 * (parent_end + child2_end)
        self.preallocate_midpoints[child1_idx, :] = mid1
        self.preallocate_midpoints[child2_idx, :] = mid2
        if not defer_hnsw and getattr(self, "hnsw_tree", None) is not None:
            pts = np.vstack([mid1, mid2]).astype(np.float32)
            ids = np.concatenate([child1_idx, child2_idx]).astype(int)
            self.hnsw_tree.add_items(pts, ids)

        # finalize counts
        self.segment_count += int(n_new)
        self.n_terminals += int(parent_idx.size)
        self.data = self.preallocate[: self.segment_count, :]

        return list(child1_idx) + list(child2_idx)

    @staticmethod
    def _equal_terminal_basis(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.asarray(w)
        dtype = w.dtype
        ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=dtype), (w.shape[0], 1))
        swap = np.abs(w[:, 2]) > 0.9
        ref[swap] = np.array([0.0, 1.0, 0.0], dtype=dtype)
        u = np.cross(ref, w)
        u_norm = np.linalg.norm(u, axis=1)
        u_norm[u_norm <= 1e-12] = 1.0
        u = u / u_norm[:, None]
        v = np.cross(w, u)
        return u, v

    def save(self, path, *, include_domain=False, domain_path=None, domain_save_kwargs=None):
        """
        Save this Tree to a compact .npz archive that can be reloaded later.
        """
        if not os.path.splitext(path)[1]:
            path = path + ".tree.npz"

        if include_domain:
            if self.domain is None:
                raise ValueError("Tree has no domain to save.")
            if domain_path is None:
                domain_path = os.path.splitext(path)[0] + ".dmn"
            kwargs = domain_save_kwargs or {}
            self.domain.save(domain_path, **kwargs)

        data, payload = self._serialize_state(domain_path=domain_path)
        blob = np.array([pickle.dumps(payload)], dtype=object)
        np.savez_compressed(path, data=data, payload=blob)
        return path

    def _serialize_state(self, *, domain_path=None):
        data_dtype = _normalize_float_dtype(getattr(self, "data_dtype", None), self.data.dtype)
        data = np.asarray(self.data, dtype=data_dtype)
        index_dtype = _normalize_int_dtype(getattr(self, "index_dtype", None), np.int64)
        payload = {
            "version": 1,
            "vessel_map": dict(self.vessel_map),
            "parameters": self.parameters,
            "physical_clearance": self.physical_clearance,
            "random_seed": self.random_seed,
            "characteristic_length": self.characteristic_length,
            "domain_clearance": self.domain_clearance,
            "n_terminals": self.n_terminals,
            "nonconvex_count": self.nonconvex_count,
            "clamped_root": self.clamped_root,
            "segment_count": int(self.segment_count),
            "max_distal_node": getattr(self, "max_distal_node", None),
            "tree_scale": getattr(self, "tree_scale", None),
            "connectivity": getattr(self, "connectivity", None),
            "probability": getattr(self, "probability", None),
            "times": getattr(self, "times", None),
            "preallocation_step": int(getattr(self, "preallocation_step", data.shape[0] + 2)),
            "convex": getattr(self, "convex", None),
            "domain_path": domain_path,
            "data_dtype": str(np.dtype(data_dtype)),
            "index_dtype": str(np.dtype(index_dtype)),
        }
        return data, payload

    @classmethod
    def load(
        cls,
        path,
        *,
        domain=None,
        domain_path=None,
        data_dtype=None,
        index_dtype=None,
        analysis_only: bool = False,
    ):
        """
        Load a Tree saved with Tree.save().
        """
        if not os.path.splitext(path)[1] and os.path.exists(path + ".tree.npz"):
            path = path + ".tree.npz"

        # A legacy payload can contain a many-million-entry vessel_map object
        # graph.  Simulation-only callers need only the vessel table, and
        # inflating/unpickling that payload dominated both load time and peak
        # memory for large validation trees.  Match the frozen TissueSim
        # analysis loader by never touching the payload in this mode.
        if analysis_only:
            from cascade.vessels.metadata import inspect_network
            from cascade.vessels.prepared import find_prepared_tree

            archive_metadata = inspect_network(path)
            source_dtype = (
                np.dtype(archive_metadata.data_dtypes[0])
                if archive_metadata.data_dtypes
                else np.dtype(np.float64)
            )
            resolved_data_dtype = _normalize_float_dtype(
                data_dtype
                if data_dtype is not None
                else os.environ.get("SVV_TREE_DATA_DTYPE"),
                source_dtype,
            )
            resolved_index_dtype = _normalize_int_dtype(
                index_dtype
                if index_dtype is not None
                else os.environ.get("SVV_TREE_INDEX_DTYPE"),
                np.int64,
            )
            prepared = find_prepared_tree(
                path,
                data_dtype=resolved_data_dtype,
                index_dtype=resolved_index_dtype,
            )
            if prepared is not None:
                # Copy-on-write keeps the persistent cache immutable while
                # allowing legacy solve paths to update private working pages.
                data = np.load(prepared.data, mmap_mode="c", allow_pickle=False)
                exact_connectivity = np.load(
                    prepared.connectivity, mmap_mode="r", allow_pickle=False
                )
                exact_node_ids = np.load(
                    prepared.node_ids, mmap_mode="r", allow_pickle=False
                )
            else:
                with np.load(path, allow_pickle=False) as npz:
                    data = npz["data"]
                exact_connectivity = None
                exact_node_ids = None
                if data.dtype != resolved_data_dtype:
                    data = np.asarray(data, dtype=resolved_data_dtype)
                else:
                    data = np.asarray(data)
            domain = cls._coerce_domain(domain)
            tree = cls(
                preallocation_step=1,
                data_dtype=resolved_data_dtype,
                index_dtype=resolved_index_dtype,
            )
            tree_data = TreeData.from_array(data)
            tree.data = tree_data
            tree.preallocate = tree_data
            tree.segment_count = int(data.shape[0])
            if exact_connectivity is not None:
                terminal_mask = (exact_connectivity[:, 0] < 0) & (
                    exact_connectivity[:, 1] < 0
                )
                inferred_terminals = int(np.count_nonzero(terminal_mask))
            elif data.ndim == 2 and data.shape[1] > 16:
                terminal_mask = np.isnan(data[:, 15]) & np.isnan(data[:, 16])
                inferred_terminals = int(np.count_nonzero(terminal_mask))
            else:
                inferred_terminals = 0
            tree.n_terminals = inferred_terminals or max((tree.segment_count + 1) // 2, 0)
            distal_nodes = (
                exact_node_ids[:, 1]
                if exact_node_ids is not None
                else (
                    data[:, 19]
                    if data.ndim == 2 and data.shape[1] > 19
                    else np.empty((0,))
                )
            )
            finite_distal = distal_nodes[np.isfinite(distal_nodes)]
            tree.max_distal_node = int(np.max(finite_distal)) if finite_distal.size else tree.segment_count
            if data.ndim == 2 and data.shape[1] > 21:
                tree.tree_scale = float(
                    np.pi
                    * np.nansum(
                        (data[:, 21] ** tree.parameters.radius_exponent)
                        * (data[:, 20] ** tree.parameters.length_exponent)
                    )
                )
            else:
                tree.tree_scale = None
            tree.preallocation_step = tree.segment_count
            tree.preallocate_midpoints = np.empty((0, 3), dtype=resolved_data_dtype)
            tree.midpoints = tree.preallocate_midpoints
            tree.connectivity = exact_connectivity
            if exact_node_ids is not None:
                tree._cascade_node_ids = exact_node_ids
            tree.vessel_map = TreeMap()
            tree.kdtm = None
            tree.hnsw_tree = None
            tree.hnsw_tree_id = None
            tree.domain = domain
            tree.probability = None
            tree._analysis_only_load = True
            tree._cascade_prepared_mmap = prepared is not None
            return tree

        with np.load(path, allow_pickle=True) as npz:
            data = npz["data"]
            payload = pickle.loads(npz["payload"][0])

        payload_data_dtype = payload.get("data_dtype") if isinstance(payload, dict) else None
        payload_index_dtype = payload.get("index_dtype") if isinstance(payload, dict) else None
        resolved_data_dtype = _normalize_float_dtype(
            data_dtype if data_dtype is not None else (payload_data_dtype or os.environ.get("SVV_TREE_DATA_DTYPE")),
            data.dtype,
        )
        resolved_index_dtype = _normalize_int_dtype(
            index_dtype if index_dtype is not None else (payload_index_dtype or os.environ.get("SVV_TREE_INDEX_DTYPE")),
            np.int64,
        )
        if data.dtype != resolved_data_dtype:
            data = np.asarray(data, dtype=resolved_data_dtype)
        if isinstance(payload, dict):
            payload["data_dtype"] = str(np.dtype(resolved_data_dtype))
            payload["index_dtype"] = str(np.dtype(resolved_index_dtype))

        if domain is None and domain_path is None:
            domain_path = payload.get("domain_path")
        if domain is None and domain_path:
            from cascade.vessels.generation.svv_adapter import Domain
            domain = Domain.load(domain_path)
        else:
            domain = cls._coerce_domain(domain)

        return cls._from_state(data, payload, domain=domain, analysis_only=analysis_only)

    @staticmethod
    def _coerce_domain(domain):
        if domain is None:
            return None
        if hasattr(domain, "characteristic_length") and hasattr(domain, "convexity"):
            return domain
        if "pyvista" in str(domain.__class__):
            from cascade.vessels.generation.svv_adapter import Domain
            dmn = Domain(domain)
            dmn.create()
            dmn.solve()
            dmn.build()
            return dmn
        raise TypeError("domain must be a svv.domain.Domain or a pyvista dataset.")

    @classmethod
    def _from_state(cls, data, payload, *, domain=None, analysis_only: bool = False):
        data_dtype = _normalize_float_dtype(payload.get("data_dtype"), data.dtype)
        index_dtype = _normalize_int_dtype(payload.get("index_dtype"), np.int64)
        if analysis_only:
            preallocation_step = 1
        else:
            preallocation_step = int(payload.get("preallocation_step", data.shape[0] + 2))
            min_prealloc = int(getattr(data, "shape", (0,))[0]) + 2
            if preallocation_step < min_prealloc:
                preallocation_step = min_prealloc
        tree = cls(
            parameters=payload.get("parameters", None),
            preallocation_step=preallocation_step,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
        )
        if tree.parameters is None:
            tree.parameters = payload.get("parameters", tree.parameters)
        tree.physical_clearance = payload.get("physical_clearance", 0.0)
        tree.random_seed = payload.get("random_seed", None)
        tree.characteristic_length = payload.get("characteristic_length", None)
        tree.domain_clearance = payload.get("domain_clearance", None)
        tree.n_terminals = payload.get("n_terminals", 0)
        tree.nonconvex_count = payload.get("nonconvex_count", 0)
        tree.clamped_root = payload.get("clamped_root", False)
        tree.max_distal_node = payload.get("max_distal_node", 0)
        tree.tree_scale = payload.get("tree_scale", None)
        tree.convex = payload.get("convex", None)

        data = TreeData.from_array(np.asarray(data, dtype=data_dtype))
        tree.data = data
        segment_count = int(payload.get("segment_count", data.shape[0]))
        if segment_count <= 0:
            segment_count = int(data.shape[0])
        tree.segment_count = segment_count
        if analysis_only:
            tree.preallocate = tree.data
            tree.preallocation_step = int(segment_count)
            tree.preallocate_midpoints = np.empty((0, 3), dtype=data_dtype)
            tree.midpoints = tree.preallocate_midpoints
        else:
            midpoints = (data[:, 0:3] + data[:, 3:6]) / 2
            tree.preallocate[:data.shape[0], :] = data
            tree.preallocate_midpoints[:data.shape[0], :] = np.asarray(midpoints, dtype=data_dtype)
            tree.midpoints = tree.preallocate_midpoints[:segment_count, :]

        connectivity = None if analysis_only else payload.get("connectivity")
        if connectivity is None and not analysis_only:
            connectivity = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(index_dtype)
        else:
            connectivity = None if connectivity is None else np.asarray(connectivity, dtype=index_dtype)
        tree.connectivity = connectivity

        tree.vessel_map = TreeMap()
        if not analysis_only:
            tree.vessel_map.update(payload.get("vessel_map", {}))

        if domain is not None:
            if analysis_only:
                tree.domain = domain
            else:
                tree.set_domain(domain)
            tree.domain = domain
            if payload.get("domain_clearance") is not None:
                tree.domain_clearance = payload.get("domain_clearance")
            if payload.get("characteristic_length") is not None:
                tree.characteristic_length = payload.get("characteristic_length")
            if payload.get("convex") is not None:
                tree.convex = payload.get("convex")

        probability = payload.get("probability")
        if probability is not None:
            tree.probability = probability
            if tree.domain is not None:
                tree.domain.cumulative_probability = np.cumsum(probability)

        if segment_count > 0 and not analysis_only:
            tree.kdtm = KDTreeManager(tree.preallocate_midpoints[:segment_count, :])
            tree.hnsw_tree = USearchTree(midpoints.astype(np.float32))
            tree.hnsw_tree_id = id(tree.hnsw_tree)
        else:
            tree.kdtm = None
            tree.hnsw_tree = None
            tree.hnsw_tree_id = None

        tree.times = payload.get("times", tree.times)
        tree._analysis_only_load = bool(analysis_only)
        return tree

    def coerce_dtypes(self, *, data_dtype=None, index_dtype=None, rebuild_indices: bool = True):
        new_data_dtype = _normalize_float_dtype(data_dtype, self.data.dtype)
        new_index_dtype = _normalize_int_dtype(index_dtype, self.index_dtype)

        self.data_dtype = new_data_dtype
        self.index_dtype = new_index_dtype

        self.data = TreeData.from_array(np.asarray(self.data, dtype=new_data_dtype))
        self.preallocate = TreeData.from_array(np.asarray(self.preallocate, dtype=new_data_dtype))
        self.preallocate_midpoints = np.asarray(self.preallocate_midpoints, dtype=new_data_dtype)
        self.midpoints = self.preallocate_midpoints[: self.segment_count, :]

        if getattr(self, "connectivity", None) is not None:
            self.connectivity = np.asarray(self.connectivity, dtype=new_index_dtype)

        if rebuild_indices:
            if self.segment_count > 0:
                midpoints = self.preallocate_midpoints[: self.segment_count, :]
                self.kdtm = KDTreeManager(midpoints)
                self.hnsw_tree = USearchTree(midpoints.astype(np.float32))
                self.hnsw_tree_id = id(self.hnsw_tree)
            else:
                self.kdtm = None
                self.hnsw_tree = None
                self.hnsw_tree_id = None

        return self
