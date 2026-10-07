"""Hematocrit propagation, caching, and nonlinear coupling."""

from __future__ import annotations

import numpy as np

from cascade.configuration import solver_state as _state

try:
    from numba import njit
except ImportError:  # pragma: no cover - dependency validation reports this earlier

    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate


def _normalize_hematocrit_model(value: str | None = None) -> str:
    mode = str(value or _state.HEMATOCRIT_MODEL).strip().lower()
    if mode in ("uniform", "uniform_tube", "constant", "diameter", "diameter_only"):
        return "uniform_tube"
    if mode in ("pries", "pries_secomb", "phase_separation", "plasma_skimming"):
        return "pries_secomb"
    raise ValueError("hematocrit model must be 'uniform_tube' or 'pries_secomb'.")


def _tube_hematocrit_from_hd_radius(radii_cm: np.ndarray, hd: np.ndarray) -> np.ndarray:
    radii_arr = np.asarray(radii_cm, dtype=float)
    hd_arr = np.clip(
        np.asarray(hd, dtype=float), _state.HEMATOCRIT_MIN, _state.HEMATOCRIT_MAX
    )
    d_um = 2.0 * radii_arr * 1.0e4
    ratio = hd_arr + (1.0 - hd_arr) * (
        1.0 + 1.7 * np.exp(-0.415 * d_um) - 0.6 * np.exp(-0.011 * d_um)
    )
    ht = hd_arr * ratio
    return np.where((radii_arr > 0.0) & np.isfinite(radii_arr), ht, 0.0)


if _state._HAVE_NUMBA:

    @njit(cache=True)
    def _phase_fraction_pries_numba(
        q_frac: float,
        d_parent_um: float,
        d_current_um: float,
        d_sibling_um: float,
        hd_parent: float,
        h_min: float,
        h_max: float,
    ) -> float:
        if q_frac <= 0.0:
            return 0.0
        if q_frac >= 1.0:
            return 1.0
        hp = hd_parent
        if hp < h_min:
            hp = h_min
        elif hp > h_max:
            hp = h_max
        dp = d_parent_um if d_parent_um > 1.0e-9 else 1.0e-9
        dc = d_current_um if d_current_um > 1.0e-9 else 1.0e-9
        ds = d_sibling_um if d_sibling_um > 1.0e-9 else 1.0e-9
        x0 = _state.PRIES_SECOMB_BIFPAR_1 * (1.0 - hp) / dp
        if x0 < 0.0:
            x0 = 0.0
        elif x0 > 0.49:
            x0 = 0.49
        if q_frac <= x0:
            return 0.0
        if q_frac >= 1.0 - x0:
            return 1.0
        diaquot = (dc * dc) / (ds * ds)
        asym = (diaquot - 1.0) / (diaquot + 1.0)
        A = _state.PRIES_SECOMB_BIFPAR_3 * asym * (1.0 - hp) / dp
        B = 1.0 + _state.PRIES_SECOMB_BIFPAR_2 * (1.0 - hp) / dp
        z = (q_frac - x0) / (1.0 - 2.0 * x0)
        if z < 1.0e-12:
            z = 1.0e-12
        elif z > 1.0 - 1.0e-12:
            z = 1.0 - 1.0e-12
        y = A + B * np.log(z / (1.0 - z))
        if y > 50.0:
            return 1.0
        if y < -50.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-y))

    @njit(cache=True)
    def _propagate_hematocrit_pries_numba(
        order: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        flows_abs: np.ndarray,
        diam_um: np.ndarray,
        hd_root: float,
        h_min: float,
        h_max: float,
    ) -> np.ndarray:
        nseg = flows_abs.shape[0]
        hd = np.empty(nseg, dtype=np.float64)
        for i in range(nseg):
            hd[i] = np.nan
        root_val = hd_root
        if root_val < h_min:
            root_val = h_min
        elif root_val > h_max:
            root_val = h_max
        for oi in range(order.shape[0]):
            idx = int(order[oi])
            if idx < 0 or idx >= nseg:
                continue
            hp = hd[idx]
            if not np.isfinite(hp):
                hp = root_val
                hd[idx] = hp
            left = int(left_child[idx])
            right = int(right_child[idx])
            has_left = left >= 0 and left < nseg
            has_right = right >= 0 and right < nseg
            if not has_left and not has_right:
                continue
            if has_left and not has_right:
                hd[left] = hp
                continue
            if has_right and not has_left:
                hd[right] = hp
                continue

            ql = flows_abs[left]
            qr = flows_abs[right]
            if not np.isfinite(ql) or ql < 0.0:
                ql = 0.0
            if not np.isfinite(qr) or qr < 0.0:
                qr = 0.0
            qsum = ql + qr
            if qsum <= 1.0e-300:
                hd[left] = hp
                hd[right] = hp
                continue
            fq_l = ql / qsum
            if fq_l <= 1.0e-12:
                fe_l = 0.0
            elif fq_l >= 1.0 - 1.0e-12:
                fe_l = 1.0
            else:
                fe_l = _phase_fraction_pries_numba(
                    fq_l,
                    diam_um[idx],
                    diam_um[left],
                    diam_um[right],
                    hp,
                    h_min,
                    h_max,
                )
            fq_r = 1.0 - fq_l
            fe_r = 1.0 - fe_l
            if fq_l > 1.0e-12:
                h_l = hp * fe_l / fq_l
            else:
                h_l = h_min
            if fq_r > 1.0e-12:
                h_r = hp * fe_r / fq_r
            else:
                h_r = h_min
            if h_l < h_min:
                h_l = h_min
            elif h_l > h_max:
                h_l = h_max
            if h_r < h_min:
                h_r = h_min
            elif h_r > h_max:
                h_r = h_max
            hd[left] = h_l
            hd[right] = h_r
        for i in range(nseg):
            if not np.isfinite(hd[i]):
                hd[i] = root_val
        return hd


def _tree_exact_connectivity(tree, data: np.ndarray) -> np.ndarray:
    conn = getattr(tree, "connectivity", None)
    if conn is not None and np.asarray(conn).shape == (data.shape[0], 3):
        return np.asarray(conn, dtype=np.int64)
    return np.nan_to_num(data[:, 15:18], nan=-1.0).astype(np.int64)


def _topdown_order_for_tree_data(
    data: np.ndarray,
    parents: np.ndarray | None = None,
    children: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    nseg = int(data.shape[0])
    if parents is None:
        parents = np.nan_to_num(data[:, 17], nan=-1.0).astype(np.int64)
    parent_rows = np.arange(nseg, dtype=np.int64)
    nonroot = parents >= 0
    if np.all(parents[nonroot] < parent_rows[nonroot]):
        return parent_rows, "index"
    if _state._HAVE_NUMBA:
        if children is None:
            left_child = np.nan_to_num(data[:, 15], nan=-1.0).astype(np.int64)
            right_child = np.nan_to_num(data[:, 16], nan=-1.0).astype(np.int64)
        else:
            left_child = np.asarray(children[:, 0], dtype=np.int64)
            right_child = np.asarray(children[:, 1], dtype=np.int64)
        roots = np.flatnonzero(parents < 0).astype(np.int64)
        order_candidate, n_order = _topdown_order_from_children_numba(
            left_child, right_child, roots, nseg
        )
        if int(n_order) == nseg:
            return order_candidate, "children_dfs"
        depths = np.asarray(data[:, 26], dtype=float)
        return np.argsort(depths, kind="mergesort").astype(
            np.int64, copy=False
        ), f"depth_sort_after_children_dfs_{int(n_order)}"
    depths = np.asarray(data[:, 26], dtype=float)
    return np.argsort(depths, kind="mergesort").astype(
        np.int64, copy=False
    ), "depth_sort"


def _hematocrit_context_for_tree(tree) -> dict[str, np.ndarray | str | int]:
    nseg = int(getattr(tree, "segment_count", 0))
    cached = getattr(tree, "_hematocrit_context", None)
    if isinstance(cached, dict) and int(cached.get("nseg", -1)) == nseg:
        return cached

    data = np.asarray(tree.data[:nseg])
    conn = _tree_exact_connectivity(tree, data)
    parents = conn[:, 2]
    left_child = conn[:, 0]
    right_child = conn[:, 1]
    order, order_mode = _topdown_order_for_tree_data(data, parents, conn[:, 0:2])
    radii = np.asarray(data[:, 21], dtype=float)
    context: dict[str, np.ndarray | str | int] = {
        "nseg": int(nseg),
        "parents": parents,
        "left_child": left_child,
        "right_child": right_child,
        "order": np.asarray(order, dtype=np.int64),
        "order_mode": str(order_mode),
        "radii": radii,
        "diam_um": np.asarray(2.0 * radii * 1.0e4, dtype=np.float64),
    }
    try:
        tree._hematocrit_context = context
    except Exception:
        pass
    return context


def _store_tree_hematocrit_cache(
    tree,
    HD: np.ndarray,
    HT: np.ndarray,
    *,
    model: str,
    flows: np.ndarray | None = None,
    fixed_flow_bc: bool = False,
) -> None:
    try:
        hd_arr = np.asarray(HD, dtype=np.float32)
        ht_arr = np.asarray(HT, dtype=np.float32)
        tree.discharge_hematocrit = hd_arr
        tree.tube_hematocrit = ht_arr
        # Chb_max stays in mol / m^3: HT is unitless and O2_CAP_PER_HCT is
        # already mol / m^3 per unit hematocrit.
        tree.Chb_max = ht_arr * float(_state.O2_CAP_PER_HCT)
        tree._hematocrit_cache_nseg = int(hd_arr.shape[0])
        tree._hematocrit_cache_model = _normalize_hematocrit_model(model)
        tree._hematocrit_cache_flows_id = id(flows) if flows is not None else None
        tree._hematocrit_cache_fixed_flow_bc = bool(fixed_flow_bc)
    except Exception:
        pass


def _get_tree_hematocrit_cache(
    tree,
    nseg: int,
    *,
    model: str,
    flows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if int(getattr(tree, "_hematocrit_cache_nseg", -1)) != int(nseg):
            return None
        if getattr(
            tree, "_hematocrit_cache_model", None
        ) != _normalize_hematocrit_model(model):
            return None
        flow_id = getattr(tree, "_hematocrit_cache_flows_id", None)
        fixed_flow_bc = bool(getattr(tree, "_hematocrit_cache_fixed_flow_bc", False))
        if not fixed_flow_bc and flows is not None and flow_id != id(flows):
            return None
        HD = np.asarray(tree.discharge_hematocrit, dtype=np.float32)
        HT = np.asarray(tree.tube_hematocrit, dtype=np.float32)
        if HD.shape[0] != nseg or HT.shape[0] != nseg:
            return None
        if not (np.all(np.isfinite(HD)) and np.all(np.isfinite(HT))):
            return None
        return HD, HT
    except Exception:
        return None


def compute_tree_hematocrit(
    tree,
    hd_root: float = _state.HD_DISCHARGE,
    *,
    flows: np.ndarray | None = None,
    model: str | None = None,
    order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty

    context = _hematocrit_context_for_tree(tree)
    radii = np.asarray(context["radii"], dtype=float)
    mode = _normalize_hematocrit_model(model)

    from cascade.concentration.vessel.network_gpu import resolve_network_accel

    if resolve_network_accel() == "gpu":
        from .hematocrit_gpu import compute_network_hematocrit_gpu

        # Exact connectivity avoids rounded coordinates and works with relabeled
        # tree rows. The public graph API handles actual converging networks.
        parents = np.asarray(context["parents"], dtype=np.int64)
        up = np.where(parents >= 0, parents + 1, 0)
        down = np.arange(nseg, dtype=np.int64) + 1
        return compute_network_hematocrit_gpu(
            up,
            down,
            np.ones(nseg) if flows is None else flows,
            radii,
            hd_root=hd_root,
            model=mode if flows is not None else "uniform_tube",
            context=context,
        )

    hd = float(hd_root)
    if mode == "pries_secomb" and flows is not None and _state._HAVE_NUMBA:
        if order is None:
            order = np.asarray(context["order"], dtype=np.int64)
        HD = _propagate_hematocrit_pries_numba(
            np.asarray(order, dtype=np.int64),
            np.asarray(context["left_child"], dtype=np.int64),
            np.asarray(context["right_child"], dtype=np.int64),
            np.abs(np.asarray(flows, dtype=float)),
            np.asarray(context["diam_um"], dtype=np.float64),
            hd,
            float(_state.HEMATOCRIT_MIN),
            float(_state.HEMATOCRIT_MAX),
        )
    elif mode == "pries_secomb" and flows is not None and not _state._HAVE_NUMBA:
        raise RuntimeError(
            "Pries-Secomb hematocrit propagation requires numba in this implementation."
        )
    else:
        HD = np.full(
            nseg, np.clip(hd, _state.HEMATOCRIT_MIN, _state.HEMATOCRIT_MAX), dtype=float
        )

    HT = _tube_hematocrit_from_hd_radius(radii, HD)

    return HD, HT


if _state._HAVE_NUMBA:

    @njit(cache=True)
    def _topdown_order_from_children_numba(
        left_child: np.ndarray,
        right_child: np.ndarray,
        roots: np.ndarray,
        nseg: int,
    ) -> tuple[np.ndarray, int]:
        order = np.empty(nseg, dtype=np.int64)
        stack = np.empty(nseg, dtype=np.int64)
        visited = np.zeros(nseg, dtype=np.uint8)
        top = 0
        for i in range(roots.shape[0] - 1, -1, -1):
            root = int(roots[i])
            if root >= 0 and root < nseg:
                stack[top] = root
                top += 1
        n_order = 0
        while top > 0:
            top -= 1
            idx = int(stack[top])
            if idx < 0 or idx >= nseg:
                continue
            if visited[idx] != 0:
                continue
            visited[idx] = 1
            order[n_order] = idx
            n_order += 1

            right = int(right_child[idx])
            if right >= 0 and right < nseg and visited[right] == 0:
                stack[top] = right
                top += 1
            left = int(left_child[idx])
            if left >= 0 and left < nseg and visited[left] == 0:
                stack[top] = left
                top += 1
        return order, n_order


__all__ = [
    "_normalize_hematocrit_model",
    "_tube_hematocrit_from_hd_radius",
    "_phase_fraction_pries_numba",
    "_propagate_hematocrit_pries_numba",
    "_tree_exact_connectivity",
    "_topdown_order_for_tree_data",
    "_hematocrit_context_for_tree",
    "_store_tree_hematocrit_cache",
    "_get_tree_hematocrit_cache",
    "compute_tree_hematocrit",
    "_topdown_order_from_children_numba",
]
