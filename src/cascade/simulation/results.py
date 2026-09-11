"""Aggregation of per-network solver results into one simulation result space."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cascade.concentration.external_field.multinetwork import (
    compact_external_field_state,
)

_DEFAULT_INDEX_DTYPE = np.dtype(np.int64)


def safe_index_array(values: Any, dtype: np.dtype) -> np.ndarray:
    """Cast indices without silently overflowing a requested int32 export."""
    values64 = np.asarray(values, dtype=np.int64)
    target = np.dtype(dtype)
    if target == np.dtype(np.int32) and values64.size:
        limits = np.iinfo(np.int32)
        if int(values64.max()) > limits.max or int(values64.min()) < limits.min:
            raise OverflowError("Index array cannot be represented as int32")
    return values64.astype(target, copy=False)


def combine_network_solutions(
    solutions: Sequence[Mapping[str, Any]],
    *,
    index_dtype: np.dtype = _DEFAULT_INDEX_DTYPE,
) -> dict[str, Any]:
    """Concatenate one or more network solutions with stable global IDs."""
    combined: dict[str, Any] = {}
    for key in ("starts", "ends", "radii", "lengths", "flows", "cin", "cout"):
        arrays = [np.asarray(solution[key]) for solution in solutions]
        combined[key] = np.concatenate(arrays, axis=0) if arrays else np.empty((0,))
    if solutions and all(
        isinstance(solution.get("cext_state"), Mapping) for solution in solutions
    ):
        compact = [
            compact_external_field_state(dict(solution)) for solution in solutions
        ]
        combined["cext_state"] = {
            "solver": str(compact[0]["solver"]),
            "backend": str(compact[0]["backend"]),
            "gl_points_si": np.concatenate(
                [state["gl_points_si"] for state in compact], axis=0
            ),
            "diffusivity_si": float(compact[0]["diffusivity_si"]),
            "window_factor": float(compact[0]["window_factor"]),
            "c_iv_gl": np.concatenate([state["c_iv_gl"] for state in compact], axis=0),
            "c_bulk_gl": np.concatenate(
                [state["c_bulk_gl"] for state in compact], axis=0
            ),
            "c_wall_gl": np.concatenate(
                [state["c_wall_gl"] for state in compact], axis=0
            ),
            "c_ext_gl": np.concatenate(
                [state["c_ext_gl"] for state in compact], axis=0
            ),
            "lambda_iv_gl": np.concatenate(
                [state["lambda_iv_gl"] for state in compact], axis=0
            ),
            "k_if_gl": np.concatenate([state["k_if_gl"] for state in compact], axis=0),
            "q_line_gl": np.concatenate(
                [state["q_line_gl"] for state in compact], axis=0
            ),
            "q_weighted_gl": np.concatenate(
                [state["q_weighted_gl"] for state in compact], axis=0
            ),
            "mono2_weight_gl": np.concatenate(
                [state["mono2_weight_gl"] for state in compact], axis=0
            ),
            "dipole2_weight_gl": np.concatenate(
                [state["dipole2_weight_gl"] for state in compact], axis=0
            ),
            "seg_cap_gl": np.concatenate(
                [state["seg_cap_gl"] for state in compact], axis=0
            ),
            "segment_vectors": np.concatenate(
                [state["segment_vectors"] for state in compact], axis=0
            ),
            "export_slim_state": True,
        }
        combined["cext_mean"] = np.concatenate(
            [np.mean(state["c_ext_gl"], axis=1) for state in compact], axis=0
        )
        combined["c_iv_minus_cext_mean"] = np.concatenate(
            [
                np.mean(state["c_iv_gl"] - state["c_ext_gl"], axis=1)
                for state in compact
            ],
            axis=0,
        )
    tree_ids: list[np.ndarray] = []
    local_ids: list[np.ndarray] = []
    for network_id, solution in enumerate(solutions):
        count = int(np.asarray(solution["starts"]).shape[0])
        tree_ids.append(np.full(count, network_id, dtype=np.int16))
        local_ids.append(safe_index_array(np.arange(count), index_dtype))
    combined["tree_id"] = (
        np.concatenate(tree_ids) if tree_ids else np.empty((0,), dtype=np.int16)
    )
    combined["local_segment_id"] = (
        np.concatenate(local_ids) if local_ids else np.empty((0,), dtype=index_dtype)
    )
    combined["global_segment_id"] = safe_index_array(
        np.arange(combined["tree_id"].shape[0]), index_dtype
    )
    return combined


__all__ = ["combine_network_solutions", "safe_index_array"]
