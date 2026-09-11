"""Reusable, reversible vascular interventions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np

from .collections import SegmentAddress, VascularNetworkSet
from .connectivity import collect_downstream_segment_ids


@dataclass(frozen=True)
class Occlusion:
    """Fractional obstruction of a globally identified vascular segment."""

    global_segment_id: int
    fraction_blocked: float
    include_downstream_when_complete: bool = True

    def __post_init__(self) -> None:
        fraction = float(self.fraction_blocked)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("Occlusion fraction_blocked must be in [0, 1].")


@dataclass
class InterventionState:
    """Resolved intervention state and restoration data for one solve."""

    applied: bool = False
    kind: str = "none"
    fraction_blocked: float = 0.0
    address: SegmentAddress | None = None
    affected_local_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )
    original_radii: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=float)
    )

    @property
    def network_id(self) -> int | None:
        return None if self.address is None else int(self.address.network_id)

    @property
    def local_segment_id(self) -> int | None:
        return None if self.address is None else int(self.address.local_id)

    @property
    def global_segment_id(self) -> int | None:
        return None if self.address is None else int(self.address.global_id)

    @property
    def mode(self) -> str:
        if not self.applied:
            return "none"
        return "full_occlusion_subtree" if self.fraction_blocked >= 1.0 else "radius_reduction"

    def metadata(self, networks: VascularNetworkSet) -> dict[str, Any]:
        effective_radius = None
        original_radius = None
        if self.original_radii.size:
            original_radius = float(self.original_radii[0])
            effective_radius = (
                0.0
                if self.fraction_blocked >= 1.0
                else original_radius * (1.0 - float(self.fraction_blocked))
            )
        preview_local = self.affected_local_ids[:20].astype(np.int64, copy=False)
        preview_global = (
            preview_local + networks.offset_for(int(self.network_id))
            if self.network_id is not None
            else np.empty((0,), dtype=np.int64)
        )
        return {
            "kind": self.kind,
            "applied": bool(self.applied),
            "mode": self.mode,
            "fraction_blocked": float(self.fraction_blocked),
            "global_segment_id": self.global_segment_id,
            "network_id": self.network_id,
            "local_segment_id": self.local_segment_id,
            "original_radius": original_radius,
            "effective_radius": effective_radius,
            "affected_segment_count": int(self.affected_local_ids.size),
            "affected_local_preview": preview_local.tolist(),
            "affected_global_preview": preview_global.tolist(),
        }

    def restore_solution_geometry(self, solution: Mapping[str, Any]) -> None:
        radii_value = solution.get("radii")
        if radii_value is None or not self.original_radii.size:
            return
        radii = np.asarray(radii_value)
        ids = self.affected_local_ids
        valid = (ids >= 0) & (ids < radii.shape[0])
        if np.any(valid):
            radii[ids[valid]] = self.original_radii[valid]


def occlusion_from_mapping(raw: Mapping[str, Any] | None) -> Occlusion | None:
    """Parse canonical and accepted legacy occlusion field names."""
    data = dict(raw or {})
    fraction = float(
        data.get("fraction_blocked", data.get("fraction_blocked_infarction", 0.0))
        or 0.0
    )
    if fraction < 0.0:
        raise ValueError("Occlusion fraction_blocked must be in [0, 1].")
    if fraction == 0.0:
        return None
    target = data.get("global_segment_id", data.get("infarction_global_segment_id"))
    if target is None:
        raise ValueError(
            "intervention.occlusion.global_segment_id is required when "
            "fraction_blocked > 0."
        )
    return Occlusion(global_segment_id=int(target), fraction_blocked=fraction)


def resolve_occlusion(
    networks: VascularNetworkSet,
    occlusion: Occlusion | None,
    *,
    geometry_only: bool = False,
) -> InterventionState:
    if occlusion is None or geometry_only or float(occlusion.fraction_blocked) <= 0.0:
        return InterventionState()
    address = networks.address(int(occlusion.global_segment_id))
    network = networks.networks[address.network_id]
    affected = np.asarray([address.local_id], dtype=np.int64)
    if (
        float(occlusion.fraction_blocked) >= 1.0
        and bool(occlusion.include_downstream_when_complete)
    ):
        affected = collect_downstream_segment_ids(network, address.local_id)
    return InterventionState(
        applied=True,
        kind="occlusion",
        fraction_blocked=float(occlusion.fraction_blocked),
        address=address,
        affected_local_ids=affected,
    )


@contextmanager
def apply_occlusion(
    networks: VascularNetworkSet,
    occlusion: Occlusion | None,
    *,
    geometry_only: bool = False,
) -> Iterator[InterventionState]:
    """Apply an occlusion for a solve and always restore anatomical radii."""
    state = resolve_occlusion(networks, occlusion, geometry_only=geometry_only)
    if not state.applied or state.network_id is None:
        yield state
        return
    network = networks.networks[state.network_id]
    data = np.asarray(network.data)
    ids = state.affected_local_ids
    state.original_radii = np.asarray(data[ids, 21], dtype=float).copy()
    try:
        if state.fraction_blocked >= 1.0:
            data[ids, 21] = 0.0
        elif ids.size:
            data[int(ids[0]), 21] = state.original_radii[0] * (
                1.0 - state.fraction_blocked
            )
        yield state
    finally:
        if ids.size:
            data[ids, 21] = state.original_radii


@contextmanager
def apply_occlusion_to_network(
    networks: VascularNetworkSet,
    occlusion: Occlusion | None,
    network_id: int,
    *,
    geometry_only: bool = False,
) -> Iterator[InterventionState]:
    """Apply a collection occlusion only while its target network is solved."""
    if occlusion is None or geometry_only:
        yield InterventionState()
        return
    address = networks.address(occlusion.global_segment_id)
    if address.network_id != int(network_id):
        yield InterventionState()
        return
    with apply_occlusion(networks, occlusion, geometry_only=False) as state:
        yield state


def zero_occluded_solution(
    solution: Mapping[str, Any], state: InterventionState
) -> None:
    """Zero transport values on fully occluded segments after a solve."""
    if not state.applied or state.fraction_blocked < 1.0:
        return
    ids = state.affected_local_ids
    for key in ("flows", "cin", "cout"):
        value = solution.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        valid = ids[(ids >= 0) & (ids < array.shape[0])]
        if valid.size:
            array[valid] = 0.0
    for state_key in ("cext_state", "cext_source_state"):
        cext_state = solution.get(state_key)
        if not isinstance(cext_state, Mapping):
            continue
        for key in (
            "cin_seg",
            "cout_seg",
            "c_iv_gl",
            "q_line_gl",
            "q_weighted_gl",
            "seg_cap_gl",
        ):
            value = cext_state.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            valid = ids[(ids >= 0) & (ids < array.shape[0])]
            if valid.size:
                array[valid] = 0.0


__all__ = [
    "InterventionState",
    "Occlusion",
    "apply_occlusion",
    "apply_occlusion_to_network",
    "occlusion_from_mapping",
    "resolve_occlusion",
    "zero_occluded_solution",
]
