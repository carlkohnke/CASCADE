"""Typed indexing for one or more vascular networks.

CASCADE treats a single tree and a multi-tree forest as the same simulation
primitive: an ordered collection of vascular networks. This module owns the
local/global segment mapping so callers do not reimplement offset arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class SegmentAddress:
    """A segment identified in collection-wide and network-local space."""

    global_id: int
    network_id: int
    local_id: int


@dataclass(frozen=True)
class VascularNetworkSet:
    """An ordered, indexable set of vascular networks."""

    networks: tuple[Any, ...]
    offsets: np.ndarray
    segment_counts: np.ndarray
    forest: Any | None = None

    @classmethod
    def from_networks(
        cls,
        networks: Iterable[Any],
        *,
        forest: Any | None = None,
    ) -> "VascularNetworkSet":
        items = tuple(networks)
        counts = np.asarray(
            [int(getattr(network, "segment_count", 0) or 0) for network in items],
            dtype=np.int64,
        )
        offsets = np.zeros(counts.size, dtype=np.int64)
        if counts.size > 1:
            offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
        return cls(items, offsets, counts, forest=forest)

    @classmethod
    def from_forest(cls, forest: Any) -> "VascularNetworkSet":
        networks = (
            network
            for network_group in forest.networks
            for network in network_group
        )
        return cls.from_networks(networks, forest=forest)

    def __len__(self) -> int:
        return len(self.networks)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.networks)

    @property
    def total_segments(self) -> int:
        return int(np.sum(self.segment_counts, dtype=np.int64))

    def offset_for(self, network_id: int) -> int:
        return int(self.offsets[int(network_id)])

    def address(self, global_segment_id: int) -> SegmentAddress:
        target = int(global_segment_id)
        if target < 0 or target >= self.total_segments:
            raise ValueError(
                f"Global segment id {target} is out of range for "
                f"{self.total_segments} total segments"
            )
        network_id = int(np.searchsorted(self.offsets, target, side="right") - 1)
        return SegmentAddress(
            global_id=target,
            network_id=network_id,
            local_id=target - int(self.offsets[network_id]),
        )

    def global_id(self, network_id: int, local_segment_id: int) -> int:
        network_id = int(network_id)
        local_segment_id = int(local_segment_id)
        if network_id < 0 or network_id >= len(self):
            raise ValueError(f"Network id {network_id} is out of range")
        count = int(self.segment_counts[network_id])
        if local_segment_id < 0 or local_segment_id >= count:
            raise ValueError(
                f"Local segment id {local_segment_id} is out of range for "
                f"network {network_id} ({count} segments)"
            )
        return int(self.offsets[network_id]) + local_segment_id


__all__ = ["SegmentAddress", "VascularNetworkSet"]
