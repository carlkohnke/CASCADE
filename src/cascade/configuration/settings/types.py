"""Data structures used to describe a named group of configurable settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SettingSection:
    name: str
    defaults: dict[str, Any]
    aliases: dict[str, str] = field(default_factory=dict)
    prefixes: tuple[str, ...] = ()
    deprecated: dict[str, str] = field(default_factory=dict)

    @property
    def constants(self) -> frozenset[str]:
        return frozenset(self.defaults)
