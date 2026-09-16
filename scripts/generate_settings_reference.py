"""Generate the exhaustive runtime-settings appendix in docs/settings.md."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


BEGIN = "<!-- BEGIN GENERATED RUNTIME SETTINGS -->"
END = "<!-- END GENERATED RUNTIME SETTINGS -->"

SECTION_SOURCES = {
    "growth": "growth.py",
    "hemodynamics": "hemodynamics.py",
    "kirchhoff": "hemodynamics.py",
    "hematocrit": "hematocrit.py",
    "oxygen": "oxygen.py",
    "tissue": "tissue.py",
    "cext": "cext.py",
    "numerics": "numerics.py",
}

SECTION_PURPOSES = {
    "growth": "Equal-terminal and equal-bifurcation growth controls.",
    "hemodynamics": "Physical pressures, flow targets, and fluid properties.",
    "kirchhoff": "Pressure/flow linear-solver choices and convergence controls.",
    "hematocrit": "Red-cell partitioning, viscosity, and Pries-Secomb parameters.",
    "oxygen": "Oxygen transport, consumption, lumen exchange, and quadrature.",
    "tissue": "Tissue sampling, neighborhood accuracy, and tissue execution.",
    "cext": "Extravascular concentration coupling and its acceleration algorithms.",
    "numerics": "Data types, compilation, timing, and tree-cache behavior.",
}

ESSENTIAL_SCIENTIFIC = {
    "ROOT_PRESSURE",
    "TERMINAL_PRESSURE",
    "QIN_TARGET",
    "CUSTOM_FLUID_DENSITY_G_CM3",
    "CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP",
    "HEMATOCRIT_MODEL",
    "HD_DISCHARGE",
    "CONCENTRATION_INLET_BY_FLUID",
    "SOLUTE_DIFFUSIVITY",
    "VMAX_MM",
    "K_M_MM",
    "FINITE_RADIUS_O2_TERMS",
    "LUMEN_WALL_CLOSURE",
    "LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S",
    "O2_CAP_PER_HCT",
    "ALPHA_MMHG",
    "P50_MMHG",
    "N_HILL",
}

COMMON_WORKFLOW = {
    "N_EQUAL_BIFURCATIONS",
    "CONCENTRATION_SOLVER",
    "CONC_MAX_FOR_NORMALIZATION",
    "DISTANCE_SAMPLE_COUNT",
    "COMPUTE_AVG_DISTANCE_TO_CHANNEL",
    "TISSUE_ACCEL_MODE",
    "CEXT_ACCEL_MODE",
    "CEXT_VESS_COUPLING_MAX_ITER",
    "CEXT_VESS_COUPLING_TOL",
}

COMPATIBILITY_INTERNAL = {
    "EXTRAVASCULAR_CONCENTRATION",
    "LUMEN_DIFFUSIVITY_CM2_S",
    "DEBUG_ADD_VESSEL",
}

PERFORMANCE_TOKENS = (
    "ACCEL_MODE",
    "BATCH",
    "CACHE",
    "CHUNK",
    "DEBUG",
    "DEFER",
    "DIAGNOSTICS",
    "DTYPE",
    "FAST",
    "GPU",
    "NUMBA",
    "PARALLEL",
    "PRECOMPUTE",
    "RECORD",
    "REPORT",
    "SAVE_TREES",
    "STREAMING",
    "THREAD",
    "TIMING",
    "USE_TREE_CACHE",
    "WORKERS",
)

PREFERRED_KEYS = {
    # This is the spelling Studio writes and it retains the model's units.
    "K_M_MM": "k_m_mm",
}

DYNAMIC_DEFAULTS = {
    "EQUAL_TERMINAL_DOMAIN_WORKERS": "max(1, CPU count - 2)",
    "CEXT_PRECOMPUTE_WORKERS": "max(1, CPU count - 2)",
    "TISSUE_PARALLEL_WORKERS": "max(1, CPU count)",
    "TISSUE_STREAMING_CHUNK_WORKERS": "max(1, CPU count - 2)",
    "TISSUE_KDTREE_WORKERS": "max(1, CPU count - 2)",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    path = ROOT / "docs" / "settings.md"
    original = path.read_text(encoding="utf-8")
    generated = _generated_appendix()
    updated = _replace_generated(original, generated)
    if args.check:
        if updated != original:
            print(
                "docs/settings.md is stale; run "
                "python scripts/generate_settings_reference.py",
                file=sys.stderr,
            )
            return 1
        return 0
    path.write_text(updated, encoding="utf-8")
    return 0


def _generated_appendix() -> str:
    settings_sections = _settings_sections()
    parts = [
        BEGIN,
        "",
        "The tables below are generated from CASCADE's runtime-setting registry.",
        "Use the shorter JSON keys shown here. Exact uppercase constant names are",
        "also accepted, but are intended mainly for compatibility and debugging.",
        "",
    ]
    descriptions_by_file = {
        filename: _source_descriptions(
            ROOT / "src" / "cascade" / "configuration" / "settings" / filename
        )
        for filename in set(SECTION_SOURCES.values())
    }
    for section_name in SECTION_SOURCES:
        section = settings_sections[section_name]
        descriptions = descriptions_by_file[SECTION_SOURCES[section_name]]
        rows = []
        for constant, default in section.defaults.items():
            if section_name == "hemodynamics" and constant.startswith("KIRCHHOFF_"):
                continue
            if section_name == "kirchhoff" and not constant.startswith("KIRCHHOFF_"):
                continue
            key = _preferred_key(section_name, constant)
            rows.append(
                (
                    _importance(constant),
                    key,
                    constant,
                    _format_default(constant, default),
                    descriptions.get(constant, _humanize(constant)),
                )
            )
        rows.sort(key=lambda row: (_importance_order(row[0]), row[1]))
        parts.extend(
            [
                (
                    f"<details><summary><code>settings.{section_name}</code> — "
                    f"{len(rows)} options</summary>"
                ),
                "",
                SECTION_PURPOSES[section_name],
                "",
                "| JSON key | Importance | Default | Meaning |",
                "| --- | --- | --- | --- |",
            ]
        )
        for importance, key, constant, default, description in rows:
            canonical = f" `{constant}`" if key.upper() != constant else ""
            parts.append(
                f"| `{key}`{canonical} | {importance} | `{default}` | "
                f"{_escape(description)} |"
            )
        parts.extend(["", "</details>", ""])

    parts.extend(
        [
            "Compatibility views:",
            "",
            "- `settings.concentration` accepts the basic concentration, diffusivity,",
            "  consumption, relaxation, and quadrature keys from `settings.oxygen`.",
            "- `settings.hemodynamics` also accepts the `settings.kirchhoff` keys.",
            "- `settings.runtime` accepts any registered exact setting name. Prefer the",
            "  named sections above in new files.",
            "",
            END,
        ]
    )
    return "\n".join(parts)


def _preferred_key(section_name: str, constant: str) -> str:
    section = _settings_sections()[section_name]
    preferred = PREFERRED_KEYS.get(constant)
    if preferred is not None and _resolves_to(section, preferred) == constant:
        return preferred
    aliases = [alias for alias, target in section.aliases.items() if target == constant]
    exact_lower = constant.lower()
    if exact_lower in aliases:
        return exact_lower
    if aliases:
        return max(aliases, key=lambda value: (len(value), value))
    for prefix in section.prefixes:
        if constant.startswith(prefix):
            candidate = constant[len(prefix) :].lower()
            if _resolves_to(section, candidate) == constant:
                return candidate
    return exact_lower


def _settings_sections() -> dict[str, Any]:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    # Documentation must describe repository defaults, not developer-machine
    # environment overrides inherited while this script happens to run.
    for name in tuple(os.environ):
        if name.startswith("SVV_"):
            os.environ.pop(name)
    from cascade.configuration.settings.registry import SETTINGS_SECTIONS

    return SETTINGS_SECTIONS


def _resolves_to(section: Any, key: str) -> str | None:
    upper = key.upper()
    if upper in section.constants:
        return upper
    target = section.aliases.get(key.lower().replace("-", "_"))
    if target in section.constants:
        return target
    for prefix in section.prefixes:
        candidate = prefix + upper
        if candidate in section.constants:
            return candidate
    return None


def _importance(constant: str) -> str:
    if constant in ESSENTIAL_SCIENTIFIC:
        return "Essential scientific"
    if constant in COMMON_WORKFLOW:
        return "Common workflow"
    if constant in COMPATIBILITY_INTERNAL:
        return "Compatibility/internal"
    if any(token in constant for token in PERFORMANCE_TOKENS):
        return "Performance/runtime"
    return "Advanced scientific/numerical"


def _importance_order(value: str) -> int:
    return {
        "Essential scientific": 0,
        "Common workflow": 1,
        "Advanced scientific/numerical": 2,
        "Performance/runtime": 3,
        "Compatibility/internal": 4,
    }[value]


def _source_descriptions(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    descriptions: dict[str, str] = {}
    key_pattern = re.compile(r'^\s*"([A-Z][A-Z0-9_]*)"\s*:')
    for index, line in enumerate(lines):
        match = key_pattern.match(line)
        if match is None:
            continue
        comments: list[str] = []
        cursor = index - 1
        while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
            comments.append(lines[cursor].lstrip()[1:].strip())
            cursor -= 1
        if comments:
            descriptions[match.group(1)] = " ".join(reversed(comments))
    return descriptions


def _format_default(constant: str, value: Any) -> str:
    if constant in DYNAMIC_DEFAULTS:
        return DYNAMIC_DEFAULTS[constant]
    if isinstance(value, type):
        text = value.__name__
    elif isinstance(value, tuple):
        text = json.dumps(list(value), separators=(",", ":"))
    elif isinstance(value, dict):
        text = json.dumps(value, separators=(",", ":"), sort_keys=True)
    elif value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, str):
        text = json.dumps(value)
    else:
        text = str(value)
    return text.replace("|", "\\|")


def _humanize(constant: str) -> str:
    return constant.replace("_", " ").lower().capitalize() + "."


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _replace_generated(original: str, generated: str) -> str:
    if BEGIN not in original or END not in original:
        raise RuntimeError(f"Missing generated markers in {ROOT / 'docs/settings.md'}")
    prefix, remainder = original.split(BEGIN, 1)
    _old, suffix = remainder.split(END, 1)
    return prefix + generated + suffix


if __name__ == "__main__":
    raise SystemExit(main())
