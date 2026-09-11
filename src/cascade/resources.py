from __future__ import annotations

from pathlib import Path


CASCADE_PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_DOMAIN_ROOT = CASCADE_PACKAGE_ROOT / "assets" / "domains"


def resolve_path(
    value: str | Path | None, *, base_dir: Path | None = None
) -> Path | None:
    """Resolve a user path relative to its settings file when available."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def resolve_domain_path(
    value: str | Path | None, *, base_dir: Path | None = None
) -> Path | None:
    """Resolve user domains, then fall back to CASCADE's packaged library."""
    path = resolve_path(value, base_dir=base_dir)
    if path is None or path.exists():
        return path

    original = Path(value).expanduser()
    if original.is_absolute():
        return path

    library_path = (PACKAGED_DOMAIN_ROOT / original).resolve()
    if library_path.exists():
        return library_path
    return path
