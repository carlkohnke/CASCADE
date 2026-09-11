"""Shared domain-boundary and volume-mesh export."""

from __future__ import annotations

from pathlib import Path


def save_domain_geometry(
    domain,
    out_dir: Path,
    *,
    boundary_filename: str,
    mesh_filename: str,
) -> dict[str, str]:
    """Save available domain geometry and return generic boundary/mesh paths."""
    outputs: dict[str, str] = {}
    boundary = getattr(domain, "boundary", None)
    mesh = getattr(domain, "mesh", None)
    if boundary is None and mesh is not None:
        try:
            boundary = mesh.extract_surface()
        except Exception:
            boundary = None
    if boundary is None:
        try:
            domain.get_boundary()
            boundary = getattr(domain, "boundary", None)
        except Exception:
            boundary = None
    if boundary is not None:
        path = out_dir / boundary_filename
        boundary.save(str(path))
        outputs["boundary"] = str(path)
    if mesh is not None:
        path = out_dir / mesh_filename
        mesh.save(str(path))
        outputs["mesh"] = str(path)
    return outputs


__all__ = ["save_domain_geometry"]
