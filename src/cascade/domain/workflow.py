"""Run-configuration-driven domain construction."""

from __future__ import annotations

from cascade.vessels._build_common import (
    Domain,
    RunConfig,
    np,
    pv,
    resolve_domain_path,
    resolve_path,
)

from cascade.configuration.bridge import (
    load_runtime_module,
)
from cascade.domain.cache import domain_cache_path

def build_domain(config: RunConfig, ts=None):
    ts = ts or load_runtime_module()
    domain_cfg = config.domain
    np.random.seed(int(domain_cfg.random_seed))
    if getattr(domain_cfg, "mesh", None) is not None:
        return ts.build_domain(mesh=domain_cfg.mesh, random_seed=int(domain_cfg.random_seed))
    kind = str(domain_cfg.kind).strip().lower()
    if kind == "cube":
        return ts.build_domain(float(domain_cfg.side_length), random_seed=int(domain_cfg.random_seed))
    if kind in {"sphere", "pv.sphere", "pyvista_sphere"}:
        radius = float(domain_cfg.radius if domain_cfg.radius is not None else float(domain_cfg.side_length) / 2.0)
        center = tuple(float(v) for v in (domain_cfg.center or [0.0, 0.0, 0.0]))
        mesh = pv.Sphere(
            radius=radius,
            center=center,
            theta_resolution=int(domain_cfg.theta_resolution),
            phi_resolution=int(domain_cfg.phi_resolution),
        )
        return ts.build_domain(mesh=mesh, random_seed=int(domain_cfg.random_seed))
    if kind in {"box", "rectangular", "rectangular_box"}:
        x_len = float(domain_cfg.x_length if domain_cfg.x_length is not None else domain_cfg.side_length)
        y_len = float(domain_cfg.y_length if domain_cfg.y_length is not None else domain_cfg.side_length)
        z_len = float(domain_cfg.z_length if domain_cfg.z_length is not None else domain_cfg.side_length)
        domain = Domain(pv.Cube(x_length=x_len, y_length=y_len, z_length=z_len))
        domain.random_seed = int(domain_cfg.random_seed)
        domain.create()
        domain.solve()
        domain.build()
        domain.set_random_generator()
        return domain

    domain_path = resolve_domain_path(domain_cfg.path, base_dir=config.settings_path.parent if config.settings_path else None)
    if domain_path is None:
        raise ValueError("domain.path is required for non-cube domains.")
    suffix = domain_path.suffix.lower()
    if suffix == ".dmn":
        domain = Domain.load(str(domain_path))
        domain.random_seed = int(domain_cfg.random_seed)
        return domain

    cached_path = None
    if bool(domain_cfg.use_cache):
        cache_dir = resolve_path(
            domain_cfg.cache_dir,
            base_dir=config.settings_path.parent if config.settings_path else None,
        )
        cached_path = domain_cache_path(domain_path, cache_dir)
        if cached_path.is_file():
            try:
                domain = Domain.load(str(cached_path))
                domain.random_seed = int(domain_cfg.random_seed)
                domain.set_random_generator()
                print(f"Using CASCADE domain cache: {cached_path}", flush=True)
                return domain
            except Exception as exc:
                print(
                    f"Warning: could not load CASCADE domain cache {cached_path} "
                    f"({exc}); rebuilding.",
                    flush=True,
                )

    mesh = pv.read(str(domain_path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    domain = ts.build_domain(mesh=mesh, random_seed=int(domain_cfg.random_seed))
    if cached_path is not None:
        try:
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            domain.save(
                str(cached_path),
                include_boundary=True,
                include_mesh=True,
                include_patch_normals=False,
            )
            print(f"Saved CASCADE domain cache: {cached_path}", flush=True)
        except Exception as exc:
            print(
                f"Warning: could not save CASCADE domain cache {cached_path} ({exc}).",
                flush=True,
            )
    return domain




__all__ = ('build_domain',)
