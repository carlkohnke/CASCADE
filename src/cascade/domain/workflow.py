"""Run-configuration-driven domain construction."""

from __future__ import annotations

from cascade.vessels._build_common import (
    Domain,
    RunConfig,
    np,
    pv,
    resolve_domain_path,
)

from cascade.configuration.bridge import (
    load_runtime_module,
)

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

    mesh = pv.read(str(domain_path))
    domain = Domain(mesh)
    domain.random_seed = int(domain_cfg.random_seed)
    domain.create()
    domain.solve()
    domain.build()
    domain.set_random_generator()
    return domain




__all__ = ('build_domain',)
