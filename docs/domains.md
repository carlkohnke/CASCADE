# Tissue domains

CASCADE can create simple domains, use the packaged `bivent.stl` biventricular
heart surface, or load a surface or volume mesh readable by PyVista.

## File-backed domain

```json
{
  "domain": {
    "type": "file",
    "path": "geometry/my_domain.vtp",
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

Relative paths are resolved from the settings file, not the current working
directory. Store the mesh beside the study configuration or use an absolute
path.

Historical `.dmn` files can be loaded for compatibility, but they are not a
supported cross-version interchange format. Prefer a PyVista-readable mesh for
new studies.

The vascular network is configured separately. See
[custom vascular geometry](custom-geometry.md) for explicit CSV and NPZ
segment networks.
