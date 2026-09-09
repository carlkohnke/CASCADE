# Domain Mesh Library

Use this folder for local PyVista-generated domain meshes that you want to reference from GFM JSON settings.

Example:

```python
from pathlib import Path
import pyvista as pv

domains = Path(__file__).resolve().parent

mesh = pv.Sphere(radius=0.5, center=(0.0, 0.0, 0.0)).triangulate().clean()
mesh.save(domains / "sphere_r0p5.vtp")
```

Then reference it from any settings file:

```json
{
  "domain": {
    "type": "file",
    "path": "sphere_r0p5.vtp",
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

When `domain.path` is relative, GFM first looks relative to the settings file. If the file is not there, it falls back to this `domains/` folder.

Large mesh files are intentionally ignored by git through the repository `.gitignore`.
