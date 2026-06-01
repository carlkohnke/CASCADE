# tetgen_worker.py
import sys
import json
import numpy as np
import pyvista as pv
import tetgen


def main(surface_path: str, out_path: str, config_path: str):
    # Load surface mesh
    surface = pv.read(surface_path)

    # Load tetrahedralize args/kwargs
    with open(config_path, "r") as f:
        cfg = json.load(f)

    args = cfg.get("args", [])
    kwargs = cfg.get("kwargs", {})

    # Run TetGen
    tgen = tetgen.TetGen(surface)
    result = tgen.tetrahedralize(*args, **kwargs)

    # Handle different return formats from tetgen versions
    # Older versions return (nodes, elems), newer may return more values
    if isinstance(result, tuple):
        nodes, elems = result[0], result[1]
    else:
        # Some versions return a grid directly
        nodes = tgen.node
        elems = tgen.elem

    # Save result
    np.savez(out_path, nodes=nodes, elems=elems)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python tetgen_worker.py <surface_path> <out_path> <config_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
