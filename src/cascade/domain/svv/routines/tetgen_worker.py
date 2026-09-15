"""Run TetGen in an isolated subprocess and serialize its mesh result.

The domain tetrahedralization helper invokes this internal worker with a
surface mesh, an output NPZ path, and JSON-encoded TetGen arguments. It is not
a public CASCADE command; process isolation keeps native TetGen failures and
temporary allocations out of the calling simulation.
"""
import sys
import json
import numpy as np
import pyvista as pv
import tetgen


def main(surface_path: str, out_path: str, config_path: str):
    """Tetrahedralize ``surface_path`` and write node/element arrays to NPZ."""
    surface = pv.read(surface_path)

    # The parent process serializes positional and keyword arguments separately
    # so this worker does not need to import CASCADE configuration objects.
    with open(config_path, "r") as f:
        cfg = json.load(f)

    args = cfg.get("args", [])
    kwargs = cfg.get("kwargs", {})

    tgen = tetgen.TetGen(surface)
    result = tgen.tetrahedralize(*args, **kwargs)

    # TetGen releases return either an array tuple (possibly with extra arrays)
    # or expose the result through the wrapper object's node/element fields.
    if isinstance(result, tuple):
        nodes, elems = result[0], result[1]
    else:
        nodes = tgen.node
        elems = tgen.elem

    np.savez(out_path, nodes=nodes, elems=elems)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python tetgen_worker.py <surface_path> <out_path> <config_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
