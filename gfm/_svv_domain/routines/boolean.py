import numpy
import trimesh
import pyvista
import pymeshfix


def convert_to_trimesh(pyvista_object):
    """
    Convert a PyVista triangle surface mesh
    to a Trimesh object.
    :param pyvista_object:
    :return:
    """
    msh = pyvista_object.triangulate()
    faces = msh.faces.reshape((-1, 4))[:, 1:].astype(int)
    vertices = msh.points.astype(float)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def convert_to_pyvista(trimesh_object):
    """
    Convert a Trimesh object to a PyVista
    triangle surface mesh.
    :param trimesh_object:
    :return:
    """
    faces = numpy.hstack([numpy.full((len(trimesh_object.faces), 1), 3),
                          trimesh_object.faces])
    vertices = trimesh_object.vertices
    return pyvista.PolyData(vertices, faces.flatten())


def boolean(pyvista_object_1, pyvista_object_2, operation='union', fix_mesh=True, engine='manifold'):
    """
    Perform a boolean operation between two
    PyVista triangle surface meshes.
    :param pyvista_object_1:
    :param pyvista_object_2:
    :param operation:
    :return:
    """
    trimesh_object_1 = convert_to_trimesh(pyvista_object_1)
    trimesh_object_2 = convert_to_trimesh(pyvista_object_2)

    def _apply(op, eng):
        if op == 'union':
            return trimesh_object_1.union(trimesh_object_2, engine=eng)
        elif op == 'intersection':
            return trimesh_object_1.intersection(trimesh_object_2, engine=eng)
        elif op == 'difference':
            return trimesh_object_1.difference(trimesh_object_2, engine=eng)
        else:
            raise ValueError("Unsupported boolean operation.")

    result_tm = None
    try:
        try:
            result_tm = _apply(operation, engine)
        except KeyError:
            # If the requested engine (e.g., 'manifold') is not registered
            # in trimesh.boolean._engines, fall back to trimesh's default
            # engine selection by omitting the explicit engine argument.
            result_tm = _apply(operation, eng=None)
    except Exception:
        result_tm = None

    if result_tm is not None:
        result = convert_to_pyvista(result_tm)
    else:
        # Fallback to PyVista/VTK boolean operations when no trimesh
        # boolean backend is available on the current platform.
        if operation == 'union':
            result = pyvista_object_1.boolean_union(pyvista_object_2)
        elif operation == 'intersection':
            result = pyvista_object_1.boolean_intersection(pyvista_object_2)
        elif operation == 'difference':
            result = pyvista_object_1.boolean_difference(pyvista_object_2)
        else:
            raise ValueError("Unsupported boolean operation.")

    if not result.is_manifold and fix_mesh:
        fix = pymeshfix.MeshFix(result)
        fix.repair(verbose=False)
        result = fix.mesh
    return result
