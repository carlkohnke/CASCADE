import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

# Import only the required VTK classes from vtkmodules to avoid
# pulling in optional IO backends (e.g., NetCDF) via umbrella import.
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkAppendFilter,
    vtkThreshold,
    vtkMarchingSquares,
)
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkTransformPolyDataFilter
from vtkmodules.vtkCommonDataModel import (
    vtkUnstructuredGrid,
    vtkDataObject,
    vtkUniformGrid,
    vtkImageData,
)
from vtkmodules.util import numpy_support

def descritize(domain, **kwargs):
    """
    This function evaluates the implicit representation
    of the domain and returns a discrete representation
    as a 2D/3D mesh.

    Parameters
    ----------
    domain : svtoolkit.Domain object
        This is the domain object to be discretized.

    Return
    ------
    mesh : vtk.vtkPolyData or vtk.vtkUnstructuredGrid
        This is the discrete representation of the domain.
        If the domain is 2D, then a vtkPolyData object is
        returned. If the domain is 3D, then a vtkUnstructuredGrid
        object is returned.

    Notes
    -----
    """
    buffer = kwargs.get("buffer", 1.0)
    resolution = kwargs.get("resolution", 20)
    mins = np.min(domain.points, axis=0)
    maxs = np.max(domain.points, axis=0)
    ranges = maxs - mins
    origin = (maxs + mins)/2
    buffers = (ranges * buffer) / 2
    scale = (ranges + 2*buffers)
    if domain.d == 2:
        mesh = threshold_2d(domain.evaluate, resolution, origin, scale, 0, -1)
    elif domain.d == 3:
        mesh = threshold_3d(domain.evaluate, resolution, origin, scale, 0, -1)
    else:
        raise ValueError("Only 2D and 3D domains are supported.")
    return mesh


def boundaries(domain, **kwargs):
    pass


def threshold_2d(function, resolution, origin, scale, upper, lower):
    """
    This function uses thresholding techniques to
    discretize a 2D subdomain of a defined 2D
    implicit domain.

    Parameters
    ----------
    function : callable
        This is the function that defines the implicit
        domain.
    resolution : int
        This is the number of points to discretize the
        two nominal dimensions of the domain.
    origin : array_like
        This is the origin of the domain.
    scale : array_like
        This is the scale of the domain.
    upper : float
        This is the upper threshold value.
    lower : float
        This is the lower threshold value.

    Returns
    -------
    mesh : vtkPolyData
        This is the discrete mesh of the 2D domain.
        The vtkPolyData object retains the scalar
        values of the function evaluated at each
        point. The scalar values are stored in the
        vtkPolyData point data as array named
        "ImplicitFunctionValue".

    Notes
    -----

    """
    # Build a plane source to evaluate the function
    plane = vtkPlaneSource()
    plane.SetXResolution(resolution)
    plane.SetYResolution(resolution)
    plane.Update()
    # Define the transform to scale the plane
    transform = vtkTransform()
    transform.Scale(scale[0], scale[1], 1)
    transform.Translate(origin[0], origin[1], origin[2])
    # Apply the transform to the plane
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputData(plane.GetOutput())
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    plane = transform_filter.GetOutput()
    # Evaluate the function on the plane
    points = numpy_support.vtk_to_numpy(transform_filter.GetOutput().GetPoints().GetData())[:, :2]
    values = function(points).flatten()
    values = numpy_support.numpy_to_vtk(values)
    values.SetName("ImplicitFunctionValue")
    # Set implicit values to the plane
    plane.GetPointData().SetScalars(values)
    # Create an UnstructuredGrid from the PolyData
    polydata_to_unstructured_grid = vtkAppendFilter()
    polydata_to_unstructured_grid.AddInputData(plane)
    polydata_to_unstructured_grid.Update()
    unstructured_grid = vtkUnstructuredGrid()
    unstructured_grid.ShallowCopy(polydata_to_unstructured_grid.GetOutput())
    # Threshold the plane
    threshold = vtkThreshold()
    threshold.SetInputData(unstructured_grid)
    threshold.ThresholdBetween(lower, upper)
    threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "ImplicitFunctionValue")
    threshold.Update()
    # Obtain the mesh
    geometry_filter = vtkGeometryFilter()
    geometry_filter.SetInputData(threshold.GetOutput())
    geometry_filter.Update()
    mesh = geometry_filter.GetOutput()
    return mesh


def threshold_3d(function, resolution, origin, scale, upper, lower):
    """
    This function uses thresholding techniques to
    discretize a 3D subdomain of a defined 3D implicit
    domain.

    Parameters
    ----------
    function : callable
        This is the function that defines the implicit
        domain.
    resolution : int
        This is the number of points to discretize the
        three nominal dimensions of the domain.
    origin : array_like
        This is the origin of the domain.
    scale : array_like
        This is the scale of the domain.
    upper : float
        This is the upper threshold value.
    lower : float
        This is the lower threshold value.

    Returns
    -------
    mesh : vtkUnstructuredGrid
        This is the discrete mesh of the 3D domain.
        The unstructured grid also retains the implicit
        values used to threshold the domain. These data
        are stored in the point data of the unstructured
        grid as a scalar array named "ImplicitFunctionValue".

    Notes
    -----

    """
    # Build the structured grid for thresholding
    uniform_grid = vtkUniformGrid()
    uniform_grid.SetDimensions(resolution, resolution, resolution)
    # Define the transform to scale the grid
    transform = vtkTransform()
    transform.Scale(scale[0], scale[1], scale[2])
    transform.Translate(origin[0], origin[1], origin[2])
    # Apply the transform to the grid
    transform_filter = vtkTransformFilter()
    transform_filter.SetInputData(uniform_grid)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    uniform_grid = transform_filter.GetOutput()
    # Evaluate the function on the grid
    points = numpy_support.vtk_to_numpy(uniform_grid.GetPoints().GetData())
    values = function(points).flatten()
    values = numpy_support.numpy_to_vtk(values)
    values.SetName("ImplicitFunctionValue")
    # Set implicit values to the grid
    uniform_grid.GetPointData().SetScalars(values)
    # Transform the grid to an unstructured grid
    append_filter = vtkAppendFilter()
    append_filter.AddInputData(uniform_grid)
    append_filter.Update()
    unstructured_grid = vtkUnstructuredGrid()
    unstructured_grid.ShallowCopy(append_filter.GetOutput())
    # Threshold the grid
    threshold = vtkThreshold()
    threshold.SetInputData(unstructured_grid)
    threshold.ThresholdBetween(lower, upper)
    threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "ImplicitFunctionValue")
    threshold.Update()
    # Obtain the mesh
    mesh = threshold.GetOutput()
    return mesh

def marching_squares(function, grid, origin, value=0.0):
    """
    This function uses the marching squares algorithm
    to discretize to determine the boundary of a 2D
    domain.

    Parameters
    ----------
    function : callable
        This is the implicit function that defines the
        domain.
    grid : numpy.ndarray
        This is an array of points that define the grid
        on which the implicit function is evaluated.

    Returns
    -------
    mesh : vtkPolyData
        This is the mesh that represents the 1D boundary
        along the 2D domain.
    """
    dimension_0_spacing = np.diff(grid[0][0, :])[0]
    dimension_1_spacing = np.diff(grid[1][:, 0])[0]
    dimension_0 = grid[0].flatten()
    dimension_1 = grid[1].flatten()
    dimensions = np.vstack((dimension_0, dimension_1)).T
    values = function(dimensions).flatten()
    image_data = vtkImageData()
    image_data.SetDimensions(grid[0].shape[0], grid[0].shape[1], 1)
    image_data.SetSpacing(dimension_0_spacing, dimension_1_spacing, 1)
    image_data.SetOrigin(origin[0], origin[1], 0)
    # Let numpy_support infer array type to avoid referencing VTK_* constants
    vtk_values = numpy_support.numpy_to_vtk(values, deep=True)
    image_data.GetPointData().SetScalars(vtk_values)
    # Create the 2D filter
    marching_squares_filter = vtkMarchingSquares()
    marching_squares_filter.SetInputData(image_data)
    marching_squares_filter.SetValue(0, value)
    # Obtain the mesh
    marching_squares_filter.Update()
    mesh = marching_squares_filter.GetOutput()
    return mesh

#def marching_cubes(function, grid, origin, value=0.0):
#    pass

def contour(function, points, resolution, value=0.0, buffer=1.5):
    n_dim = points.shape[1]
    if n_dim == 2:
        bounds = [np.min(points[:, 0]), np.max(points[:, 0]),
                  np.min(points[:, 1]), np.max(points[:, 1])]
        origin = [(bounds[1]+bounds[0])/2,
                  (bounds[3]+bounds[2])/2,
                  0]
        mesh = pv.Plane(i_resolution=resolution, j_resolution=resolution)
        mesh.points[:, 0] *= (bounds[1] - bounds[0])*buffer
        mesh.points[:, 1] *= (bounds[3] - bounds[2])*buffer
        mesh.points += origin
        mesh.point_data["values"] = function(mesh.points[:, :2])
        boundary = mesh.contour([value])
    elif n_dim == 3:
        bounds = [np.min(points, axis=0), np.max(points, axis=0)]
        bounds_range = np.abs(bounds[1] - bounds[0])*buffer
        x = np.linspace(bounds[0][0] - bounds_range[0], bounds[1][0] + bounds_range[0], resolution)
        y = np.linspace(bounds[0][1] - bounds_range[1], bounds[1][1] + bounds_range[1], resolution)
        z = np.linspace(bounds[0][2] - bounds_range[2], bounds[1][2] + bounds_range[2], resolution)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        strucutrured_grid = pv.StructuredGrid(x, y, z)
        strucutrured_grid.point_data["values"] = function(strucutrured_grid.points)
        boundary = strucutrured_grid.contour([value])
        mesh = strucutrured_grid
        """
        origin = (bounds[1] + bounds[0]) / 2
        ranges = (bounds[1] - bounds[0])*buffer
        spacing = ranges / resolution
        dimensions = [resolution, resolution, resolution]
        #origin = np.array(origin) - np.array(ranges)/2
        mesh = pv.ImageData(spacing=spacing, dimensions=dimensions, origin=origin)
        grid = np.mgrid[origin[0] - ranges[0]/2:origin[0] + ranges[0]/2:resolution*1j,
                        origin[1] - ranges[1]/2:origin[1] + ranges[1]/2:resolution*1j,
                        origin[2] - ranges[2]/2:origin[2] + ranges[2]/2:resolution*1j]
        grid = grid.reshape(3, -1).T
        values = function(grid)
        values = values.reshape(resolution, resolution, resolution)
        mesh.point_data["values"] = values.flatten()
        #boundary = mesh.contour([value], method="flying_edges")
        verts, faces, normals, _ = marching_cubes(values, level=value, spacing=spacing,
                                                  allow_degenerate=False)
        offset = (verts.max(axis=0) + verts.min(axis=0)) / 2
        shift = (origin - offset)
        verts = (verts) + shift
        face_data = np.zeros((faces.shape[0], 4), dtype=int)
        face_data[:, 0] = 3
        face_data[:, 1:] = faces
        boundary = pv.PolyData(verts, face_data)
        boundary.point_data["values"] = function(verts).flatten()
        """
    else:
        boundary = None
        mesh = None
    return boundary, mesh
