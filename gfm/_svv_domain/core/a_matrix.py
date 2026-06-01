import numpy as np
from .m_matrix import m_matrix
from .n_matrix import n_matrix


def a_matrix(points, rbf_degree=3):
    """
    Construct the full interpolation matrix for Duchon spline interpolation.

    This function builds the full interpolation matrix `A` used in Duchon spline interpolation
    over a set of points defining a (D-1)-dimensional manifold embedded in D-dimensional space.
    The matrix `A` is assembled from the sub-matrices `M` and `N`, which are constructed based
    on the input points and the specified radial basis function (RBF) degree.


    Parameters
    ----------
    points : ndarray of shape (n_points, dim)
        Coordinates of the points defining the manifold. Each row corresponds to a point
        in D-dimensional space.

    rbf_degree : int or float, optional
        Degree of the Duchon interpolant function. Controls the smoothness of the interpolation.
        Default is tri-harmonic kernel rbf_degree=3.

    Returns
    -------
    A : ndarray of shape ((n_points + 1) * (dim + 1), (n_points + 1) * (dim + 1))
        The full interpolation matrix assembled from sub-matrices `M` and `N`.
        This matrix is used to compute the interpolation coefficients for the Duchon spline.

    Notes
    -----
    Duchon splines generalize thin-plate splines to higher dimensions, allowing interpolation
    on manifolds of arbitrary dimension [1]_. The construction of the interpolation matrix involves
    creating two sub-matrices:

    - **M**: A block matrix representing the interaction between points via the RBF.
    - **N**: A matrix incorporating polynomial terms to ensure the interpolation passes through
      the given points.

    The full matrix `A` is assembled as:

    .. math::

        A = \\begin{bmatrix}
                M & N \\\\
                N^T & 0
            \\end{bmatrix}

    where the zero block is of appropriate size to complete the matrix.

    References
    ----------
    .. [1] Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev spaces.
           In *Constructive Theory of Functions of Several Variables* (pp. 85-100). Springer, Berlin, Heidelberg.
           `Springer Link <https://doi.org/10.1007/BFb0086566>`_

    Examples
    --------
    **Example 1: Using 2D Points**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix

        # Define a set of points in 2D space
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Compute the interpolation matrix with the default RBF degree
        A = a_matrix(points)

        print("Shape of the interpolation matrix:", A.shape)
        # Output: Shape of the interpolation matrix: (20, 20)

    **Example 2: Using 3D Points**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix

        # Define a set of 3D points
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        # Compute the interpolation matrix for 3D points
        A = a_matrix(points)

        print("Shape of the interpolation matrix:", A.shape)
        # Output: Shape of the interpolation matrix: (24, 24)

    **Example 3: Changing the RBF Degree**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix

        # Define a set of 2D points
        points = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Compute the interpolation matrix with a higher RBF degree
        A_high_degree = a_matrix(points, rbf_degree=5)

        # Compute the interpolation matrix with a lower RBF degree
        A_low_degree = a_matrix(points, rbf_degree=2)

        print("Shape with higher degree (5):", A_high_degree.shape)
        print("Shape with lower degree (2):", A_low_degree.shape)
        # Output:
        # Shape with higher degree (5): (24, 24)
        # Shape with lower degree (2): (24, 24)

    **Example 4: Practical Interpolation Task**

    Suppose you have a function defined on a set of 2D points, and you want to interpolate this function over the entire plane.

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix
        from scipy.linalg import solve

        # Define a set of 2D points
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])

        # Values of the function at the given points
        values = np.array([0, 1, 1, 0, 0.5])

        # Build the interpolation matrix
        A = a_matrix(points)

        # Right-hand side vector
        b = np.zeros(A.shape[0])
        b[:len(values)*(points.shape[1]+1):points.shape[1]+1] = values

        # Solve for the interpolation coefficients
        coeffs = solve(A, b)

        print("Interpolation coefficients:")
        print(coeffs)

    See Also
    --------
    :func:`svtoolkit.domain.core.h_matrix.h_matrix` : Function to compute the H matrix for bending energy minimization.

    """
    n = points.shape[0]
    d = points.shape[1]
    m_ = m_matrix(points, rbf_degree=rbf_degree)
    n_ = n_matrix(points)
    a_ = np.zeros(((n + 1) * (d + 1), (n + 1) * (d + 1)))
    a_[:n * (d + 1), :n * (d + 1)] = m_
    a_[:n * (d + 1), n * (d + 1):] = n_
    a_[n * (d + 1):, :n * (d + 1)] = n_.T
    return a_
