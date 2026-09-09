import numpy as np


def n_matrix(points):
    r"""
    Construct the N matrix used in Duchon spline interpolation.

    This function computes the N matrix for a given set of points defining a (D-1)-dimensional manifold embedded in
    D-dimensional space. The N matrix incorporates polynomial terms to ensure that the interpolation passes through
    the given points and satisfies the necessary smoothness conditions.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        Coordinates of the points defining the manifold. Each row corresponds to a point in D-dimensional space.

    Returns
    -------
    n_ : ndarray of shape ((D+1)*N, D+1)
        The N matrix used in the construction of the full interpolation matrix :math:`A`. This matrix has dimensions
        suited for combining with the M matrix.

    Notes
    -----
    The N matrix is constructed as follows:

    - **First N rows** (associated with function values at the data points):

      .. math::

          N_{i, j} = x_i^{(j)}, \quad \text{for} \quad j = 0, \dots, D-1

          N_{i, D} = 1

      where :math:`x_i^{(j)}` is the :math:`j`-th coordinate of the :math:`i`-th point.

    - **Remaining D × N rows** (associated with derivative constraints):

      .. math::

          N_{N + (i \cdot D) + k, j} = \delta_{j, k}, \quad \text{for} \quad k = 0, \dots, D-1

      where :math:`\delta_{j, k}` is the Kronecker delta function, :math:`i = 0, \dots, N-1`.

    The N matrix is used in assembling the full interpolation matrix :math:`A` as:

    .. math::

        A = \begin{bmatrix}
                M & N \\
                N^\top & 0
            \end{bmatrix}

    where :math:`M` is the matrix computed by the :func:`m_matrix` function.

    Examples
    --------
    **Example 1: Constructing the N Matrix for 2D Points**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.n_matrix import n_matrix

        # Define a set of 2D points
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Compute the N matrix
        N = n_matrix(points)

        print("Shape of N matrix:", N.shape)
        print(N)

    **Example 2: Using N Matrix in Interpolation**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.n_matrix import n_matrix
        from svtoolkit.domain.core.a_matrix import a_matrix

        # Define a set of points in 3D space
        points = np.random.rand(5, 3)

        # Compute the N matrix
        N = n_matrix(points)

        # Compute the full interpolation matrix A
        A = a_matrix(points)

        # Verify dimensions
        print("Shape of N matrix:", N.shape)
        print("Shape of A matrix:", A.shape)

    See Also
    --------
    :func:`svtoolkit.domain.core.a_matrix.a_matrix` : Function to compute the full interpolation matrix A.
    :func:`svtoolkit.domain.core.m_matrix.m_matrix` : Function to compute the M matrix.
    :func:`svtoolkit.domain.core.h_matrix.h_matrix` : Function to compute the H matrix for bending energy minimization.

    """
    n = points.shape[0]
    d = points.shape[1]
    n_ = np.zeros(((d + 1) * n, (d + 1)))
    for i in range(n):
        for j in range(d):
            n_[i, j] = points[i, j]
        n_[i, -1] = 1
    for i in range(n):
        n_[n + i * d:n + i * d + d, :d] = np.eye(d)
    return n_
