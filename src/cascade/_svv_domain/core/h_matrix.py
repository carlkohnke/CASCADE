import numpy as np


def h_matrix(a_, n, d, lam):
    r"""
    Compute the H matrix for Duchon's variational problem.

    This function computes the H matrix, also known as Duchon's matrix, which forms the linear system
    of equations to minimize in the variational problem of thin-plate spline interpolation. The matrix
    :math:`H` is used to represent the bending energy minimization problem and should be positive
    semi-definite.

    Parameters
    ----------
    a_ : ndarray of shape ((n + 1) * (d + 1), (n + 1) * (d + 1))
        The full interpolation matrix :math:`A` for the points defining the (D-1)-dimensional manifold
        embedded in D-dimensional space. It is assembled from sub-matrices M and N.

    n : int
        The number of points defining the (D-1)-dimensional manifold.

    d : int
        The dimension of the space in which the (D-1)-dimensional manifold is embedded.

    lam : float
        The regularization parameter (also known as the thin-plate relaxation parameter).
        It determines the trade-off between fitting the manifold through the control points
        and smoothing the interpolation. A higher value of :math:`\lambda` leads to a smoother
        manifold that may not pass exactly through the control points.

    Returns
    -------
    h_ : ndarray of shape ((d * n), (d * n))
        The H matrix representing the linear system for the bending energy minimization problem.

    j00_ : ndarray
        The top-left block of the inverse of :math:`A`, corresponding to the data-data interactions.

    j01_ : ndarray
        The top-right block of the inverse of :math:`A`, corresponding to the data-polynomial interactions.

    j11_ : ndarray
        The bottom-right block of the inverse of :math:`A`, corresponding to the polynomial-polynomial interactions.

    a_inv : ndarray
        The inverse of the interpolation matrix :math:`A`.

    Notes
    -----
    The function computes the H matrix using the following steps:

    1. Compute the inverse of the interpolation matrix :math:`A`:

       .. math::
           A^{-1}

    2. Partition the inverse matrix into blocks:

       - :math:`J = A^{-1}_{0: n \times (d+1), 0: n \times (d+1)}`
       - :math:`J_{00} = J_{0: n \times d, 0: n \times d}`
       - :math:`J_{01} = J_{0: n \times d, n \times d: n \times (d+1)}`
       - :math:`J_{11} = J_{n \times d: n \times (d+1), n \times d: n \times (d+1)}`

    3. Compute the H matrix based on the regularization parameter :math:`\lambda`:

       - If :math:`\lambda > 0`:

         .. math::
             H = J_{11} - \lambda J_{01}^T \left( I + \lambda J_{00} \right)^{-1} J_{01}

       - If :math:`\lambda = 0`:

         .. math::
             H = J_{11}

    The H matrix is positive semi-definite and is used in the bending energy minimization problem for thin-plate
    splines.

    References
    ----------
    .. [1] Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev spaces.
           In *Constructive Theory of Functions of Several Variables* (pp. 85-100). Springer, Berlin, Heidelberg.
           `Springer Link <https://doi.org/10.1007/BFb0086566>`_

    Examples
    --------
    **Example 1: Computing the H Matrix with Regularization**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix
        from svtoolkit.domain.core.h_matrix import h_matrix

        # Define parameters
        n = 5       # Number of points
        d = 2       # Dimension (e.g., 2D space)
        lam = 0.1   # Regularization parameter

        # Generate sample points
        points = np.random.rand(n, d)

        # Compute the interpolation matrix A
        A = a_matrix(points)

        # Compute the H matrix
        h, j00, j01, j11, a_inv = h_matrix(A, n, d, lam)

        print("H matrix shape:", h.shape)
        # Output: H matrix shape: (10, 10)

    **Example 2: Without Regularization**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix
        from svtoolkit.domain.core.h_matrix import h_matrix

        # Define parameters
        n = 4       # Number of points
        d = 3       # Dimension (e.g., 3D space)
        lam = 0.0   # No regularization

        # Generate sample points
        points = np.random.rand(n, d)

        # Compute the interpolation matrix A
        A = a_matrix(points)

        # Compute the H matrix
        h, j00, j01, j11, a_inv = h_matrix(A, n, d, lam)

        print("H matrix shape:", h.shape)
        # Output: H matrix shape: (12, 12)

    **Example 3: Solving the Bending Energy Minimization Problem**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.a_matrix import a_matrix
        from svtoolkit.domain.core.h_matrix import h_matrix
        from scipy.linalg import eigh

        # Define parameters
        n = 6
        d = 2
        lam = 0.05

        # Generate sample points
        points = np.random.rand(n, d)

        # Compute the interpolation matrix A
        A = a_matrix(points)

        # Compute the H matrix
        h, _, _, _, _ = h_matrix(A, n, d, lam)

        # Solve the eigenvalue problem
        eigenvalues, eigenvectors = eigh(h)

        print("Eigenvalues of H matrix:", eigenvalues)

    See Also
    --------
    :func:`svtoolkit.domain.core.a_matrix.a_matrix` : Function to compute the full interpolation matrix A.
    :func:`svtoolkit.domain.core.m_matrix.m_matrix` : Function to compute the M sub-matrix.
    :func:`svtoolkit.domain.core.n_matrix.n_matrix` : Function to compute the N sub-matrix.

    """
    a_inv = np.linalg.inv(a_)
    j_ = a_inv[:n * (d + 1), :n * (d + 1)]
    j00_ = j_[:n, :n]
    j01_ = j_[:n, n:].copy()
    j11_ = j_[n:, n:]
    if lam > 0:
        inv = np.linalg.inv(np.eye(j00_.shape[0]) + lam * j00_)
        h_ = j11_ - (lam * j01_.T) @ inv @ j01_
    else:
        h_ = j11_
    return h_, j00_, j01_, j11_, a_inv
