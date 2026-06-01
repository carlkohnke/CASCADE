import numpy as np
from scipy.spatial.distance import cdist


def m00(points, rbf_degree=3):
    r"""
    Compute the M\ :sub:`00` sub-matrix of the M matrix using a radial basis function (RBF).

    The M\ :sub:`00` sub-matrix is an :math:`N \times N` matrix computed based on the distances
    between points raised to the specified RBF degree. This sub-matrix represents
    the interactions between the points in the context of thin-plate spline interpolation.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        The coordinates of the points defining the (D-1)-dimensional manifold embedded
        in D-dimensional space.

    rbf_degree : float, optional
        The degree of the radial basis function. Controls the smoothness of the interpolation.
        Default is 3.

    Returns
    -------
    m00_ : ndarray of shape (N, N)
        The M\ :sub:`00` sub-matrix of the M matrix.

    Notes
    -----
    The M\ :sub:`00` sub-matrix is computed as:

    .. math::

        M_{00}[i, j] = \| \mathbf{x}_i - \mathbf{x}_j \|^{p}

    where:

    - :math:`\mathbf{x}_i` and :math:`\mathbf{x}_j` are the coordinates of points :math:`i` and :math:`j`.
    - :math:`p` is the RBF degree (`rbf_degree`).

    Examples
    --------
    **Example:** Compute the M\ :sub:`00` sub-matrix for a set of 2D points.

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.m_matrix import m00

        points = np.array([[0, 0], [1, 0], [0, 1]])
        m00_submatrix = m00(points, rbf_degree=3)
        print(m00_submatrix)

    """
    return cdist(points, points) ** rbf_degree


def m01(points, rbf_degree=3):
    r"""
    Compute the M\ :sub:`01` sub-matrix of the M matrix using a radial basis function (RBF).

    The M\ :sub:`01` sub-matrix is an :math:`N \times D \times N` matrix representing the interaction
    between the points and their derivatives.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        The coordinates of the points defining the (D-1)-dimensional manifold embedded
        in D-dimensional space.

    rbf_degree : float, optional
        The degree of the radial basis function. Controls the smoothness of the interpolation.
        Default is 3.

    Returns
    -------
    m01_ : ndarray of shape (N, D*N)
        The M\ :sub:`01` sub-matrix of the M matrix.

    Notes
    -----
    The M\ :sub:`01` sub-matrix is computed as:

    .. math::

        M_{01}[i, jD + k] = -p (\mathbf{x}_i^{(k)} - \mathbf{x}_j^{(k)}) \| \mathbf{x}_i - \mathbf{x}_j \|^{p - 2}

    where:

    - :math:`\mathbf{x}_i^{(k)}` is the :math:`k`-th coordinate of point :math:`i`.
    - :math:`p` is the RBF degree (`rbf_degree`).
    - :math:`D` is the dimension of the space.

    Examples
    --------
    **Example:** Compute the M\ :sub:`01` sub-matrix for a set of 2D points.

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.m_matrix import m01

        points = np.array([[0, 0], [1, 0], [0, 1]])
        m01_submatrix = m01(points, rbf_degree=3)
        print(m01_submatrix)

    """
    n = points.shape[0]
    d = points.shape[1]
    m01_ = np.zeros((n, n * d))
    for i in range(n):
        for j in range(n):
            diff = points[i, :] - points[j, :]
            diff_sum = np.sum(diff ** 2)
            if diff_sum == 0 and (rbf_degree/2 - 1) < 0:
                norm1 = np.inf
            else:
                norm1 = diff_sum**(rbf_degree / 2 - 1)
            for k in range(d):
                if i == j:
                    m01_[i, j * d + k] = 0
                else:
                    m01_[i, j * d + k] = -rbf_degree * (points[i, k] - points[j, k]) * norm1
    return m01_


def m11(points, rbf_degree=3):
    r"""
    Compute the M\ :sub:`11` sub-matrix of the M matrix using a radial basis function (RBF).

    The M\ :sub:`11` sub-matrix is an :math:`D N \times D N` matrix representing the interaction
    between the derivatives of the points.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        The coordinates of the points defining the (D-1)-dimensional manifold embedded
        in D-dimensional space.

    rbf_degree : float, optional
        The degree of the radial basis function. Controls the smoothness of the interpolation.
        Default is 3.

    Returns
    -------
    m11_ : ndarray of shape (D*N, D*N)
        The M\ :sub:`11` sub-matrix of the M matrix.

    Notes
    -----
    The M\ :sub:`11` sub-matrix is computed as:

    .. math::

        M_{11}[iD + k, jD + l] = -2 p (p - 2) (\mathbf{x}_i^{(k)} - \mathbf{x}_j^{(k)}) (\mathbf{x}_i^{(l)} -
                                 \mathbf{x}_j^{(l)}) \| \mathbf{x}_i - \mathbf{x}_j \|^{p - 4} -
                                 p \delta_{kl} \| \mathbf{x}_i - \mathbf{x}_j \|^{p - 2}

    where:

    - :math:`\mathbf{x}_i^{(k)}` is the :math:`k`-th coordinate of point :math:`i`.
    - :math:`p` is the RBF degree (`rbf_degree`).
    - :math:`\delta_{kl}` is the Kronecker delta function.
    - :math:`D` is the dimension of the space.

    Examples
    --------
    **Example:** Compute the M\ :sub:`11` sub-matrix for a set of 2D points.

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.m_matrix import m11

        points = np.array([[0, 0], [1, 0], [0, 1]])
        m11_submatrix = m11(points, rbf_degree=3)
        print(m11_submatrix)

    """
    n = points.shape[0]
    d = points.shape[1]
    m11_ = np.zeros((n * d, n * d))
    for i in range(n):
        for j in range(i, n):
            diff = points[i, :] - points[j, :]
            diff_sum = np.sum(diff ** 2)
            if i == j:
                if (rbf_degree / 2 - 1) < 0:
                    norm1 = np.inf
                else:
                    norm1 = diff_sum ** (rbf_degree / 2 - 1)
                if (rbf_degree / 2 - 2) < 0:
                    norm2 = np.inf
                else:
                    norm2 = diff_sum ** (rbf_degree / 2 - 2)
            else:
                if diff_sum == 0 and (rbf_degree / 2 - 1) < 0:
                    norm1 = np.inf
                else:
                    norm1 = diff_sum ** (rbf_degree / 2 - 1)
                if diff_sum == 0 and (rbf_degree / 2 - 2) < 0:
                    norm2 = np.inf
                else:
                    norm2 = diff_sum ** (rbf_degree / 2 - 2)
            for k in range(d):
                for l in range(d):
                    if i == j:
                        m11_[j * d + l, i * d + k] = 0
                    elif k == l:
                        m11_[j * d + l, i * d + k] = -2 * (rbf_degree / 2 - 1) * rbf_degree * diff[k] ** 2 * norm2 - \
                                                     rbf_degree * norm1
                    else:
                        m11_[j * d + l, i * d + k] = -2 * (rbf_degree / 2 - 1) * rbf_degree * diff[k] * diff[l] * norm2
    return m11_


def m_matrix(points, rbf_degree=3):
    r"""
    Construct the full M matrix for the (D-1)-dimensional manifold embedded in D-dimensional space.

    The M matrix is constructed by combining the M\ :sub:`00`, M\ :sub:`01`, and M\ :sub:`11` sub-matrices, representing
    interactions between points and their derivatives in the context of thin-plate spline interpolation.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        The coordinates of the points defining the (D-1)-dimensional manifold embedded
        in D-dimensional space.

    rbf_degree : float, optional
        The degree of the radial basis function. Controls the smoothness of the interpolation.
        Default is 3.

    Returns
    -------
    m_ : ndarray of shape (N*(D+1), N*(D+1))
        The full M matrix.

    Notes
    -----
    The full M matrix is assembled as:

    .. math::

        M = \begin{bmatrix}
                M_{00} & M_{01} \\
                M_{01}^T & M_{11} + M_{11}^T
            \end{bmatrix}

    where:

    - :math:`M_{00}` is the sub-matrix computed by :func:`m00`.
    - :math:`M_{01}` is the sub-matrix computed by :func:`m01`.
    - :math:`M_{11}` is the sub-matrix computed by :func:`m11`.

    Examples
    --------
    **Example:** Construct the full M matrix for a set of 2D points.

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.core.m_matrix import m_matrix

        points = np.array([[0, 0], [1, 0], [0, 1]])
        m_full = m_matrix(points, rbf_degree=3)
        print(m_full)

    See Also
    --------
    m00 : Compute the M\ :sub:`00` sub-matrix.
    m01 : Compute the M\ :sub:`01` sub-matrix.
    m11 : Compute the M\ :sub:`11` sub-matrix.
    a_matrix : Compute the full interpolation matrix A.

    """
    n = points.shape[0]
    d = points.shape[1]
    m_ = np.zeros((n * (d + 1), n * (d + 1)))
    m00_ = m00(points, rbf_degree)
    m01_ = m01(points, rbf_degree)
    m11_ = m11(points, rbf_degree)
    m_[:n, :n] = m00_
    m_[:n, n:] = m01_
    m_[n:, :n] = m01_.T
    m_[n:, n:] = m11_
    m_[n:, n:] += m11_.T
    return m_
