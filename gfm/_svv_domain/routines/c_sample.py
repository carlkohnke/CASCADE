import numpy as np

def pick_from_tetrahedron(simplicies: np.ndarray, rdx: np.ndarray) -> np.ndarray:
    i = simplicies.shape[0]
    for ii in range(i):
        if rdx[ii, 1, 0] + rdx[ii, 2, 0] > 1:
            rdx[ii, 1, 0] = 1.0 - rdx[ii, 1, 0]
            rdx[ii, 2, 0] = 1.0 - rdx[ii, 2, 0]
        if rdx[ii, 2, 0] + rdx[ii, 3, 0] > 1:
            tmp = rdx[ii, 3, 0]
            rdx[ii, 3, 0] = 1.0 - rdx[ii, 1, 0] - rdx[ii, 2, 0]
            rdx[ii, 2, 0] = 1.0 - tmp
        elif rdx[ii, 1, 0] + rdx[ii, 2, 0] + rdx[ii, 3, 0] > 1:
            tmp = rdx[ii, 3, 0]
            rdx[ii, 3, 0] = rdx[ii, 1, 0] + rdx[ii, 2, 0] + rdx[ii, 3, 0] - 1.0
            rdx[ii, 1, 0] = 1.0 - rdx[ii, 2, 0] - tmp
        rdx[ii, 0, 0] = 1.0 - rdx[ii, 1, 0] - rdx[ii, 2, 0] - rdx[ii, 3, 0]
    points = np.sum(rdx * simplicies, axis=1)
    return points


def pick_from_triangle(simplices: np.ndarray, rdx: np.ndarray) -> np.ndarray:
    i = simplices.shape[0]
    for ii in range(i):
        if rdx[ii, 1, 0] + rdx[ii, 2, 0] > 1:
            rdx[ii, 1, 0] = 1.0 - rdx[ii, 1, 0]
            rdx[ii, 2, 0] = 1.0 - rdx[ii, 2, 0]
        rdx[ii, 0, 0] = 1.0 - rdx[ii, 1, 0] - rdx[ii, 2, 0]
    points = np.sum(rdx * simplices, axis=1)
    return points


def pick_from_line(simplices: np.ndarray, rdx: np.ndarray) -> np.ndarray:
    i = simplices.shape[0]
    for ii in range(i):
        rdx[ii, 0, 0] = 1.0 - rdx[ii, 1, 0]
    points = np.sum(rdx * simplices, axis=1)
    return points
