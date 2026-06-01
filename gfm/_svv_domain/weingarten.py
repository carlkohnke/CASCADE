import numpy as np
#import numba as nb
from scipy.spatial import cKDTree
from scipy import spatial
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import open3d as o3d
import sklearn.neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.mixture import GaussianMixture


def bases(points):
    n = points.shape[0]
    d = points.shape[1]
    k = int(d*np.round(np.log(n)))
    kdt = spatial.KDTree(points)
    dist, idx = kdt.query(points, k=k+1)
    for i in range(n):
        local_points = points[idx[i,:],:]
        local_points = local_points - np.ones((k+1,1))@local_points[0,:].reshape(1,-1)
        local_points = local_points[1:,:]
        local_points = local_points/np.linalg.norm(local_points,axis=1).reshape(-1,1)
        u, s, vh = np.linalg.svd(local_points)


def generate_bases(points, quiet=True):
    n_points = points.shape[0]
    dim = points.shape[1]
    int_dim = dim - 1
    tangent_basis = np.zeros((dim, int_dim, n_points))
    normal_basis = np.zeros((dim, dim - int_dim, n_points))
    K = int(min(20, dim * np.log10(n_points))) # can have problems with low numbers of points
    u, u_index, u_counts = np.unique(points,axis=1,return_counts=True,return_index=True)
    KDT = spatial.KDTree(u)
    dist, idx = KDT.query(u, k=K + 1)
    count = 0
    if quiet:
        for i in range(u.shape[0]):
            tmp = points[idx[i, :], :]
            # Account for duplicate points for multiple normal estimation
            # and feature edge detection
            # if u_counts[i] > 1:
            # ensure that the count of the points is greater than the
            # number of neighbors to avoid singular matrix solutions
            tmp = tmp - np.ones((K + 1, 1)) @ tmp[0, :].reshape(1, -1)
            tmp = tmp[1:, :]
            U, S, H = np.linalg.svd(tmp)
            H = H.T
            diag_idx = S.argsort()[::-1]
            sort_diag = np.sort(S)[::-1]
            H = H[:, diag_idx]
            tangent_basis[:, :, i] = H[:, :int_dim]
            normal_basis[:, :, i] = H[:, dim - 1].reshape(-1, 1)
    else:
        for i in tqdm(range(n_points), desc='Generating bases       '):
            tmp = points[idx[i, :], :]
            tmp = tmp - np.ones((K + 1, 1)) @ tmp[0, :].reshape(1, -1)
            tmp = tmp[1:, :]
            U, S, H = np.linalg.svd(tmp)
            H = H.T
            diag_idx = S.argsort()[::-1]
            sort_diag = np.sort(S)[::-1]
            H = H[:, diag_idx]
            tangent_basis[:, :, i] = H[:, :int_dim]
            normal_basis[:, :, i] = H[:, dim - 1].reshape(-1, 1)
    return tangent_basis, normal_basis, idx, KDT


def estimate_weingarten_map(points, quiet=True):
    n_points = points.shape[0]
    dim = points.shape[1]
    tangent_basis, normal_basis, idx, KDT = generate_bases(points, quiet=quiet)
    tangent_dim = tangent_basis.shape[1]
    normal_dim = normal_basis.shape[1]
    weingarten_map = np.zeros((tangent_dim, tangent_dim,
                               normal_dim, n_points))
    if quiet:
        for i in range(n_points):
            for j in range(normal_dim):
                tmp_point = points[i, :]
                tmp_tangent_basis = tangent_basis[:, :, i]
                tmp_normal_basis = normal_basis[:, j, i]
                tmp_neighborhood = idx[i, :]
                tmp_neighborhood_size = len(tmp_neighborhood)
                tmp_local_normals = normal_basis[:, :, tmp_neighborhood]
                tmp_normal_extension = np.zeros((tmp_neighborhood_size, dim))
                for k in range(tmp_neighborhood_size):
                    projection = tmp_local_normals[:, :, k] @ \
                                 tmp_local_normals[:, :, k].T @ \
                                 tmp_normal_basis.reshape(-1, 1)
                    tmp_normal_extension[k, :] = projection.T
                tmp_diff_normal = np.zeros((tmp_neighborhood_size - 1, dim))
                tmp_diff_position = np.zeros((tmp_neighborhood_size - 1, dim))
                for k in range(tmp_neighborhood_size - 1):
                    tmp_diff_normal[k, :] = tmp_normal_extension[k + 1, :] - tmp_normal_basis.reshape(1, -1)
                    tmp_diff_position[k, :] = points[tmp_neighborhood[k + 1], :] - tmp_point
                tmp_normal_projection = tmp_diff_normal @ tmp_tangent_basis
                tmp_position_projection = tmp_diff_position @ tmp_tangent_basis
                A = -(np.linalg.inv(tmp_position_projection.T @ tmp_position_projection)) @ \
                    tmp_position_projection.T @ tmp_normal_projection
                weingarten_map[:, :, j, i] = (1 / 2) * (A + A.T)
    else:
        for i in tqdm(range(n_points), desc='Building Weingarten Map'):
            for j in range(normal_dim):
                tmp_point = points[i, :]
                tmp_tangent_basis = tangent_basis[:, :, i]
                tmp_normal_basis = normal_basis[:, j, i]
                tmp_neighborhood = idx[i, :]
                tmp_neighborhood_size = len(tmp_neighborhood)
                tmp_local_normals = normal_basis[:, :, tmp_neighborhood]
                tmp_normal_extension = np.zeros((tmp_neighborhood_size, dim))
                for k in range(tmp_neighborhood_size):
                    projection = tmp_local_normals[:, :, k] @ \
                                 tmp_local_normals[:, :, k].T @ \
                                 tmp_normal_basis.reshape(-1, 1)
                    tmp_normal_extension[k, :] = projection.T
                tmp_diff_normal = np.zeros((tmp_neighborhood_size - 1, dim))
                tmp_diff_position = np.zeros((tmp_neighborhood_size - 1, dim))
                for k in range(tmp_neighborhood_size - 1):
                    tmp_diff_normal[k, :] = tmp_normal_extension[k + 1, :] - \
                                            tmp_normal_basis.reshape(1, -1)
                    tmp_diff_position[k, :] = points[tmp_neighborhood[k + 1], :] - \
                                              tmp_point
                tmp_normal_projection = tmp_diff_normal @ tmp_tangent_basis
                tmp_position_projection = tmp_diff_position @ tmp_tangent_basis
                A = -(np.linalg.inv(tmp_position_projection.T @ \
                                    tmp_position_projection)) @ \
                    tmp_position_projection.T @ \
                    tmp_normal_projection
                weingarten_map[:, :, j, i] = (1 / 2) * (A + A.T)
    return weingarten_map, idx, KDT


def estimate_gaussian_curvature(points, quiet=True):
    n_points = points.shape[0]
    guassian_curvature = np.zeros((n_points, 1))
    w_map, idx, KDT = estimate_weingarten_map(points, quiet=quiet)
    for i in range(n_points):
        guassian_curvature[i] = np.linalg.det(w_map[:, :, 0, i])
    return guassian_curvature, idx, KDT


def show_gaussian_curvature(points):
    guassian_curvature, idx, KDT = estimate_gaussian_curvature(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    coolwarm = cm.get_cmap('coolwarm', 1000)
    gc_min = min(guassian_curvature)
    gc_max = max(guassian_curvature)
    colors = coolwarm(guassian_curvature)
    cax = ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], cmap=coolwarm, c=guassian_curvature)
    cbar = fig.colorbar(cax, ax=ax, ticks=[gc_min, gc_max])
    return fig, ax


def estimate_normals(points,radius=0.1, nn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn))
    pcd.orient_normals_consistent_tangent_plane(nn)
    normals = np.asarray(pcd.normals)
    return points, normals


def fit_surface(points,degree=3,nn=20):
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # fit a polynomial surface to the points

    poly = PolynomialFeatures(degree=degree)
    curvatures = []
    for i in range(points.shape[0]):
        neighbors = points[indices[i,:],:]
        centered_points = neighbors - points[i,:]

        X = poly.fit_transform(centered_points)