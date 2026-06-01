import numpy as np
from typing import Tuple
from itertools import combinations
from time import perf_counter
import scipy.spatial as spatial
import tqdm
from .c_allocate import (
    norm, argwhere_nonzeros,
    argwhere_value_double,
    any_value_double,
    duplicate_map,
    _allocate_patch,
    _allocate_angle
)


# [TODO] the allocator needs to be accelerated to handle large point clouds
def allocate(*args: Tuple[np.ndarray, ...], min_patch_size: int = 10, max_patch_size: int = 20,
             overlap: float = 0.2, feature_angle: float = 30) -> list:
    """
    Create a list of patches from a set of points and normals, if provided, from a point cloud.

    Parameters
    ----------
        points : np.ndarray
            A numpy array of shape (n, d) containing the points of the point cloud.
        normals : np.ndarray
            A numpy array of shape (n, d) containing the normals of the point cloud.
        min_patch_size : int
            The minimum number of points in a patch.
        max_patch_size : int
            The maximum number of points in a patch.
        overlap : float
            The maximum percentage of overlap between patches as a value between 0 and 1.
        feature_angle : float
            The maximum angle allowed between point-wise normal vectors
    Returns
    -------
        patches : list
            A list of tuples containing the points and normals of each patch.
    """
    if len(args) == 0:
        print("Error: No data provided.")
        return [None, None]
    elif len(args) == 1:
        points = args[0]
        normals = None
    else:
        points = args[0]
        normals = args[1]
        magnitudes = norm(normals)
        if any_value_double(magnitudes.flatten(), 0.0):
            e = argwhere_value_double(magnitudes.flatten(), 0.0)
            print("Error: Normals with zero magnitude found at indices:\n{}.".format(e))
            print("Deleting points with zero magnitude normals...")
            print("Points:\n{}".format(points[e, :]))
            print("Normals:\n{}".format(normals[e, :]))
            indices = argwhere_nonzeros(magnitudes.flatten())
            points = points[indices, :]
            normals = normals[indices, :]
            magnitudes = norm(normals)
        normals = normals / magnitudes
    unique_points, unique_inverse, unique_counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    max_duplicates = np.max(unique_counts)
    duplicates, duplicate_set = duplicate_map(unique_inverse, unique_counts)
    has_duplicates = len(duplicates) > 0
    if normals is None:
        if has_duplicates:
            print("Warning: Duplicate points found with no normals provided.")
            print("Removing duplicate points...")
            print("Done.")
    else:
        if has_duplicates:
            remove_duplicates = []
            for key in duplicates.keys():
                repeated_indices = duplicates[key]
                repeated_normals = normals[repeated_indices, :]
                comb = combinations(list(range(repeated_normals.shape[0])), 2)
                for i, j in comb:
                    dot = np.dot(repeated_normals[i, :], repeated_normals[j, :])
                    dot = np.clip(dot, -1, 1)
                    angle = np.arccos(dot) * (180 / np.pi)
                    if not np.isclose(angle, 0):
                        if angle < feature_angle:
                            feature_angle = angle
                    else:
                        print("Error: Duplicate points with identical normals found.")
                        print('Points:\n{} \nRepeated Normals:\n{}\nAngle:{}\nCombination:{}'.format(points[repeated_indices, :], repeated_normals,angle,tuple([i,j])))
                        print("Removing duplicate point and normal {}...".format(repeated_indices[j]))
                        remove_duplicates.append(repeated_indices[j])
                        duplicate_set.remove(repeated_indices[j])
                        duplicates[key].remove(repeated_indices[j])
            if len(remove_duplicates) > 0:
                idx = np.arange(points.shape[0]).tolist()
                for i in remove_duplicates:
                    idx.remove(i)
                points = points[idx, :]
                normals = normals[idx, :]
    kdtree = spatial.cKDTree(points)
    overlap = np.clip(overlap, 0, 1)
    max_patch_size = np.min([max_patch_size, points.shape[0]])
    min_patch_size = np.min([min_patch_size, max_patch_size//2])
    patch_points = []
    patch_normals = []
    point_list = np.arange(points.shape[0]).tolist()
    point_set = set(point_list)
    remaining_points = []
    progress_bar = tqdm.tqdm(total=len(point_set), desc='Allocating patches', unit='point', leave=False)
    while len(point_set) > 0:
        progress_start = len(point_set)
        point_idx = point_set.pop()
        _, indices = kdtree.query(points[point_idx, :], k=max_patch_size)
        if normals is not None:
            dots = np.dot(normals[point_idx, :], normals[indices, :].T)
            dots = np.clip(dots, -1, 1)
            angles = np.arccos(dots) * (180 / np.pi)
            indices = np.array(_allocate_angle(point_idx, indices, points[indices, :], normals[indices, :],
                                               feature_angle)).astype(np.int64)
            _, unique_indices = np.unique(points[indices, :], axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)
            indices = indices[unique_indices]
        if len(indices) < min_patch_size:
            remaining_points.append(point_idx)
            continue
        else:
            patch_points.append(points[indices, :])
            if normals is not None:
                patch_normals.append(normals[indices, :])
            else:
                patch_normals.append(None)
            point_set = _allocate_patch(indices, overlap, point_set, duplicate_set)
        progress_end = len(point_set)
        progress_bar.update(progress_start - progress_end)
    progress_bar.close()
    progress_bar = tqdm.tqdm(total=len(remaining_points), desc='Allocating remaining patches', unit='point', leave=False)
    while len(remaining_points) > 0:
        progress_start = len(remaining_points)
        point_idx = remaining_points.pop()
        _, indices = kdtree.query(points[point_idx, :], k=max_patch_size)
        if normals is not None:
            dots = np.dot(normals[point_idx, :], normals[indices, :].T)
            dots = np.clip(dots, -1, 1)
            angles = np.arccos(dots) * (180 / np.pi)
            if np.sum(angles < 180) >= 3:
                indices = np.array(_allocate_angle(point_idx, indices, points[indices, :], normals[indices, :],
                                                   feature_angle)).astype(np.int64)
                _, unique_indices = np.unique(points[indices, :], axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                indices = indices[unique_indices]
            else:
                _, unique_indices = np.unique(points[indices, :], axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                indices = indices[unique_indices]
                indices = indices[:3]
        patch_points.append(points[indices, :])
        if normals is not None:
            patch_normals.append(normals[indices, :])
        else:
            patch_normals.append(None)
        progress_end = len(remaining_points)
        progress_bar.update(progress_start - progress_end)
    return list(zip(patch_points, patch_normals))
