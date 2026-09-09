import numpy as np
from typing import Tuple, Dict, List, Set

def norm(data: np.ndarray) -> np.ndarray:
    return np.linalg.norm(data, axis=1, keepdims=True)


def argwhere_nonzeros(data: np.ndarray) -> np.ndarray:
    return np.nonzero(data)[0].astype(np.int64)


def argwhere_value_double(data: np.ndarray, value: float) -> np.ndarray:
    return np.where(data == value)[0].astype(np.int64)


def argwhere_value_int(data: np.ndarray, value: int) -> np.ndarray:
    return np.where(data == value)[0].astype(np.int64)


def any_value_double(data: np.ndarray, value: float) -> bool:
    return bool(np.any(data == value))


def any_value_int(data: np.ndarray, value: int) -> bool:
    return bool(np.any(data == value))


def c_dict(data: np.ndarray) -> Dict[int, List[float]]:
    return {i: list(map(float, row)) for i, row in enumerate(data)}


def duplicate_map(unique_inverse: np.ndarray, unique_counts: np.ndarray):
    dup: Dict[int, List[int]] = {}
    dup_set: Set[int] = set()
    for key, cnt in enumerate(unique_counts):
        if cnt <= 1:
            continue
        idxs = np.where(unique_inverse == key)[0].tolist()
        dup[key] = idxs
        dup_set.update(idxs)
    return dup, dup_set


def _allocate_patch(indices: np.ndarray, overlap: float, point_set: Set[int], duplicates_set: Set[int]):
    i = indices.shape[0]
    overlap_indices = int(i * overlap)
    for k in range(1, overlap_indices):
        idx = int(indices[k])
        if idx in point_set:
            if duplicates_set:
                if idx not in duplicates_set:
                    point_set.discard(idx)
            else:
                point_set.discard(idx)
    return point_set


def closest_point(index: int, included: Set[int], points: np.ndarray):
    if not included:
        return index, 0.0
    p = points[index]
    min_distance = 0.0
    closest_index = index
    for ii in included:
        diff = points[ii] - p
        distance = float(np.sqrt(float(np.dot(diff, diff))))
        if distance > min_distance and min_distance == 0.0:
            min_distance = distance
            closest_index = int(ii)
        elif distance == min_distance:
            closest_index = int(ii)
        elif distance > min_distance:
            continue
        else:
            min_distance = distance
            closest_index = int(ii)
    return closest_index, min_distance


def get_angle(index_1: int, index_2: int, normals: np.ndarray) -> float:
    v1 = normals[index_1]
    v2 = normals[index_2]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    ratio = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(ratio)))


def _allocate_angle(idx: int, indices: np.ndarray, points: np.ndarray, normals: np.ndarray, feature_angle: float):
    include_points: Set[int] = set()
    include_indices: Set[int] = set()
    allocated: List[int] = []
    # Add the seed idx
    for ii in range(indices.shape[0]):
        if int(indices[ii]) == int(idx):
            include_points.add(ii)
            include_indices.add(int(indices[ii]))
            allocated.append(int(indices[ii]))
            break
    for ii in range(1, indices.shape[0]):
        closest_idx, closest_distance = closest_point(ii, include_points, points)
        if closest_distance == 0.0:
            continue
        angle = get_angle(ii, closest_idx, normals)
        if angle > feature_angle:
            continue
        include_points.add(ii)
        include_indices.add(int(indices[ii]))
        allocated.append(int(indices[ii]))
    return allocated
