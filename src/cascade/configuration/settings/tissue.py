"""Default tissue oxygen sampling, Green's-function, and GPU settings."""

from __future__ import annotations

import os
import numpy as np

DEFAULTS = {
    # Default number of tissue sample points for random tissue oxygen evaluation.
    "DISTANCE_SAMPLE_COUNT": 1000000,
    # Number of tissue points processed together in some CPU distance/oxygen kernels.
    "DISTANCE_CHUNK_SIZE": 256,
    # If true, compute average distance from sample points to the nearest vessel centerline.
    "COMPUTE_AVG_DISTANCE_TO_CHANNEL": False,
    # If true, use a KD-tree nearest-neighbor search instead of a dense all-vessel distance matrix for large trees.
    "DISTANCE_USE_KDTREE_FOR_LARGE_TREES": True,
    # Largest vessel count where brute-force distance-to-vessel calculations are still allowed.
    "DISTANCE_BRUTE_FORCE_MAX_SEGMENTS": 20000,
    # Minimum number of candidate vessels considered per tissue point in KD-tree distance searches.
    "DISTANCE_KDTREE_MIN_CANDIDATES": 64,
    # Candidate multiplier for KD-tree searches; larger values reduce missed nearest vessels but cost more time.
    "DISTANCE_KDTREE_CANDIDATE_MULT": 8,
    # Hard cap on candidate vessels per tissue point in KD-tree distance searches.
    "DISTANCE_KDTREE_MAX_CANDIDATES": 4096,
    # Maximum number of nearby vessels kept per tissue point for tissue oxygen calculations.
    "NEAREST_TISSUE_VESSELS": 250,
    # Cutoff radius measured in oxygen decay lengths; farther vessels are ignored for tissue oxygen.
    "WINDOW_FACTOR": 6,
    # Tissue quadrature after a Cext solve: independent resamples to GL_ORDER;
    # legacy_cext reuses the Cext source nodes for oracle compatibility only.
    "CEXT_TISSUE_QUADRATURE_MODE": "independent",
    # Candidate multiplier used while building tissue nearest-vessel caches.
    "TISSUE_KDTREE_CANDIDATE_MULT": 2,
    # Worker count for CPU tissue calculations that can run in parallel.
    "TISSUE_PARALLEL_WORKERS": max(os.cpu_count() or 1, 1),
    # If true, use numba-compiled tissue kernels when available.
    "TISSUE_USE_NUMBA": True,
    # If true, process very large tissue point sets in streaming chunks to reduce peak memory.
    "TISSUE_STREAMING_ENABLED": False,
    # Minimum tissue point count before streaming mode is considered.
    "TISSUE_STREAMING_MIN_POINTS": 100000,
    # Target number of point-vessel candidate slots per streaming chunk; controls memory per chunk.
    "TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS": 500000,
    # Smallest allowed point count for a streaming chunk.
    "TISSUE_STREAMING_MIN_CHUNK_POINTS": 256,
    # Largest allowed point count for a streaming chunk.
    "TISSUE_STREAMING_MAX_CHUNK_POINTS": 2048,
    # If true, remove vessels outside the local oxygen window during streaming to reduce work.
    "TISSUE_STREAMING_PRUNE_BY_WINDOW": True,
    # Print streaming progress every this many chunks.
    "TISSUE_STREAMING_LOG_EVERY_CHUNKS": 25,
    # Worker count for streaming tissue chunks.
    "TISSUE_STREAMING_CHUNK_WORKERS": max((os.cpu_count() or 1) - 2, 1),
    # Numba thread count assigned inside each streaming worker process.
    "TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER": 1,
    # Number of tissue points per chunk when building CPU tissue geometry caches.
    "TISSUE_CACHE_CHUNK_SIZE": 512,
    # Worker count used for KD-tree queries while building tissue caches.
    "TISSUE_KDTREE_WORKERS": max((os.cpu_count() or 1) - 2, 1),
    # Tissue oxygen backend selection: "gpu", "cpu", or "auto".
    "TISSUE_ACCEL_MODE": "gpu",
    # Number of tissue points processed per GPU kernel launch.
    "TISSUE_GPU_CHUNK_POINTS": 8192,
    # Lower bound for GPU chunk size when automatic chunk sizing is used.
    "TISSUE_GPU_MIN_CHUNK_POINTS": 512,
    # Number of tissue points to cross-check against CPU results for GPU validation; 0 disables validation.
    "TISSUE_GPU_VALIDATE_POINTS": 0,
    # If true, use a cell-list spatial grid to find Cext source vessels near tissue points on GPU.
    "TISSUE_CEXT_CELL_LIST_ENABLE": True,
    # Desired average number of vessel sources per spatial cell in the GPU Cext tissue cell list.
    "TISSUE_CEXT_CELL_TARGET_OCCUPANCY": 4,
    # Minimum grid resolution per dimension for the GPU Cext tissue cell list.
    "TISSUE_CEXT_CELL_MIN_GRID": 16,
    # Maximum grid resolution per dimension for the GPU Cext tissue cell list.
    "TISSUE_CEXT_CELL_MAX_GRID": 256,
    # Maximum number of neighboring cell layers searched around each tissue point.
    "TISSUE_CEXT_CELL_MAX_RAD_CELLS": 8,
    # Minimum segment length in SI units; shorter segments are treated as degenerate for tissue kernels.
    "TISSUE_MIN_SEGMENT_LENGTH_SI": 1.0e-12,
    # If true, force tissue caches to store floats as float64 instead of the memory-saving default.
    "TISSUE_CACHE_FORCE_FLOAT64": False,
    # Floating-point dtype used inside tissue cache arrays.
    "TISSUE_CACHE_FLOAT_DTYPE": np.float32,
    # Integer dtype used inside tissue cache index arrays.
    "TISSUE_CACHE_INDEX_DTYPE": np.int32,
}

ALIASES = {
    "distance_sample_count": "DISTANCE_SAMPLE_COUNT",
    "distance_chunk_size": "DISTANCE_CHUNK_SIZE",
    "compute_avg_distance_to_channel": "COMPUTE_AVG_DISTANCE_TO_CHANNEL",
    "nearest_tissue_vessels": "NEAREST_TISSUE_VESSELS",
    "nearest_vessels": "NEAREST_TISSUE_VESSELS",
    "window_factor": "WINDOW_FACTOR",
    "cext_quadrature_mode": "CEXT_TISSUE_QUADRATURE_MODE",
    "cext_tissue_quadrature_mode": "CEXT_TISSUE_QUADRATURE_MODE",
    "accel": "TISSUE_ACCEL_MODE",
    "accel_mode": "TISSUE_ACCEL_MODE",
    "gpu_chunk_points": "TISSUE_GPU_CHUNK_POINTS",
    "gpu_validate_points": "TISSUE_GPU_VALIDATE_POINTS",
    "cache_force_float64": "TISSUE_CACHE_FORCE_FLOAT64",
    "cache_float_dtype": "TISSUE_CACHE_FLOAT_DTYPE",
    "cache_index_dtype": "TISSUE_CACHE_INDEX_DTYPE",
}

PREFIXES = ("TISSUE_", "DISTANCE_")

DEPRECATED = {}
