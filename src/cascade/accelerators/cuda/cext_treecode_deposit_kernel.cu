extern "C" __global__ void cext_treecode_deposit_kernel(
    const int* point_leaf_ids,
    const float* point_pos,
    const float* point_lambda,
    const float* point_q,
    const unsigned char* point_active,
    const float* node_center,
    const float* bin_edges,
    int n_bins,
    int n_nodes,
    int n_points,
    float* total_mass,
    float* total_dipole,
    float* active_mass,
    float* active_dipole,
    int* active_count
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;
    int leaf = point_leaf_ids[idx];
    float lam = point_lambda[idx];
    float q = point_q[idx];
    if (leaf < 0 || leaf >= n_nodes || !(q != 0.0f) || !isfinite(q) || !isfinite(lam) || lam <= 0.0f) return;
    int bin_idx = 0;
    for (int bi = 0; bi < n_bins - 1; ++bi) {
        if (lam > bin_edges[bi + 1]) bin_idx = bi + 1;
    }
    float rx = point_pos[idx * 3 + 0] - node_center[leaf * 3 + 0];
    float ry = point_pos[idx * 3 + 1] - node_center[leaf * 3 + 1];
    float rz = point_pos[idx * 3 + 2] - node_center[leaf * 3 + 2];
    int base = bin_idx * n_nodes + leaf;
    atomicAdd(total_mass + base, q);
    atomicAdd(total_dipole + base * 3 + 0, q * rx);
    atomicAdd(total_dipole + base * 3 + 1, q * ry);
    atomicAdd(total_dipole + base * 3 + 2, q * rz);
    if (point_active[idx] != 0) {
        atomicAdd(active_mass + base, q);
        atomicAdd(active_dipole + base * 3 + 0, q * rx);
        atomicAdd(active_dipole + base * 3 + 1, q * ry);
        atomicAdd(active_dipole + base * 3 + 2, q * rz);
        atomicAdd(active_count + leaf, 1);
    }
}
