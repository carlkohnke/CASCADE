extern "C" __global__ void cext_hybrid_deposit_runtime_stencil_kernel(
    const float* q_weighted_gl,
    const float* lambda_iv_gl,
    const int* node_seg_ids,
    const unsigned char* active_source_mask,
    const float* gl_points_flat,
    const float* lambda_bin_edges,
    int n_bins,
    int grid_n,
    int grid_cells,
    int assignment_mode,
    int n_nodes,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    float* out_mass
) {
    int node_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_idx >= n_nodes) return;
    int seg_idx = node_seg_ids[node_idx];
    if (active_source_mask[seg_idx] == 0) return;
    float q = q_weighted_gl[node_idx];
    float lam = lambda_iv_gl[node_idx];
    if (!isfinite(q) || !isfinite(lam) || q == 0.0f || lam <= 0.0f) return;
    int bin_idx = 0;
    for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
        if (lam > lambda_bin_edges[edge_idx]) bin_idx = edge_idx;
        else break;
    }
    float gx = (gl_points_flat[3*node_idx + 0] - origin_x) / spacing - 0.5f;
    float gy = (gl_points_flat[3*node_idx + 1] - origin_y) / spacing - 0.5f;
    float gz = (gl_points_flat[3*node_idx + 2] - origin_z) / spacing - 0.5f;
    int ix[3], iy[3], iz[3];
    float wx[3], wy[3], wz[3];
    int nx, ny, nz;
    cext_axis_assignment(gx, grid_n, assignment_mode, ix, wx, &nx);
    cext_axis_assignment(gy, grid_n, assignment_mode, iy, wy, &ny);
    cext_axis_assignment(gz, grid_n, assignment_mode, iz, wz, &nz);
    int base = bin_idx * grid_cells;
    for (int ax = 0; ax < nx; ++ax) {
        if (ix[ax] < 0 || ix[ax] >= grid_n || !(wx[ax] != 0.0f) || !isfinite(wx[ax])) continue;
        for (int ay = 0; ay < ny; ++ay) {
            if (iy[ay] < 0 || iy[ay] >= grid_n || !(wy[ay] != 0.0f) || !isfinite(wy[ay])) continue;
            float wxy = wx[ax] * wy[ay];
            for (int az = 0; az < nz; ++az) {
                if (iz[az] < 0 || iz[az] >= grid_n || !(wz[az] != 0.0f) || !isfinite(wz[az])) continue;
                float sw = wxy * wz[az];
                int flat = (ix[ax] * grid_n + iy[ay]) * grid_n + iz[az];
                atomicAdd(&out_mass[base + flat], q * sw);
            }
        }
    }
}
