// Deposit active vessel source strengths into lambda-binned FFT grids.
// hybrid_geometry.py uses precomputed interpolation indices and weights here.
extern "C" __global__ void cext_hybrid_deposit_kernel(
    const float* q_weighted_gl,
    const float* lambda_iv_gl,
    const int* node_seg_ids,
    const unsigned char* active_source_mask,
    const int* stencil_flat_idx,
    const float* stencil_weight,
    const float* lambda_bin_edges,
    int n_bins,
    int grid_cells,
    int stencil_n,
    int n_nodes,
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
        if (lam > lambda_bin_edges[edge_idx]) {
            bin_idx = edge_idx;
        } else {
            break;
        }
    }
    int base = bin_idx * grid_cells;
    int stencil_base = node_idx * stencil_n;
    for (int slot = 0; slot < stencil_n; ++slot) {
        int flat = stencil_flat_idx[stencil_base + slot];
        if (flat < 0) continue;
        float weight = stencil_weight[stencil_base + slot];
        if (!(weight != 0.0f) || !isfinite(weight)) continue;
        atomicAdd(&out_mass[base + flat], q * weight);
    }
}
