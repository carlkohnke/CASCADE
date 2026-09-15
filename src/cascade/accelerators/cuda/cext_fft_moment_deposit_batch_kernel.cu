// Deposit a batch of orientation moments into lambda-binned FFT source grids.
// hybrid_deposit.py compiles this kernel for batched GPU deposition.
extern "C" __global__ void cext_fft_moment_deposit_batch_kernel(
    const float* o2_weight_gl,
    const float* node_tx,
    const float* node_ty,
    const float* node_tz,
    const float* lambda_iv_gl,
    const int* node_seg_ids,
    const unsigned char* active_source_mask,
    const int* stencil_flat_idx,
    const float* stencil_weight,
    const float* lambda_bin_edges,
    int moment_start,
    int moment_count,
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
    float w0 = o2_weight_gl[node_idx];
    float lam = lambda_iv_gl[node_idx];
    float tx = node_tx[node_idx];
    float ty = node_ty[node_idx];
    float tz = node_tz[node_idx];
    if (!isfinite(w0) || !isfinite(lam) || !isfinite(tx) || !isfinite(ty) || !isfinite(tz) || w0 == 0.0f || lam <= 0.0f) return;

    float coeffs[7];
    coeffs[0] = 1.0f;
    coeffs[1] = tx * tx;
    coeffs[2] = ty * ty;
    coeffs[3] = tz * tz;
    coeffs[4] = tx * ty;
    coeffs[5] = tx * tz;
    coeffs[6] = ty * tz;

    int bin_idx = 0;
    for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
        if (lam > lambda_bin_edges[edge_idx]) {
            bin_idx = edge_idx;
        } else {
            break;
        }
    }
    int stencil_base = node_idx * stencil_n;
    for (int slot = 0; slot < stencil_n; ++slot) {
        int flat = stencil_flat_idx[stencil_base + slot];
        if (flat < 0) continue;
        float sw = stencil_weight[stencil_base + slot];
        if (!(sw != 0.0f) || !isfinite(sw)) continue;
        for (int local_idx = 0; local_idx < moment_count; ++local_idx) {
            int moment_idx = moment_start + local_idx;
            if (moment_idx < 0 || moment_idx >= 7) continue;
            int out_base = (local_idx * n_bins + bin_idx) * grid_cells;
            atomicAdd(&out_mass[out_base + flat], w0 * coeffs[moment_idx] * sw);
        }
    }
}
