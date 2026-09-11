extern "C" __global__ void cext_fft_discrete_self_runtime_stencil_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* source_weight_gl,
    const float* lambda_iv_gl,
    const unsigned char* active_source_mask,
    const float* lambda_bin_edges,
    const float* green_grids,
    int n_bins,
    int grid_n,
    int grid_cells,
    int gl_order,
    int target_count,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    int assignment_mode,
    int source_assignment_mode,
    float* out_self
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_count * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    if (active_source_mask[target_seg] == 0) return;

    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float gx = (tx - origin_x) / spacing - 0.5f;
    float gy = (ty - origin_y) / spacing - 0.5f;
    float gz = (tz - origin_z) / spacing - 0.5f;
    int ti[3], tj[3], tk[3];
    float wi[3], wj[3], wk[3];
    int target_stencil_n = 2;
    if (assignment_mode == 1) {
        int dummy;
        cext_axis_assignment(gx, grid_n, 1, ti, wi, &target_stencil_n);
        cext_axis_assignment(gy, grid_n, 1, tj, wj, &dummy);
        cext_axis_assignment(gz, grid_n, 1, tk, wk, &dummy);
    } else {
        int i0 = (int)floorf(gx);
        int j0 = (int)floorf(gy);
        int k0 = (int)floorf(gz);
        float fx = gx - i0;
        float fy = gy - j0;
        float fz = gz - k0;
        if (i0 < 0) { i0 = 0; fx = 0.0f; }
        if (j0 < 0) { j0 = 0; fy = 0.0f; }
        if (k0 < 0) { k0 = 0; fz = 0.0f; }
        if (i0 >= grid_n - 1) { i0 = grid_n - 2; fx = 1.0f; }
        if (j0 >= grid_n - 1) { j0 = grid_n - 2; fy = 1.0f; }
        if (k0 >= grid_n - 1) { k0 = grid_n - 2; fz = 1.0f; }
        target_stencil_n = 2;
        ti[0] = i0; ti[1] = i0 + 1; ti[2] = -1;
        tj[0] = j0; tj[1] = j0 + 1; tj[2] = -1;
        tk[0] = k0; tk[1] = k0 + 1; tk[2] = -1;
        wi[0] = 1.0f - fx; wi[1] = fx; wi[2] = 0.0f;
        wj[0] = 1.0f - fy; wj[1] = fy; wj[2] = 0.0f;
        wk[0] = 1.0f - fz; wk[1] = fz; wk[2] = 0.0f;
    }

    float acc = 0.0f;
    for (int source_node = 0; source_node < gl_order; ++source_node) {
        int source_flat = target_seg * gl_order + source_node;
        float source_weight = source_weight_gl[source_flat];
        float lam = lambda_iv_gl[source_flat];
        if (!isfinite(source_weight) || source_weight == 0.0f || !isfinite(lam) || lam <= 0.0f) continue;
        int bin_idx = 0;
        for (int edge_idx = 1; edge_idx < n_bins; ++edge_idx) {
            if (lam > lambda_bin_edges[edge_idx]) bin_idx = edge_idx;
            else break;
        }
        int green_base = bin_idx * grid_cells;
        float sgx = (gl_points_si[3*source_flat + 0] - origin_x) / spacing - 0.5f;
        float sgy = (gl_points_si[3*source_flat + 1] - origin_y) / spacing - 0.5f;
        float sgz = (gl_points_si[3*source_flat + 2] - origin_z) / spacing - 0.5f;
        int si[3], sj[3], sk[3];
        float swi[3], swj[3], swk[3];
        int snx, sny, snz;
        cext_axis_assignment(sgx, grid_n, source_assignment_mode, si, swi, &snx);
        cext_axis_assignment(sgy, grid_n, source_assignment_mode, sj, swj, &sny);
        cext_axis_assignment(sgz, grid_n, source_assignment_mode, sk, swk, &snz);
        for (int sx_i = 0; sx_i < snx; ++sx_i) {
            if (si[sx_i] < 0 || si[sx_i] >= grid_n || !(swi[sx_i] != 0.0f) || !isfinite(swi[sx_i])) continue;
            for (int sy_i = 0; sy_i < sny; ++sy_i) {
                if (sj[sy_i] < 0 || sj[sy_i] >= grid_n || !(swj[sy_i] != 0.0f) || !isfinite(swj[sy_i])) continue;
                float swxy = swi[sx_i] * swj[sy_i];
                for (int sz_i = 0; sz_i < snz; ++sz_i) {
                    if (sk[sz_i] < 0 || sk[sz_i] >= grid_n || !(swk[sz_i] != 0.0f) || !isfinite(swk[sz_i])) continue;
                    int sx = si[sx_i];
                    int sy = sj[sy_i];
                    int sz = sk[sz_i];
                    float deposited = source_weight * swxy * swk[sz_i];
                    for (int ax = 0; ax < target_stencil_n; ++ax) {
                        if (ti[ax] < 0 || ti[ax] >= grid_n || !(wi[ax] != 0.0f) || !isfinite(wi[ax])) continue;
                        int dx = ti[ax] - sx;
                        if (dx < 0) dx += grid_n;
                        float wx = wi[ax];
                        for (int ay = 0; ay < target_stencil_n; ++ay) {
                            if (tj[ay] < 0 || tj[ay] >= grid_n || !(wj[ay] != 0.0f) || !isfinite(wj[ay])) continue;
                            int dy = tj[ay] - sy;
                            if (dy < 0) dy += grid_n;
                            float wxy = wx * wj[ay];
                            for (int az = 0; az < target_stencil_n; ++az) {
                                if (tk[az] < 0 || tk[az] >= grid_n || !(wk[az] != 0.0f) || !isfinite(wk[az])) continue;
                                int dz = tk[az] - sz;
                                if (dz < 0) dz += grid_n;
                                float target_weight = wxy * wk[az];
                                int green_idx = green_base + ((dx * grid_n + dy) * grid_n + dz);
                                acc += deposited * target_weight * green_grids[green_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    if (isfinite(acc)) out_self[flat] += acc;
}
