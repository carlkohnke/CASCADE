// Combine stencil pairs sharing the same displacement before reading Green grids.
extern "C" __device__ int cext_axis_correlation(
    const int* target, const float* tw, int nt,
    const int* source, const float* sw, int ns,
    int grid_n, int* delta, float* weight
) {
    int low = target[0] - source[ns - 1];
    int count = nt + ns - 1;
    for (int i = 0; i < count; ++i) {
        int d = low + i;
        delta[i] = d < 0 ? d + grid_n : d;
        weight[i] = 0.0f;
    }
    for (int a = 0; a < nt; ++a) {
        if (target[a] < 0 || target[a] >= grid_n || !isfinite(tw[a])) continue;
        for (int b = 0; b < ns; ++b) {
            if (source[b] < 0 || source[b] >= grid_n || !isfinite(sw[b])) continue;
            weight[target[a] - source[b] - low] += tw[a] * sw[b];
        }
    }
    return count;
}

extern "C" __global__ void cext_fft_discrete_self_correlated_kernel(
    const int* target_seg_ids, const float* gl_points_si,
    const float* source_weight_gl, const float* segment_vectors,
    const float* lambda_iv_gl, const unsigned char* active_source_mask,
    const float* lambda_bin_edges, const float* green_grids,
    int moment_idx, int n_bins, int grid_n, int grid_cells,
    int gl_order, int target_count, float origin_x, float origin_y,
    float origin_z, float spacing, int assignment_mode,
    int source_assignment_mode, float* out_self
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    if (flat >= target_count * gl_order) return;
    int target_seg = target_seg_ids[flat / gl_order];
    if (!active_source_mask[target_seg]) return;
    int target_node = flat % gl_order;
    float coeff = 1.0f;
    if (moment_idx > 0) {
        float vx = segment_vectors[3 * target_seg];
        float vy = segment_vectors[3 * target_seg + 1];
        float vz = segment_vectors[3 * target_seg + 2];
        float len = sqrtf(vx * vx + vy * vy + vz * vz);
        if (!(len > 1e-30f) || !isfinite(len)) return;
        float ux = vx / len, uy = vy / len, uz = vz / len;
        if (moment_idx == 1) coeff = ux * ux;
        else if (moment_idx == 2) coeff = uy * uy;
        else if (moment_idx == 3) coeff = uz * uz;
        else if (moment_idx == 4) coeff = ux * uy;
        else if (moment_idx == 5) coeff = ux * uz;
        else if (moment_idx == 6) coeff = uy * uz;
        else return;
    }
    if (coeff == 0.0f || !isfinite(coeff)) return;
    float origin[3] = {origin_x, origin_y, origin_z};
    int target[3][3], nt[3];
    float tw[3][3];
    for (int axis = 0; axis < 3; ++axis) {
        float g = (gl_points_si[(target_seg * gl_order + target_node) * 3 + axis] - origin[axis]) / spacing - 0.5f;
        if (assignment_mode == 0) g = fminf(grid_n - 1.0f, fmaxf(0.0f, g));
        cext_axis_assignment(g, grid_n, assignment_mode, target[axis], tw[axis], &nt[axis]);
        if (assignment_mode == 0 && target[axis][0] >= grid_n - 1) {
            target[axis][0] = grid_n - 2;
            target[axis][1] = grid_n - 1;
            tw[axis][0] = 0.0f; tw[axis][1] = 1.0f;
        }
    }
    float acc = 0.0f;
    for (int node = 0; node < gl_order; ++node) {
        int source_flat = target_seg * gl_order + node;
        float weight = source_weight_gl[source_flat] * coeff;
        float lam = lambda_iv_gl[source_flat];
        if (weight == 0.0f || !isfinite(weight) || !(lam > 0.0f) || !isfinite(lam)) continue;
        int bin = 0;
        for (int b = 1; b < n_bins; ++b) {
            if (lam > lambda_bin_edges[b]) bin = b; else break;
        }
        int delta[3][5], count[3];
        float cw[3][5];
        for (int axis = 0; axis < 3; ++axis) {
            int source[3], ns;
            float sw[3];
            float g = (gl_points_si[source_flat * 3 + axis] - origin[axis]) / spacing - 0.5f;
            cext_axis_assignment(g, grid_n, source_assignment_mode, source, sw, &ns);
            count[axis] = cext_axis_correlation(target[axis], tw[axis], nt[axis], source, sw, ns, grid_n, delta[axis], cw[axis]);
        }
        float response = 0.0f;
        for (int x = 0; x < count[0]; ++x) {
            if (cw[0][x] == 0.0f) continue;
            for (int y = 0; y < count[1]; ++y) {
                float xy = cw[0][x] * cw[1][y];
                if (xy == 0.0f) continue;
                int row = (delta[0][x] * grid_n + delta[1][y]) * grid_n;
                for (int z = 0; z < count[2]; ++z) {
                    if (cw[2][z] == 0.0f) continue;
                    response += xy * cw[2][z] * green_grids[bin * grid_cells + row + delta[2][z]];
                }
            }
        }
        acc += weight * response;
    }
    if (isfinite(acc)) out_self[flat] += acc;
}
