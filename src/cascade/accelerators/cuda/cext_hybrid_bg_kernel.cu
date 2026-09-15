// Sample the FFT far/background field at vessel quadrature nodes.
// hybrid_deposit.py compiles this kernel as the global half of hybrid Cext.
extern "C" __global__ void cext_hybrid_bg_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* phi_grids,
    const float* mass_grids,
    const float* lambda_bins,
    int n_bins,
    int grid_n,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    float near_radius_si,
    int gl_order,
    int target_n,
    float diffusivity_si,
    float* out_bg
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_n * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];

    float gx = (tx - origin_x) / spacing - 0.5f;
    float gy = (ty - origin_y) / spacing - 0.5f;
    float gz = (tz - origin_z) / spacing - 0.5f;
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

    int rad_cells = (int)ceilf(near_radius_si / spacing);
    float total_bg = 0.0f;
    float near_bg = 0.0f;
    for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
        int base = bin_idx * grid_n * grid_n * grid_n;
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;
        int idx000 = base + ((i0 * grid_n + j0) * grid_n + k0);
        int idx001 = base + ((i0 * grid_n + j0) * grid_n + k1);
        int idx010 = base + ((i0 * grid_n + j1) * grid_n + k0);
        int idx011 = base + ((i0 * grid_n + j1) * grid_n + k1);
        int idx100 = base + ((i1 * grid_n + j0) * grid_n + k0);
        int idx101 = base + ((i1 * grid_n + j0) * grid_n + k1);
        int idx110 = base + ((i1 * grid_n + j1) * grid_n + k0);
        int idx111 = base + ((i1 * grid_n + j1) * grid_n + k1);
        float c00 = phi_grids[idx000] * (1.0f - fx) + phi_grids[idx100] * fx;
        float c01 = phi_grids[idx001] * (1.0f - fx) + phi_grids[idx101] * fx;
        float c10 = phi_grids[idx010] * (1.0f - fx) + phi_grids[idx110] * fx;
        float c11 = phi_grids[idx011] * (1.0f - fx) + phi_grids[idx111] * fx;
        float c0 = c00 * (1.0f - fy) + c10 * fy;
        float c1 = c01 * (1.0f - fy) + c11 * fy;
        total_bg += c0 * (1.0f - fz) + c1 * fz;

        float lambda_bin = lambda_bins[bin_idx];
        if (lambda_bin < 1.0e-30f) lambda_bin = 1.0e-30f;
        int ilo = i0 - rad_cells;
        int jlo = j0 - rad_cells;
        int klo = k0 - rad_cells;
        int ihi = i0 + rad_cells + 1;
        int jhi = j0 + rad_cells + 1;
        int khi = k0 + rad_cells + 1;
        if (ilo < 0) ilo = 0;
        if (jlo < 0) jlo = 0;
        if (klo < 0) klo = 0;
        if (ihi >= grid_n) ihi = grid_n - 1;
        if (jhi >= grid_n) jhi = grid_n - 1;
        if (khi >= grid_n) khi = grid_n - 1;
        for (int ix = ilo; ix <= ihi; ++ix) {
            float cx = origin_x + (ix + 0.5f) * spacing;
            float dx = cx - tx;
            for (int iy = jlo; iy <= jhi; ++iy) {
                float cy = origin_y + (iy + 0.5f) * spacing;
                float dy = cy - ty;
                for (int iz = klo; iz <= khi; ++iz) {
                    float cz = origin_z + (iz + 0.5f) * spacing;
                    float dz = cz - tz;
                    float r = sqrtf(dx * dx + dy * dy + dz * dz);
                    if (r > near_radius_si || r < 1.0e-30f) continue;
                    float mass = mass_grids[base + ((ix * grid_n + iy) * grid_n + iz)];
                    if (!(mass != 0.0f)) continue;
                    near_bg += mass * expf(-r / lambda_bin) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                }
            }
        }
    }
    float total = total_bg - near_bg;
    if (!(total == total) || !isfinite(total)) total = 0.0f;
    out_bg[target_idx * gl_order + target_node] = total;
}
