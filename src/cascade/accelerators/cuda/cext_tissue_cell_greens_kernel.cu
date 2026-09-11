extern "C" __global__ void cext_tissue_cell_greens_kernel(
    const float* points_si,
    const int* cell_ptr,
    const int* cell_node_ids,
    const float* cell_source_reach,
    const float* cell_segment_reach,
    const float* starts_si,
    const float* segment_vectors,
    const float* seg_len_sq,
    const float* seg_len,
    const float* radii_si,
    const float* gl_points_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    float diffusivity_si,
    float window_factor,
    float origin_x,
    float origin_y,
    float origin_z,
    float spacing,
    int grid_n,
    int rad_cells,
    int n_points,
    int gl_order,
    unsigned char* keep_mask,
    float* out
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_points) return;

    float px = points_si[row * 3 + 0];
    float py = points_si[row * 3 + 1];
    float pz = points_si[row * 3 + 2];

    int cx = (int)floorf((px - origin_x) / spacing);
    int cy = (int)floorf((py - origin_y) / spacing);
    int cz = (int)floorf((pz - origin_z) / spacing);
    int lo_x = cx - rad_cells;
    int lo_y = cy - rad_cells;
    int lo_z = cz - rad_cells;
    int hi_x = cx + rad_cells;
    int hi_y = cy + rad_cells;
    int hi_z = cz + rad_cells;
    if (lo_x < 0) lo_x = 0;
    if (lo_y < 0) lo_y = 0;
    if (lo_z < 0) lo_z = 0;
    if (hi_x >= grid_n) hi_x = grid_n - 1;
    if (hi_y >= grid_n) hi_y = grid_n - 1;
    if (hi_z >= grid_n) hi_z = grid_n - 1;

    float total = 0.0f;
    float cap_max = 0.0f;
    int inside_any = 0;

    if (lo_x <= hi_x && lo_y <= hi_y && lo_z <= hi_z) {
        for (int ix = lo_x; ix <= hi_x; ++ix) {
            for (int iy = lo_y; iy <= hi_y; ++iy) {
                for (int iz = lo_z; iz <= hi_z; ++iz) {
                    int cell_flat = (ix * grid_n + iy) * grid_n + iz;
                    int row_start = cell_ptr[cell_flat];
                    int row_end = cell_ptr[cell_flat + 1];
                    if (row_start >= row_end) continue;

                    float cell_min_x = origin_x + ((float)ix) * spacing;
                    float cell_min_y = origin_y + ((float)iy) * spacing;
                    float cell_min_z = origin_z + ((float)iz) * spacing;
                    float cell_max_x = cell_min_x + spacing;
                    float cell_max_y = cell_min_y + spacing;
                    float cell_max_z = cell_min_z + spacing;
                    float ddx = 0.0f;
                    float ddy = 0.0f;
                    float ddz = 0.0f;
                    if (px < cell_min_x) ddx = cell_min_x - px;
                    else if (px > cell_max_x) ddx = px - cell_max_x;
                    if (py < cell_min_y) ddy = cell_min_y - py;
                    else if (py > cell_max_y) ddy = py - cell_max_y;
                    if (pz < cell_min_z) ddz = cell_min_z - pz;
                    else if (pz > cell_max_z) ddz = pz - cell_max_z;
                    float cell_dist = sqrtf(ddx * ddx + ddy * ddy + ddz * ddz);
                    int do_source = cell_dist <= cell_source_reach[cell_flat];
                    int do_inside = (!inside_any) && (cell_dist <= cell_segment_reach[cell_flat]);
                    if (!do_source && !do_inside) continue;

                    for (int pos = row_start; pos < row_end; ++pos) {
                        int node_idx = cell_node_ids[pos];
                        if (node_idx < 0) continue;
                        int seg_i = node_idx / gl_order;

                        if (do_inside && !inside_any) {
                            float sx0 = starts_si[seg_i * 3 + 0];
                            float sx1 = starts_si[seg_i * 3 + 1];
                            float sx2 = starts_si[seg_i * 3 + 2];
                            float vx0 = segment_vectors[seg_i * 3 + 0];
                            float vx1 = segment_vectors[seg_i * 3 + 1];
                            float vx2 = segment_vectors[seg_i * 3 + 2];
                            float len2 = fmaxf(seg_len_sq[seg_i], 1.0e-30f);
                            float wx0 = px - sx0;
                            float wx1 = py - sx1;
                            float wx2 = pz - sx2;
                            float proj = (wx0 * vx0 + wx1 * vx1 + wx2 * vx2) / len2;
                            proj = fminf(1.0f, fmaxf(0.0f, proj));
                            float cx0 = sx0 + proj * vx0;
                            float cx1 = sx1 + proj * vx1;
                            float cx2 = sx2 + proj * vx2;
                            float dxs = px - cx0;
                            float dys = py - cx1;
                            float dzs = pz - cx2;
                            float dseg = sqrtf(dxs * dxs + dys * dys + dzs * dzs);
                            float rlim = fminf(radii_si[seg_i], seg_len[seg_i]);
                            if (dseg <= rlim) inside_any = 1;
                        }
                        if (!do_source) continue;

                        float lam = lambda_iv_gl[node_idx];
                        if (!(lam > 0.0f) || !isfinite(lam)) continue;
                        float sx = gl_points_si[node_idx * 3 + 0];
                        float sy = gl_points_si[node_idx * 3 + 1];
                        float sz = gl_points_si[node_idx * 3 + 2];
                        float dx = px - sx;
                        float dy = py - sy;
                        float dz = pz - sz;
                        float r = sqrtf(dx * dx + dy * dy + dz * dz);
                        if (r <= 1.0e-12f || r > window_factor * lam) continue;
                        float q = q_weighted_gl[node_idx];
                        float o2_weight = mono2_weight_gl[node_idx] + dipole2_weight_gl[node_idx];
                        if (!isfinite(q) || !isfinite(o2_weight) || (q == 0.0f && o2_weight == 0.0f)) continue;
                        float kernel = expf(-r / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                        total += q * kernel;
                        if (o2_weight != 0.0f) {
                            float vx0 = segment_vectors[seg_i * 3 + 0];
                            float vx1 = segment_vectors[seg_i * 3 + 1];
                            float vx2 = segment_vectors[seg_i * 3 + 2];
                            float vlen = sqrtf(vx0 * vx0 + vx1 * vx1 + vx2 * vx2);
                            if (vlen > 1.0e-30f) {
                                float tdotr = (vx0 * dx + vx1 * dy + vx2 * dz) / vlen;
                                float mu2 = (tdotr * tdotr) / (r * r);
                                if (mu2 > 1.0f) mu2 = 1.0f;
                                float inv_r = 1.0f / r;
                                float inv_l = 1.0f / lam;
                                float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                                          + (1.0f - mu2) * inv_l * inv_l) * kernel;
                                total += o2_weight * pH;
                            }
                        }
                        float cap = seg_cap_gl[seg_i];
                        if (cap > cap_max) cap_max = cap;
                    }
                }
            }
        }
    }

    if (inside_any) {
        keep_mask[row] = 0;
        out[row] = 0.0f;
        return;
    }
    keep_mask[row] = 1;
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    if (cap_max > 0.0f && total > cap_max) total = cap_max;
    out[row] = total;
}
