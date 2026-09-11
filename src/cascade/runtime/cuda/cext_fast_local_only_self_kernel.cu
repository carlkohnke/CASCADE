extern "C" __global__ void cext_fast_local_only_self_kernel(
    const float* gl_points,
    const float* segment_vectors,
    const float* radii,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    const unsigned char* active_source_mask,
    const int* cell_ptr,
    const int* cell_node_ids,
    const float* cell_max_lambda,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const float spacing,
    const int grid_n,
    const int local_rad_cells,
    const float near_radius_si,
    const float lambda_window,
    const int gl_order,
    const int nseg,
    const float diffusivity,
    float* c_ext_total,
    float* c_ext_cap
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nseg * gl_order;
    if (idx >= total) return;
    int target_seg = idx / gl_order;
    int target_gl = idx - target_seg * gl_order;
    int pidx = target_seg * gl_order + target_gl;
    float px = gl_points[3*pidx + 0];
    float py = gl_points[3*pidx + 1];
    float pz = gl_points[3*pidx + 2];
    int ix = (int)floorf((px - origin_x) / spacing);
    int iy = (int)floorf((py - origin_y) / spacing);
    int iz = (int)floorf((pz - origin_z) / spacing);
    if (ix < 0) ix = 0; if (ix >= grid_n) ix = grid_n - 1;
    if (iy < 0) iy = 0; if (iy >= grid_n) iy = grid_n - 1;
    if (iz < 0) iz = 0; if (iz >= grid_n) iz = grid_n - 1;
    float rcut2 = near_radius_si * near_radius_si;
    float target_radius = radii[target_seg];
    float total_val = 0.0f;
    float cap_val = 0.0f;
    const float four_pi = 12.566370614359172f;
    for (int dz = -local_rad_cells; dz <= local_rad_cells; ++dz) {
        int cz = iz + dz;
        if (cz < 0 || cz >= grid_n) continue;
        for (int dy = -local_rad_cells; dy <= local_rad_cells; ++dy) {
            int cy = iy + dy;
            if (cy < 0 || cy >= grid_n) continue;
            for (int dx = -local_rad_cells; dx <= local_rad_cells; ++dx) {
                int cx = ix + dx;
                if (cx < 0 || cx >= grid_n) continue;
                int cell = (cx * grid_n + cy) * grid_n + cz;
                float cell_lam = cell_max_lambda[cell];
                if (!(cell_lam > 0.0f) || !isfinite(cell_lam)) continue;
                float cell_cutoff = lambda_window * cell_lam;
                float center_x = origin_x + ((float)cx + 0.5f) * spacing;
                float center_y = origin_y + ((float)cy + 0.5f) * spacing;
                float center_z = origin_z + ((float)cz + 0.5f) * spacing;
                float ddx = fabsf(px - center_x) - 0.5f * spacing;
                float ddy = fabsf(py - center_y) - 0.5f * spacing;
                float ddz = fabsf(pz - center_z) - 0.5f * spacing;
                if (ddx < 0.0f) ddx = 0.0f;
                if (ddy < 0.0f) ddy = 0.0f;
                if (ddz < 0.0f) ddz = 0.0f;
                if ((ddx*ddx + ddy*ddy + ddz*ddz) > cell_cutoff * cell_cutoff) continue;
                int start = cell_ptr[cell];
                int end = cell_ptr[cell + 1];
                for (int ptr = start; ptr < end; ++ptr) {
                    int source_node = cell_node_ids[ptr];
                    int source_seg = source_node / gl_order;
                    if (source_seg == target_seg) continue;
                    if (active_source_mask[source_seg] == 0) continue;
                    int source_gl = source_node - source_seg * gl_order;
                    float lam = lambda_iv_gl[source_seg * gl_order + source_gl];
                    if (!(lam > 0.0f) || !isfinite(lam)) continue;
                    float source_cutoff = lambda_window * lam;
                    float sx = gl_points[3*source_node + 0];
                    float sy = gl_points[3*source_node + 1];
                    float sz = gl_points[3*source_node + 2];
                    float rx = px - sx;
                    float ry = py - sy;
                    float rz = pz - sz;
                    float dist2 = rx*rx + ry*ry + rz*rz;
                    float radius_sum = target_radius + radii[source_seg];
                    float rr2 = dist2 + radius_sum * radius_sum;
                    if (rr2 > rcut2 || rr2 > source_cutoff * source_cutoff) continue;
                    float rr = sqrtf(rr2);
                    if (!(rr > 0.0f) || !isfinite(rr)) continue;
                    float qw = q_weighted_gl[source_seg * gl_order + source_gl];
                    float kernel = expf(-rr / lam) / (four_pi * diffusivity * rr);
                    total_val += qw * kernel;
                    float o2_weight = mono2_weight_gl[source_seg * gl_order + source_gl]
                                      + dipole2_weight_gl[source_seg * gl_order + source_gl];
                    if (o2_weight != 0.0f) {
                        float vx = segment_vectors[source_seg * 3 + 0];
                        float vy = segment_vectors[source_seg * 3 + 1];
                        float vz = segment_vectors[source_seg * 3 + 2];
                        float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                        if (vlen > 1.0e-30f) {
                            float tdotr = (vx * rx + vy * ry + vz * rz) / vlen;
                            float mu2 = (tdotr * tdotr) / (rr * rr);
                            if (mu2 > 1.0f) mu2 = 1.0f;
                            float inv_r = 1.0f / rr;
                            float inv_l = 1.0f / lam;
                            float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                                      + (1.0f - mu2) * inv_l * inv_l) * kernel;
                            total_val += o2_weight * pH;
                        }
                    }
                    float src_cap = seg_cap_gl[source_seg];
                    if (src_cap > cap_val) cap_val = src_cap;
                }
            }
        }
    }
    if (!isfinite(total_val) || total_val < 0.0f) total_val = 0.0f;
    c_ext_total[idx] = total_val;
    c_ext_cap[idx] = cap_val;
}
