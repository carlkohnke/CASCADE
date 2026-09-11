extern "C" __global__ void cext_tissue_greens_kernel(
    const float* points_si,
    const int* nearest_idx,
    const unsigned char* keep_mask,
    const float* gl_points_si,
    const float* segment_vectors,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    float diffusivity_si,
    float window_factor,
    int n_points,
    int keep_k,
    int gl_order,
    float* out
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_points) return;
    if (!keep_mask[row]) {
        out[row] = 0.0f;
        return;
    }

    float px = points_si[row * 3 + 0];
    float py = points_si[row * 3 + 1];
    float pz = points_si[row * 3 + 2];
    float total = 0.0f;
    float cap_max = 0.0f;

    for (int local_i = 0; local_i < keep_k; ++local_i) {
        int seg_i = nearest_idx[row * keep_k + local_i];
        if (seg_i < 0) continue;
        int contributed = 0;
        for (int g = 0; g < gl_order; ++g) {
            int node_idx = seg_i * gl_order + g;
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
                float vx = segment_vectors[seg_i * 3 + 0];
                float vy = segment_vectors[seg_i * 3 + 1];
                float vz = segment_vectors[seg_i * 3 + 2];
                float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                if (vlen > 1.0e-30f) {
                    float tdotr = (vx * dx + vy * dy + vz * dz) / vlen;
                    float mu2 = (tdotr * tdotr) / (r * r);
                    if (mu2 > 1.0f) mu2 = 1.0f;
                    float inv_r = 1.0f / r;
                    float inv_l = 1.0f / lam;
                    float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                              + (1.0f - mu2) * inv_l * inv_l) * kernel;
                    total += o2_weight * pH;
                }
            }
            contributed = 1;
        }
        if (contributed) {
            float cap = seg_cap_gl[seg_i];
            if (cap > cap_max) cap_max = cap;
        }
    }

    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    if (cap_max > 0.0f && total > cap_max) total = cap_max;
    out[row] = total;
}
