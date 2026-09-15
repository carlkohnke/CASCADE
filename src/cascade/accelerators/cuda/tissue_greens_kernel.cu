// Evaluate tissue oxygen over prefiltered nearest-vessel candidates.
// concentration/tissue/gpu.py compiles this standard sparse GPU path.
extern "C" __device__ float interp_lut(float x, const float* xs, const float* ys, int n) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __global__ void tissue_greens_kernel(
    const float* points_si,
    const float* starts_si,
    const float* segment_vectors,
    const float* seg_len,
    const float* radii_si,
    const int* nearest_idx,
    const float* proj_raw,
    const float* d_center,
    const unsigned char* keep_mask,
    const float* cin_pos,
    const float* alpha_edge,
    const float* flow_sign,
    float diffusivity_si,
    float km,
    float vmax,
    float window_factor,
    float lam_ref,
    const float* gl_nodes,
    const float* gl_weights,
    int gl_order,
    const float* xs_lut,
    const float* k0_lut,
    const float* ratio_lut,
    int lut_n,
    int n_points,
    int keep_k,
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
    float cap_max = 1.0e-6f;
    float total = 0.0f;

    for (int local_i = 0; local_i < keep_k; ++local_i) {
        int flat = row * keep_k + local_i;
        int seg_i = nearest_idx[flat];
        if (seg_i < 0) continue;
        float L = seg_len[seg_i];
        if (L <= 0.0f) continue;

        float proj = proj_raw[flat];
        if (flow_sign[seg_i] < 0.0f) proj = 1.0f - proj;
        proj = fminf(1.0f, fmaxf(0.0f, proj));
        float s_star = proj * L;

        float Cc_star = cin_pos[seg_i] * expf(-alpha_edge[seg_i] * s_star);
        float denom_gate = km + fmaxf(Cc_star, 1.0e-12f);
        denom_gate = fmaxf(denom_gate, 1.0e-30f);
        float lam_gate = sqrtf(diffusivity_si / fmaxf(vmax / denom_gate, 1.0e-30f));
        if (d_center[flat] > window_factor * lam_gate) continue;
        if (Cc_star > cap_max) cap_max = Cc_star;

        float halfW = window_factor * lam_ref;
        float s0 = fmaxf(0.0f, s_star - halfW);
        float s1 = fminf(L, s_star + halfW);
        if (s1 <= s0 + 1.0e-15f) continue;

        float mid = 0.5f * (s0 + s1);
        float half = 0.5f * (s1 - s0);
        float total_local = 0.0f;

        float sx0 = starts_si[seg_i * 3 + 0];
        float sx1 = starts_si[seg_i * 3 + 1];
        float sx2 = starts_si[seg_i * 3 + 2];
        float vx0 = segment_vectors[seg_i * 3 + 0];
        float vx1 = segment_vectors[seg_i * 3 + 1];
        float vx2 = segment_vectors[seg_i * 3 + 2];
        float radius = radii_si[seg_i];
        float cin_seg = cin_pos[seg_i];
        float alpha = alpha_edge[seg_i];

        for (int g = 0; g < gl_order; ++g) {
            float s = mid + half * gl_nodes[g];
            float t = s / L;
            float xs0 = sx0 + t * vx0;
            float xs1 = sx1 + t * vx1;
            float xs2 = sx2 + t * vx2;
            float r0 = px - xs0;
            float r1 = py - xs1;
            float r2 = pz - xs2;
            float r = sqrtf(r0 * r0 + r1 * r1 + r2 * r2);
            if (r <= 1.0e-12f) continue;

            float cc_s = cin_seg * expf(-alpha * s);
            float denom_loc = km + fmaxf(cc_s, 1.0e-12f);
            denom_loc = fmaxf(denom_loc, 1.0e-30f);
            float lam_loc = sqrtf(diffusivity_si / fmaxf(vmax / denom_loc, 1.0e-30f));
            float phi_loc = fmaxf(radius / fmaxf(lam_loc, 1.0e-30f), 1.0e-12f);
            float r_over_lam = r / fmaxf(lam_loc, 1.0e-30f);
            float k0_num = interp_lut(r_over_lam, xs_lut, k0_lut, lut_n);
            float k0_den = fmaxf(interp_lut(phi_loc, xs_lut, k0_lut, lut_n), 1.0e-30f);
            float Ci_R = cc_s * (k0_num / k0_den);

            float denom_corr = km + fminf(3.0f * Ci_R, cc_s);
            denom_corr = fmaxf(denom_corr, 1.0e-30f);
            float lam_corr = sqrtf(diffusivity_si / fmaxf(vmax / denom_corr, 1.0e-30f));
            float phi = fmaxf(radius / fmaxf(lam_corr, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut(phi, xs_lut, ratio_lut, lut_n);
            float wall_factor = (diffusivity_si / fmaxf(lam_corr, 1.0e-30f)) * ratio;
            float q_s = (2.0f * 3.14159265358979323846f * radius) * wall_factor * cc_s;
            float kernel = expf(-r / fmaxf(lam_corr, 1.0e-30f)) / (4.0f * 3.14159265358979323846f * r);
            float integrand = q_s * (kernel / diffusivity_si);
            total_local += gl_weights[g] * integrand;
        }
        total += half * total_local;
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out[row] = fminf(total, cap_max);
}
