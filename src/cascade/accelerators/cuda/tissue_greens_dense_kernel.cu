// Exact fused tissue Green's-function path for dense all-segment evaluation.
// concentration/tissue/gpu.py compiles this source when every segment is kept.
extern "C" __device__ float interp_lut_dense(
    float x,
    const float* xs,
    const float* ys,
    int n
) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
    }
    float x0 = xs[lo];
    float x1 = xs[lo + 1];
    float y0 = ys[lo];
    float y1 = ys[lo + 1];
    if (x1 == x0) return y0;
    return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
}

/*
 * Exact fused path for networks where every valid segment is retained by the
 * public nearest-vessel contract.  It deliberately combines centerline
 * projection, lumen masking, window rejection, and Green's-function
 * quadrature so no (point x segment) candidate matrices are materialized.
 */
extern "C" __global__ void tissue_greens_dense_kernel(
    const float* points_si,
    const float* starts_si,
    const float* segment_vectors,
    const float* seg_len_sq,
    const float* seg_len,
    const float* radii_si,
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
    int n_segments,
    unsigned char* keep_mask,
    float* out
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_points) return;

    float px = points_si[row * 3 + 0];
    float py = points_si[row * 3 + 1];
    float pz = points_si[row * 3 + 2];
    float cap_max = 1.0e-6f;
    float total = 0.0f;
    int inside_any = 0;

    for (int seg_i = 0; seg_i < n_segments; ++seg_i) {
        float len2 = seg_len_sq[seg_i];
        float L = seg_len[seg_i];
        if (!(len2 > 0.0f) || !(L > 0.0f)) continue;

        float sx0 = starts_si[seg_i * 3 + 0];
        float sx1 = starts_si[seg_i * 3 + 1];
        float sx2 = starts_si[seg_i * 3 + 2];
        float vx0 = segment_vectors[seg_i * 3 + 0];
        float vx1 = segment_vectors[seg_i * 3 + 1];
        float vx2 = segment_vectors[seg_i * 3 + 2];
        float wx0 = px - sx0;
        float wx1 = py - sx1;
        float wx2 = pz - sx2;
        float proj_raw = (wx0 * vx0 + wx1 * vx1 + wx2 * vx2) / len2;
        float proj_clip = fminf(1.0f, fmaxf(0.0f, proj_raw));
        float cx0 = sx0 + proj_clip * vx0;
        float cx1 = sx1 + proj_clip * vx1;
        float cx2 = sx2 + proj_clip * vx2;
        float dx = px - cx0;
        float dy = py - cx1;
        float dz = pz - cx2;
        float d_center = sqrtf(dx * dx + dy * dy + dz * dz);

        float radius_limit = fminf(radii_si[seg_i], L);
        if (
            !inside_any && proj_raw >= 0.0f && proj_raw <= 1.0f
            && d_center <= radius_limit
        ) {
            inside_any = 1;
            break;
        }

        float proj_flow = flow_sign[seg_i] < 0.0f ? 1.0f - proj_raw : proj_raw;
        proj_flow = fminf(1.0f, fmaxf(0.0f, proj_flow));
        float s_star = proj_flow * L;
        float Cc_star = cin_pos[seg_i] * expf(-alpha_edge[seg_i] * s_star);
        float denom_gate = fmaxf(km + fmaxf(Cc_star, 1.0e-12f), 1.0e-30f);
        float lam_gate = sqrtf(
            diffusivity_si / fmaxf(vmax / denom_gate, 1.0e-30f)
        );
        if (d_center > window_factor * lam_gate) continue;
        if (Cc_star > cap_max) cap_max = Cc_star;

        float halfW = window_factor * lam_ref;
        float s0 = fmaxf(0.0f, s_star - halfW);
        float s1 = fminf(L, s_star + halfW);
        if (s1 <= s0 + 1.0e-15f) continue;
        float mid = 0.5f * (s0 + s1);
        float half = 0.5f * (s1 - s0);
        float total_local = 0.0f;
        float radius = radii_si[seg_i];
        float cin_seg = cin_pos[seg_i];
        float alpha = alpha_edge[seg_i];

        for (int g = 0; g < gl_order; ++g) {
            float s = mid + half * gl_nodes[g];
            float t = s / L;
            float qx = sx0 + t * vx0;
            float qy = sx1 + t * vx1;
            float qz = sx2 + t * vx2;
            float rx = px - qx;
            float ry = py - qy;
            float rz = pz - qz;
            float r = sqrtf(rx * rx + ry * ry + rz * rz);
            if (r <= 1.0e-12f) continue;

            float cc_s = cin_seg * expf(-alpha * s);
            float denom_loc = fmaxf(km + fmaxf(cc_s, 1.0e-12f), 1.0e-30f);
            float lam_loc = sqrtf(
                diffusivity_si / fmaxf(vmax / denom_loc, 1.0e-30f)
            );
            float phi_loc = fmaxf(radius / fmaxf(lam_loc, 1.0e-30f), 1.0e-12f);
            float r_over_lam = r / fmaxf(lam_loc, 1.0e-30f);
            float k0_num = interp_lut_dense(r_over_lam, xs_lut, k0_lut, lut_n);
            float k0_den = fmaxf(
                interp_lut_dense(phi_loc, xs_lut, k0_lut, lut_n), 1.0e-30f
            );
            float Ci_R = cc_s * (k0_num / k0_den);

            float denom_corr = fmaxf(km + fminf(3.0f * Ci_R, cc_s), 1.0e-30f);
            float lam_corr = sqrtf(
                diffusivity_si / fmaxf(vmax / denom_corr, 1.0e-30f)
            );
            float phi = fmaxf(radius / fmaxf(lam_corr, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut_dense(phi, xs_lut, ratio_lut, lut_n);
            float wall_factor = (
                diffusivity_si / fmaxf(lam_corr, 1.0e-30f)
            ) * ratio;
            float q_s = 2.0f * 3.14159265358979323846f * radius * wall_factor * cc_s;
            float green = expf(-r / fmaxf(lam_corr, 1.0e-30f))
                / (4.0f * 3.14159265358979323846f * r);
            total_local += gl_weights[g] * q_s * (green / diffusivity_si);
        }
        total += half * total_local;
    }

    if (inside_any) {
        keep_mask[row] = 0;
        out[row] = 0.0f;
        return;
    }
    keep_mask[row] = 1;
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out[row] = fminf(total, cap_max);
}
