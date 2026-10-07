// Refresh per-quadrature-node Cext source weights during nonlinear iteration.
// state.py compiles this fused cache update to reduce Python/GPU round trips.
extern "C" __device__ float cext_cache_interp_lut(float x, const float* xs, const float* ys, int n) {
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

extern "C" __global__ void cext_iteration_cache_kernel(
    const float* c_bulk_in,
    const float* c_wall_in,
    const float* c_ext_gl,
    const float* radii_si,
    const float* ds_gl,
    const float* xs_lut,
    const float* ratio_lut,
    int lut_n,
    int nseg,
    int gl_order,
    float diffusivity_si,
    float vmax,
    float km,
    float vess_floor,
    int use_lambda_tissue,
    int include_mono2,
    int include_dipole2,
    float* q_weighted_gl,
    float* mono2_weight_gl,
    float* dipole2_weight_gl,
    float* lambda_iv_gl,
    float* seg_cap_gl,
    float* lambda_if_gl,
    float* k_if_gl,
    float* q_line_gl
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nseg * gl_order;
    if (flat >= total) return;
    int seg_idx = flat / gl_order;
    int node_idx = flat - seg_idx * gl_order;

    float c_bulk = c_bulk_in[flat];
    if (!(c_bulk >= vess_floor) || !isfinite(c_bulk)) c_bulk = vess_floor;
    float c_wall = c_wall_in[flat];
    if (!(c_wall >= vess_floor) || !isfinite(c_wall)) c_wall = vess_floor;
    float c_ext = c_ext_gl[flat];
    if (!(c_ext > 0.0f) || !isfinite(c_ext)) c_ext = 0.0f;

    float denom_if = km + fmaxf(c_wall, 0.0f);
    if (denom_if < 1.0e-30f) denom_if = 1.0e-30f;
    float k1_if = vmax / denom_if;
    if (k1_if < 1.0e-30f) k1_if = 1.0e-30f;
    float lambda_if = sqrtf(diffusivity_si / k1_if);
    if (lambda_if < 1.0e-30f) lambda_if = 1.0e-30f;

    float lambda_iv = lambda_if;
    if (use_lambda_tissue != 0) {
        float eff = sqrtf(fmaxf(km + fmaxf(c_wall, 0.0f), 1.0e-30f) * fmaxf(km + c_ext, 1.0e-30f));
        float k1_t = vmax / fmaxf(eff, 1.0e-30f);
        if (k1_t < 1.0e-30f) k1_t = 1.0e-30f;
        lambda_iv = sqrtf(diffusivity_si / k1_t);
        if (lambda_iv < 1.0e-30f) lambda_iv = 1.0e-30f;
    }

    float radius = radii_si[seg_idx];
    if (!(radius > 0.0f) || !isfinite(radius)) radius = 0.0f;
    float phi = radius / lambda_if;
    if (phi < 1.0e-12f) phi = 1.0e-12f;
    float ratio = cext_cache_interp_lut(phi, xs_lut, ratio_lut, lut_n);
    float k_if = (2.0f * 3.14159265358979323846f * radius) * (diffusivity_si / lambda_if) * ratio;
    float q_line = k_if * (c_wall - c_ext);
    float ds = ds_gl[flat];
    float q_weight = q_line * ds;
    float a2 = radius * radius;

    q_weighted_gl[flat] = q_weight;
    lambda_if_gl[flat] = lambda_if;
    k_if_gl[flat] = k_if;
    q_line_gl[flat] = q_line;
    lambda_iv_gl[flat] = lambda_iv;
    mono2_weight_gl[flat] = include_mono2 != 0 ? 0.25f * a2 * q_weight : 0.0f;
    dipole2_weight_gl[flat] = include_dipole2 != 0 ? a2 * 3.14159265358979323846f * diffusivity_si * c_wall * ds : 0.0f;

    if (node_idx == 0) {
        float cap = vess_floor;
        for (int j = 0; j < gl_order; ++j) {
            int idx = seg_idx * gl_order + j;
            float cb = c_bulk_in[idx];
            if (!(cb >= vess_floor) || !isfinite(cb)) cb = vess_floor;
            float cw = c_wall_in[idx];
            if (!(cw >= vess_floor) || !isfinite(cw)) cw = vess_floor;
            cap = fmaxf(cap, fmaxf(cb, cw));
        }
        seg_cap_gl[seg_idx] = cap;
    }
}
