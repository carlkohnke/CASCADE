// Propagate frozen-field vessel oxygen by tree level with well-mixed closure.
// frozen.py compiles this kernel for the GPU top-down Cext solver.
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

extern "C" __global__ void frozen_topdown_kernel(
    const int* seg_ids,
    int n_level,
    const int* parents,
    const float* flows_si,
    const float* radii_si,
    const float* lengths_si,
    const float* gl_t,
    const float* c_ext_gl,
    int gl_order,
    float diffusivity_si,
    float vmax,
    float km,
    float inlet_concentration,
    const float* chb_max,
    int is_blood,
    float alpha_mmhg,
    float vess_floor,
    const float* xs_lut,
    const float* ratio_lut,
    int lut_n,
    float* cin_seg,
    float* cout_seg,
    float* c_iv_gl
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_level) return;
    int seg_idx = seg_ids[idx];
    int parent = parents[seg_idx];
    float cin_local = inlet_concentration;
    if (parent >= 0) cin_local = cout_seg[parent];
    if (cin_local < vess_floor) cin_local = vess_floor;
    cin_seg[seg_idx] = cin_local;

    float flow_mag_si = fabsf(flows_si[seg_idx]);
    if (flow_mag_si <= 1.0e-30f) flow_mag_si = 1.0e-30f;
    float radius_si = radii_si[seg_idx];
    float length_si = lengths_si[seg_idx];
    float c_running = cin_local;
    float prev_s = 0.0f;

    for (int node_idx = 0; node_idx < gl_order; ++node_idx) {
        float s_target = gl_t[node_idx] * length_si;
        float ds_step = s_target - prev_s;
        if (ds_step < 0.0f) ds_step = 0.0f;
        float c_ext_local = c_ext_gl[seg_idx * gl_order + node_idx];
        if (c_ext_local < 0.0f) c_ext_local = 0.0f;

        float c_pos = c_running;
        if (c_pos < 0.0f) c_pos = 0.0f;
        float denom = km + c_pos;
        if (denom < 1.0e-30f) denom = 1.0e-30f;
        float k1 = vmax / denom;
        if (k1 < 1.0e-30f) k1 = 1.0e-30f;
        float lambda_if = sqrtf(diffusivity_si / k1);
        if (lambda_if < 1.0e-30f) lambda_if = 1.0e-30f;
        float phi = radius_si / lambda_if;
        if (phi < 1.0e-12f) phi = 1.0e-12f;
        float ratio = interp_lut(phi, xs_lut, ratio_lut, lut_n);
        float k_if = (2.0f * 3.14159265358979323846f * radius_si) * (diffusivity_si / lambda_if) * ratio;
        float beta_if = k_if / flow_mag_si;
        if (is_blood > 0) {
            float P = c_running / alpha_mmhg;
            float den = (P * P * P + 150.0f * P + 23400.0f);
            float dsdP = 70200.0f * (P * P + 50.0f) / (den * den);
            float buffer = 1.0f + fmaxf(chb_max[seg_idx], 0.0f) * dsdP / alpha_mmhg;
            if (buffer < 1.0e-30f) buffer = 1.0e-30f;
            beta_if /= buffer;
        }
        float exponent = -beta_if * ds_step;
        if (exponent < -150.0f) exponent = -150.0f;
        else if (exponent > 50.0f) exponent = 50.0f;
        float c_node = c_ext_local + (c_running - c_ext_local) * expf(exponent);
        if (c_node < vess_floor) c_node = vess_floor;
        c_iv_gl[seg_idx * gl_order + node_idx] = c_node;
        c_running = c_node;
        prev_s = s_target;
    }

    float ds_tail = length_si - prev_s;
    if (ds_tail < 0.0f) ds_tail = 0.0f;
    float c_ext_tail = 0.0f;
    if (gl_order > 0) {
        c_ext_tail = c_ext_gl[seg_idx * gl_order + gl_order - 1];
        if (c_ext_tail < 0.0f) c_ext_tail = 0.0f;
    }
    float c_pos = c_running;
    if (c_pos < 0.0f) c_pos = 0.0f;
    float denom = km + c_pos;
    if (denom < 1.0e-30f) denom = 1.0e-30f;
    float k1 = vmax / denom;
    if (k1 < 1.0e-30f) k1 = 1.0e-30f;
    float lambda_if = sqrtf(diffusivity_si / k1);
    if (lambda_if < 1.0e-30f) lambda_if = 1.0e-30f;
    float phi = radius_si / lambda_if;
    if (phi < 1.0e-12f) phi = 1.0e-12f;
    float ratio = interp_lut(phi, xs_lut, ratio_lut, lut_n);
    float k_if = (2.0f * 3.14159265358979323846f * radius_si) * (diffusivity_si / lambda_if) * ratio;
    float beta_tail = k_if / flow_mag_si;
    if (is_blood > 0) {
        float P = c_running / alpha_mmhg;
        float den = (P * P * P + 150.0f * P + 23400.0f);
        float dsdP = 70200.0f * (P * P + 50.0f) / (den * den);
        float buffer = 1.0f + fmaxf(chb_max[seg_idx], 0.0f) * dsdP / alpha_mmhg;
        if (buffer < 1.0e-30f) buffer = 1.0e-30f;
        beta_tail /= buffer;
    }
    float exponent_tail = -beta_tail * ds_tail;
    if (exponent_tail < -150.0f) exponent_tail = -150.0f;
    else if (exponent_tail > 50.0f) exponent_tail = 50.0f;
    float c_out = c_ext_tail + (c_running - c_ext_tail) * expf(exponent_tail);
    if (c_out < vess_floor) c_out = vess_floor;
    cout_seg[seg_idx] = c_out;
}
