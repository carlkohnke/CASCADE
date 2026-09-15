// Propagate frozen-field vessel oxygen by tree level with the Graetz closure.
// frozen.py compiles this kernel for the GPU top-down Cext solver.
extern "C" __device__ float interp_lut_graetz(float x, const float* xs, const float* ys, int n) {
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
    float t = (x - x0) / (x1 - x0);
    return y0 + t * (y1 - y0);
}

extern "C" __device__ int graetz_basis_index(
    float Bi,
    int key_min,
    int key_max,
    int per_decade,
    float min_bi,
    float max_bi
) {
    float b = fminf(fmaxf(Bi, min_bi), max_bi);
    int key = (int)lrintf(log10f(b) * (float)per_decade);
    if (key < key_min) key = key_min;
    if (key > key_max) key = key_max;
    return key - key_min;
}

extern "C" __device__ float lambda_from_wall_graetz(
    float c_wall,
    float diffusivity_si,
    float vmax,
    float km
) {
    float c = fmaxf(c_wall, 0.0f);
    float denom = fmaxf(km + c, 1.0e-30f);
    float rate = fmaxf(vmax / denom, 1.0e-30f);
    return fmaxf(sqrtf(diffusivity_si / rate), 1.0e-30f);
}

extern "C" __device__ float severinghaus_buffer_graetz(
    float c_bulk,
    float chb_max_local,
    float alpha_mmhg,
    int is_blood
) {
    if (is_blood <= 0) return 1.0f;
    float P = c_bulk / alpha_mmhg;
    float den = (P * P * P + 150.0f * P + 23400.0f);
    float dsdP = 70200.0f * (P * P + 50.0f) / fmaxf(den * den, 1.0e-30f);
    float buffer = 1.0f + fmaxf(chb_max_local, 0.0f) * dsdP / alpha_mmhg;
    return fmaxf(buffer, 1.0e-30f);
}

extern "C" __device__ void graetz_propagate_step(
    const float* profile,
    float* out_profile,
    float c_ext,
    float ds_step,
    float radius_si,
    float flow_mag_si,
    float lumen_diffusivity_si,
    float buffer,
    int basis_idx,
    int n_radial,
    int n_modes,
    const float* mu2_table,
    const float* phi_table,
    const float* project_table,
    const float* cup_table,
    float vess_floor,
    float* out_bulk,
    float* out_wall
) {
    float pi = 3.14159265358979323846f;
    float U = flow_mag_si / fmaxf(pi * radius_si * radius_si, 1.0e-30f);
    float xi = lumen_diffusivity_si * fmaxf(ds_step, 0.0f) / fmaxf(buffer * radius_si * radius_si * U, 1.0e-30f);
    float coeff[16];
    for (int m = 0; m < n_modes; ++m) {
        float acc = 0.0f;
        for (int j = 0; j < n_radial; ++j) {
            float y = profile[j] - c_ext;
            acc += project_table[(basis_idx * n_modes + m) * n_radial + j] * y;
        }
        float arg = mu2_table[basis_idx * n_modes + m] * xi;
        if (arg > 80.0f) arg = 80.0f;
        coeff[m] = acc * expf(-arg);
    }
    float bulk = 0.0f;
    for (int j = 0; j < n_radial; ++j) {
        float y = 0.0f;
        for (int m = 0; m < n_modes; ++m) {
            y += phi_table[(basis_idx * n_radial + j) * n_modes + m] * coeff[m];
        }
        float val = c_ext + y;
        if (val < vess_floor) val = vess_floor;
        out_profile[j] = val;
        bulk += cup_table[basis_idx * n_radial + j] * val;
    }
    if (bulk < vess_floor) bulk = vess_floor;
    *out_bulk = bulk;
    *out_wall = out_profile[n_radial - 1];
}

extern "C" __global__ void frozen_topdown_graetz_kernel(
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
    float lumen_diffusivity_si,
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
    int key_min,
    int key_max,
    int per_decade,
    float min_bi,
    float max_bi,
    int n_radial,
    int n_modes,
    int max_fp_iters,
    float fp_tol,
    const float* mu2_table,
    const float* phi_table,
    const float* project_table,
    const float* cup_table,
    float* cin_seg,
    float* cout_seg,
    float* c_bulk_gl,
    float* c_wall_gl,
    int debug_enabled,
    int* debug_gl_fp_iters,
    float* debug_gl_fp_resid,
    float* debug_gl_buffer,
    int* debug_tail_fp_iters,
    float* debug_tail_fp_resid,
    float* debug_tail_buffer
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
    float radius_si = radii_si[seg_idx];
    float length_si = lengths_si[seg_idx];
    float profile[32];
    float trial[32];
    float best[32];
    for (int j = 0; j < n_radial; ++j) profile[j] = cin_local;
    float bulk_running = cin_local;
    float prev_s = 0.0f;
    int use_wellmixed = (flow_mag_si <= 1.0e-30f || radius_si <= 1.0e-30f || length_si <= 0.0f || lumen_diffusivity_si <= 1.0e-30f);

    for (int node_idx = 0; node_idx < gl_order; ++node_idx) {
        float s_target = gl_t[node_idx] * length_si;
        float ds_step = s_target - prev_s;
        if (ds_step < 0.0f) ds_step = 0.0f;
        float c_ext_local = c_ext_gl[seg_idx * gl_order + node_idx];
        if (c_ext_local < 0.0f) c_ext_local = 0.0f;
        float c_wall = bulk_running;
        float c_bulk = bulk_running;
        if (use_wellmixed) {
            c_bulk_gl[seg_idx * gl_order + node_idx] = c_bulk;
            c_wall_gl[seg_idx * gl_order + node_idx] = c_wall;
            prev_s = s_target;
            continue;
        }
        float wall_guess = fmaxf(profile[n_radial - 1], 0.0f);
        float lambda_guess = lambda_from_wall_graetz(wall_guess, diffusivity_si, vmax, km);
        float accepted_bulk = bulk_running;
        float accepted_wall = wall_guess;
        float final_rel = 0.0f;
        float step_buffer = severinghaus_buffer_graetz(
            bulk_running,
            chb_max[seg_idx],
            alpha_mmhg,
            is_blood
        );
        int used_fp_iters = 0;
        for (int fp = 0; fp < max_fp_iters; ++fp) {
            float phi_l = fmaxf(radius_si / fmaxf(lambda_guess, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut_graetz(phi_l, xs_lut, ratio_lut, lut_n);
            float k_if = (2.0f * 3.14159265358979323846f * radius_si)
                       * (diffusivity_si / fmaxf(lambda_guess, 1.0e-30f)) * ratio;
            float Bi = k_if / fmaxf(2.0f * 3.14159265358979323846f * lumen_diffusivity_si, 1.0e-30f);
            int basis_idx = graetz_basis_index(Bi, key_min, key_max, per_decade, min_bi, max_bi);
            graetz_propagate_step(
                profile,
                trial,
                c_ext_local,
                ds_step,
                radius_si,
                flow_mag_si,
                lumen_diffusivity_si,
                step_buffer,
                basis_idx,
                n_radial,
                n_modes,
                mu2_table,
                phi_table,
                project_table,
                cup_table,
                vess_floor,
                &c_bulk,
                &c_wall
            );
            float lambda_new = lambda_from_wall_graetz(fmaxf(c_wall, 0.0f), diffusivity_si, vmax, km);
            for (int j = 0; j < n_radial; ++j) best[j] = trial[j];
            accepted_bulk = c_bulk;
            accepted_wall = c_wall;
            used_fp_iters = fp + 1;
            final_rel = fabsf(lambda_new - lambda_guess) / fmaxf(lambda_guess, 1.0e-30f);
            if (final_rel < fp_tol) break;
            lambda_guess = 0.5f * lambda_guess + 0.5f * lambda_new;
            lambda_guess = fmaxf(lambda_guess, 1.0e-30f);
        }
        for (int j = 0; j < n_radial; ++j) profile[j] = best[j];
        bulk_running = accepted_bulk;
        c_bulk_gl[seg_idx * gl_order + node_idx] = accepted_bulk;
        c_wall_gl[seg_idx * gl_order + node_idx] = accepted_wall;
        if (debug_enabled > 0) {
            int out_idx = seg_idx * gl_order + node_idx;
            debug_gl_fp_iters[out_idx] = used_fp_iters;
            debug_gl_fp_resid[out_idx] = final_rel;
            debug_gl_buffer[out_idx] = step_buffer;
        }
        prev_s = s_target;
    }

    float ds_tail = length_si - prev_s;
    if (ds_tail < 0.0f) ds_tail = 0.0f;
    float c_ext_tail = 0.0f;
    if (gl_order > 0) c_ext_tail = fmaxf(c_ext_gl[seg_idx * gl_order + gl_order - 1], 0.0f);
    if (!use_wellmixed && ds_tail > 0.0f) {
        float tail_wall_guess = fmaxf(profile[n_radial - 1], 0.0f);
        float tail_lambda_guess = lambda_from_wall_graetz(tail_wall_guess, diffusivity_si, vmax, km);
        float tail_buffer = severinghaus_buffer_graetz(
            bulk_running,
            chb_max[seg_idx],
            alpha_mmhg,
            is_blood
        );
        float tail_bulk = bulk_running;
        float tail_wall = tail_wall_guess;
        float accepted_tail_bulk = tail_bulk;
        float accepted_tail_wall = tail_wall;
        float tail_rel = 0.0f;
        int tail_used_fp_iters = 0;
        for (int fp = 0; fp < max_fp_iters; ++fp) {
            float phi_l = fmaxf(radius_si / fmaxf(tail_lambda_guess, 1.0e-30f), 1.0e-12f);
            float ratio = interp_lut_graetz(phi_l, xs_lut, ratio_lut, lut_n);
            float k_if = (2.0f * 3.14159265358979323846f * radius_si)
                       * (diffusivity_si / fmaxf(tail_lambda_guess, 1.0e-30f)) * ratio;
            float Bi = k_if / fmaxf(2.0f * 3.14159265358979323846f * lumen_diffusivity_si, 1.0e-30f);
            int basis_idx = graetz_basis_index(Bi, key_min, key_max, per_decade, min_bi, max_bi);
            graetz_propagate_step(
                profile,
                trial,
                c_ext_tail,
                ds_tail,
                radius_si,
                flow_mag_si,
                lumen_diffusivity_si,
                tail_buffer,
                basis_idx,
                n_radial,
                n_modes,
                mu2_table,
                phi_table,
                project_table,
                cup_table,
                vess_floor,
                &tail_bulk,
                &tail_wall
            );
            float tail_lambda_new = lambda_from_wall_graetz(fmaxf(tail_wall, 0.0f), diffusivity_si, vmax, km);
            for (int j = 0; j < n_radial; ++j) best[j] = trial[j];
            accepted_tail_bulk = tail_bulk;
            accepted_tail_wall = tail_wall;
            tail_used_fp_iters = fp + 1;
            tail_rel = fabsf(tail_lambda_new - tail_lambda_guess) / fmaxf(tail_lambda_guess, 1.0e-30f);
            if (tail_rel < fp_tol) break;
            tail_lambda_guess = 0.5f * tail_lambda_guess + 0.5f * tail_lambda_new;
            tail_lambda_guess = fmaxf(tail_lambda_guess, 1.0e-30f);
        }
        for (int j = 0; j < n_radial; ++j) profile[j] = best[j];
        bulk_running = accepted_tail_bulk;
        if (debug_enabled > 0) {
            debug_tail_fp_iters[seg_idx] = tail_used_fp_iters;
            debug_tail_fp_resid[seg_idx] = tail_rel;
            debug_tail_buffer[seg_idx] = tail_buffer;
        }
    }
    cout_seg[seg_idx] = fmaxf(bulk_running, vess_floor);
}
