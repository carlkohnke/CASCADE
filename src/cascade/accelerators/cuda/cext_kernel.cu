// Evaluate Cext directly over a precomputed sparse source-candidate graph.
// frozen.py loads this source through CuPy/NVRTC for GPU frozen-field solves.
extern "C" __device__ float interp_lut(float x, const float* xs, const float* ys, int n) {
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

extern "C" __global__ void cext_kernel(
    const int* target_seg_ids,
    const int* row_ptr,
    const int* col_idx,
    const float* gl_points_si,
    const float* midpoints_si,
    const float* segment_vectors,
    const float* radii_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* mono2_weight_gl,
    const float* dipole2_weight_gl,
    const float* seg_cap_gl,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    int gl_order,
    int exclude_width,
    int batch_n,
    float diffusivity_si,
    float window_factor,
    float* out
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = batch_n * gl_order;
    if (flat >= total_targets) return;
    int batch_idx = flat / gl_order;
    int target_node = flat - batch_idx * gl_order;
    int target_seg = target_seg_ids[batch_idx];
    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_lambda = lambda_iv_gl[target_seg * gl_order + target_node];
    if (target_lambda < 1.0e-30f) target_lambda = 1.0e-30f;
    float target_radius = radii_si[target_seg];
    float target_mid_x = midpoints_si[target_seg * 3 + 0];
    float target_mid_y = midpoints_si[target_seg * 3 + 1];
    float target_mid_z = midpoints_si[target_seg * 3 + 2];

    float total = 0.0f;
    float cap_max = 0.0f;
    int row_start = row_ptr[batch_idx];
    int row_end = row_ptr[batch_idx + 1];
    int excl_n = (int)exclude_count[target_seg];
    for (int pos = row_start; pos < row_end; ++pos) {
        int source_seg = col_idx[pos];
        int excluded = 0;
        for (int ei = 0; ei < excl_n; ++ei) {
            if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                excluded = 1;
                break;
            }
        }
        if (excluded) continue;
        float radius_sum = target_radius + radii_si[source_seg];
        float mdx = midpoints_si[source_seg * 3 + 0] - target_mid_x;
        float mdy = midpoints_si[source_seg * 3 + 1] - target_mid_y;
        float mdz = midpoints_si[source_seg * 3 + 2] - target_mid_z;
        float coarse_r = sqrtf(mdx * mdx + mdy * mdy + mdz * mdz + radius_sum * radius_sum);
        float source_lambda_max = 1.0e-30f;
        for (int source_node = 0; source_node < gl_order; ++source_node) {
            float source_lambda_probe = lambda_iv_gl[source_seg * gl_order + source_node];
            if (source_lambda_probe > source_lambda_max) source_lambda_max = source_lambda_probe;
        }
        float coarse_lambda = target_lambda;
        if (source_lambda_max > coarse_lambda) coarse_lambda = source_lambda_max;
        if (coarse_r > window_factor * coarse_lambda) continue;
        float seg_cap = seg_cap_gl[source_seg];
        int seg_contributed = 0;
        for (int source_node = 0; source_node < gl_order; ++source_node) {
            float source_lambda = lambda_iv_gl[source_seg * gl_order + source_node];
            if (source_lambda < 1.0e-30f) source_lambda = 1.0e-30f;
            float sx = gl_points_si[(source_seg * gl_order + source_node) * 3 + 0];
            float sy = gl_points_si[(source_seg * gl_order + source_node) * 3 + 1];
            float sz = gl_points_si[(source_seg * gl_order + source_node) * 3 + 2];
            float dx = tx - sx;
            float dy = ty - sy;
            float dz = tz - sz;
            float r_center2 = dx * dx + dy * dy + dz * dz;
            float r = sqrtf(r_center2 + radius_sum * radius_sum);
            if (r > window_factor * source_lambda) continue;
            if (r < 1.0e-30f) r = 1.0e-30f;
            float kernel = expf(-r / source_lambda) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
            int source_flat = source_seg * gl_order + source_node;
            total += q_weighted_gl[source_flat] * kernel;
            float o2_weight = mono2_weight_gl[source_flat] + dipole2_weight_gl[source_flat];
            if (o2_weight != 0.0f) {
                float vx = segment_vectors[source_seg * 3 + 0];
                float vy = segment_vectors[source_seg * 3 + 1];
                float vz = segment_vectors[source_seg * 3 + 2];
                float vlen = sqrtf(vx * vx + vy * vy + vz * vz);
                if (vlen > 1.0e-30f) {
                    float tdotr = (vx * dx + vy * dy + vz * dz) / vlen;
                    float mu2 = (tdotr * tdotr) / (r * r);
                    if (mu2 > 1.0f) mu2 = 1.0f;
                    float inv_r = 1.0f / r;
                    float inv_l = 1.0f / source_lambda;
                    float pH = ((1.0f - 3.0f * mu2) * (inv_r * inv_r + inv_l * inv_r)
                              + (1.0f - mu2) * inv_l * inv_l) * kernel;
                    total += o2_weight * pH;
                }
            }
            seg_contributed = 1;
        }
        if (seg_contributed && seg_cap > cap_max) cap_max = seg_cap;
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    if (cap_max > 0.0f && total > cap_max) total = cap_max;
    out[target_seg * gl_order + target_node] = total;
}
