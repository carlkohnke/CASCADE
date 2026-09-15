// Traverse the prepared treecode and evaluate Cext at vessel quadrature nodes.
// treecode.py compiles this evaluation stage for approximate far-field solves.
extern "C" __global__ void cext_treecode_kernel(
    const int* target_seg_ids,
    const float* target_points_si,
    const float* radii_si,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    const float* source_points_si,
    const int* source_seg_ids,
    const float* source_lambda,
    const float* source_q,
    const int* node_start,
    const int* node_end,
    const float* node_center,
    const float* node_half,
    const int* node_children,
    const unsigned char* node_is_leaf,
    const int* node_point_count,
    const int* node_active_count,
    const unsigned char* point_active,
    const float* node_total_mass,
    const float* node_total_dipole,
    const float* node_active_mass,
    const float* node_active_dipole,
    const float* lambda_centers,
    int n_bins,
    int gl_order,
    int exclude_width,
    int node_count,
    int target_count,
    int order_mode,
    float theta,
    float near_radius_si,
    float diffusivity_si,
    float window_factor,
    float global_cap,
    float* out_total,
    unsigned long long* metrics
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_count * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];
    float tx = target_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = target_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = target_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_radius = radii_si[target_seg];
    int excl_n = (int)exclude_count[target_seg];
    float total = 0.0f;
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        int node = stack[--top];
        if (node < 0 || node >= node_count) continue;
        float cx = node_center[node * 3 + 0];
        float cy = node_center[node * 3 + 1];
        float cz = node_center[node * 3 + 2];
        float dx = cx - tx;
        float dy = cy - ty;
        float dz = cz - tz;
        float dist2 = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(dist2 + 1.0e-24f);
        float node_r = 1.7320508075688772f * node_half[node];
        float sep = dist - node_r;
        if (sep < 1.0e-12f) sep = 1.0e-12f;
        int is_leaf = node_is_leaf[node] != 0;
        int total_count = node_point_count[node];
        int active_count = node_active_count[node];
        int inactive_count = total_count - active_count;
        if (sep > near_radius_si && inactive_count > 0) {
            for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
                int base = bin_idx * node_count + node;
                float mass = node_total_mass[base] - node_active_mass[base];
                if (!(mass != 0.0f) || !isfinite(mass)) continue;
                float lam = lambda_centers[bin_idx];
                if (lam < 1.0e-30f) lam = 1.0e-30f;
                float g = expf(-dist / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * dist);
                float contrib = mass * g;
                if (order_mode > 0) {
                    float dotp =
                        (node_total_dipole[base * 3 + 0] - node_active_dipole[base * 3 + 0]) * dx
                        + (node_total_dipole[base * 3 + 1] - node_active_dipole[base * 3 + 1]) * dy
                        + (node_total_dipole[base * 3 + 2] - node_active_dipole[base * 3 + 2]) * dz;
                    contrib += g * (1.0f / lam + 1.0f / dist) * (dotp / dist);
                }
                total += contrib;
            }
            atomicAdd(metrics + 0, (unsigned long long)1);
        }
        if (active_count <= 0 && sep > near_radius_si) {
            continue;
        }
        if (!is_leaf && sep > near_radius_si && active_count > 0 && (node_r / sep) <= theta) {
            for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
                int base = bin_idx * node_count + node;
                float mass = node_active_mass[base];
                if (!(mass != 0.0f) || !isfinite(mass)) continue;
                float lam = lambda_centers[bin_idx];
                if (lam < 1.0e-30f) lam = 1.0e-30f;
                float g = expf(-dist / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * dist);
                float contrib = mass * g;
                if (order_mode > 0) {
                    float dotp =
                        node_active_dipole[base * 3 + 0] * dx
                        + node_active_dipole[base * 3 + 1] * dy
                        + node_active_dipole[base * 3 + 2] * dz;
                    contrib += g * (1.0f / lam + 1.0f / dist) * (dotp / dist);
                }
                total += contrib;
            }
            atomicAdd(metrics + 0, (unsigned long long)1);
            continue;
        }
        if (is_leaf) {
            int need_all = sep <= near_radius_si;
            int start = node_start[node];
            int stop = node_end[node];
            for (int pos = start; pos < stop; ++pos) {
                if (!need_all && point_active[pos] == 0) continue;
                int source_seg = source_seg_ids[pos];
                int excluded = 0;
                for (int ei = 0; ei < excl_n; ++ei) {
                    if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                        excluded = 1;
                        break;
                    }
                }
                if (excluded) continue;
                float radius_sum = target_radius + radii_si[source_seg];
                float sx = source_points_si[pos * 3 + 0];
                float sy = source_points_si[pos * 3 + 1];
                float sz = source_points_si[pos * 3 + 2];
                float sdx = sx - tx;
                float sdy = sy - ty;
                float sdz = sz - tz;
                float r = sqrtf(sdx * sdx + sdy * sdy + sdz * sdz + radius_sum * radius_sum);
                float lam = source_lambda[pos];
                if (lam < 1.0e-30f) lam = 1.0e-30f;
                if (r > window_factor * lam) continue;
                float q = source_q[pos];
                if (!(q != 0.0f) || !isfinite(q)) continue;
                total += q * expf(-r / lam) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                atomicAdd(metrics + 2, (unsigned long long)1);
            }
            continue;
        }
        atomicAdd(metrics + 1, (unsigned long long)1);
        for (int slot = 0; slot < 8; ++slot) {
            int child = node_children[node * 8 + slot];
            if (child < 0) continue;
            if (node_active_count[child] <= 0) {
                float chx = node_center[child * 3 + 0];
                float chy = node_center[child * 3 + 1];
                float chz = node_center[child * 3 + 2];
                float cdx = chx - tx;
                float cdy = chy - ty;
                float cdz = chz - tz;
                float cdist = sqrtf(cdx * cdx + cdy * cdy + cdz * cdz + 1.0e-24f);
                float child_r = 1.7320508075688772f * node_half[child];
                float child_sep = cdist - child_r;
                if (child_sep > near_radius_si) continue;
            }
            if (top < 64) stack[top++] = child;
        }
    }
    if (global_cap > 0.0f) total = fminf(total, global_cap);
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out_total[target_idx * gl_order + target_node] = total;
}
