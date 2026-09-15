// Evaluate the exact near field used to correct the hybrid FFT approximation.
// hybrid_deposit.py compiles this kernel for cell-binned local Cext sources.
extern "C" __global__ void cext_local_corr_kernel(
    const int* target_seg_ids,
    const float* gl_points_si,
    const float* radii_si,
    const float* lambda_iv_gl,
    const float* q_weighted_gl,
    const float* seg_cap_gl,
    const int* exclude_idx,
    const unsigned char* exclude_count,
    const unsigned char* active_source_mask,
    const int* cell_ptr,
    const int* cell_node_ids,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_n,
    int rad_cells,
    float near_radius_si,
    int gl_order,
    int exclude_width,
    int target_n,
    float diffusivity_si,
    float* out_total,
    float* out_cap
) {
    int flat = blockDim.x * blockIdx.x + threadIdx.x;
    int total_targets = target_n * gl_order;
    if (flat >= total_targets) return;
    int target_idx = flat / gl_order;
    int target_node = flat - target_idx * gl_order;
    int target_seg = target_seg_ids[target_idx];

    float tx = gl_points_si[(target_seg * gl_order + target_node) * 3 + 0];
    float ty = gl_points_si[(target_seg * gl_order + target_node) * 3 + 1];
    float tz = gl_points_si[(target_seg * gl_order + target_node) * 3 + 2];
    float target_radius = radii_si[target_seg];

    int cx = (int)floorf((tx - grid_origin_x) / grid_spacing);
    int cy = (int)floorf((ty - grid_origin_y) / grid_spacing);
    int cz = (int)floorf((tz - grid_origin_z) / grid_spacing);
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
    int excl_n = (int)exclude_count[target_seg];
    for (int ix = lo_x; ix <= hi_x; ++ix) {
        for (int iy = lo_y; iy <= hi_y; ++iy) {
            for (int iz = lo_z; iz <= hi_z; ++iz) {
                int cell_flat = (ix * grid_n + iy) * grid_n + iz;
                int row_start = cell_ptr[cell_flat];
                int row_end = cell_ptr[cell_flat + 1];
                for (int pos = row_start; pos < row_end; ++pos) {
                    int source_node_id = cell_node_ids[pos];
                    int source_seg = source_node_id / gl_order;
                    int source_node = source_node_id - source_seg * gl_order;
                    if (active_source_mask[source_seg] == 0) continue;
                    int excluded = 0;
                    for (int ei = 0; ei < excl_n; ++ei) {
                        if (exclude_idx[target_seg * exclude_width + ei] == source_seg) {
                            excluded = 1;
                            break;
                        }
                    }
                    if (excluded) continue;
                    float radius_sum = target_radius + radii_si[source_seg];
                    float seg_cap = seg_cap_gl[source_seg];
                    float source_lambda = lambda_iv_gl[source_node_id];
                    if (source_lambda < 1.0e-30f) source_lambda = 1.0e-30f;
                    float sx = gl_points_si[source_node_id * 3 + 0];
                    float sy = gl_points_si[source_node_id * 3 + 1];
                    float sz = gl_points_si[source_node_id * 3 + 2];
                    float dx = sx - tx;
                    float dy = sy - ty;
                    float dz = sz - tz;
                    float r = sqrtf(dx * dx + dy * dy + dz * dz + radius_sum * radius_sum);
                    if (r > near_radius_si) continue;
                    if (r < 1.0e-30f) r = 1.0e-30f;
                    float kernel = expf(-r / source_lambda) / (4.0f * 3.14159265358979323846f * diffusivity_si * r);
                    total += q_weighted_gl[source_node_id] * kernel;
                    if (seg_cap > cap_max) cap_max = seg_cap;
                }
            }
        }
    }
    if (!(total >= 0.0f) || !isfinite(total)) total = 0.0f;
    out_total[target_idx * gl_order + target_node] = total;
    out_cap[target_idx * gl_order + target_node] = cap_max;
}
