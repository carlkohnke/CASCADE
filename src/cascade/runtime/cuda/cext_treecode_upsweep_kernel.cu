extern "C" __global__ void cext_treecode_upsweep_kernel(
    const int* level_node_ids,
    const int* node_children,
    const float* node_center,
    int n_bins,
    int n_nodes_level,
    int n_nodes_total,
    float* total_mass,
    float* total_dipole,
    float* active_mass,
    float* active_dipole,
    int* active_count
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_nodes_level) return;
    int node = level_node_ids[idx];
    int count_sum = 0;
    for (int bi = 0; bi < n_bins; ++bi) {
        float tm = 0.0f, am = 0.0f;
        float tdx = 0.0f, tdy = 0.0f, tdz = 0.0f;
        float adx = 0.0f, ady = 0.0f, adz = 0.0f;
        for (int slot = 0; slot < 8; ++slot) {
            int child = node_children[node * 8 + slot];
            if (child < 0) continue;
            if (bi == 0) count_sum += active_count[child];
            int child_base = bi * n_nodes_total + child;
            float child_tm = total_mass[child_base];
            float child_am = active_mass[child_base];
            float dx = node_center[child * 3 + 0] - node_center[node * 3 + 0];
            float dy = node_center[child * 3 + 1] - node_center[node * 3 + 1];
            float dz = node_center[child * 3 + 2] - node_center[node * 3 + 2];
            tm += child_tm;
            am += child_am;
            tdx += total_dipole[child_base * 3 + 0] + child_tm * dx;
            tdy += total_dipole[child_base * 3 + 1] + child_tm * dy;
            tdz += total_dipole[child_base * 3 + 2] + child_tm * dz;
            adx += active_dipole[child_base * 3 + 0] + child_am * dx;
            ady += active_dipole[child_base * 3 + 1] + child_am * dy;
            adz += active_dipole[child_base * 3 + 2] + child_am * dz;
        }
        int base = bi * n_nodes_total + node;
        total_mass[base] = tm;
        active_mass[base] = am;
        total_dipole[base * 3 + 0] = tdx;
        total_dipole[base * 3 + 1] = tdy;
        total_dipole[base * 3 + 2] = tdz;
        active_dipole[base * 3 + 0] = adx;
        active_dipole[base * 3 + 1] = ady;
        active_dipole[base * 3 + 2] = adz;
    }
    active_count[node] = count_sum;
}
