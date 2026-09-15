// Reduce active source decay lengths to one conservative maximum per grid cell.
// hybrid_deposit.py uses the result to bound local-only source searches.
extern "C" __device__ int atomicMaxFloatBits(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    int val_i = __float_as_int(val);
    while (__int_as_float(old) < val) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, val_i);
        if (assumed == old) break;
    }
    return old;
}
extern "C" __global__ void cext_local_cell_max_lambda_kernel(
    const int* local_node_cell_flat,
    const int* node_seg_ids,
    const float* lambda_iv_gl,
    const unsigned char* active_source_mask,
    const int n_nodes,
    const int gl_order,
    float* cell_max_lambda
) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= n_nodes) return;
    int cell = local_node_cell_flat[node];
    if (cell < 0) return;
    int source_seg = node_seg_ids[node];
    if (active_source_mask[source_seg] == 0) return;
    int source_gl = node - source_seg * gl_order;
    if (source_gl < 0 || source_gl >= gl_order) return;
    float lam = lambda_iv_gl[source_seg * gl_order + source_gl];
    if (!(lam > 0.0f) || !isfinite(lam)) return;
    atomicMaxFloatBits(&cell_max_lambda[cell], lam);
}
