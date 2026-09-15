// Shared cloud-in-cell and triangular-shaped-cloud axis assignment helpers.
// hybrid_geometry.py prepends this header to runtime-stencil CUDA kernels.
extern "C" __device__ float cext_tsc_weight(float dist) {
    dist = fabsf(dist);
    if (dist < 0.5f) return 0.75f - dist * dist;
    if (dist < 1.5f) {
        float t = 1.5f - dist;
        return 0.5f * t * t;
    }
    return 0.0f;
}
extern "C" __device__ void cext_axis_assignment(float g, int grid_n, int assignment_mode, int* idx, float* w, int* n) {
    if (assignment_mode == 1) {
        int c = (int)floorf(g + 0.5f);
        *n = 3;
        for (int s = 0; s < 3; ++s) {
            int ii = c + s - 1;
            idx[s] = ii;
            w[s] = cext_tsc_weight(g - (float)ii);
        }
    } else {
        int i0 = (int)floorf(g);
        float f = g - (float)i0;
        *n = 2;
        idx[0] = i0;
        idx[1] = i0 + 1;
        idx[2] = -1;
        w[0] = 1.0f - f;
        w[1] = f;
        w[2] = 0.0f;
    }
}
