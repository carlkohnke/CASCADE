// One thread per graph node. CSR gather avoids atomic flux accumulation.
#ifdef CASCADE_NODE_FLOAT64
typedef double real_t;
#else
typedef float real_t;
#endif

__device__ double content(double c, double capacity, double alpha) {
    double p = c / alpha;
    double num = p*p*p + 150.0*p;
    return c + capacity * num / (num + 23400.0);
}

__device__ double invert_content(double target, double capacity, double alpha, double guess) {
    if (target <= 0.0) return 0.0;
    if (capacity <= 0.0) return target;
    double lo = fmax(target-capacity, 0.0), hi = target;
    double c = fmin(fmax(guess, lo), hi);
    for (int it=0; it<64; ++it) {
        double p=c/alpha, den=p*p*p+150.0*p+23400.0;
        double residual=content(c, capacity, alpha)-target;
        if (fabs(residual) <= 1.e-12*fmax(target, 1.e-30)) return c;
        if (residual>0.0) hi=c; else lo=c;
        double slope=1.0+capacity/alpha*70200.0*(p*p+50.0)/(den*den);
        double trial=c-residual/slope;
        c = (trial>lo && trial<hi) ? trial : 0.5*(lo+hi);
    }
    return 0.5*(lo+hi);
}

extern "C" __global__ void network_junction_kernel(
    int nnode, const int* node_ids, const int* offsets, const int* edges, const real_t* q,
    const real_t* cout, const real_t* capacity, const real_t* denominator,
    const real_t* node_capacity, const unsigned char* inlet_mask,
    const real_t* current, real_t* updated, real_t* delta,
    real_t* relative_residual, double alpha, double inlet_value,
    double omega, int conserve_total
) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx>=nnode) return;
    int node=node_ids[idx];
    double old=current[node], value=old, incoming=0.0;
    int start=offsets[node], stop=offsets[node+1];
    if (inlet_mask[node]) value=inlet_value;
    else if (start<stop && denominator[node]>0.0) {
        for (int i=start; i<stop; ++i) {
            int e=edges[i];
            incoming += q[e] * (conserve_total ? content(cout[e], capacity[e], alpha) : (double)cout[e]);
        }
        double target=incoming / denominator[node];
        int e=edges[start];
        if (stop-start==1 && q[e]==denominator[node] &&
            (!conserve_total || capacity[e]==node_capacity[node])) value=cout[e];
        else value=conserve_total ? invert_content(target, node_capacity[node], alpha, old) : target;
    }
    delta[node]=(real_t)fabs(value-old);
    double outgoing=denominator[node] * (conserve_total ? content(old, node_capacity[node], alpha) : old);
    relative_residual[node]=(start<stop && !inlet_mask[node]) ?
        (real_t)(fabs(incoming-outgoing)/fmax(fmax(fabs(incoming),fabs(outgoing)),1.e-30)) : (real_t)0;
    updated[node]=(real_t)(inlet_mask[node] ? inlet_value :
        (omega==1.0 ? value : old+omega*(value-old)));
}
