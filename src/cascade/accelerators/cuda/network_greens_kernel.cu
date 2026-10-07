extern "C" __global__ void network_greens_kernel(
    int n, const int* ids, const int* up, const double* nodes,
    const double* q, const double* radii, const double* lengths,
    const double* cap, double diffusivity, double vmax, double km,
    double alpha, int blood, int steps, const double* xs, const double* ys,
    int lut_n, double* cin, double* cout) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;
    int e=ids[idx];
    double c=fmax(nodes[up[e]],0.);
    cin[e]=c;
    if(q[e]<=0. || radii[e]<=0. || lengths[e]<=0.) {cout[e]=c;return;}
    double lam=sqrt(diffusivity/fmax(vmax/fmax(km+c,1.e-30),1.e-30));
    double phi=fmax(radii[e]/fmax(lam,1.e-30),1.e-12);
    double ratio;
    if(phi<=xs[0]) ratio=ys[0];
    else if(phi>=xs[lut_n-1]) ratio=ys[lut_n-1];
    else {
        int lo=0,hi=lut_n-1;
        while(hi-lo>1) {int mid=(hi+lo)/2;if(xs[mid]<=phi) lo=mid;else hi=mid;}
        ratio=ys[lo]+(phi-xs[lo])/(xs[lo+1]-xs[lo])*(ys[lo+1]-ys[lo]);
    }
    double base=6.283185307179586*radii[e]/fmax(q[e],1.e-30)*diffusivity/fmax(lam,1.e-30)*ratio;
    int count=blood ? steps : 1;
    for(int j=0;j<count;++j) {
        double p=c/alpha,den=p*p*p+150.*p+23400.;
        double buffer=blood ? 1.+fmax(cap[e],0.)/alpha*70200.*(p*p+50.)/(den*den) : 1.;
        c*=exp(fmin(fmax(-base/fmax(buffer,1.e-30)*lengths[e]/count,-150.),50.));
    }
    cout[e]=c;
}
