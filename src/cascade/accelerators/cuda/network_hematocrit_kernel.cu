extern "C" __global__ void network_hematocrit_kernel(
    int n, const int* nodes, const int* in_offsets, const int* in_edges,
    const int* out_offsets, const int* out_edges, const double* q,
    const double* diameter, const double* source_flow, double* hd, double root_hd, double hmin,
    double hmax, double par1, double par2, double par3, int phase) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;
    int node=nodes[idx], istart=in_offsets[node], istop=in_offsets[node+1];
    int start=out_offsets[node], stop=out_offsets[node+1];
    double qi=0., rbc=0., dp=0., qo=0.;
    for(int j=istart;j<istop;++j) {
        int e=in_edges[j];qi+=q[e];rbc+=q[e]*hd[e];dp+=q[e]*diameter[e];
    }
    for(int j=start;j<stop;++j) qo+=q[out_edges[j]];
    if(qo<=0.) return;
    double hp=qi>0. ? (rbc+source_flow[node]*root_hd)/qo : root_hd;
    if(phase && stop-start==2) {
        int l=out_edges[start],r=out_edges[start+1];
        double f=q[l]/qo, fe=f;
        double parent=qi>0. ? dp/qi : sqrt(diameter[l]*diameter[l]+diameter[r]*diameter[r]);
        parent=fmax(parent,1.e-9);
        double x0=fmin(fmax(par1*(1.-hp)/parent,0.),.49);
        if(f<=x0) fe=0.;
        else if(f>=1.-x0) fe=1.;
        else {
            double dquot=diameter[l]*diameter[l]/fmax(diameter[r]*diameter[r],1.e-30);
            double a=par3*(dquot-1.)/(dquot+1.)*(1.-hp)/parent;
            double b=1.+par2*(1.-hp)/parent;
            double z=fmin(fmax((f-x0)/(1.-2.*x0),1.e-12),1.-1.e-12);
            fe=1./(1.+exp(-fmin(fmax(a+b*log(z/(1.-z)),-50.),50.)));
        }
        // Bound the RBC partition, rather than independently clipping HD:
        // both daughter bounds and exact RBC flux conservation then coexist.
        if(hp>0.) {
            double low=fmax(hmin*f/hp,1.-hmax*(1.-f)/hp);
            double high=fmin(hmax*f/hp,1.-hmin*(1.-f)/hp);
            fe=fmin(fmax(fe,low),high);
        }
        hd[l]=hp*fe/f;
        hd[r]=hp*(1.-fe)/(1.-f);
    } else {
        for(int j=start;j<stop;++j) hd[out_edges[j]]=hp;
    }
}
