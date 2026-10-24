// N128/M8/A128 correctness prototype: one thread owns one complete atom.
// All block threads reach every barrier, including partial batches/empty atoms.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include "block128_math.h"

namespace {
using namespace cst128;

__device__ double alpha(const float* p, const double* norm, int a) {
    double raw=::sqrt(norm[a*6]*norm[a*6+3]);
    return static_cast<double>(p[a*4])/::fmax(raw,FLOOR);
}

// Prepare only six norm scalars per atom; no global Native/Co/Ci/Beta/Phi.
__global__ void prepare_norms(const float* p, const double* fit, double* norm) {
    int a=threadIdx.x;
    double width=.1+9.9*sigmoid(p[a*4+3]);
    Component comp[4];
    double co[4][M+1], ci[4][M+1];
    for (int c=0;c<4;++c) {
        comp[c]=component(width,c);
        coefficients(fit,width,comp[c],co[c],ci[c]);
    }
    for (int axis=0;axis<2;++axis) {
        double center=p[a*4+1+axis], n=0., nm=0., ni=0.;
        for (int i=0;i<N;++i) {
            double f[3]={};
            for (int c=0;c<4;++c) {
                double term[3];
                series(i,center,comp[c],co[c],ci[c],term);
                for (int k=0;k<3;++k) f[k]+=term[k];
            }
            n+=f[0]*f[0]; nm+=f[0]*f[1]; ni+=f[0]*f[2];
        }
        norm[(a*2+axis)*3]=n;
        norm[(a*2+axis)*3+1]=nm;
        norm[(a*2+axis)*3+2]=ni;
    }
}

// Element ownership: (batch,basis); parent even modes inherit compatible children.
__device__ void build_z_tile(const float* source, double* z, int node) {
    int bank=node_bank(node), first=node_start(node), size=bank_size(bank);
    for (int q=threadIdx.x;q<BATCH*R;q+=blockDim.x) {
        int b=q/R, r=q%R, mode=r<=M ? r : r-M;
        double value=0.;
        if (bank>=4 && mode%2==0) {
            int cb=bank-4, child=bank_offset(cb)+first/bank_size(cb);
            int cr=mode/2+(r<=M ? 0 : M);
            value=z[(child*BATCH+b)*R+cr]+z[((child+1)*BATCH+b)*R+cr];
        } else {
            for (int i=first;i<first+size;++i)
                value+=static_cast<double>(source[b*N+i])*basis(bank,r,i);
        }
        z[node*BATCH*R+q]=value;
    }
}

// The calling thread owns all sites/modes/batches for this component/atom.
__device__ void contract_component(const float* source, const double* z,
    double center, Component comp, const double* co, const double* ci,
    bool derivatives, double result[3][BATCH]) {
    int lo,hi; bounds(center,comp.length,lo,hi);
    int left=lo,right=lo;
    if (comp.bank>=0) {
        int size=bank_size(comp.bank);
        int first=(lo+size-1)/size, last=hi/size;
        if (last>first) {
            for (int t=first;t<last;++t) {
                int node=bank_offset(comp.bank)+t;
                for (int r=0;r<R;++r) {
                    int mode=r<=M ? r : r-M;
                    double beta_fields[3];
                    beta(comp,center,r,co[mode],ci[mode],beta_fields);
                    for (int b=0;b<BATCH;++b)
                        for (int k=0;k<(derivatives?3:1);++k)
                            result[k][b]+=z[(node*BATCH+b)*R+r]*beta_fields[k];
                }
            }
            left=first*size; right=last*size;
        }
    }
    for (int i=lo;i<hi;++i) {
        if (i>=left && i<right) continue;
        double f[3]; series(i,center,comp,co,ci,f);
        for (int b=0;b<BATCH;++b)
            for (int k=0;k<(derivatives?3:1);++k)
                result[k][b]+=static_cast<double>(source[b*N+i])*f[k];
    }
}

__device__ void write_parameter_partial(const float* p, const double* norm,
    double h[3][BATCH], double g[3][BATCH], double* dp, int batch_group) {
    int a=threadIdx.x;
    double nu=norm[a*6], nv=norm[a*6+3];
    double raw=::sqrt(nu*nv), denom=::fmax(raw,FLOOR);
    double al=static_cast<double>(p[a*4])/denom;
    double s=0., om=0., im=0., w=0.;
    for (int b=0;b<BATCH;++b) {
        s+=h[0][b]*g[0][b];
        om+=h[0][b]*g[1][b]; im+=h[1][b]*g[0][b];
        w+=h[2][b]*g[0][b]+h[0][b]*g[2][b];
    }
    double lmout=0., lmin=0., lw=0.;
    if (raw>FLOOR && nu>0. && nv>0.) {
        lmout=norm[a*6+1]/nu; lmin=norm[a*6+4]/nv;
        lw=norm[a*6+2]/nu+norm[a*6+5]/nv;
    }
    double sig=sigmoid(p[a*4+3]);
    int offset=(batch_group*A+a)*4;
    dp[offset]=s/denom;
    dp[offset+1]=al*(om-s*lmout);
    dp[offset+2]=al*(im-s*lmin);
    dp[offset+3]=al*(w-s*lw)*9.9*sig*(1.-sig);
}

// One writer per C element; finite-support interiors only.
__device__ void build_c(const float* p, const double* fit, const double* norm,
    const double* values, double* cbuf, int axis, int output_begin) {
    for (int q=threadIdx.x;q<BANKS*BATCH*R;q+=blockDim.x) {
        int bank=q/(BATCH*R), b=q/R%BATCH, r=q%R;
        int size=bank_size(bank), start=output_begin/size*size;
        int mode=r<=M ? r : r-M;
        double sum=0.;
        for (int a=0;a<A;++a) {
            double width=.1+9.9*sigmoid(p[a*4+3]);
            double center=p[a*4+1+axis];
            double scaled=alpha(p,norm,a)*values[a*BATCH+b];
            for (int k=0;k<4;++k) {
                Component cp=component(width,k);
                if (cp.bank!=bank) continue;
                int lo,hi; bounds(center,cp.length,lo,hi);
                if (lo>start || hi<start+size) continue;
                double co,ci,bt[3]; coefficient(fit,width,cp,mode,co,ci);
                beta(cp,center,r,co,ci,bt);
                sum+=scaled*bt[0];
            }
        }
        cbuf[q]=sum; // assigns every element, including unused/empty nodes
    }
}

__device__ void store_output(const float* p, const double* fit, const double* norm,
    const double* values, const double* cbuf, float* output,
    int batch_count, int batch_begin, int output_begin, int axis) {
    int b=threadIdx.x/32, i=output_begin+threadIdx.x%32;
    double acc=0.;
    for (int bank=0;bank<BANKS;++bank)
        for (int r=0;r<R;++r) acc+=cbuf[(bank*BATCH+b)*R+r]*basis(bank,r,i);
    for (int a=0;a<A;++a) {
        double width=.1+9.9*sigmoid(p[a*4+3]), center=p[a*4+1+axis];
        double direct=0.;
        for (int k=0;k<4;++k) {
            Component cp=component(width,k);
            if (!(cp.weight>0. || cp.width_derivative!=0.)) continue;
            int lo,hi; bounds(center,cp.length,lo,hi);
            if (i<lo || i>=hi) continue;
            if (cp.bank>=0) {
                int size=bank_size(cp.bank), start=i/size*size;
                if (lo<=start && hi>=start+size) continue; // already in C
            }
            double co[M+1],ci[M+1],f[3];
            coefficients(fit,width,cp,co,ci);
            series(i,center,cp,co,ci,f);
            direct+=f[0];
        }
        acc+=alpha(p,norm,a)*values[a*BATCH+b]*direct;
    }
    if (batch_begin+b<batch_count) output[(batch_begin+b)*N+i]=static_cast<float>(acc);
}

template<bool BACKWARD>
__global__ void block128(const float* x, const float* p, const float* dy,
    const double* fit, const double* norm, float* output, double* dp, int batch_count) {
    extern __shared__ __align__(16) unsigned char arena[];
    int a=threadIdx.x, tile=blockIdx.x%4, batch_group=blockIdx.x/4;
    int batch_begin=batch_group*BATCH;
    bool owner=BACKWARD && tile==0;
    float* sx=reinterpret_cast<float*>(arena);
    float* sy=reinterpret_cast<float*>(arena+(owner?2048:0));
    double* zx=reinterpret_cast<double*>(arena+(owner?4096:2048));
    double* zg=owner ? reinterpret_cast<double*>(arena+19328) : zx;
    double* values=reinterpret_cast<double*>(arena+(owner?34560:17280));
    // [init] Each thread reads four source elements per required side.
    for (int q=a;q<BATCH*N;q+=blockDim.x) {
        int b=batch_begin+q/N, i=q%N;
        if (!BACKWARD || owner) sx[q]=b<batch_count ? x[b*N+i] : 0.f;
        if (BACKWARD) sy[q]=b<batch_count ? dy[b*N+i] : 0.f;
    }
    __syncthreads();
    for (int node=0;node<NODES;++node) {
        if (!BACKWARD || owner) build_z_tile(sx,zx,node);
        if (BACKWARD) build_z_tile(sy,zg,node);
        __syncthreads();
    }
    // [atom] This thread handles the entire atom, all four batches and both sides.
    double h[3][BATCH]={}, g[3][BATCH]={};
    double width=.1+9.9*sigmoid(p[a*4+3]);
    for (int k=0;k<4;++k) {
        Component cp=component(width,k);
        if (!(cp.weight>0. || cp.width_derivative!=0.)) continue;
        double co[M+1],ci[M+1]; coefficients(fit,width,cp,co,ci);
        if (!BACKWARD || owner)
            contract_component(sx,zx,p[a*4+2],cp,co,ci,owner,h);
        if (BACKWARD)
            contract_component(sy,zg,p[a*4+1],cp,co,ci,owner,g);
    }
    for (int b=0;b<BATCH;++b) values[a*BATCH+b]=BACKWARD ? g[0][b] : h[0][b];
    if (owner) write_parameter_partial(p,norm,h,g,dp,batch_group);
    __syncthreads(); // no thread can read Z after this point
    // [C] Reuse Z_X (or Z_dY for dX-only blocks) only after the above barrier.
    build_c(p,fit,norm,values,zx,BACKWARD?1:0,tile*32);
    __syncthreads();
    store_output(p,fit,norm,values,zx,output,batch_count,batch_begin,tile*32,BACKWARD?1:0);
}

__global__ void reduce_parameters(const double* partial, float* dp, int groups) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=A*4) return;
    double sum=0.;
    for (int g=0;g<groups;++g) sum+=partial[g*A*4+i];
    dp[i]=static_cast<float>(sum);
}
} // namespace

std::vector<at::Tensor> block128_forward_cuda(at::Tensor x, at::Tensor p, at::Tensor fit) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    auto y=at::empty_like(x);
    auto norm=at::empty({128,2,3},x.options().dtype(at::kDouble));
    prepare_norms<<<1,128,0,stream>>>(p.data_ptr<float>(),fit.data_ptr<double>(),norm.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    block128<false><<<4*((x.size(0)+3)/4),128,cst128::FORWARD_SHARED,stream>>>(
        x.data_ptr<float>(),p.data_ptr<float>(),nullptr,fit.data_ptr<double>(),
        norm.data_ptr<double>(),y.data_ptr<float>(),nullptr,x.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y,norm};
}

std::vector<at::Tensor> block128_backward_cuda(at::Tensor x, at::Tensor p, at::Tensor dy,
                                             at::Tensor fit, at::Tensor norm) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    int groups=(x.size(0)+3)/4;
    auto dx=at::empty_like(x), dp=at::empty_like(p);
    auto partial=at::empty({groups,128,4},x.options().dtype(at::kDouble));
    block128<true><<<4*groups,128,cst128::BACKWARD_SHARED,stream>>>(
        x.data_ptr<float>(),p.data_ptr<float>(),dy.data_ptr<float>(),fit.data_ptr<double>(),
        norm.data_ptr<double>(),dx.data_ptr<float>(),partial.data_ptr<double>(),x.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_parameters<<<4,128,0,stream>>>(partial.data_ptr<double>(),dp.data_ptr<float>(),groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx,dp};
}
