#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../radius_cell_mass/common.cuh"
#include "../radius_cell_mass/profile_values.cuh"
using torch::Tensor;
using namespace radius_cell_mass;
namespace connected {
constexpr int G=104, START=4;
struct Conf { Axis a; bool connect; double delta,upper,ratio; int prefix,nodes; };
Conf config(int n,int m,bool connect,std::vector<double> s) {
    Conf c{}; c.a=parse_axis({double(n),1.,1.,0.,double(m),2.});
    c.connect=connect; c.delta=s[0];c.upper=s[1];c.ratio=s[2];
    c.prefix=c.a.banks[2*START].node_offset;
    c.nodes=c.a.node_count-c.prefix;
    return c;
}
__device__ double smooth(double x) { x=fmin(1.,fmax(0.,x)); if(x>.5) {double u=1-x;return 1-u*u*u*(10+u*(-15+6*u));} return x*x*x*(10+x*(-15+6*x)); }
__device__ double slope(double x) { return x>0&&x<1 ? 30*x*x*(1-x)*(1-x):0; }
__device__ void fill_branch(float *g,Conf c,int tile,int first,int count,double r,double mu,double w,double wr,double wm) {
    int level=__ffs(tile)-1, bi=level*2+count-2;
    double cv[17],dr[17]; profile_coefficients_generic(r,count*double(tile),c.a.modes,cv,dr);
    int width=tile<16 ? tile*count : 2*c.a.modes+1;
    g[0]=float(tile);g[1]=float(first);g[2]=float(bi);g[3]=float(width);g[4]=float(count);
    if(tile<16) {
        for(int j=0;j<width;++j) {
            double value=cv[0],rd=0,md=0;
            for(int k=1;k<=c.a.modes;++k) {
                double omega=2*PI*k/(count*tile),arg=PI*k/(count*tile);
                double filter=sin(arg)/arg,phase=omega*(mu-(first*tile+j));
                value+=2*cv[k]*filter*cos(phase);
                rd+=2*dr[k]*filter*cos(phase);
                md-=2*cv[k]*filter*omega*sin(phase);
            }
            g[5+j]=float(w*value);g[5+width+j]=float(w*rd+wr*value);g[5+2*width+j]=float(w*md+wm*value);
        }
    } else {
        g[5]=float(w*cv[0]);g[5+width]=float(wr*cv[0]);g[5+2*width]=float(wm*cv[0]);
        for(int k=1;k<=c.a.modes;++k) {
            double omega=2*PI*k/(count*tile),phase=omega*(mu-first*tile),cp=cos(phase),sp=sin(phase);
            g[5+2*k-1]=float(2*w*cv[k]*cp);g[5+2*k]=float(2*w*cv[k]*sp);
            g[5+width+2*k-1]=float(2*(w*dr[k]+wr*cv[k])*cp);g[5+width+2*k]=float(2*(w*dr[k]+wr*cv[k])*sp);
            g[5+2*width+2*k-1]=float(2*cv[k]*(wm*cp-w*omega*sp));g[5+2*width+2*k]=float(2*cv[k]*(wm*sp+w*omega*cp));
        }
    }
}
__global__ void geometry(const float *p,float *g,uint32_t *mask,int atoms,Conf c,int side) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=atoms)return;
    double r=p[4*a+1],orig=p[4*a+(side?2:3)];
    CUDA_KERNEL_ASSERT(isfinite(r)&&r>0&&r<=c.a.period/2&&isfinite(orig)&&isfinite(p[4*a]));
    double mu=fmod(orig,c.a.period);if(mu<0)mu+=c.a.period;
    float *base=g+int64_t(a)*4*G;
    for(int j=0;j<4;++j)base[j*G]=0;
    if(!c.connect) {
        auto m=classify(r,mu,c.a);auto bank=c.a.banks[m.bank];
        fill_branch(base,c,bank.tile_sites,m.first_tile,bank.count,r,mu,1,0,0);
        if(bank.level>=START)atomicOr(mask,uint32_t(1)<<m.bank);
        return;
    }
    int tile=1;while(r>c.upper*tile)tile*=2;
    double v=log(r/(c.upper/c.ratio*tile))/log(c.ratio),w=smooth(v),wr=slope(v)/(r*log(c.ratio));
    int slot=0;
    for(int si=0;si<2;++si) {
        int t=tile<<si;double sw=si?w:1-w,sr=si?wr:-wr;
        if(sw==0&&sr==0)continue;
        double q=(mu+.5)/t-1.5;int first=int(floor(q));double u=(q-first-(.5-c.delta))/(2*c.delta),pw=smooth(u),pm=slope(u)/(2*c.delta*t);
        for(int wi=0;wi<2;++wi) {
            double wp=wi?pw:1-pw,wm=wi?pm:-pm;
            if(wp==0&&wm==0)continue;
            fill_branch(base+slot*G,c,t,first+wi,3,r,mu,sw*wp,sr*wp,sw*wm);++slot;
        }
        if(t>=16)atomicOr(mask,uint32_t(1)<<(2*(__ffs(t)-1)+1));
    }
}
// Shared moments on immutable tiles. Inactive banks are masked on device.
__global__ void moments(const float *x,float *z,const uint32_t *mask,int batch,Conf c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x,F=c.a.basis_count();
    int64_t total=int64_t(batch)*c.nodes*F;if(idx>=total)return;
    int f=idx%F,node=c.prefix+(idx/F)%c.nodes,b=idx/(F*c.nodes);
    int bi=0;while(bi+1<c.a.bank_count&&node>=c.a.banks[bi+1].node_offset)++bi;
    if(!(*mask&(uint32_t(1)<<bi))){z[idx]=0;return;}
    auto bank=c.a.banks[bi];int tile=node-bank.node_offset,k=(f+1)/2;
    float sum=0,omega=float(2*PI*k/bank.period),filter=k?float(sin(PI*k/bank.period)/(PI*k/bank.period)):1;
    float stepc=cosf(omega),steps=sinf(omega),cp=1,sp=0;
    for(int j=0;j<bank.tile_sites;++j) {
        sum+=x[int64_t(b)*c.a.sites+tile*bank.tile_sites+j]*(f==0?1:(f%2?cp:sp))*filter;
        float next=cp*stepc-sp*steps;sp=sp*stepc+cp*steps;cp=next;
    }
    z[idx]=sum;
}
__device__ void project(const float *x,const float *z,const float *g,int b,Conf c,float &v,float &r,float &m) {
    v=r=m=0;int F=c.a.basis_count();
    for(int slot=0;slot<4;++slot) {
        const float *p=g+slot*G;int t=int(p[0]);if(!t)continue;
        int first=int(p[1]),bi=int(p[2]),width=int(p[3]),count=int(p[4]);
        if(t<16) {
            for(int j=0;j<width;++j) {
                float signal=x[int64_t(b)*c.a.sites+positive_mod(first*t+j,c.a.sites)];
                v+=signal*p[5+j];r+=signal*p[5+width+j];m+=signal*p[5+2*width+j];
            }
        } else {
            auto bank=c.a.banks[bi];
            for(int part=0;part<count;++part) {
                int node=bank.node_offset+positive_mod(first+part,bank.tile_count);
                const float *mom=z+(int64_t(b)*c.nodes+node-c.prefix)*F;
                v+=p[5]*mom[0];r+=p[5+width]*mom[0];m+=p[5+2*width]*mom[0];
                for(int k=1;k<=c.a.modes;++k) {
                    float angle=float(2*PI*k*part/count),cp=cosf(angle),sp=sinf(angle);
                    float mc=cp*mom[2*k-1]-sp*mom[2*k],ms=sp*mom[2*k-1]+cp*mom[2*k];
                    v+=p[5+2*k-1]*mc+p[5+2*k]*ms;
                    r+=p[5+width+2*k-1]*mc+p[5+width+2*k]*ms;
                    m+=p[5+2*width+2*k-1]*mc+p[5+2*width+2*k]*ms;
                }
            }
        }
    }
}
__global__ void response(const float*x,const float*z,const float*g,float*h,int atoms,int batch,Conf c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=atoms*batch)return;
    float v,r,m;project(x,z,g+int64_t(idx%atoms)*4*G,idx/atoms,c,v,r,m);h[idx]=v;
}
__global__ void coefficients(const float*p,const float*g,const float*h,float*y,float*C,int atoms,int batch,Conf c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=atoms*batch)return;
    int a=idx%atoms,b=idx/atoms,F=c.a.basis_count();float value=p[4*a]*h[idx];
    for(int slot=0;slot<4;++slot) {
        const float *geo=g+(int64_t(a)*4+slot)*G;int t=int(geo[0]);if(!t)continue;
        int first=int(geo[1]),bi=int(geo[2]),width=int(geo[3]),count=int(geo[4]);
        if(t<16) {
            for(int j=0;j<width;++j)atomicAdd(y+int64_t(b)*c.a.sites+positive_mod(first*t+j,c.a.sites),value*geo[5+j]);
        } else {
            auto bank=c.a.banks[bi];
            for(int part=0;part<count;++part) {
                int node=bank.node_offset+positive_mod(first+part,bank.tile_count);
                float *out=C+(int64_t(b)*c.nodes+node-c.prefix)*F;
                atomicAdd(out,value*geo[5]);
                for(int k=1;k<=c.a.modes;++k) {
                    float angle=float(2*PI*k*part/count),cp=cosf(angle),sp=sinf(angle),cc=geo[5+2*k-1],ss=geo[5+2*k];
                    atomicAdd(out+2*k-1,value*(cc*cp+ss*sp));atomicAdd(out+2*k,value*(-cc*sp+ss*cp));
                }
            }
        }
    }
}
__global__ void expand(const float*C,float*y,const uint32_t*mask,int batch,Conf c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=batch*c.a.sites)return;
    int b=idx/c.a.sites,site=idx%c.a.sites,F=c.a.basis_count();float sum=0;
    for(int bi=START*2;bi<c.a.bank_count;++bi) {
        if(!(*mask&(uint32_t(1)<<bi)))continue;
        auto bank=c.a.banks[bi];int node=bank.node_offset+site/bank.tile_sites,j=site%bank.tile_sites;
        const float *coef=C+(int64_t(b)*c.nodes+node-c.prefix)*F;
        sum+=coef[0];
        for(int k=1;k<=c.a.modes;++k) {
            float angle=float(2*PI*k*j/bank.period),arg=float(PI*k/bank.period),filter=sinf(arg)/arg;
            sum+=filter*(coef[2*k-1]*cosf(angle)+coef[2*k]*sinf(angle));
        }
    }
    y[idx]+=sum;
}
__global__ void gradients(const float *p,const float *x,const float *dy,const float *zx,const float *zy,const float *gi,const float *go,float *dp,int atoms,int batch,Conf ci,Conf co) {
    int a=blockIdx.x,b=threadIdx.x;float ga=0,gr=0,gm=0,gn=0;
    if(b<batch) {
        float h,hr,hm,g,grr,gmout;
        project(x,zx,gi+int64_t(a)*4*G,b,ci,h,hr,hm);
        project(dy,zy,go+int64_t(a)*4*G,b,co,g,grr,gmout);
        ga=h*g;gr=p[4*a]*(hr*g+h*grr);gm=p[4*a]*h*gmout;gn=p[4*a]*hm*g;
    }
    __shared__ float sums[4][256];sums[0][b]=ga;sums[1][b]=gr;sums[2][b]=gm;sums[3][b]=gn;__syncthreads();
    for(int stride=128;stride;stride>>=1){if(b<stride)for(int j=0;j<4;++j)sums[j][b]+=sums[j][b+stride];__syncthreads();}
    if(b==0)for(int j=0;j<4;++j)dp[4*a+j]=sums[j][0];
}
void check(Tensor x,Tensor p,int ni,int no,int m) {
    TORCH_CHECK(x.is_cuda()&&p.is_cuda()&&x.device()==p.device()&&x.scalar_type()==torch::kFloat&&p.scalar_type()==torch::kFloat&&x.is_contiguous()&&p.is_contiguous(),"CUDA contiguous FP32 required");
    TORCH_CHECK(x.dim()==2&&p.dim()==2&&p.size(1)==4&&p.size(0)>0&&p.size(0)<=131072&&x.size(1)==ni&&x.size(0)>0&&x.size(0)<=256,"shape outside qualified cohort");
    TORCH_CHECK(ni>=16&&no>=16&&ni<=16384&&no<=16384&&(m==8||m==16),"unsupported chart/modes");
}
Tensor make_moments(Tensor x,Tensor mask,Conf c) {
    auto z=torch::empty({x.size(0),c.nodes,c.a.basis_count()},x.options());
    moments<<<(z.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),z.data_ptr<float>(),mask.data_ptr<uint32_t>(),x.size(0),c);
    return z;
}
Tensor get_response(Tensor x,Tensor z,Tensor g,int atoms,Conf c) {
    auto h=torch::empty({x.size(0),atoms},x.options());
    response<<<(h.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),z.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),atoms,x.size(0),c);
    return h;
}
Tensor expansion(Tensor p,Tensor g,Tensor h,Tensor mask,Conf c) {
    auto y=torch::zeros({h.size(0),c.a.sites},h.options()),C=torch::zeros({h.size(0),c.nodes,c.a.basis_count()},h.options());
    coefficients<<<(h.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),y.data_ptr<float>(),C.data_ptr<float>(),p.size(0),h.size(0),c);
    expand<<<(y.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(C.data_ptr<float>(),y.data_ptr<float>(),mask.data_ptr<uint32_t>(),h.size(0),c);
    return y;
}
std::vector<Tensor> forward(Tensor x,Tensor p,int ni,int no,int m,bool connected,std::vector<double>s) {
    check(x,p,ni,no,m);c10::cuda::CUDAGuard guard(x.device());auto ci=config(ni,m,connected,s),co=config(no,m,connected,s);
    auto gi=torch::empty({p.size(0),4,G},p.options()),go=torch::empty_like(gi),mi=torch::zeros({1},p.options().dtype(torch::kUInt32)),mo=torch::zeros_like(mi);
    auto stream=at::cuda::getCurrentCUDAStream();
    geometry<<<(p.size(0)+63)/64,64,0,stream>>>(p.data_ptr<float>(),gi.data_ptr<float>(),mi.data_ptr<uint32_t>(),p.size(0),ci,0);
    geometry<<<(p.size(0)+63)/64,64,0,stream>>>(p.data_ptr<float>(),go.data_ptr<float>(),mo.data_ptr<uint32_t>(),p.size(0),co,1);
    auto z=make_moments(x,mi,ci),h=get_response(x,z,gi,p.size(0),ci),y=expansion(p,go,h,mo,co);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,gi,go,mi,mo};
}
std::vector<Tensor> backward(Tensor x,Tensor p,Tensor dy,Tensor gi,Tensor go,Tensor mi,Tensor mo,int ni,int no,int m,bool connected,std::vector<double>s) {
    check(x,p,ni,no,m);c10::cuda::CUDAGuard guard(x.device());auto ci=config(ni,m,connected,s),co=config(no,m,connected,s);
    auto zx=make_moments(x,mi,ci),zy=make_moments(dy,mo,co),g=get_response(dy,zy,go,p.size(0),co),dx=expansion(p,gi,g,mi,ci),dp=torch::empty_like(p);
    gradients<<<p.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),x.data_ptr<float>(),dy.data_ptr<float>(),zx.data_ptr<float>(),zy.data_ptr<float>(),gi.data_ptr<float>(),go.data_ptr<float>(),dp.data_ptr<float>(),p.size(0),x.size(0),ci,co);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dp};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&connected::forward);m.def("backward",&connected::backward);}
