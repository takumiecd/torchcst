#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../radius_cell_mass/common.cuh"
#include "../connected_cell_mass_reuse/profile_m16.cuh"
using torch::Tensor;
using namespace radius_cell_mass;
namespace connected_coalesced {
using connected_reuse::coefficients_m16;
constexpr int G=104, START=4;
constexpr float PI_F=3.14159265358979323846f;
struct Conf { Axis a; bool connect; double delta,upper,ratio; int prefix,nodes; };
Conf config(int n,int m,bool connect,std::vector<double> s) {
    Conf c{}; c.a=parse_axis({double(n),1.,1.,0.,double(m),2.});
    c.connect=connect; c.delta=s[0];c.upper=s[1];c.ratio=s[2];
    c.prefix=c.a.banks[2*START].node_offset;
    c.nodes=c.a.node_count-c.prefix;
    return c;
}
__device__ __forceinline__ float smooth32(float x) {
    x=fminf(1.f,fmaxf(0.f,x));float u=1-x;
    return x>.5f ? 1-u*u*u*(10+u*(-15+6*u)) : x*x*x*(10+x*(-15+6*x));
}
__device__ __forceinline__ float slope32(float x) {return x>0&&x<1 ? 30*x*x*(1-x)*(1-x):0;}
__device__ void fill32(float *g,int tile,int first,float mu,float w,float wr,float wm,const float *cv,const float *dr) {
    int bi=2*(__ffs(tile)-1)+1,width=tile<16?3*tile:33;
    g[0]=tile;g[1]=first;g[2]=bi;g[3]=width;g[4]=3;
    float period=3.f*tile;
    if(tile<16) {
        for(int j=0;j<width;++j) {
            float value=cv[0],rd=0,md=0;
            for(int k=1;k<=16;++k) {
                float omega=2*PI_F*k/period,arg=PI_F*k/period;
                float sn,cs;sincosf(omega*(mu-(first*tile+j)),&sn,&cs);
                float filter=sinf(arg)/arg;
                value+=2*cv[k]*filter*cs;rd+=2*dr[k]*filter*cs;md-=2*cv[k]*filter*omega*sn;
            }
            g[5+j]=w*value;g[5+width+j]=w*rd+wr*value;g[5+2*width+j]=w*md+wm*value;
        }
    } else {
        g[5]=w*cv[0];g[5+width]=wr*cv[0];g[5+2*width]=wm*cv[0];
        for(int k=1;k<=16;++k) {
            float omega=2*PI_F*k/period,cs,sn;sincosf(omega*(mu-first*tile),&sn,&cs);
            g[5+2*k-1]=2*w*cv[k]*cs;g[5+2*k]=2*w*cv[k]*sn;
            g[5+width+2*k-1]=2*(w*dr[k]+wr*cv[k])*cs;g[5+width+2*k]=2*(w*dr[k]+wr*cv[k])*sn;
            g[5+2*width+2*k-1]=2*cv[k]*(wm*cs-w*omega*sn);g[5+2*width+2*k]=2*cv[k]*(wm*sn+w*omega*cs);
        }
    }
}
// The density coefficients depend only on R/T, not on center or chart side.
// Compute once per active scale, reuse across both windows AND both axes.
__global__ void geometry32(const float *p,float *gi,float *go,uint32_t *mi,uint32_t *mo,int atoms,Conf ci,Conf co) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=atoms)return;
    float r=p[4*a+1];bool valid=isfinite(r)&&r>0&&r<=.5f*ci.a.sites&&r<=.5f*co.a.sites;
    for(int j=0;j<4;++j)valid&=isfinite(p[4*a+j]);CUDA_KERNEL_ASSERT(valid);
    float *gis=gi+int64_t(a)*4*G,*gos=go+int64_t(a)*4*G;
    for(int j=0;j<4;++j){gis[j*G]=0;gos[j*G]=0;}
    int tile=1;while(r>.9f*tile)tile*=2;
    float v=logf(r/(.72f*tile))/logf(1.25f),w=smooth32(v),wr=slope32(v)/(r*logf(1.25f));
    int slots[2]={0,0};
    for(int si=0;si<2;++si){
        int t=tile<<si;float sw=si?w:1-w,sr=si?wr:-wr;if(sw==0&&sr==0)continue;
        float cv[17],dr[17];coefficients_m16(r,3.f*t,cv,dr);
        for(int side=0;side<2;++side){
            int n=side?co.a.sites:ci.a.sites;float mu=fmodf(p[4*a+(side?2:3)],float(n));if(mu<0)mu+=n;
            float q=(mu+.5f)/t-1.5f;int first=int(floorf(q));
            float u=(q-first-.4f)/.2f,pw=smooth32(u),pm=slope32(u)/(.2f*t);
            for(int wi=0;wi<2;++wi){
                float wp=wi?pw:1-pw,wm=wi?pm:-pm;if(wp==0&&wm==0)continue;
                float *g=(side?gos:gis)+slots[side]++*G;
                fill32(g,t,first+wi,mu,sw*wp,sr*wp,sw*wm,cv,dr);
            }
            if(t>=16)atomicOr(side?mo:mi,uint32_t(1)<<(2*(__ffs(t)-1)+1));
        }
    }
}

// K=3 rotations are exact constants, independent of atom, radius and row.
__device__ __forceinline__ void rotation(int k,int part,float &cs,float &sn) {
    int q=(k*part)%3;cs=q?-.5f:1.f;sn=q==1?.8660254037844386f:q==2?-.8660254037844386f:0.f;
}
__global__ void close_mask(uint32_t *mask,Conf c) {
    uint32_t v=*mask;
    for(int bi=c.a.bank_count-1;bi>=2*START;--bi)if(v&(uint32_t(1)<<bi))
        for(int child=bi-2;child>=2*START;child-=2)v|=uint32_t(1)<<child;
    *mask=v;
}
// X and shared moments have contiguous rows: X[site,b], Z[node,f,b].
// Each warp shares geometry and loads adjacent rows rather than strided sites.
template<bool Derivatives>
__device__ __forceinline__ void project(const float*x,const float*z,const float*g,int b,int batch,Conf c,float &v,float &r,float &m) {
    v=r=m=0;
    for(int slot=0;slot<4;++slot){
        const float*p=g+slot*G;int t=int(p[0]);if(!t)continue;
        int first=int(p[1]),bi=int(p[2]),width=int(p[3]);
        if(t<16){
            for(int j=0;j<width;++j){
                float xj=x[int64_t(positive_mod(first*t+j,c.a.sites))*batch+b];
                v+=xj*p[5+j];
                if constexpr(Derivatives){r+=xj*p[5+width+j];m+=xj*p[5+2*width+j];}
            }
        }else{
            auto bank=c.a.banks[bi];
            for(int part=0;part<3;++part){
                int node=bank.node_offset+positive_mod(first+part,bank.tile_count)-c.prefix;
                const float*mom=z+int64_t(node)*33*batch+b;
                float dc=mom[0];v+=p[5]*dc;
                if constexpr(Derivatives){r+=p[5+width]*dc;m+=p[5+2*width]*dc;}
                for(int k=1;k<=16;++k){
                    float cs,sn;rotation(k,part,cs,sn);
                    float cc=mom[int64_t(2*k-1)*batch],ss=mom[int64_t(2*k)*batch];
                    float mc=cs*cc-sn*ss,ms=sn*cc+cs*ss;
                    v+=p[5+2*k-1]*mc+p[5+2*k]*ms;
                    if constexpr(Derivatives){
                        r+=p[5+width+2*k-1]*mc+p[5+width+2*k]*ms;
                        m+=p[5+2*width+2*k-1]*mc+p[5+2*width+2*k]*ms;
                    }
                }
            }
        }
    }
}
__global__ void response(const float*x,const float*z,const float*g,float*h,int atoms,int batch,Conf c){
    int a=blockIdx.x,b=threadIdx.x;if(b>=batch)return;
    float v,r,m;project<false>(x,z,g+int64_t(a)*4*G,b,batch,c,v,r,m);
    h[int64_t(a)*batch+b]=v;
}
__global__ void gradients(const float*p,const float*x,const float*dy,const float*zx,const float*zy,const float*gi,const float*go,const float*cached_g,float*dp,int atoms,int batch,Conf ci,Conf co){
    int a=blockIdx.x,b=threadIdx.x;float sums[4]={};
    if(b<batch){
        float h,hr,hm,g,gr,gm;
        project<true>(x,zx,gi+int64_t(a)*4*G,b,batch,ci,h,hr,hm);
        project<true>(dy,zy,go+int64_t(a)*4*G,b,batch,co,g,gr,gm);
        g=cached_g[int64_t(a)*batch+b];
        sums[0]=h*g;sums[1]=p[4*a]*(hr*g+h*gr);sums[2]=p[4*a]*h*gm;sums[3]=p[4*a]*hm*g;
    }
    __shared__ float partial[4][8];
    for(int j=0;j<4;++j){
        for(int d=16;d;d>>=1)sums[j]+=__shfl_down_sync(0xffffffff,sums[j],d);
        if((b&31)==0)partial[j][b/32]=sums[j];
    }
    __syncthreads();
    if(b<32)for(int j=0;j<4;++j){
        float value=b<8?partial[j][b]:0;
        for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
        if(b==0)dp[4*a+j]=value;
    }
}
// DC/even harmonics inherit child moments; batch lanes remain contiguous.
__global__ void level_moments(const float*x,float*z,const uint32_t*mask,int batch,Conf c,int level){
    int bi=2*level+1;if(!(*mask&(uint32_t(1)<<bi)))return;
    int b=blockIdx.y*32+threadIdx.x,basis=blockIdx.z*4+threadIdx.y;
    if(b>=batch||basis>=33)return;
    auto bank=c.a.banks[bi];int tile=blockIdx.x,node=bank.node_offset+tile,k=(basis+1)/2;
    float sum=0;
    if(level>START&&k%2==0){
        auto child=c.a.banks[bi-2];
        for(int part=0;part<2;++part){
            const float*mom=z+int64_t(child.node_offset+tile*2+part-c.prefix)*33*batch+b;
            if(basis==0){sum+=mom[0];continue;}
            float cs,sn;rotation(k/2,part,cs,sn);
            float cc=mom[int64_t(2*(k/2)-1)*batch],ss=mom[int64_t(2*(k/2))*batch];
            sum+=basis%2?cs*cc-sn*ss:sn*cc+cs*ss;
        }
    }else{
        float omega=2*PI_F*k/float(bank.period),arg=PI_F*k/float(bank.period),filter=k?sinf(arg)/arg:1;
        float stepc,steps;sincosf(omega,&steps,&stepc);float cs=1,sn=0;
        for(int j=0;j<bank.tile_sites;++j){
            sum+=x[int64_t(tile*bank.tile_sites+j)*batch+b]*(basis==0?1:(basis%2?cs:sn))*filter;
            float next=cs*stepc-sn*steps;sn=sn*stepc+cs*steps;cs=next;
        }
    }
    z[(int64_t(node-c.prefix)*33+basis)*batch+b]=sum;
}
__global__ void owner_keys(const float*g,int64_t*keys,int atoms,Conf c) {
    int v=threadIdx.x+blockIdx.x*blockDim.x;if(v>=4*atoms)return;
    const float *p=g+int64_t(v)*G;int t=int(p[0]);
    keys[v]=t?c.a.banks[int(p[2])].node_offset+positive_mod(int(p[1]),c.a.sites/t):c.a.node_count;
}
__global__ void owner_expand(const float*p,const float*g,const float*h,const int64_t*rows,const int64_t*offsets,float*y,float*C,const uint32_t*mask,int atoms,int batch,Conf c) {
    int node=blockIdx.x,b0=blockIdx.y*8,bi=0;
    while(bi+1<c.a.bank_count&&node>=c.a.banks[bi+1].node_offset)++bi;
    auto bank=c.a.banks[bi];int t=bank.tile_sites,tile=node-bank.node_offset;
    if(bank.count!=3)return;
    if(bank.level>=START&&!(*mask&(uint32_t(1)<<bi)))return;
    int width=t<16?t:33;
    for(int task=threadIdx.x;task<8*width;task+=blockDim.x){
        int b=b0+task/width,f=task%width;if(b>=batch)continue;float value=0;
        for(int part=0;part<3;++part){
            int first_node=bank.node_offset+positive_mod(tile-part,bank.tile_count);
            float cs=1,sn=0;if(t>=16&&f)rotation((f+1)/2,part,cs,sn);
            for(int64_t index=offsets[first_node];index<offsets[first_node+1];++index){
                int v=int(rows[index]),a=v/4;const float *geo=g+int64_t(v)*G;
                float cc;
                if(t<16)cc=geo[5+part*t+f];
                else if(f==0)cc=geo[5];
                else{int k=(f+1)/2;float a0=geo[5+2*k-1],a1=geo[5+2*k];cc=f%2?a0*cs+a1*sn:-a0*sn+a1*cs;}
                value+=(p[4*a]*h[int64_t(a)*batch+b])*cc;
            }
        }
        if(t<16)atomicAdd(y+int64_t(b)*c.a.sites+tile*t+f,value);
        else C[(int64_t(b)*c.nodes+node-c.prefix)*33+f]=value;
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
            float angle=float(2*PI_F*k*j/float(bank.period)),arg=float(PI_F*k/float(bank.period)),filter=sinf(arg)/arg;
            sum+=filter*(coef[2*k-1]*cosf(angle)+coef[2*k]*sinf(angle));
        }
    }
    y[idx]+=sum;
}
void check(Tensor x,Tensor p,int ni,int no,int m) {
    TORCH_CHECK(x.is_cuda()&&p.is_cuda()&&x.device()==p.device()&&x.scalar_type()==torch::kFloat&&p.scalar_type()==torch::kFloat&&x.is_contiguous()&&p.is_contiguous(),"CUDA contiguous FP32 required");
    TORCH_CHECK(x.dim()==2&&p.dim()==2&&p.size(1)==4&&p.size(0)>0&&p.size(0)<=131072&&x.size(1)==ni&&x.size(0)>0&&x.size(0)<=256,"shape outside qualified cohort");
    TORCH_CHECK(ni>=16&&no>=16&&ni<=16384&&no<=16384&&(m==8||m==16),"unsupported chart/modes");
}
std::vector<Tensor> ranges(Tensor geometry,int atoms,Conf c) {
    auto keys=torch::empty({4LL*atoms},geometry.options().dtype(torch::kLong));
    owner_keys<<<(keys.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(geometry.data_ptr<float>(),keys.data_ptr<int64_t>(),atoms,c);
    auto sorted=at::sort(keys,true,-1,false);
    auto boundaries=torch::arange(c.a.node_count+1,keys.options());
    auto offsets=torch::searchsorted(std::get<0>(sorted),boundaries);
    return {std::get<1>(sorted),offsets};
}
Tensor moments(Tensor xt,Tensor mask,Conf c){
    int batch=xt.size(1);auto z=torch::empty({c.nodes,33,batch},xt.options());
    for(int level=START;level<c.a.bank_count/2;++level){
        dim3 grid(c.a.banks[2*level+1].tile_count,(batch+31)/32,9);
        level_moments<<<grid,dim3(32,4),0,at::cuda::getCurrentCUDAStream()>>>(xt.data_ptr<float>(),z.data_ptr<float>(),mask.data_ptr<uint32_t>(),batch,c,level);
    }
    return z;
}
Tensor get_response(Tensor xt,Tensor z,Tensor geometry,int atoms,Conf c){
    int batch=xt.size(1);auto h=torch::empty({atoms,batch},xt.options());
    response<<<atoms,256,0,at::cuda::getCurrentCUDAStream()>>>(xt.data_ptr<float>(),z.data_ptr<float>(),geometry.data_ptr<float>(),h.data_ptr<float>(),atoms,batch,c);
    return h;
}
Tensor expansion(Tensor p,Tensor g,Tensor h,Tensor mask,Tensor rows,Tensor offsets,Conf c){
    int batch=h.size(1);auto y=torch::zeros({batch,c.a.sites},h.options()),C=torch::empty({batch,c.nodes,33},h.options());
    owner_expand<<<dim3(c.a.node_count,(batch+7)/8),128,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),rows.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),y.data_ptr<float>(),C.data_ptr<float>(),mask.data_ptr<uint32_t>(),p.size(0),batch,c);
    expand<<<(y.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(C.data_ptr<float>(),y.data_ptr<float>(),mask.data_ptr<uint32_t>(),batch,c);
    return y;
}
std::vector<Tensor> forward(Tensor x,Tensor p,int ni,int no,int m,bool connected,std::vector<double>s,int stage){
    check(x,p,ni,no,m);TORCH_CHECK(stage==0&&m==16&&connected&&s==std::vector<double>({.1,.9,1.25}),"requires default coalesced M16");
    c10::cuda::CUDAGuard guard(x.device());auto ci=config(ni,m,connected,s),co=config(no,m,connected,s);
    auto gi=torch::empty({p.size(0),4,G},p.options()),go=torch::empty_like(gi),mi=torch::zeros({1},p.options().dtype(torch::kUInt32)),mo=torch::zeros_like(mi);
    auto stream=at::cuda::getCurrentCUDAStream();
    geometry32<<<(p.size(0)+63)/64,64,0,stream>>>(p.data_ptr<float>(),gi.data_ptr<float>(),go.data_ptr<float>(),mi.data_ptr<uint32_t>(),mo.data_ptr<uint32_t>(),p.size(0),ci,co);
    close_mask<<<1,1,0,stream>>>(mi.data_ptr<uint32_t>(),ci);close_mask<<<1,1,0,stream>>>(mo.data_ptr<uint32_t>(),co);
    auto ri=ranges(gi,p.size(0),ci),ro=ranges(go,p.size(0),co);
    auto xt=x.transpose(0,1).contiguous();auto z=moments(xt,mi,ci),h=get_response(xt,z,gi,p.size(0),ci);
    auto y=expansion(p,go,h,mo,ro[0],ro[1],co);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,gi,go,mi,mo,ri[0],ri[1],ro[0],ro[1]};
}
std::vector<Tensor> backward(Tensor x,Tensor p,Tensor dy,Tensor gi,Tensor go,Tensor mi,Tensor mo,Tensor rowsi,Tensor offsetsi,Tensor rowso,Tensor offsetso,int ni,int no,int m,bool connected,std::vector<double>s,int stage){
    check(x,p,ni,no,m);c10::cuda::CUDAGuard guard(x.device());auto ci=config(ni,m,connected,s),co=config(no,m,connected,s);
    auto xt=x.transpose(0,1).contiguous(),yt=dy.transpose(0,1).contiguous();
    auto zx=moments(xt,mi,ci),zy=moments(yt,mo,co),g=get_response(yt,zy,go,p.size(0),co);
    auto dx=expansion(p,gi,g,mi,rowsi,offsetsi,ci),dp=torch::empty_like(p);
    gradients<<<p.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),xt.data_ptr<float>(),yt.data_ptr<float>(),zx.data_ptr<float>(),zy.data_ptr<float>(),gi.data_ptr<float>(),go.data_ptr<float>(),g.data_ptr<float>(),dp.data_ptr<float>(),p.size(0),x.size(0),ci,co);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dp};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&connected_coalesced::forward);m.def("backward",&connected_coalesced::backward);}
