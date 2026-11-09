#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../radius_cell_mass/common.cuh"
#include "profile_m16.cuh"
using torch::Tensor;
using namespace radius_cell_mass;
namespace connected_reuse {
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

// Shared moments on immutable tiles. Inactive banks are masked on device.
__global__ void moments(const float *x,float *z,const uint32_t *mask,int batch,Conf c) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x,F=c.a.basis_count();
    int64_t total=int64_t(batch)*c.nodes*F;if(idx>=total)return;
    int f=idx%F,node=c.prefix+(idx/F)%c.nodes,b=idx/(F*c.nodes);
    int bi=0;while(bi+1<c.a.bank_count&&node>=c.a.banks[bi+1].node_offset)++bi;
    if(!(*mask&(uint32_t(1)<<bi))){z[idx]=0;return;}
    auto bank=c.a.banks[bi];int tile=node-bank.node_offset,k=(f+1)/2;
    float sum=0,omega=float(2*PI_F*k/float(bank.period)),filter=k?float(sin(PI_F*k/float(bank.period))/(PI_F*k/float(bank.period))):1;
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
                    float angle=float(2*PI_F*k*part/count),cp=cosf(angle),sp=sinf(angle);
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
                    float angle=float(2*PI_F*k*part/count),cp=cosf(angle),sp=sinf(angle),cc=geo[5+2*k-1],ss=geo[5+2*k];
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
            float angle=float(2*PI_F*k*j/float(bank.period)),arg=float(PI_F*k/float(bank.period)),filter=sinf(arg)/arg;
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
// K=3 rotations are exact constants, independent of atom, radius and row.
__device__ __forceinline__ void rotation(int k,int part,float &cs,float &sn) {
    int q=(k*part)%3;cs=q?-.5f:1.f;sn=q==1?.8660254037844386f:q==2?-.8660254037844386f:0.f;
}
// One thread reuses a single atom's geometry across eight complete rows.
__global__ void response8(const float*x,const float*z,const float*g,float*h,int atoms,int batch,Conf c) {
    int a=blockIdx.x*blockDim.x+threadIdx.x,b0=blockIdx.y*8;if(a>=atoms)return;
    float value[8]={};constexpr int F=33;
    for(int slot=0;slot<4;++slot){
        const float *geo=g+(int64_t(a)*4+slot)*G;int t=int(geo[0]);if(!t)continue;
        int first=int(geo[1]),bi=int(geo[2]),width=int(geo[3]);
        if(t<16){
            for(int j=0;j<width;++j){
                int site=positive_mod(first*t+j,c.a.sites);float mass=geo[5+j];
                #pragma unroll
                for(int r=0;r<8;++r)if(b0+r<batch)value[r]+=mass*x[int64_t(b0+r)*c.a.sites+site];
            }
        }else{
            auto bank=c.a.banks[bi];
            for(int part=0;part<3;++part){
                int node=bank.node_offset+positive_mod(first+part,bank.tile_count)-c.prefix;
                float dc=geo[5];
                #pragma unroll
                for(int r=0;r<8;++r)if(b0+r<batch)value[r]+=dc*z[(int64_t(b0+r)*c.nodes+node)*F];
                for(int k=1;k<=16;++k){
                    float cs,sn;rotation(k,part,cs,sn);
                    float a0=geo[5+2*k-1],a1=geo[5+2*k];
                    float cc=a0*cs+a1*sn,ss=-a0*sn+a1*cs;
                    #pragma unroll
                    for(int r=0;r<8;++r)if(b0+r<batch){
                        const float *mom=z+(int64_t(b0+r)*c.nodes+node)*F;
                        value[r]+=cc*mom[2*k-1]+ss*mom[2*k];
                    }
                }
            }
        }
    }
    #pragma unroll
    for(int r=0;r<8;++r)if(b0+r<batch)h[int64_t(b0+r)*atoms+a]=value[r];
}
__device__ __forceinline__ void project_fast(const float*x,const float*z,const float*g,int b,Conf c,float &v,float &r,float &m) {
    v=r=m=0;
    for(int slot=0;slot<4;++slot){
        const float *p=g+slot*G;int t=int(p[0]);if(!t)continue;
        int first=int(p[1]),bi=int(p[2]),width=int(p[3]);
        if(t<16){
            for(int j=0;j<width;++j){
                float xj=x[int64_t(b)*c.a.sites+positive_mod(first*t+j,c.a.sites)];
                v+=xj*p[5+j];r+=xj*p[5+width+j];m+=xj*p[5+2*width+j];
            }
        }else{
            auto bank=c.a.banks[bi];
            for(int part=0;part<3;++part){
                int node=bank.node_offset+positive_mod(first+part,bank.tile_count)-c.prefix;
                const float *mom=z+(int64_t(b)*c.nodes+node)*33;
                v+=p[5]*mom[0];r+=p[5+width]*mom[0];m+=p[5+2*width]*mom[0];
                for(int k=1;k<=16;++k){
                    float cs,sn;rotation(k,part,cs,sn);
                    float mc=cs*mom[2*k-1]-sn*mom[2*k],ms=sn*mom[2*k-1]+cs*mom[2*k];
                    v+=p[5+2*k-1]*mc+p[5+2*k]*ms;
                    r+=p[5+width+2*k-1]*mc+p[5+width+2*k]*ms;
                    m+=p[5+2*width+2*k-1]*mc+p[5+2*width+2*k]*ms;
                }
            }
        }
    }
}
__global__ void gradients_reuse(const float *p,const float *x,const float *dy,const float *zx,const float *zy,const float *gi,const float *go,float *dp,int atoms,int batch,Conf ci,Conf co) {
    int a=blockIdx.x,b=threadIdx.x;float ga=0,gr=0,gm=0,gn=0;
    if(b<batch){
        float h,hr,hm,g,grr,gmout;
        project_fast(x,zx,gi+int64_t(a)*4*G,b,ci,h,hr,hm);
        project_fast(dy,zy,go+int64_t(a)*4*G,b,co,g,grr,gmout);
        ga=h*g;gr=p[4*a]*(hr*g+h*grr);gm=p[4*a]*h*gmout;gn=p[4*a]*hm*g;
    }
    __shared__ float sums[4][256];sums[0][b]=ga;sums[1][b]=gr;sums[2][b]=gm;sums[3][b]=gn;__syncthreads();
    for(int stride=128;stride;stride>>=1){if(b<stride)for(int j=0;j<4;++j)sums[j][b]+=sums[j][b+stride];__syncthreads();}
    if(b==0)for(int j=0;j<4;++j)dp[4*a+j]=sums[j][0];
}
__global__ void close_mask(uint32_t *mask,Conf c) {
    uint32_t v=*mask;
    for(int bi=c.a.bank_count-1;bi>=2*START;--bi)if(v&(uint32_t(1)<<bi))
        for(int child=bi-2;child>=2*START;child-=2)v|=uint32_t(1)<<child;
    *mask=v;
}
// Inherit DC/even harmonics from child tiles; only unmatched frequencies read X.
__global__ void level_moments(const float*x,float*z,const uint32_t*mask,int batch,Conf c,int level) {
    if(!(*mask&(uint32_t(3)<<(2*level))))return;
    int f=threadIdx.x+blockIdx.x*blockDim.x,F=33,tiles=c.a.banks[2*level].tile_count,nodes=2*tiles;
    if(f>=batch*nodes*F)return;
    int basis=f%F,local=(f/F)%nodes,b=f/(F*nodes),bi=2*level+(local>=tiles),tile=local%tiles;
    if(!(*mask&(uint32_t(1)<<bi)))return;
    auto bank=c.a.banks[bi];int node=bank.node_offset+tile,k=(basis+1)/2;
    float sum=0;
    if(level>START&&k%2==0){
        auto child=c.a.banks[bi-2];
        for(int part=0;part<2;++part){
            const float *mom=z+(int64_t(b)*c.nodes+child.node_offset+tile*2+part-c.prefix)*F;
            if(basis==0){sum+=mom[0];continue;}
            float cs,sn;rotation(k/2,part,cs,sn);
            float cc=mom[2*(k/2)-1],ss=mom[2*(k/2)];
            sum+=basis%2?cs*cc-sn*ss:sn*cc+cs*ss;
        }
    }else{
        float omega=2*PI_F*k/float(bank.period),filter=k?sinf(PI_F*k/float(bank.period))/(PI_F*k/float(bank.period)):1;
        float stepc,steps;sincosf(omega,&steps,&stepc);float cs=1,sn=0;
        for(int j=0;j<bank.tile_sites;++j){
            sum+=x[int64_t(b)*c.a.sites+tile*bank.tile_sites+j]*(basis==0?1:(basis%2?cs:sn))*filter;
            float next=cs*stepc-sn*steps;sn=sn*stepc+cs*steps;cs=next;
        }
    }
    z[(int64_t(b)*c.nodes+node-c.prefix)*F+basis]=sum;
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
                value+=(p[4*a]*h[int64_t(b)*atoms+a])*cc;
            }
        }
        if(t<16)atomicAdd(y+int64_t(b)*c.a.sites+tile*t+f,value);
        else C[(int64_t(b)*c.nodes+node-c.prefix)*33+f]=value;
    }
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
std::vector<Tensor> ranges(Tensor geometry,int atoms,Conf c) {
    auto keys=torch::empty({4LL*atoms},geometry.options().dtype(torch::kLong));
    owner_keys<<<(keys.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(geometry.data_ptr<float>(),keys.data_ptr<int64_t>(),atoms,c);
    auto sorted=at::sort(keys,true,-1,false);
    auto boundaries=torch::arange(c.a.node_count+1,keys.options());
    auto offsets=torch::searchsorted(std::get<0>(sorted),boundaries);
    return {std::get<1>(sorted),offsets};
}
Tensor moments_reuse(Tensor x,Tensor mask,Conf c) {
    auto z=torch::empty({x.size(0),c.nodes,33},x.options());
    for(int level=START;level<c.a.bank_count/2;++level){
        int64_t work=x.size(0)*2*c.a.banks[2*level].tile_count*33;
        level_moments<<<(work+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),z.data_ptr<float>(),mask.data_ptr<uint32_t>(),x.size(0),c,level);
    }
    return z;
}
Tensor response_reuse(Tensor x,Tensor z,Tensor geo,int atoms,Conf c) {
    auto h=torch::empty({x.size(0),atoms},x.options());
    response8<<<dim3((atoms+31)/32,(x.size(0)+7)/8),32,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),z.data_ptr<float>(),geo.data_ptr<float>(),h.data_ptr<float>(),atoms,x.size(0),c);
    return h;
}
Tensor expansion_reuse(Tensor p,Tensor g,Tensor h,Tensor mask,Tensor rows,Tensor offsets,Conf c) {
    auto y=torch::zeros({h.size(0),c.a.sites},h.options()),C=torch::empty({h.size(0),c.nodes,33},h.options());
    owner_expand<<<dim3(c.a.node_count,(h.size(0)+7)/8),128,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),g.data_ptr<float>(),h.data_ptr<float>(),rows.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),y.data_ptr<float>(),C.data_ptr<float>(),mask.data_ptr<uint32_t>(),p.size(0),h.size(0),c);
    expand<<<(y.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(C.data_ptr<float>(),y.data_ptr<float>(),mask.data_ptr<uint32_t>(),h.size(0),c);
    return y;
}
std::vector<Tensor> forward(Tensor x,Tensor p,int ni,int no,int m,bool connected,std::vector<double>s,bool reuse) {
    check(x,p,ni,no,m);TORCH_CHECK(m==16&&connected&&s==std::vector<double>({.1,.9,1.25}),"requires default connected M16");
    c10::cuda::CUDAGuard guard(x.device());auto ci=config(ni,m,connected,s),co=config(no,m,connected,s);
    auto gi=torch::empty({p.size(0),4,G},p.options()),go=torch::empty_like(gi),mi=torch::zeros({1},p.options().dtype(torch::kUInt32)),mo=torch::zeros_like(mi);
    auto stream=at::cuda::getCurrentCUDAStream();
    geometry32<<<(p.size(0)+63)/64,64,0,stream>>>(p.data_ptr<float>(),gi.data_ptr<float>(),go.data_ptr<float>(),mi.data_ptr<uint32_t>(),mo.data_ptr<uint32_t>(),p.size(0),ci,co);
    Tensor rowsi=torch::empty({0},p.options().dtype(torch::kLong)),offsetsi=rowsi,rowso=rowsi,offsetso=rowsi;
    Tensor z,h,y;
    if(reuse){
        close_mask<<<1,1,0,stream>>>(mi.data_ptr<uint32_t>(),ci);close_mask<<<1,1,0,stream>>>(mo.data_ptr<uint32_t>(),co);
        auto ri=ranges(gi,p.size(0),ci),ro=ranges(go,p.size(0),co);rowsi=ri[0];offsetsi=ri[1];rowso=ro[0];offsetso=ro[1];
        z=moments_reuse(x,mi,ci);h=response_reuse(x,z,gi,p.size(0),ci);y=expansion_reuse(p,go,h,mo,rowso,offsetso,co);
    }else{z=make_moments(x,mi,ci);h=get_response(x,z,gi,p.size(0),ci);y=expansion(p,go,h,mo,co);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,gi,go,mi,mo,rowsi,offsetsi,rowso,offsetso};
}
std::vector<Tensor> backward(Tensor x,Tensor p,Tensor dy,Tensor gi,Tensor go,Tensor mi,Tensor mo,Tensor rowsi,Tensor offsetsi,Tensor rowso,Tensor offsetso,int ni,int no,int m,bool connected,std::vector<double>s,bool reuse) {
    check(x,p,ni,no,m);c10::cuda::CUDAGuard guard(x.device());auto ci=config(ni,m,connected,s),co=config(no,m,connected,s);
    auto zx=reuse?moments_reuse(x,mi,ci):make_moments(x,mi,ci),zy=reuse?moments_reuse(dy,mo,co):make_moments(dy,mo,co);
    auto g=reuse?response_reuse(dy,zy,go,p.size(0),co):get_response(dy,zy,go,p.size(0),co);
    auto dx=reuse?expansion_reuse(p,gi,g,mi,rowsi,offsetsi,ci):expansion(p,gi,g,mi,ci),dp=torch::empty_like(p);
    if(reuse)gradients_reuse<<<p.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),x.data_ptr<float>(),dy.data_ptr<float>(),zx.data_ptr<float>(),zy.data_ptr<float>(),gi.data_ptr<float>(),go.data_ptr<float>(),dp.data_ptr<float>(),p.size(0),x.size(0),ci,co);
    else gradients<<<p.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),x.data_ptr<float>(),dy.data_ptr<float>(),zx.data_ptr<float>(),zy.data_ptr<float>(),gi.data_ptr<float>(),go.data_ptr<float>(),dp.data_ptr<float>(),p.size(0),x.size(0),ci,co);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dp};
}

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&connected_reuse::forward);m.def("backward",&connected_reuse::backward);}
