#pragma once
// Positive homogeneous coefficients of scaled T16(cos(theta)/sqrt(z))^2.
// Row k is degree16-k in z and u=1-z; normalization cancels the common scale.
// Exact rational derivation and FP64 oracle validation live in tests.
namespace connected_reuse {
__device__ __constant__ float m16_terms[17][17] = {
    {0.5f, 8.0f, 1020.0f, 42840.0f, 881790.0f, 10581480.0f, 81477396.0f, 426786360.0f, 1577585295.0f, 4206894120.0f, 8180071900.0f, 1.160082924e+10f, 1.186448445e+10f, 8518091400.0f, 4071834900.0f, 1163381400.0f, 150270097.5f},
    {0.0f, 8.0f, 1008.0f, 41496.0f, 829920.0f, 9585576.0f, 70294224.0f, 346450104.0f, 1187828928.0f, 2887084200.0f, 5004279280.0f, 6141615480.0f, 5211067680.0f, 2906172360.0f, 958078800.0f, 141430680.0f},
    {0.0f, 8.0f, 988.0f, 39520.0f, 760760.0f, 8368360.0f, 57741684.0f, 263961984.0f, 824881200.0f, 1787242600.0f, 2680863900.0f, 2729606880.0f, 1799059080.0f, 691945800.0f, 117858900.0f},
    {0.0f, 8.0f, 960.0f, 36960.0f, 677600.0f, 7013160.0f, 44884224.0f, 187017600.0f, 520977600.0f, 976833000.0f, 1215614400.0f, 961440480.0f, 437018400.0f, 86843400.0f},
    {0.0f, 8.0f, 924.0f, 33880.0f, 584430.0f, 5610528.0f, 32728080.0f, 121561440.0f, 293049900.0f, 455855400.0f, 440660220.0f, 240360120.0f, 56448210.0f},
    {0.0f, 8.0f, 880.0f, 30360.0f, 485760.0f, 4250400.0f, 22102080.0f, 71042400.0f, 142084800.0f, 171685800.0f, 114457200.0f, 32256120.0f},
    {0.0f, 8.0f, 828.0f, 26496.0f, 386400.0f, 3013920.0f, 13562640.0f, 36167040.0f, 56188080.0f, 46823400.0f, 16128060.0f},
    {0.0f, 8.0f, 768.0f, 22400.0f, 291200.0f, 1965600.0f, 7338240.0f, 15200640.0f, 16286400.0f, 7012200.0f},
    {0.0f, 8.0f, 700.0f, 18200.0f, 204750.0f, 1146600.0f, 3325140.0f, 4750200.0f, 2629575.0f},
    {0.0f, 8.0f, 624.0f, 14040.0f, 131040.0f, 570024.0f, 1140048.0f, 841464.0f},
    {0.0f, 8.0f, 540.0f, 10080.0f, 73080.0f, 219240.0f, 226548.0f},
    {0.0f, 8.0f, 448.0f, 6496.0f, 32480.0f, 50344.0f},
    {0.0f, 8.0f, 348.0f, 3480.0f, 8990.0f},
    {0.0f, 8.0f, 240.0f, 1240.0f},
    {0.0f, 8.0f, 124.0f},
    {0.0f, 8.0f},
    {0.25f},
};

__device__ __forceinline__ void coefficients_m16(float radius,float period,float *c,float *dr) {
    constexpr float PI_F=3.14159265358979323846f;
    float cs,sn;sincosf(PI_F*radius/period,&sn,&cs);
    float z=cs*cs,u=1-z,dz=-(2*PI_F/period)*cs*sn;
    float zp[17],up[17];zp[0]=up[0]=1;
    #pragma unroll
    for(int j=1;j<=16;++j){zp[j]=zp[j-1]*z;up[j]=up[j-1]*u;}
    float zero=0,dzero=0;
    for(int k=0;k<=16;++k){
        float raw=0,derivative=0;int degree=16-k;
        for(int j=0;j<=degree;++j){
            int p=degree-j;float a=m16_terms[k][j];
            raw+=a*zp[p]*up[j];
            derivative+=a*((p?p*zp[p-1]*up[j]:0)-(j?j*zp[p]*up[j-1]:0));
        }
        if(k==0){zero=raw;dzero=derivative;c[0]=1/period;dr[0]=0;}
        else{c[k]=(raw/zero)/period;dr[k]=((derivative-raw*(dzero/zero))/zero)*(dz/period);}
    }
}
} // namespace connected_reuse
