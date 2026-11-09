from pathlib import Path

import sympy as s

m = 16
x, z, u = s.symbols("x z u")
a, b = s.Integer(2), x
for n in range(2, m + 1):
    a, b = b, s.expand(x * b - z * a / 4)
poly = s.Poly(s.expand(b * b), x)
rows = []
for k in range(m + 1):
    q = s.expand(
        sum(
            c * s.binomial(power[0], power[0] // 2 - k) / 2 ** power[0]
            for power, c in poly.terms()
            if power[0] // 2 >= k
        )
        * 4 ** (m - 1)
    )
    d = m - k
    h = s.Poly(
        s.expand(
            sum(
                c * z ** power[0] * (z + u) ** (d - power[0])
                for power, c in s.Poly(q, z).terms()
            )
        ),
        z,
        u,
    )
    co = [h.coeff_monomial(z ** (d - j) * u**j) for j in range(d + 1)]
    assert all(c >= 0 for c in co)
    rows.append(co)
p = Path(
    "src/torchcst/_backends/cuda/algorithms/linear/connected_cell_mass_reuse/profile_m16.cuh"
)
lines = [
    "#pragma once",
    "// Positive homogeneous coefficients of scaled T16(cos(theta)/sqrt(z))^2.",
    "// Row k is degree16-k in z and u=1-z; normalization cancels the common scale.",
    "// Exact rational derivation and FP64 oracle validation live in tests.",
    "namespace connected_reuse {",
    "__device__ __constant__ float m16_terms[17][17] = {",
]
for co in rows:
    lines.append(
        "    {"
        + ", ".join(
            f"{float(c):.10g}f"
            if "." in f"{float(c):.10g}" or "e" in f"{float(c):.10g}"
            else f"{float(c):.10g}.0f"
            for c in co
        )
        + "},"
    )
lines += [
    "};",
    r"""
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
""",
]
p.write_text("\n".join(lines))
# Exact coefficients for the CPU tests, generated once without a runtime sympy dependency.
Path("tests/fixtures/connected-m16-positive.json").write_text(
    __import__("json").dumps([[float(c) for c in co] for co in rows], indent=2) + "\n"
)
print("generated", p)
