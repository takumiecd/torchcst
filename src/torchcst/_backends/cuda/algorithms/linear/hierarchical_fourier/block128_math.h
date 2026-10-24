// Scalar mathematics shared by the CUDA prototype and independent CPU checks.
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define CST128_HD __host__ __device__ __forceinline__
#else
#define CST128_HD inline
#endif

namespace cst128 {
constexpr int N = 128, A = 128, BATCH = 4, M = 8, R = 17;
constexpr int BANKS = 12, NODES = 28, DEGREE = 32;
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double QLO = .29, FLOOR = 1e-6;
constexpr int FORWARD_SHARED = 21376, BACKWARD_SHARED = 38656;

struct Component {
    double length, weight, width_derivative;
    int bank;
};

CST128_HD double sigmoid(double x) { return 1. / (1. + ::exp(-x)); }
CST128_HD double clamp(double x, double lo, double hi) {
    return ::fmin(::fmax(x, lo), hi);
}
CST128_HD void smooth(double z, double& value, double& derivative) {
    z = clamp(z, 0., 1.);
    value = z*z*z*(10. + z*(-15. + 6.*z));
    derivative = 30.*z*z*(1.-z)*(1.-z);
}

CST128_HD Component component(double width, int c) {
    const double a0 = (.2/2. + 1.25*.05)/(1./.3 - 1.);
    int level = 0;
    for (int j=1; j<12; ++j)
        if (width >= a0 * (1<<j)) level = j;
    int lev = c < 2 ? (level > 0 ? level-1 : 0) : level;
    double threshold = a0*(1<<level);
    double h, dh;
    double interval = level > 0 ? .15*threshold : 1.;
    smooth((width-threshold)/interval, h, dh);
    h = level > 0 ? h : 1.;
    dh = level > 0 ? dh/interval : 0.;
    double outer = c < 2 ? 1.-h : h;
    double douter = c < 2 ? -dh : dh;
    double tile = .2*(1<<lev), step = .05*(1<<lev);
    int n = static_cast<int>(::ceil((width+tile/2.)/step));
    double w, dw;
    smooth((width-n*step+tile/2.+.25*step)/(.25*step), w, dw);
    dw /= .25*step;
    double inner = c%2 ? w : 1.-w;
    double dinner = c%2 ? dw : -dw;
    int period = n+c%2;
    Component result{period*step, outer*inner, douter*inner+outer*dinner, -1};
    if (result.weight > 0 && lev >= 4 && lev <= 6 && period >= 4 && period <= 7)
        result.bank = (lev-4)*4+period-4;
    return result;
}

// The fixed continuous-q Chebyshev table is the existing degree-32 M8 fit.
// b and bi are the weighted coefficient and its live-I derivative, respectively.
CST128_HD void coefficient(const double* fit, double width, Component c, int mode,
                            double& b, double& bi) {
    double t = (2.*width/c.length-(1.+QLO))/(1.-QLO);
    double b1=0., b2=0., d1=0., d2=0.;
    for (int j=DEGREE-1; j>=1; --j) {
        double v = 2.*t*b1-b2+fit[j*(M+1)+mode];
        double d = 2.*b1+2.*t*d1-d2;
        b2=b1; b1=v; d2=d1; d1=d;
    }
    double raw = fit[mode]+t*b1-b2;
    double rawq = (b1+t*d1-d2)*2./(1.-QLO);
    b = raw*c.weight;
    bi = rawq/c.length*c.weight+raw*c.width_derivative;
}

CST128_HD void coefficients(const double* fit, double width, Component c,
                             double* b, double* bi) {
    for (int n=0; n<=M; ++n) coefficient(fit, width, c, n, b[n], bi[n]);
}

CST128_HD void series(int site, double center, Component c, const double* b,
                      const double* bi, double* value) {
    value[0]=value[1]=value[2]=0.;
    double delta = site*.1-center;
    if (!(::fabs(delta) < c.length)) return;
    double theta = PI*delta/c.length, cs=::cos(theta), ss=::sin(theta);
    double cn=1., sn=0.;
    for (int n=0; n<=M; ++n) {
        value[0] += b[n]*cn;
        value[1] += b[n]*sn*(PI*n/c.length);
        value[2] += bi[n]*cn;
        double next = cn*cs-sn*ss;
        sn=sn*cs+cn*ss; cn=next;
    }
}

CST128_HD void bounds(double center, double length, int& lo, int& hi) {
    lo = static_cast<int>(clamp(::floor((center-length)/.1)+1., 0., N));
    hi = static_cast<int>(clamp(::ceil((center+length)/.1), 0., N));
}
CST128_HD int bank_size(int bank) { return 32 << (bank/4); }
CST128_HD int bank_offset(int bank) {
    return bank < 4 ? 4*bank : bank < 8 ? 16+2*(bank-4) : 24+bank-8;
}
CST128_HD double bank_length(int bank) { return (3.2+.8*(bank%4))*(1<<(bank/4)); }
CST128_HD int node_bank(int node) {
    return node < 16 ? node/4 : node < 24 ? 4+(node-16)/2 : 8+node-24;
}
CST128_HD int node_start(int node) {
    int bank=node_bank(node);
    return (node-bank_offset(bank))*bank_size(bank);
}
CST128_HD double basis(int bank, int r, int site) {
    int n = r<=M ? r : r-M;
    double theta = PI*n*(site*.1)/bank_length(bank);
    return r<=M ? ::cos(theta) : ::sin(theta);
}

// Global-coordinate Fourier basis. No extension beyond finite support.
CST128_HD void beta(Component c, double center, int r, double b, double bi,
                     double* result) {
    int n = r<=M ? r : r-M;
    double omega=PI*n/c.length, cs=::cos(omega*center), ss=::sin(omega*center);
    result[0]=b*(r<=M ? cs : ss);
    result[1]=b*omega*(r<=M ? -ss : cs);
    result[2]=bi*(r<=M ? cs : ss);
}
} // namespace cst128
