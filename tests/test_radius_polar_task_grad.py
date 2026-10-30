"""CPU source smoke for detached-radius task work, without loading CUDA."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex


def test_value_only_coefficients_and_both_gradient_specializations_compile(tmp_path):
    """Exercise actual C++ arithmetic; GPU code generation remains a separate gate."""
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    common = (
        (directory / "common.cuh")
        .read_text()
        .replace("#include <torch/extension.h>", "")
    )
    # Compile the real include graph in the reverse entry order too. Only the
    # Torch include is removed; these source helpers themselves need no Torch.
    (tmp_path / "common.cuh").write_text(common)
    (tmp_path / "limits.cuh").write_text((directory / "limits.cuh").read_text())
    (tmp_path / "profile_values.cuh").write_text(
        (directory / "profile_values.cuh").read_text()
    )
    common = '#include "profile_values.cuh"\n#include "common.cuh"\n'
    hybrid = (directory / "hybrid.cuh").read_text()
    phase = hybrid[
        hybrid.index("struct HarmonicPhase") : hybrid.index("inline int checked_start")
    ]
    projections = hybrid[
        hybrid.index("template<bool Derivatives = true>") : hybrid.index(
            "template<class T, bool Direct>"
        )
    ]
    pipeline = (directory / "pipeline.cu").read_text()
    projection = pipeline[
        pipeline.index("template<bool Derivatives = true>") : pipeline.index(
            "template<class T>\n__global__ void response_kernel"
        )
    ]
    batched = (directory / "batched_parameter_grad.cuh").read_text()
    program = (
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __noinline__
#define TORCH_CHECK(condition, ...) assert(condition)
struct { int x = 0; } blockIdx, threadIdx;
struct { int x = 1; } blockDim;
inline void sincos(double x, double *s, double *c);
"""
        + common
        + r"""
inline void sincos(double x, double *s, double *c) { *s = sin(x); *c = cos(x); }
namespace radius_cell_mass {
"""
        + phase
        + projections
        + projection
        + "\n}\n"
        + batched
        + r"""
using namespace radius_cell_mass;
template<class T> void check_kernel(bool polar, int modes, int start, int batch) {
    Axis axis = parse_axis({16, .37, 1, -1.2, double(modes), 2});
    axis.polar = polar; axis.amplitude_max = 1; axis.w_c = .3;
    axis.kappa = 2; axis.lower_kappa = 2; axis.upper_decay_power = .5;
    axis.radius_min = .1; axis.radius_birth = .5;
    axis.radius_max = 2; axis.upper_floor = .5;
    T p[4] = {T(.3), T(.4), T(.7), T(-.2)};
    T x[33 * 16], dy[33 * 16], full[4], task[4];
    double h[33], g[33], zx[33 * 100 * 33], zy[33 * 100 * 33];
    for (int j = 0; j < 33 * 16; ++j) { x[j] = T(sin(j)); dy[j] = T(cos(j)); }
    for (int j = 0; j < 33; ++j) { h[j] = sin(j + .1); g[j] = cos(j + .2); }
    for (int j = 0; j < 33 * 100 * 33; ++j) { zx[j] = sin(j + .3); zy[j] = cos(j + .4); }
    batched_parameter_kernel<T, true>(p, x, dy, zx, zy, h, g, full, 1, batch, axis, axis, start, start);
    // Responses discard all derivative outputs. Values must preserve bytes
    // with nullptr derivative inputs, for direct and both compact layouts.
    double c[17], dr[17], unused_r, unused_mu;
    double R = radius(p, 0, axis), mu = wrapped_center(center(p, 0, axis, false), axis);
    Membership member = classify(R, center(p, 0, axis, false), axis);
    profile_coefficients(R, axis.banks[member.bank].period, modes, c, dr);
    double a = direct_project(x, 0, axis, member, mu, c, dr, unused_r, unused_mu);
    double v = direct_project<T, false>(x, 0, axis, member, mu, c, nullptr, unused_r, unused_mu);
    assert(memcmp(&a, &v, sizeof(double)) == 0);
    if (start < axis.bank_count / 2) {
        a = compact_project(zx, 0, axis, member, mu, c, dr, 0, unused_r, unused_mu);
        v = compact_project<false>(zx, 0, axis, member, mu, c, nullptr, 0, unused_r, unused_mu);
        assert(memcmp(&a, &v, sizeof(double)) == 0);
    }
    a = project(zx, 0, axis, member, mu, c, dr, unused_r, unused_mu);
    v = project<false>(zx, 0, axis, member, mu, c, nullptr, unused_r, unused_mu);
    assert(memcmp(&a, &v, sizeof(double)) == 0);
    if (polar) {
        batched_parameter_kernel<T, false>(p, x, dy, zx, zy, h, g, task, 1, batch, axis, axis, start, start);
        assert(memcmp(full, task, sizeof full) == 0);
    }
}
int main() {
    for (double period : {1., 128., 47.36, 3072.}) {
        for (double ratio : {0., 1e-14, .01, .125, .25, .499999999999, .5}) {
            for (int modes : {1, 3, 8, 16}) {
                double expected[17], derivative[17], actual[17];
                profile_coefficients(ratio * period, period, modes, expected, derivative);
                profile_values(ratio * period, period, modes, actual);
                for (int k = 0; k <= modes; ++k) {
                    assert(std::isfinite(actual[k]));
                    assert(memcmp(actual + k, expected + k, sizeof(double)) == 0);
                }
            }
        }
    }
    static_assert(sizeof(RadiusCoefficients<false>) == 1);
    static_assert(sizeof(RadiusRows<false>) == 1);
    for (int modes : {1, 3, 8, 16})
      for (bool polar : {false, true}) for (int start : {0, 2, 5})
        for (int batch : {0, 1, 4, 5, 31, 32, 33}) {
            check_kernel<float>(polar, modes, start, batch);
            check_kernel<double>(polar, modes, start, batch);
        }
}
"""
    )
    source = tmp_path / "polar_task_smoke.cpp"
    binary = tmp_path / "polar_task_smoke"
    source.write_text(program)
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
