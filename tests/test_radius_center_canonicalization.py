"""Stable periodic-relative coordinates select one window and its phases."""

import math
import os
import shlex
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from test_radius_cell_mass import axis
from test_radius_cell_mass_cuda import GPU

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
)
from torchcst._backends.torch.kernels.radius_cell_mass import (
    _relative_centers,
    axis_masses,
)


def exact_relative(center, origin, period):
    return (
        Fraction.from_float(center) - Fraction.from_float(origin)
    ) % Fraction.from_float(period)


def native_relative(center, origin, period):
    value = math.fmod(math.fmod(center, period) - math.fmod(origin, period), period)
    return value + period if value < 0 else value


@pytest.mark.parametrize(
    "center,origin,period",
    [
        (79 * 13.7, -1.2, 1024.0),
        (math.nextafter(79 * 13.7, -math.inf), -1.2, 1024.0),
        (math.nextafter(79 * 13.7, math.inf), -1.2, 1024.0),
        (1e300, 0.5, 1024.0),
        (-1e300, 0.5, 1024.0),
        (1e300, 0.5, 128.0),
        (sys.float_info.max, -sys.float_info.max, 1024.0),
        (-sys.float_info.max, sys.float_info.max, 1024.0),
        (59.5, 0.0, 1024.0),
        (math.nextafter(59.5, -math.inf), 0.0, 1024.0),
        (math.nextafter(59.5, math.inf), 0.0, 1024.0),
    ],
)
def test_reference_wrap_matches_exact_encoded_floats_and_native(center, origin, period):
    centers = torch.tensor([center], dtype=torch.float64, requires_grad=True)
    result = _relative_centers(centers, origin, period)
    expected = float(exact_relative(center, origin, period))
    assert result.item() == expected == native_relative(center, origin, period)
    result.backward(torch.ones_like(result))
    assert centers.grad.item() == 1.0


def test_tile_boundary_uses_stable_relative_coordinate_without_snapping():
    center = 79 * 13.7
    actual = _relative_centers(
        torch.tensor([center], dtype=torch.float64), -1.2, 1024.0
    )
    assert actual.item().hex() == "0x1.dbffffffffffap+5"
    assert ((center - (-1.2)) % 1024.0).hex() == "0x1.dc00000000000p+5"
    assert math.floor((actual.item() + 0.5) / 4 - 1) == 13
    assert math.floor((((center + 1.2) % 1024) + 0.5) / 4 - 1) == 14
    grid = axis(1024, spacing=1.0, origin=-1.2)
    radius = torch.tensor([2.3285715051147386], dtype=torch.float64)
    masses = axis_masses(grid, radius, torch.tensor([center], dtype=torch.float64))
    assert torch.isfinite(masses).all()
    # The entire 12-cell finite Fourier window is retained; changing its first
    # tile is a real operator change, including the small nonzero tail cells.
    assert masses[0, 52:56].abs().max() > 1e-8
    assert torch.equal(masses[0, 64:68], torch.zeros(4, dtype=torch.float64))


def test_scalar_libm_repair_keeps_signed_zero_and_unit_vjp():
    center = torch.tensor(
        [-sys.float_info.max], dtype=torch.float64, requires_grad=True
    )
    actual = _relative_centers(center, 0.0, 1024.0)
    assert actual.item() == 0
    assert bool(torch.signbit(actual).item())
    actual.backward(torch.ones_like(actual))
    assert center.grad.item() == 1.0


@pytest.mark.parametrize("period", [1e-200, 0.37, 1024.0, 1e200, sys.float_info.max])
def test_wrap_huge_finite_inputs_and_overflow_keep_finite_values_and_unit_gradient(
    period,
):
    centers = torch.tensor(
        [sys.float_info.max, -sys.float_info.max, 0.75 * period, -0.75 * period],
        dtype=torch.float64,
        requires_grad=True,
    )
    for origin in (sys.float_info.max, -sys.float_info.max, -0.75 * period):
        actual = _relative_centers(centers, origin, period)
        assert bool(torch.isfinite(actual).all())
        assert bool(((actual >= 0) & (actual <= period)).all())
        expected = torch.tensor(
            [float(exact_relative(float(c), origin, period)) for c in centers.detach()],
            dtype=torch.float64,
        )
        # This helper-only extreme-period bound is an FP64 operation-count
        # check, not a relaxation of any Linear/gradient/optimizer gate.
        assert bool(((actual - expected).abs() <= 8 * math.ulp(period)).all())
        gradient = torch.autograd.grad(actual, centers, torch.ones_like(actual))[0]
        assert torch.equal(gradient, torch.ones_like(centers))


def test_actual_native_wrap_is_unchanged_and_matches_reference_on_valid_axes(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("host C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    for name in ("common.cuh", "limits.cuh", "profile_values.cuh"):
        (tmp_path / name).write_text(
            (directory / name).read_text().replace("#include <torch/extension.h>", "")
        )
    cases = [
        (79 * 13.7, -1.2, 1024.0),
        (1e300, 0.5, 1024.0),
        (-1e300, 0.5, 1024.0),
        (sys.float_info.max, -sys.float_info.max, 1024.0),
        (-sys.float_info.max, sys.float_info.max, 1024.0),
    ]
    rows = ",\n".join(
        "{"
        + ",".join(float(v).hex() for v in (*values, native_relative(*values)))
        + "}"
        for values in cases
    )
    source = (
        r"""
#include <cassert>
#include <cstring>
#define __host__
#define __device__
#define __constant__
#define __noinline__
#define TORCH_CHECK(c,...) assert(c)
#include "common.cuh"
int main() {
    using namespace radius_cell_mass;
    double cases[][4]={
"""
        + rows
        + r"""
    };
    for(auto &r:cases) {
        Axis a{};a.origin=r[1];a.period=r[2];double value=wrapped_center(r[0],a);
        assert(memcmp(&value,r+3,sizeof(double))==0);
    }
}
"""
    )
    path, binary = tmp_path / "wrap.cpp", tmp_path / "wrap"
    path.write_text(source)
    result = subprocess.run(
        [*compiler, "-std=c++17", "-O2", str(path), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@GPU
def test_original_boundary_fixture_all_vjps_against_stable_public_reference():
    from torchcst import (
        ChebyshevRadiusSpec,
        CSTLinear,
        RadiusBounds,
        chart_presets,
        presets,
    )

    # The encoded center/origin pair above occurs at atom 79 in this skewed
    # real Linear fixture. Keep the legacy native Plan: this regression does
    # not depend on an experimental owner-fusion implementation.
    chart = chart_presets.regular_grid(((1024,),), origin=-1.2, geometry="flat_torus")
    reference = CSTLinear(
        chart,
        chart,
        atoms=129,
        dtype=torch.float64,
        backend="factored",
        kernel=presets.polar_radius_cell_mass(
            amplitude_max=1.0,
            w_c=0.3,
            input_radius_bounds=RadiusBounds(
                minimum=0.1,
                birth=4.0,
                maximum=512.0,
                upper_floor=1.0,
            ),
            profile=ChebyshevRadiusSpec(modes=8),
        ),
    )
    with torch.no_grad():
        reference.atoms.p[:, 0] = 0.15
        reference.atoms.p[:, 1] = (1 - 0.15**2) ** 0.5
        reference.atoms.p[:, 2] = torch.arange(129, dtype=torch.float64) * 13.7
        reference.atoms.p[:, 3] = torch.where(
            torch.arange(129) % 2 == 0,
            torch.tensor(-1.2, dtype=torch.float64),
            torch.tensor(1022.8 - 1e-4, dtype=torch.float64),
        )
    plan = hierarchy_from_operator(reference.declaration())
    gen = torch.Generator().manual_seed(2081)
    x = torch.randn(4, 1024, dtype=torch.float64, generator=gen, requires_grad=True)
    dy = torch.randn(4, 1024, dtype=torch.float64, generator=gen)
    expected = reference(x)
    gradients = torch.autograd.grad(expected, (x, reference.atoms.p), dy)
    cx = x.detach().cuda().requires_grad_()
    cp = reference.atoms.p.detach().cuda().requires_grad_()
    actual = ex.radius_linear(plan, cx, cp, recipe=GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
    got = torch.autograd.grad(actual, (cx, cp), dy.cuda())
    for a, b in zip((actual, *got), (expected, *gradients)):
        torch.testing.assert_close(a.cpu(), b, atol=2e-9, rtol=2e-8)
