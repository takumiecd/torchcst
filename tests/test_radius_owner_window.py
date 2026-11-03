"""Window compression, actual CPU coefficient code and independent CUDA VJPs."""

import copy
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from test_radius_cell_mass_cuda import GPU
from test_radius_owner_response_coeffs import model

from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW17,
    GLOBAL_OWNER_WINDOW_FOLD,
    GLOBAL_OWNER_WINDOW_RECOMPUTE,
)

PLANS = [GLOBAL_OWNER_WINDOW17, GLOBAL_OWNER_WINDOW_FOLD, GLOBAL_OWNER_WINDOW_RECOMPUTE]


def test_actual_window_coefficients_all_banks_and_original_frequency_derivatives(
    tmp_path,
):
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not shutil.which(compiler[0]):
        pytest.skip("C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    for name in ("common.cuh", "limits.cuh", "profile_values.cuh"):
        (tmp_path / name).write_text(
            (directory / name).read_text().replace("#include <torch/extension.h>", "")
        )
    hybrid = (directory / "hybrid.cuh").read_text()
    phase = hybrid[
        hybrid.index("struct HarmonicPhase") : hybrid.index("inline int checked_start")
    ]
    window = (directory / "owner_window.cuh").read_text()
    helpers = window[
        window.index("struct OwnerWindowLayout") : window.index("template <class T>")
    ]
    source = (
        r"""
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#define __host__
#define __device__
#define __constant__
#define __noinline__
#define TORCH_CHECK(c,...) assert(c)
using std::min;
inline void radius_test_sincos(double x,double*s,double*c) {*s=sin(x);*c=cos(x);}
#define sincos radius_test_sincos
#include "common.cuh"
namespace radius_cell_mass {
"""
        + phase
        + helpers
        + r"""
}
using namespace radius_cell_mass;
int main() {
 double worst=0, dworst=0; int checks=0;
 for (int N:{8,32,128,1024}) for(double spacing:{.125,.37,2.0}) for(double origin:{0.,-1.2}) {
  Axis a=parse_axis({double(N),spacing,1.,origin,8.,2.});
  if(N==1024) assert(owner_window_layout(a,true).total_fields==23902);
  for(int bi=0;bi<a.bank_count;++bi) {
   Bank bank=a.banks[bi]; int L=bank.count*bank.tile_sites;
   double c[17]; profile_values(.73*bank.tile_sites*spacing,bank.period,8,c);
   for(int first:{-1,0,bank.tile_count-1}) for(double fraction:{.1,.7}) {
    double mu=(first+.5+fraction)*bank.tile_sites*spacing;
    double q[17],dq[17]; owner_window_coefficients(c,mu,first,a,bank,true,q,dq);
    int rank=min(L,17);
    for(int j=0;j<L;++j) {
     double actual=q[0], derivative=dq[0], want=spacing*c[0], dwant=0;
     for(int k=1;2*k-1<rank;++k) {
      double cp=cos(2*PI*k*j/L),sp=sin(2*PI*k*j/L);
      actual+=q[2*k-1]*cp; derivative+=dq[2*k-1]*cp;
      if(2*k<rank) {actual+=q[2*k]*sp;derivative+=dq[2*k]*sp;}
     }
     for(int k=1;k<=8;++k) {
      double arg=PI*k/L,omega=2*PI*k/bank.period;
      double v=2*c[k]*spacing*sin(arg)/arg, angle=omega*(mu-(first*bank.tile_sites+j)*spacing);
      want+=v*cos(angle);dwant-=v*omega*sin(angle);
     }
     worst=std::max(worst,fabs(actual-want));dworst=std::max(dworst,fabs(derivative-dwant));++checks;
     assert(fabs(actual-want)<2e-11); assert(fabs(derivative-dwant)<3e-10);
    }
   }
  }
 }
 printf("{\"cell_checks\":%d,\"max_abs\":%.17g,\"derivative_max_abs\":%.17g}\n",checks,worst,dworst);
}
"""
    )
    path = tmp_path / "fold.cpp"
    path.write_text(source)
    binary = tmp_path / "fold"
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    record = json.loads(result.stdout)
    assert record["cell_checks"] > 100000
    print(record)


@GPU
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_window_vjp_and_retained_offload(plan, dtype, batch, needs_x, needs_p, skew):
    reference = model(dtype, skew).double()
    candidate = copy.deepcopy(reference).to(dtype).cuda()
    reference.atoms.p.data = candidate.atoms.p.detach().cpu().double()
    reference.atoms.p.requires_grad_(needs_p)
    candidate.atoms.p.requires_grad_(needs_p)
    candidate.selector = FixedSelector(plan, registry=make_algorithm_registry())
    gen = torch.Generator().manual_seed(2081)
    xx = torch.randn(batch, 1024, dtype=dtype, generator=gen)
    dy = torch.randn(batch, 1024, dtype=dtype, generator=gen)
    rx = xx.double().requires_grad_(needs_x)
    ry = reference(rx)
    expected = torch.autograd.grad(
        ry, tuple(t for t in (rx, reference.atoms.p) if t.requires_grad), dy.double()
    )
    x = xx.cuda().requires_grad_(needs_x)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (t.detach().cpu().clone(), t.device), lambda a: a[0].to(a[1])
    ):
        y = candidate(x)
    args = tuple(t for t in (x, candidate.atoms.p) if t.requires_grad)
    actual = torch.autograd.grad(y, args, dy.cuda(), retain_graph=True)
    repeated = torch.autograd.grad(y, args, dy.cuda())
    for a, b in zip(actual, repeated):
        assert torch.equal(a, b)
    tol = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    for a, b in zip((y, *actual), (ry, *expected)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.cpu().double(), b, **tol)


@GPU
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize(
    "contract",
    [
        "public_retained_hooks",
        "raw_version_once_diff",
        "no_grad",
        "inference",
        "frozen",
        "empty_atoms",
        "empty_batch",
    ],
)
def test_window_live_contracts(monkeypatch, plan, contract):
    import test_radius_owner_response_derivatives as derivative

    monkeypatch.setattr(derivative, "NEW", plan)
    derivative.test_existing_independent_contracts_on_explicit_derivative_plan(
        monkeypatch, contract
    )


@pytest.mark.parametrize("plan", PLANS)
def test_window_scope_and_explicit_native_leaf(plan, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from torchcst import LinearInputs
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        RadiusCellMassAlgorithm,
        hierarchy_from_operator,
    )
    from torchcst._backends.schema import DeviceInfo

    m = model(torch.float32)
    x = torch.zeros(4, 1024)
    context = m.build_context(LinearInputs(x))
    context = replace(context, device=DeviceInfo("cuda", 0, "metadata-only"))
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(context, plan.recipe).supported
    assert not algorithm.supports(
        replace(context, input_shape=(5, 1024)), plan.recipe
    ).supported
    registry = make_algorithm_registry()
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    calls = []
    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            checked_autograd_linear_owner_window=lambda *args: calls.append(args) or x
        ),
    )
    axes = hierarchy_from_operator(m.declaration())
    assert ex.radius_linear(axes, x, m.atoms.p, recipe=plan.recipe) is x
    assert len(calls) == 1 and calls[0][-1] == PLANS.index(plan)


@GPU
@pytest.mark.parametrize("plan", PLANS)
def test_window_saved_payload(plan):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    m = model(torch.float32).cuda()
    axes = hierarchy_from_operator(m.declaration())
    x = torch.randn(4, 1024, device="cuda", requires_grad=True)
    shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: shapes.append(tuple(t.shape)) or t, lambda t: t
    ):
        y = ex.radius_linear(axes, x, m.atoms.p, recipe=plan.recipe)
    recompute = plan == GLOBAL_OWNER_WINDOW_RECOMPUTE
    assert shapes.count((4, 7)) == (0 if recompute else 2)
    assert shapes.count((4, 23902)) == int(recompute)
    y.sum().backward()
