"""Complete response derivatives replace saved moments only in explicit mode7."""

import copy
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU
from test_radius_owner_response_coeffs import model

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from torchcst import FixedSelector, LinearInputs, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_RESPONSE_COEFFICIENTS as OLD,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_RESPONSE_DERIVATIVES as NEW,
)
from torchcst._backends.schema import DeviceInfo

ROOT = Path(__file__).resolve().parents[1]


def test_derivative_plan_scope_roundtrip_and_actual_saved_payload():
    run = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-response-derivatives-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == OLD
    assert run.entry("radius-global-owner-response-derivatives").plan == NEW
    assert run.dense and decode_snapshot(run.snapshot()) == run
    m = model()
    context = m.build_context(LinearInputs(torch.zeros(4, 1024, dtype=torch.float64)))
    context = replace(context, device=DeviceInfo("cuda", 0, "metadata-only"))
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(context, NEW.recipe).supported
    assert not algorithm.supports(
        replace(context, input_shape=(5, 1024)), NEW.recipe
    ).supported
    assert not algorithm.supports(
        replace(context, atom_count=65537), NEW.recipe
    ).supported
    for fields in (
        {"native_entry": False},
        {"native_autograd": False},
        {"input_direct_levels": 1},
        {"output_direct_levels": 3},
        {"moment_method": "direct"},
        {"output_tile_sites": 16},
    ):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(replace(NEW.recipe, **fields))
    # DH replaces saved Zx. Zy is temporary until reverse response/C completes.
    assert 2 * 4 * 52428 * 8 - 2 * 4 * 1022 * 17 * 8 == 2_243_456
    assert 2_243_456 + 4 * 1022 * 17 * 8 == 2_799_424


def test_checked_entry_distinguishes_new_mode_and_live_fallback(monkeypatch):
    from torch.overrides import TorchFunctionMode

    m = model(torch.float32)
    plan = hierarchy_from_operator(m.declaration())
    x = torch.zeros(4, 1024)
    calls = []
    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            checked_autograd_linear=lambda *args: calls.append(args) or x
        ),
    )
    assert ex.checked_radius_linear(plan, x, m.atoms.p, recipe=NEW.recipe) is x
    assert calls[-1][-1] == 7
    assert ex.checked_radius_linear(plan, x, m.atoms.p, recipe=OLD.recipe) is x
    assert calls[-1][-1] == 6

    class CustomMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    def fallback(actual_plan, actual_x, actual_p, *, recipe):
        assert actual_plan is plan and actual_x is x and actual_p is m.atoms.p
        assert recipe.schedule == "global_batched_cached_active_moments_grad4"
        assert not recipe.native_entry and not recipe.native_autograd
        return "live-fallback"

    monkeypatch.setattr(ex, "radius_linear", fallback)
    with CustomMode():
        assert (
            ex.checked_radius_linear(plan, x, m.atoms.p, recipe=NEW.recipe)
            == "live-fallback"
        )


def test_actual_complete_derivative_helpers_and_kernels_on_cpu(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("host C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    for name in ("common.cuh", "limits.cuh", "profile_values.cuh"):
        (tmp_path / name).write_text(
            (directory / name).read_text().replace("#include <torch/extension.h>", "")
        )
    hybrid = (directory / "hybrid.cuh").read_text()
    phase = hybrid[
        hybrid.index("struct HarmonicPhase") : hybrid.index("inline int checked_start")
    ]
    moments = hybrid[
        hybrid.index(
            "template<class T>\n__device__ inline double compact_moment_element"
        ) : hybrid.index("template<class T>\n__global__ void compact_moments_kernel")
    ]
    owner = (directory / "owner_response_coefficients.cuh").read_text().split(
        "// Host orchestration:"
    )[0] + "\n}\n"
    derivatives = (directory / "owner_response_derivatives.cuh").read_text().split(
        "// Host orchestration:"
    )[0] + "\n}\n"
    source = (
        r"""
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __noinline__
#define __shared__
#define __syncthreads() ((void)0)
#define __syncwarp() ((void)0)
#define TORCH_CHECK(c,...) assert(c)
struct { int x=0; } blockIdx,threadIdx;
struct { int x=128; } blockDim;
template<class T> T min(T a,T b) {return a<b?a:b;}
inline void radius_test_sincos(double x,double*s,double*c) {*s=sin(x);*c=cos(x);}
#define sincos radius_test_sincos
#include "common.cuh"
namespace radius_cell_mass {
constexpr int BLOCK_LOCAL_C_STRIDE=33;
inline int bank_for_node(int node,const Axis&a) {int b=0;while(b+1<a.bank_count&&node>=a.banks[b+1].node_offset)++b;return b;}
"""
        + phase
        + moments
        + "\n}\n"
        + owner
        + derivatives
        + r"""
namespace radius_cell_mass {double owner_scratch[5000];}
using namespace radius_cell_mass;
template<class T> void check(int B,bool reverse) {
    Axis a=parse_axis({1024,.37,1,-1.2,8,2,1.,.3,3.,3.,1.,.03,1.48,189.44,.37});
    int A=7,J=a.node_count-prefix_nodes(a,2),F=17;
    std::vector<T> p(A*4),x(B*a.sites),gradient(A*4);
    double amp[7]={1.,.3,.15,.15,.15,0.,.15},q[7]={1.,1.,1.,1.8,2.6,4.,4.};
    for(int atom=0;atom<A;++atom) {
        p[4*atom]=T(amp[atom]*sqrt(q[atom]));
        p[4*atom+1]=T(sqrt(1-amp[atom]*amp[atom])*sqrt(q[atom]));
        p[4*atom+2]=T(a.origin+atom*13.7);
        p[4*atom+3]=T(a.origin+(atom%2?a.period-1e-4:0));
    }
    for(int b=0;b<B;++b)for(int i=0;i<a.sites;++i)x[b*a.sites+i]=T(sin(i*.13+b*.17));
    std::vector<double> z(B*J*F),h(B*A),dh(B*A),g(B*A),dg(B*A);
    for(int b=0;b<B;++b)for(int node=prefix_nodes(a,2);node<a.node_count;++node)for(int f=0;f<F;++f)
        z[(b*J+node-prefix_nodes(a,2))*F+f]=compact_moment_element(x.data(),z.data(),b,node,f,a,2,false);
    for(int atom=0;atom<A;++atom) {
        double R=radius(p.data(),atom,a),original=center(p.data(),atom,a,reverse);
        Membership m=classify(R,original,a);Bank bank=a.banks[m.bank];
        double c[17],value[4]={},slope[4]={},old[4]={},unused[4]={};
        double mu=wrapped_center(original,a);
        profile_values(R,bank.period,8,c);
        owner_complete_response_derivatives(x.data(),z.data(),value,slope,B,a,m,mu,c,2,true);
        owner_complete_response_derivatives(x.data(),z.data(),old,unused,B,a,m,mu,c,2,false);
        for(int b=0;b<B;++b) {
            double expected=0,derivative=0;
            for(int part=0;part<bank.count;++part)for(int local=0;local<bank.tile_sites;++local) {
                int site=(m.first_tile+part)*bank.tile_sites+local;
                double mass=a.spacing*c[0],dm=0;
                for(int k=1;k<=8;++k) {
                    double omega=2*PI*k/bank.period,arg=PI*k*a.spacing/bank.period;
                    double factor=2*c[k]*a.spacing*sin(arg)/arg;
                    mass+=factor*cos(omega*(mu-site*a.spacing));
                    dm-=factor*omega*sin(omega*(mu-site*a.spacing));
                }
                double signal=double(x[b*a.sites+positive_mod(site,a.sites)]);
                expected+=signal*mass;derivative+=signal*dm;
            }
            assert(std::isfinite(value[b])&&std::isfinite(slope[b]));
            assert(value[b]==old[b]&&unused[b]==0);
            assert(fabs(value[b]-expected)<2e-10*fmax(1.,fabs(expected)));
            assert(fabs(slope[b]-derivative)<2e-9*fmax(1.,fabs(derivative)));
            h[b*A+atom]=value[b];dh[b*A+atom]=slope[b];
            g[b*A+atom]=.7+b*.13;dg[b*A+atom]=-.2+b*.11;
        }
        threadIdx.x=atom;
        owner_derivative_parameter_kernel(p.data(),h.data(),g.data(),dh.data(),dg.data(),gradient.data(),A,B,a);
        double da=0,dmi=0,dmo=0;
        for(int b=0;b<B;++b) {int i=b*A+atom;da+=h[i]*g[i];dmi+=amplitude(p.data(),atom,a)*g[i]*dh[i];dmo+=amplitude(p.data(),atom,a)*h[i]*dg[i];}
        double s=p[4*atom],t=p[4*atom+1],safe=s*s+t*t;
        T expected[4]={T(da/sqrt(safe)*(1-s*s/safe)),T(-da/sqrt(safe)*s*t/safe),T(dmi),T(dmo)};
        for(int k=0;k<4;++k)assert(fabs(double(gradient[4*atom+k])-expected[k])<1e-6*fmax(1.,fabs(expected[k])));
    }
    // Instantiate the real cooperative kernel without pretending to emulate
    // CTA barriers on CPU: every owner is empty and exits before its barrier.
    threadIdx.x=0;std::vector<int64_t> empty(a.node_count+1),rows(A);
    owner_response_derivatives_kernel(p.data(),x.data(),z.data(),h.data(),dh.data(),nullptr,rows.data(),empty.data(),A,B,a,a,2,reverse,true);
    for(int atom=0;atom<A;++atom) {threadIdx.x=atom;owner_derivative_project_kernel(p.data(),x.data(),z.data(),h.data(),dh.data(),A,B,a,reverse);}
}
int main() {static_assert(OWNER_SHARED_BYTES==34432);for(int B:{0,1,3,4})for(bool reverse:{false,true}) {check<float>(B,reverse);check<double>(B,reverse);}}
"""
    )
    path, binary = tmp_path / "derivatives.cpp", tmp_path / "derivatives"
    path.write_text(source)
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(path),
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


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_all_vjps_boundary_images_skew_retained_and_offload(
    dtype, batch, needs_x, needs_p, skew
):
    original = model(dtype, skew)
    reference = copy.deepcopy(original).double()
    reference.atoms.p.requires_grad_(needs_p)
    candidate = original.cuda()
    candidate.atoms.p.requires_grad_(needs_p)
    candidate.selector = FixedSelector(NEW, registry=make_algorithm_registry())
    generator = torch.Generator().manual_seed(2081)
    host_x = torch.randn(batch, 1024, dtype=dtype, generator=generator)
    host_dy = torch.randn(batch, 1024, dtype=dtype, generator=generator)
    rx = host_x.double().requires_grad_(needs_x)
    expected = reference(rx)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(v for v in (rx, reference.atoms.p) if v.requires_grad),
        host_dy.double(),
    )
    shapes = []

    def pack(value):
        shapes.append(tuple(value.shape))
        return value.detach().cpu().clone(), value.device

    def unpack(value):
        return value[0].to(value[1])

    x = host_x.cuda().requires_grad_(needs_x)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual = candidate(x)
    selected = tuple(v for v in (x, candidate.atoms.p) if v.requires_grad)
    gradients = torch.autograd.grad(actual, selected, host_dy.cuda(), retain_graph=True)
    repeated = torch.autograd.grad(actual, selected, host_dy.cuda())
    for a, b in zip(gradients, repeated):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    assert (batch, 1022, 17) not in shapes
    assert (batch, 4094, 3, 17) not in shapes
    if batch:
        assert shapes.count((batch, len(candidate.atoms.p))) == (2 if needs_p else 1)
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    for got, want in zip((actual, *gradients), (expected, *expected_gradients)):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.cpu().double(), want, **tolerance)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3, 4])
def test_project_derivative_is_independent_of_amplitude_and_matches_cpu_cells(
    dtype, batch
):
    from torchcst._backends.torch.kernels.radius_cell_mass import axis_masses
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )
    from torchcst.kernels.radius_cell_mass import radius_cell_mass_axes

    m = model(dtype)
    plan = hierarchy_from_operator(m.declaration())
    signal = torch.randn(batch, 1024, dtype=dtype)
    p = m.atoms.p.detach().double()
    _, _, radius = decoded_values(m.kernel, m.atoms.p.detach())
    center = p[:, 3].clone().requires_grad_()
    axis = radius_cell_mass_axes(
        (m.input_chart.declaration(), m.output_chart.declaration()),
        m.kernel.declaration(),
    )[1]
    expected = signal.double() @ axis_masses(axis, radius.detach(), center).T
    # Each atom response depends only on its own center. Differentiating the
    # row sum extracts all per-atom center derivatives, including amplitude0.
    slopes = torch.stack(
        [
            torch.autograd.grad(row.sum(), center, retain_graph=True)[0]
            for row in expected
        ]
    )
    actual, derivative = ex._extension().owner_derivative_project(
        signal.cuda(), m.atoms.p.detach().cuda(), plan.output.native_config()
    )
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    for got, want in ((actual, expected), (derivative, slopes)):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.cpu(), want, **tolerance)


@GPU
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
def test_raw_saved_schema_replaces_z_and_only_saves_derivatives_for_atom_edges(
    needs_x, needs_p
):
    m = model(torch.float32).cuda()
    m.atoms.p.requires_grad_(needs_p)
    plan = hierarchy_from_operator(m.declaration())
    x = torch.randn(4, 1024, device="cuda", requires_grad=needs_x)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        y = ex.checked_radius_linear(plan, x, m.atoms.p, recipe=NEW.recipe)
    assert len(saved) == (5 if needs_x and needs_p else 4)
    assert sum(tuple(t.shape) == (4, 7) for t in saved) == (2 if needs_p else 1)
    assert sum(t.dtype == torch.int64 for t in saved) == int(needs_x)
    assert not any(tuple(t.shape) == (4, 1022, 17) for t in saved)
    y.sum().backward()


@GPU
@pytest.mark.parametrize(
    "fault",
    ["x_stride", "p_stride", "h_shape", "dh_shape", "dh_dtype", "dh_stride", "keys"],
)
def test_saved_hook_metadata_checked_before_any_reverse_launch(fault, tmp_path):
    m = model(torch.float32).cuda()
    x = torch.randn(4, 1024, device="cuda", requires_grad=True)
    plan = hierarchy_from_operator(m.declaration())
    records = []

    def pack(value):
        index = len(records)
        records.append(value)
        return value, index

    def unpack(record):
        value, index = record
        target = {
            "x_stride": 0,
            "p_stride": 1,
            "h_shape": 2,
            "dh_shape": 3,
            "dh_dtype": 3,
            "dh_stride": 3,
            "keys": 4,
        }[fault]
        if index != target:
            return value
        if fault.endswith("stride"):
            return value.new_empty(value.shape[0], value.shape[1] * 2)[:, ::2].copy_(
                value
            )
        if fault == "dh_dtype":
            return value.float()
        return value[:-1]

    before = m.atoms.p.detach().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = ex.checked_radius_linear(plan, x, m.atoms.p, recipe=NEW.recipe)
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile,
        pytest.raises(RuntimeError, match="contiguous|response|keys"),
    ):
        torch.autograd.grad(y.sum(), (x, m.atoms.p))
    # sum/backward may launch its own reduction/broadcast; the rejected owner
    # must never reach routing, response, expansion or parameter CUDA work.
    # CPU autograd annotations contain the C++ Function's namespace too. Read
    # actual CUDA kernel records, not inclusive/user annotations with GPU time.
    trace = tmp_path / "rejected-owner-backward.json"
    profile.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    assert not any(
        event.get("cat") == "kernel" and "radius_cell_mass::" in event.get("name", "")
        for event in events
    )
    assert torch.equal(before.view(torch.uint8), m.atoms.p.detach().view(torch.uint8))


@GPU
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
def test_existing_independent_contracts_on_explicit_derivative_plan(
    monkeypatch, contract
):
    # Reuse the qualified assertions without editing or weakening the old
    # tests. Only their selected explicit Plan changes for this invocation.
    import test_radius_owner_response_coeffs as old

    monkeypatch.setattr(old, "OWNER", NEW)
    if contract == "public_retained_hooks":
        old.test_owner_public_hooks_retained_read_leases_and_fresh_live_parameters()
    elif contract == "raw_version_once_diff":
        old.test_owner_raw_saved_variable_version_and_higher_derivative_guards()
    elif contract in ("no_grad", "inference", "frozen"):
        old.test_owner_nonexecutable_native_context(contract)
    else:
        old.test_owner_raw_native_empty_shapes_have_zero_complete_vjps(
            4 if contract == "empty_atoms" else 0,
            0 if contract == "empty_atoms" else 7,
        )


@GPU
@pytest.mark.parametrize(
    "fault",
    [
        "batch",
        "sites",
        "polar",
        "modes",
        "spacing",
        "origin",
        "bounds",
        "cutoff",
        "method",
        "tile",
    ],
)
@pytest.mark.parametrize("checked", [False, True])
def test_both_raw_entries_reject_geometry_before_classification(fault, checked):
    m = model(torch.float32).cuda()
    axes = hierarchy_from_operator(m.declaration())
    ci, co = list(axes.input.native_config()), list(axes.output.native_config())
    x = torch.randn(5 if fault == "batch" else 4, 1024, device="cuda")
    start, merge, tile = 2, True, 32
    if fault == "sites":
        ci[0] = 128
    elif fault == "polar":
        ci = ci[:6]
    elif fault == "modes":
        ci[4] = 3
    elif fault == "spacing":
        co[1] *= 2
    elif fault == "origin":
        co[3] += 1
    elif fault == "bounds":
        co[12] += 1
    elif fault == "cutoff":
        start = 3
    elif fault == "method":
        merge = False
    elif fault == "tile":
        tile = 16
    native = ex._extension()
    function = native.checked_autograd_linear if checked else native.autograd_linear
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile,
        pytest.raises(RuntimeError),
    ):
        function(x, m.atoms.p, ci, co, start, 2, merge, tile, 7)
    assert not any(
        str(event.device_type).endswith(".CUDA") for event in profile.events()
    )
