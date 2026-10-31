"""Owner response/C prototype: bounded scratch, complete caches, no atom copies."""

import copy
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

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.training.cell_mass_mnist import make_radius_cst
from torchcst import (
    ChebyshevRadiusSpec,
    CSTLinear,
    FixedSelector,
    LinearInputs,
    RadiusBounds,
    chart_presets,
    make_algorithm_registry,
    presets,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_RESPONSE_COEFFICIENTS as OWNER,
)
from torchcst._backends.schema import DeviceInfo

ROOT = Path(__file__).resolve().parents[1]


def model(dtype=torch.float64, skew=False):
    chart = chart_presets.regular_grid(((1024,),), origin=-1.2, geometry="flat_torus")
    result = CSTLinear(
        chart,
        chart,
        atoms=129 if skew else 7,
        dtype=dtype,
        backend="factored",
        kernel=presets.polar_radius_cell_mass(
            amplitude_max=1.0,
            w_c=0.3,
            input_radius_bounds=RadiusBounds(
                minimum=0.1, birth=4.0, maximum=512.0, upper_floor=1.0
            ),
            profile=ChebyshevRadiusSpec(modes=8),
        ),
    )
    with torch.no_grad():
        if skew:
            result.atoms.p[:, 0] = 0.15
            result.atoms.p[:, 1] = (1 - 0.15**2) ** 0.5
            result.atoms.p[:, 2] = torch.arange(129, dtype=dtype) * 13.7
            # These two unwrapped owners share the last wrapped owner. P=12
            # does not divide Q=1024; each atom must keep its own phase anchor.
            result.atoms.p[:, 3] = torch.where(
                torch.arange(129) % 2 == 0,
                torch.tensor(-1.2, dtype=dtype),
                torch.tensor(1022.8 - 1e-4, dtype=dtype),
            )
        else:
            result.atoms.p.copy_(
                torch.tensor(
                    [
                        [1, 0, -1.2, 1022.8],
                        [0.3, 0.9539392014169457, -1025.2, -1.2001],
                        [0.15, 0.9886859966642595, 1022.79, 1022.81],
                        [0.15, 1.33, 4096, -4096],
                        [0.15, 1.6, 7.3, 8.2],
                        [0, 2, 32, 64],
                        [0.3, 1.977371993328519, 128, 256],
                    ],
                    dtype=dtype,
                )
            )
    return result


def test_owner_plan_roundtrip_scope_budgets_and_old_defaults():
    run = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-response-coeffs-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == GLOBAL_ACTIVE_GRAD4_NATIVE
    assert run.entry("radius-global-owner-response-coeffs").plan == OWNER
    assert run.dense and decode_snapshot(run.snapshot()) == run
    assert 4 * 4094 * 3 * 17 * 8 == 6_681_408
    assert 8 * (4 * 3 * 17 + 4 * 17 * 33 + 128 * 14) + 4 * 128 == 34_432
    assert (
        make_radius_cst(128, 7).selector._plan.recipe.schedule
        == "block_local_cached_parallel"
    )
    context = model().build_context(
        LinearInputs(torch.zeros(4, 1024, dtype=torch.float64))
    )
    context = replace(context, device=DeviceInfo("cuda", 0, "metadata-only"))
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(context, OWNER.recipe).supported
    assert algorithm.supports(
        replace(context, atom_count=65536), OWNER.recipe
    ).supported
    assert not algorithm.supports(
        replace(context, atom_count=65537), OWNER.recipe
    ).supported
    assert not algorithm.supports(
        replace(context, input_shape=(5, 1024)), OWNER.recipe
    ).supported
    for changed in (
        {"moment_method": "direct"},
        {"input_direct_levels": 1},
        {"output_direct_levels": 5},
        {"output_tile_sites": 16},
        {"native_autograd": False},
        {"native_entry": False},
    ):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(replace(OWNER.recipe, **changed))


def test_owner_support_rejects_changed_live_axes_and_other_families():
    from test_radius_large_shapes import metadata_context

    algorithm = RadiusCellMassAlgorithm()
    assert not algorithm.supports(metadata_context(), OWNER.recipe).supported
    small = make_radius_cst(128, 7)
    context = small.build_context(LinearInputs(torch.zeros(4, 128)))
    context = replace(context, device=DeviceInfo("cuda", 0, "metadata-only"))
    assert not algorithm.supports(context, OWNER.recipe).supported
    changed = model()
    with torch.no_grad():
        changed.kernel.radius_birth_output.add_(0.25)
    context = changed.build_context(
        LinearInputs(torch.zeros(4, 1024, dtype=torch.float64))
    )
    context = replace(context, device=DeviceInfo("cuda", 0, "metadata-only"))
    assert not algorithm.supports(context, OWNER.recipe).supported


@pytest.mark.parametrize("custom", ["subclass", "function_mode", "dispatch_mode"])
def test_owner_custom_tensor_and_modes_retain_existing_python_fallback(
    monkeypatch, custom
):
    from contextlib import nullcontext

    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class Custom(torch.Tensor):
        pass

    m = model(torch.float32)
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    plan = hierarchy_from_operator(m.declaration())
    x = torch.zeros(4, 1024)
    if custom == "subclass":
        x = x.as_subclass(Custom)
    context = (
        FunctionMode()
        if custom == "function_mode"
        else DispatchMode()
        if custom == "dispatch_mode"
        else nullcontext()
    )

    def fallback(actual_plan, actual_x, actual_p, *, recipe):
        assert actual_plan is plan and actual_x is x and actual_p is m.atoms.p
        assert recipe.schedule == "global_batched_cached_active_moments_grad4"
        assert not recipe.native_autograd and not recipe.native_entry
        return "fallback"

    monkeypatch.setattr(ex, "radius_linear", fallback)
    monkeypatch.setattr(
        ex, "_extension", lambda: pytest.fail("raw native entry before fallback")
    )
    with context:
        assert (
            ex.checked_radius_linear(plan, x, m.atoms.p, recipe=OWNER.recipe)
            == "fallback"
        )


def test_owner_checked_cpu_wrapper_uses_new_mode(monkeypatch):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

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
    assert ex.checked_radius_linear(plan, x, m.atoms.p, recipe=OWNER.recipe) is x
    assert calls[0][-1] == 6


def test_actual_owner_helpers_math_mapping_and_kernel_syntax_on_cpu(tmp_path):
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
    moment = hybrid[
        hybrid.index(
            "template<class T>\n__device__ inline double compact_moment_element"
        ) : hybrid.index("template<class T>\n__global__ void compact_moments_kernel")
    ]
    owner = (directory / "owner_response_coefficients.cuh").read_text().split(
        "// Host orchestration:"
    )[0] + "\n}\n"
    source = (
        r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
#include <cmath>
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
inline void radius_test_sincos(double x,double*s,double*c) {*s=std::sin(x);*c=std::cos(x);}
#define sincos radius_test_sincos
#include "common.cuh"
namespace radius_cell_mass {
constexpr int BLOCK_LOCAL_C_STRIDE=33;
inline int bank_for_node(int node,const Axis&a) {int b=0;while(b+1<a.bank_count && node>=a.banks[b+1].node_offset)++b;return b;}
"""
        + phase
        + moment
        + "\n}\n"
        + owner
        + r"""
namespace radius_cell_mass {double owner_scratch[5000];}
using namespace radius_cell_mass;
double mass(const double*c,int first,int part,int local,Axis a,Bank bank,double mu) {
    int site=(first+part)*bank.tile_sites+local;
    double v=a.spacing*c[0];
    for(int k=1;k<=a.modes;++k) {
        double arg=PI*k*a.spacing/bank.period;
        v+=2*c[k]*a.spacing*sin(arg)/arg*cos(2*PI*k/bank.period*(mu-site*a.spacing));
    }
    return v;
}
template<class T> void check(int B,bool skew,bool reverse) {
    Axis a=parse_axis({1024,.37,1,-1.2,8,2,1.,.3,3.,3.,1.,.03,1.48,189.44,.37});
    int A=skew?129:7,F=17,start=2,J=a.node_count-prefix_nodes(a,start);
    std::vector<T> p(A*4),x(B*a.sites);
    for(int b=0;b<B;++b)for(int i=0;i<a.sites;++i)x[b*a.sites+i]=T(sin(i*.13+b*.17));
    double amps[7]={1.,.3,.15,.15,.15,0.,.15},q[7]={1.,1.,1.,1.8,2.6,4.,4.};
    for(int atom=0;atom<A;++atom) {
        int r=skew?2:atom;
        p[4*atom]=T(amps[r]*sqrt(q[r]));p[4*atom+1]=T(sqrt(1-amps[r]*amps[r])*sqrt(q[r]));
        p[4*atom+2]=T(a.origin+atom*13.7);
        p[4*atom+3]=T(a.origin+(atom%2?a.period-1e-4:0));
    }
    std::vector<double> z(B*J*F),partial(B*a.node_count*3*F,std::numeric_limits<double>::quiet_NaN()),h(B*A),expected(B*a.sites);
    std::vector<int64_t> offsets(a.node_count+1,0);
    for(int b=0;b<B;++b)for(int node=prefix_nodes(a,start);node<a.node_count;++node)for(int f=0;f<F;++f)
        z[(b*J+node-prefix_nodes(a,start))*F+f]=compact_moment_element(x.data(),z.data(),b,node,f,a,start,false);
    for(int atom=0;atom<A;++atom) {
        double R=radius(p.data(),atom,a),src=center(p.data(),atom,a,reverse),dest=center(p.data(),atom,a,!reverse);
        Membership sm=classify(R,src,a),dm=classify(R,dest,a);Bank bank=a.banks[dm.bank];
        double c[17],cached[OWNER_STRIDE]={},values[4]={};
        profile_values(R,bank.period,8,c);
        owner_complete_response(x.data(),z.data(),values,B,a,sm,wrapped_center(src,a),c,start);
        int node=bank.node_offset+positive_mod(dm.first_tile,bank.tile_count);
        int fields=bank.level<start?bank.tile_sites:F;
        if(!offsets[node+1])for(int b=0;b<B;++b)for(int part=0;part<bank.count;++part)for(int f=0;f<fields;++f)
            partial[owner_partial_index(b,node,part,f,a)]=0;
        ++offsets[node+1];
        for(int k=0;k<=8;++k)cached[k]=c[k];cached[9]=wrapped_center(dest,a);
        for(int b=0;b<B;++b) {
            double independent=0;
            Bank sb=a.banks[sm.bank];
            for(int part=0;part<sb.count;++part)for(int local=0;local<sb.tile_sites;++local) {
                int site=positive_mod((sm.first_tile+part)*sb.tile_sites+local,a.sites);
                independent+=double(x[b*a.sites+site])*mass(c,sm.first_tile,part,local,a,sb,wrapped_center(src,a));
            }
            assert(std::isfinite(values[b]) && fabs(values[b]-independent)<=2e-10*fmax(1.,fabs(independent)));
            h[b*A+atom]=values[b];cached[10+b]=amplitude(p.data(),atom,a)*values[b];
            for(int part=0;part<bank.count;++part)for(int local=0;local<bank.tile_sites;++local) {
                int site=positive_mod((dm.first_tile+part)*bank.tile_sites+local,a.sites);
                expected[b*a.sites+site]+=amplitude(p.data(),atom,a)*independent*mass(c,dm.first_tile,part,local,a,bank,wrapped_center(dest,a));
            }
        }
        for(int part=0;part<bank.count;++part) {
            double columns[17];owner_columns(cached,dm.first_tile,part,a,bank,columns,1,0);
            for(int f=0;f<fields;++f) {
                double value=bank.level<start?owner_direct_mass(cached,dm.first_tile,part,f,a,bank):columns[f];
                for(int b=0;b<B;++b)partial[owner_partial_index(b,node,part,f,a)]+=cached[10+b]*value;
            }
        }
    }
    for(int i=1;i<=a.node_count;++i)offsets[i]+=offsets[i-1];
    for(int b=0;b<B;++b)for(int site=0;site<a.sites;++site) {
        double actual=owner_partial_site(partial.data(),offsets.data(),b,site,a,start);
        assert(std::isfinite(actual));
        assert(fabs(actual-expected[b*a.sites+site])<=2e-9*fmax(1.,fabs(expected[b*a.sites+site])));
    }
    // Instantiate both real kernels. No cooperative CUDA execution is claimed
    // by this CPU smoke: all owners are empty and the final grid has B=0.
    std::vector<int64_t> empty(a.node_count+1,0),rows(A);
    owner_response_coefficients_kernel(p.data(),x.data(),z.data(),h.data(),partial.data(),rows.data(),empty.data(),A,B,a,a,start,reverse);
    owner_expand_kernel(partial.data(),offsets.data(),x.data(),0,a,start);
}
int main() {
    static_assert(OWNER_SHARED_BYTES==34432);
    for(bool skew:{false,true})for(bool reverse:{false,true})for(int batch:{0,1,4}) {
        check<float>(batch,skew,reverse);check<double>(batch,skew,reverse);
    }
}
"""
    )
    path, binary = tmp_path / "owner.cpp", tmp_path / "owner"
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
@pytest.mark.parametrize("batch", [0, 1, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_owner_all_vjps_direct_wide_images_skew_retained_offload(
    dtype, batch, needs_x, needs_p, skew
):
    original = model(dtype, skew)
    reference = copy.deepcopy(original).double()
    reference.atoms.p.requires_grad_(needs_p)
    candidate = original.cuda()
    candidate.atoms.p.requires_grad_(needs_p)
    candidate.selector = FixedSelector(OWNER, registry=make_algorithm_registry())
    gen = torch.Generator().manual_seed(2081)
    host_x = torch.randn(batch, 1024, dtype=dtype, generator=gen)
    host_dy = torch.randn(batch, 1024, dtype=dtype, generator=gen)
    rx = host_x.double().requires_grad_(needs_x)
    expected = reference(rx)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(v for v in (rx, reference.atoms.p) if v.requires_grad),
        host_dy.double(),
    )
    x = host_x.cuda().requires_grad_(needs_x)
    shapes = []

    def pack(value):
        shapes.append(tuple(value.shape))
        return value.detach().cpu().clone(), value.device

    def unpack(value):
        return value[0].to(value[1])

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual = candidate(x)
    selected = tuple(v for v in (x, candidate.atoms.p) if v.requires_grad)
    gradients = torch.autograd.grad(actual, selected, host_dy.cuda(), retain_graph=True)
    repeated = torch.autograd.grad(actual, selected, host_dy.cuda())
    for a, b in zip(gradients, repeated):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    if batch:
        assert (batch, 1022, 17) in shapes
    assert (batch, 4094, 3, 17) not in shapes
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    for actual_value, expected_value in zip(
        (actual, *gradients), (expected, *expected_gradients)
    ):
        assert bool(torch.isfinite(actual_value).all())
        torch.testing.assert_close(
            actual_value.cpu().double(), expected_value, **tolerance
        )


@GPU
@pytest.mark.parametrize("batch,count", [(0, 7), (4, 0)])
def test_owner_raw_native_empty_shapes_have_zero_complete_vjps(batch, count):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    atoms = torch.empty((0, 4)) if count == 0 else count
    m = make_radius_cst(1024, atoms, radius_execution="global-response4").cuda()
    plan = hierarchy_from_operator(m.declaration())
    x = torch.randn(batch, 1024, device="cuda", requires_grad=True)
    y = ex.checked_radius_linear(plan, x, m.atoms.p, recipe=OWNER.recipe)
    gx, gp = torch.autograd.grad(y.sum(), (x, m.atoms.p))
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(gx, torch.zeros_like(x))
    assert torch.equal(gp, torch.zeros_like(m.atoms.p))


@GPU
@pytest.mark.parametrize("mode", ["no_grad", "inference", "frozen"])
def test_owner_nonexecutable_native_context(mode):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    cpu = model(torch.float32)
    candidate = copy.deepcopy(cpu).cuda()
    plan = hierarchy_from_operator(candidate.declaration())
    x = torch.randn(4, 1024)
    if mode == "frozen":
        cpu.requires_grad_(False)
        candidate.requires_grad_(False)
        expected = cpu(x)
        actual = ex.checked_radius_linear(
            plan, x.cuda(), candidate.atoms.p, recipe=OWNER.recipe
        )
    else:
        context = torch.no_grad if mode == "no_grad" else torch.inference_mode
        with context():
            expected = cpu(x)
            actual = ex.checked_radius_linear(
                plan, x.cuda(), candidate.atoms.p, recipe=OWNER.recipe
            )
    assert not actual.requires_grad
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-4)


@GPU
@pytest.mark.parametrize(
    "fault",
    [
        "signal_stride",
        "parameter_stride",
        "moment_dtype",
        "response_shape",
        "key_count",
    ],
)
def test_owner_saved_hook_metadata_checked_before_reverse_owner_launch(fault):
    candidate = model(torch.float32).cuda()
    candidate.selector = FixedSelector(OWNER, registry=make_algorithm_registry())
    x = torch.randn(4, 1024, device="cuda", requires_grad=True)

    def pack(value):
        shape = tuple(value.shape)
        kind = (
            "signal_stride"
            if shape == (4, 1024)
            else "parameter_stride"
            if shape == (7, 4)
            else "moment_dtype"
            if shape == (4, 1022, 17)
            else "response_shape"
            if shape == (4, 7)
            else "key_count"
            if shape == (7,) and value.dtype == torch.int64
            else None
        )
        return value, kind

    def unpack(record):
        value, kind = record
        if kind != fault:
            return value
        if kind in ("signal_stride", "parameter_stride"):
            result = value.new_empty(value.shape[0], value.shape[1] * 2)[:, ::2]
            return result.copy_(value)
        if kind == "moment_dtype":
            return value.float()
        if kind == "response_shape":
            return value[:, :-1]
        return value[:-1]

    before = candidate.atoms.p.detach().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = candidate(x)
    with pytest.raises(RuntimeError, match="contiguous|compact|response|keys"):
        torch.autograd.grad(y.sum(), (x, candidate.atoms.p))
    assert torch.equal(
        before.view(torch.uint8), candidate.atoms.p.detach().view(torch.uint8)
    )


@GPU
@pytest.mark.parametrize(
    "fault",
    [
        "batch",
        "sites",
        "polar",
        "modes",
        "base",
        "spacing",
        "origin",
        "bounds",
        "cutoff",
        "method",
        "tile",
    ],
)
def test_owner_raw_native_rejects_unsupported_before_classification(fault):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

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
    elif fault == "base":
        ci[2] = 2
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
    before = m.atoms.p.detach().clone()
    native = ex._extension()
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof,
        pytest.raises(RuntimeError),
    ):
        native.checked_autograd_linear(x, m.atoms.p, ci, co, start, 2, merge, tile, 6)
    assert not any(str(event.device_type).endswith(".CUDA") for event in prof.events())
    assert torch.equal(before.view(torch.uint8), m.atoms.p.detach().view(torch.uint8))


@GPU
def test_owner_public_hooks_retained_read_leases_and_fresh_live_parameters():
    candidate = model(torch.float32).cuda()
    candidate.selector = FixedSelector(OWNER, registry=make_algorithm_registry())
    x = torch.randn(4, 1024, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    calls = []
    handle = candidate.register_forward_hook(lambda *args: calls.append("forward"))
    y = candidate(x)
    torch.autograd.grad(y, (x, candidate.atoms.p), dy, retain_graph=True)
    with pytest.raises(RuntimeError, match="outstanding|retained"):
        candidate.atom_state._require_boundary()
    torch.autograd.grad(y, (x, candidate.atoms.p), dy)
    candidate.atom_state._require_boundary()
    assert calls == ["forward"]
    with torch.no_grad():
        candidate.atoms.p[:, :2].mul_(1.05)
        candidate.atoms.p[:, 2:].add_(1.25)
    reference = model(torch.float64)
    reference.load_state_dict(candidate.state_dict())
    reference.selector, reference.backend = None, "factored"
    rx = x.detach().cpu().double().requires_grad_()
    expected = reference(rx)
    expected_gradients = torch.autograd.grad(
        expected, (rx, reference.atoms.p), dy.cpu().double()
    )
    actual = candidate(x)
    gradients = torch.autograd.grad(actual, (x, candidate.atoms.p), dy)
    for got, want in zip((actual, *gradients), (expected, *expected_gradients)):
        torch.testing.assert_close(got.cpu().double(), want, atol=3e-5, rtol=3e-4)
    assert calls == ["forward", "forward"]
    handle.remove()


@GPU
def test_owner_raw_saved_variable_version_and_higher_derivative_guards():
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    m = model().cuda()
    plan = hierarchy_from_operator(m.declaration())
    x = torch.randn(4, 1024, dtype=torch.float64, device="cuda", requires_grad=True)
    p = m.atoms.p.detach().clone().requires_grad_()
    y = ex.checked_radius_linear(plan, x, p, recipe=OWNER.recipe)
    with torch.no_grad():
        p.add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()
    y = ex.checked_radius_linear(plan, x, p, recipe=OWNER.recipe)
    gradient = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)
