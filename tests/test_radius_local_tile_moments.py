"""Opt-in local tile moments retain complete FP64 responses and cutoff2 C."""

import copy
import math
import os
import shlex
import shutil
import subprocess
import sysconfig
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from test_radius_cell_mass_cuda import GPU

from benchmarks.training.cell_mass_mnist import make_radius_cst
from torchcst import FixedSelector, LinearInputs, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.expansion_layout import (
    batched_coefficient_layout,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    BLOCK_LOCAL_BATCHED_CACHED_PARALLEL,
    LOCAL_TILE_MOMENTS,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.recipe import (
    RadiusCellMassRecipe,
)
from torchcst._backends.schema import DeviceInfo


def mixed_model(dtype=torch.float64):
    model = make_radius_cst(128, 11, dtype=dtype)
    # Reach every possible support hierarchy level0..6, including zero and
    # negative amplitudes. q changes activity, while s/sqrt(q) fixes amplitude.
    with torch.no_grad():
        for i, (amp, q) in enumerate(
            [
                (1.0, 1.0),
                (0.3, 1.0),
                (0.15, 1.0),
                (0.15, 1.8),
                (0.15, 2.6),
                (0.15, 3.4),
                (0.15, 4.0),
                (0.0, 1.0),
                (0.0, 4.0),
                (-0.15, 3.0),
                (0.5, 1.0),
            ]
        ):
            model.atoms.p[i, :2] = torch.tensor(
                [amp * math.sqrt(q), math.sqrt(1 - amp * amp) * math.sqrt(q)],
                dtype=dtype,
            )
        model.atoms.p[:, 2:] = torch.tensor(
            [
                [0.0, 128.0],
                [-128.0, -1e-7],
                [127.99, 128.01],
                [512.0, -512.0],
                [-0.0, 0.0],
                [256.0, -256.0],
                [7.3, 8.2],
                [32.0, 64.0],
                [96.0, 16.0],
                [-0.01, 128.3],
                [32.1, -31.9],
            ],
            dtype=dtype,
        )
    return model


def metadata_context(batch=32):
    model = mixed_model()
    return replace(
        model.build_context(LinearInputs(torch.zeros(batch, 128, dtype=torch.float64))),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )


def test_local_tile_plan_schema_default_and_payload_budgets():
    algorithm = RadiusCellMassAlgorithm()
    registry = make_algorithm_registry()
    assert (
        registry.loads_plan(registry.dumps_plan(LOCAL_TILE_MOMENTS))
        == LOCAL_TILE_MOMENTS
    )
    assert tuple(f.name for f in fields(RadiusCellMassRecipe)) == (
        "moment_method",
        "save_response",
        "input_direct_levels",
        "output_direct_levels",
        "schedule",
        "output_tile_sites",
        "native_autograd",
        "native_entry",
    )
    assert RadiusCellMassRecipe().schedule == "sequential"
    assert (
        make_radius_cst(128, 7).selector._plan.recipe.schedule
        == "block_local_cached_parallel"
    )
    assert algorithm.supports(metadata_context(), LOCAL_TILE_MOMENTS.recipe).supported
    assert not algorithm.supports(
        metadata_context(33), LOCAL_TILE_MOMENTS.recipe
    ).supported
    for count in (819, 4003, 4096):
        assert algorithm.supports(
            replace(metadata_context(), atom_count=count), LOCAL_TILE_MOMENTS.recipe
        ).supported
    assert not algorithm.supports(
        replace(metadata_context(), atom_count=4097), LOCAL_TILE_MOMENTS.recipe
    ).supported
    model = mixed_model()
    inputs, outputs = model.support_radii()
    for radius in (inputs, outputs):
        assert set(radius.detach().log2().ceil().clamp_min(0).int().tolist()) == set(
            range(7)
        )
    axis = hierarchy_from_operator(mixed_model().declaration()).input
    assert axis.active_nodes(5) == 14 and axis.active_nodes(2) == 126
    for batch in (0, 1, 4, 5, 31, 32):
        shared_bytes = 8 * (batch * 3 * 17 + 16 * 17 + 17)
        assert shared_bytes <= 15368 < 48 * 1024
        assert batched_coefficient_layout(axis, 2, 32).bytes < 48 * 1024
    assert 16 * 32 * (126 - 14) * 17 == 974848  # eliminated global Z payload
    assert 16 * 32 * 819 + 16 * 32 * 14 * 17 == 541184  # H/G plus both retained Z


@pytest.mark.parametrize(
    "change",
    [
        {"input_direct_levels": 1},
        {"output_direct_levels": 5},
        {"output_tile_sites": 16},
        {"moment_method": "direct"},
        {"native_entry": False},
        {"native_autograd": False},
    ],
)
def test_local_tile_recipe_rejects_unsupported_execution_before_work(change):
    with pytest.raises(ValueError):
        RadiusCellMassAlgorithm().validate_recipe(
            replace(LOCAL_TILE_MOMENTS.recipe, **change)
        )


def test_local_tile_generic_tensor_entrance_uses_existing_python_batched_fallback(
    monkeypatch,
):
    class Custom(torch.Tensor):
        pass

    x = torch.zeros(5, 128).as_subclass(Custom)
    model = mixed_model(torch.float32)
    plan = hierarchy_from_operator(model.declaration())
    calls = []

    def fallback(actual_plan, actual_x, actual_p, *, recipe):
        calls.append(recipe)
        assert actual_plan is plan and actual_x is x and actual_p is model.atoms.p
        assert recipe.schedule == "global_batched_cached_response4_grad4"
        assert not recipe.native_entry and not recipe.native_autograd
        return "fallback"

    monkeypatch.setattr(ex, "radius_linear", fallback)
    monkeypatch.setattr(
        ex, "_extension", lambda: pytest.fail("native work before fallback")
    )
    assert (
        ex.checked_radius_linear(
            plan, x, model.atoms.p, recipe=LOCAL_TILE_MOMENTS.recipe
        )
        == "fallback"
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "batch,offload",
    [(0, False), (1, False), (4, False), (5, True), (31, False), (32, False)],
)
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@GPU
def test_local_tile_all_vjps_wrap_zero_amplitude_retained_and_offloaded(
    batch, offload, needs_x, needs_p, dtype
):
    cpu = mixed_model(dtype).double()
    cpu.selector, cpu.backend = None, "factored"
    cpu.atoms.p.requires_grad_(needs_p)
    gen = torch.Generator().manual_seed(1909)
    host_x = (
        torch.randn(batch, 128, dtype=dtype, generator=gen)
        .double()
        .requires_grad_(needs_x)
    )
    host_dy = torch.randn(batch, 128, dtype=dtype, generator=gen).double()
    expected_y = cpu(host_x)
    expected_grad = torch.autograd.grad(
        expected_y, tuple(v for v in (host_x, cpu.atoms.p) if v.requires_grad), host_dy
    )
    model = mixed_model(dtype).cuda()
    model.atoms.p.requires_grad_(needs_p)
    model.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    x = host_x.detach().to(device="cuda", dtype=dtype).requires_grad_(needs_x)
    selected = tuple(v for v in (x, model.atoms.p) if v.requires_grad)
    saved_shapes = []

    def pack(value):
        saved_shapes.append((value.dtype, tuple(value.shape)))
        return (value.detach().cpu().clone(), value.device) if offload else value

    def unpack(value):
        return value[0].to(value[1]) if offload else value

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual_y = model(x)
    dy = host_dy.to(device="cuda", dtype=dtype)
    first = torch.autograd.grad(actual_y, selected, dy, retain_graph=True)
    second = torch.autograd.grad(actual_y, selected, dy)
    for a, b in zip(first, second):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 2e-9, "rtol": 2e-8}
    )
    for actual, expected in zip((actual_y, *first), (expected_y, *expected_grad)):
        torch.testing.assert_close(actual.cpu().double(), expected, **tolerance)
    if batch:
        # Public empty-batch execution returns before the native Function and
        # legitimately saves no moments/owner keys. Its VJPs are checked above.
        assert (torch.float64, (batch, 14, 17)) in saved_shapes
        assert (torch.float64, (batch, 126, 17)) not in saved_shapes
        # Keys are required for atom-only owner grouping too; no packed P snapshot.
        assert (torch.int64, (11,)) in saved_shapes
    if needs_p:
        gp = first[int(needs_x)]
        radial = (gp[:, :2] * model.atoms.p.detach()[:, :2]).sum(1)
        torch.testing.assert_close(
            radial,
            torch.zeros_like(radial),
            atol=3e-5 if dtype == torch.float32 else 2e-12,
            rtol=0,
        )


@pytest.mark.parametrize("mode", ["no_grad", "inference", "frozen"])
@GPU
def test_local_tile_nonexecutable_context_matches_public_reference(mode):
    cpu = mixed_model(torch.float32)
    model = copy.deepcopy(cpu).cuda()
    cpu.selector, cpu.backend = None, "factored"
    model.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    x = torch.randn(5, 128)
    if mode == "frozen":
        cpu.requires_grad_(False)
        model.requires_grad_(False)
        expected, actual = cpu(x), model(x.cuda())
    else:
        context = torch.no_grad if mode == "no_grad" else torch.inference_mode
        with context():
            expected, actual = cpu(x), model(x.cuda())
    assert not actual.requires_grad
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize(
    "fault",
    [
        "batch",
        "sites",
        "modes",
        "polar",
        "base",
        "merge",
        "cutoff",
        "tile",
        "method",
        "parameters",
    ],
)
@GPU
def test_local_tile_checked_entry_rejects_unsupported_before_parameter_writes(fault):
    model = mixed_model(torch.float32).cuda()
    p = model.atoms.p
    axis = hierarchy_from_operator(model.declaration()).input
    config_i, config_o = list(axis.native_config()), list(axis.native_config())
    x = torch.randn(33 if fault == "batch" else 5, 128, device="cuda")
    start, tile, merge = 2, 32, True
    if fault == "sites":
        config_i[0] = 64
    elif fault == "modes":
        config_i[4] = 3
    elif fault == "polar":
        config_i = config_i[:6]
    elif fault == "base":
        config_i[2] = 2
    elif fault == "merge":
        config_i[2], config_i[5] = 2, 4
    elif fault == "cutoff":
        start = 5
    elif fault == "tile":
        tile = 16
    elif fault == "method":
        merge = False
    elif fault == "parameters":
        config_o[6] *= 2
    saved = p.detach().clone()
    with pytest.raises(RuntimeError):
        ex._extension().checked_autograd_linear_local_tile(
            x, p, config_i, config_o, start, 2, merge, tile
        )
    assert torch.equal(p.detach().view(torch.uint8), saved.view(torch.uint8))
    torch.cuda.synchronize()


@GPU
def test_local_tile_empty_atoms_and_public_native_optimizer_state_match_old_plan():
    from torchcst import CSTAdamW
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )

    results = []
    gen = torch.Generator().manual_seed(1912)
    inputs = [torch.randn(b, 128, generator=gen) for b in (4, 5, 32)]
    targets = [torch.randn_like(x) for x in inputs]
    for selected in (BLOCK_LOCAL_BATCHED_CACHED_PARALLEL, LOCAL_TILE_MOMENTS):
        model = mixed_model(torch.float32).cuda()
        if not selected.recipe.native_entry:
            selected = replace(
                selected,
                recipe=replace(
                    selected.recipe, native_autograd=True, native_entry=True
                ),
            )
        model.selector = FixedSelector(selected, registry=make_algorithm_registry())
        opt = CSTAdamW(
            model.parameters(),
            model=model,
            lr=1e-4,
            weight_decay=0.0,
            amplitude_weight_decay=0.01,
            fused=True,
            check_finite=False,
            update_selector=FixedSelector(
                RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
            ),
        )
        values = {}
        for step, (x, target) in enumerate(zip(inputs, targets)):
            opt.zero_grad(set_to_none=True)
            xx = x.cuda().requires_grad_()
            y = model(xx)
            (y - target.cuda()).square().mean().backward()
            values[step, "Y"] = y.detach().cpu()
            values[step, "dX"] = xx.grad.cpu()
            values[step, "dP"] = model.atoms.p.grad.cpu()
            opt.step()
            values[step, "P"] = model.atoms.p.detach().cpu().clone()
            for name, value in opt.state[model.atoms.p].items():
                values[step, name] = value.detach().cpu().clone()
        results.append(values)
    for name in results[0]:
        torch.testing.assert_close(
            results[1][name], results[0][name], atol=4e-6, rtol=4e-5
        )
    empty = make_radius_cst(128, torch.empty(0, 4)).cuda()
    empty.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    y = empty(x)
    gx, gp = torch.autograd.grad(y.sum(), (x, empty.atoms.p))
    assert torch.equal(y, torch.zeros_like(y)) and torch.equal(gx, torch.zeros_like(gx))
    assert gp.shape == (0, 4)


@pytest.mark.parametrize(
    "sites,modes,polar,base,merge",
    [
        (64, 8, True, 1, 2),
        (128, 3, True, 1, 2),
        (128, 8, False, 1, 2),
        (128, 8, True, 2, 4),
    ],
)
def test_local_tile_support_rejects_other_operator_families(
    sites, modes, polar, base, merge
):
    from torchcst import (
        ChebyshevRadiusSpec,
        CSTLinear,
        RadiusBounds,
        chart_presets,
        presets,
    )

    chart = chart_presets.regular_grid(((sites,),), geometry="flat_torus")
    options = {
        "profile": ChebyshevRadiusSpec(modes=modes),
        "input_base_tile_sites": base,
        "output_base_tile_sites": base,
        "merge_factor": merge,
    }
    kernel = (
        presets.polar_radius_cell_mass(
            amplitude_max=1.0,
            w_c=0.3,
            input_radius_bounds=RadiusBounds(
                minimum=0.5, birth=4.0, maximum=min(50.0, sites / 2), upper_floor=1.0
            ),
            **options,
        )
        if polar
        else presets.radius_cell_mass(**options)
    )
    layer = CSTLinear(chart, chart, atoms=11, kernel=kernel)
    context = replace(
        layer.build_context(LinearInputs(torch.zeros(5, sites))),
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    result = RadiusCellMassAlgorithm().supports(context, LOCAL_TILE_MOMENTS.recipe)
    assert not result.supported
    assert any("N128 Polar M8 base1 merge2" in reason for reason in result.reasons)


def test_actual_local_tile_math_and_owner_kernel_syntax_on_cpu(tmp_path):
    """Actual device helpers against old contractions; no CUDA compiler/runtime."""
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("host C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    (tmp_path / "common.cuh").write_text(
        (directory / "common.cuh")
        .read_text()
        .replace("#include <torch/extension.h>", "")
    )
    for name in ("limits.cuh", "profile_values.cuh", "local_tile_geometry.cuh"):
        (tmp_path / name).write_text((directory / name).read_text())
    hybrid = (directory / "hybrid.cuh").read_text()
    phase = hybrid[
        hybrid.index("struct HarmonicPhase") : hybrid.index("inline int checked_start")
    ]
    moments = hybrid[
        hybrid.index(
            "template<class T>\n__device__ inline double compact_moment_element"
        ) : hybrid.index("template<class T>\n__global__ void compact_moments_kernel")
    ]
    project = hybrid[
        hybrid.index("template<bool Derivatives = true>") : hybrid.index(
            "template<class T, bool Direct>"
        )
    ]
    owners = (directory / "local_tile_moments.cuh").read_text()
    owners = owners[
        owners.index("template<class T, int Stage>") : owners.index(
            "inline void check_local_tile_axes"
        )
    ]
    source = (
        r"""
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __noinline__
#define __shared__
#define __syncthreads() ((void)0)
#define TORCH_CHECK(c, ...) assert(c)
struct { int x = 0; } threadIdx, blockIdx;
struct { int x = 32; } blockDim;
inline void radius_test_sincos(double x, double *s, double *c) { *s = sin(x); *c = cos(x); }
#define sincos radius_test_sincos
#include "common.cuh"
namespace radius_cell_mass {
inline int bank_for_node(int node, const Axis &axis) {
    int bank = 0;
    while (bank + 1 < axis.bank_count && node >= axis.banks[bank + 1].node_offset) ++bank;
    return bank;
}
"""
        + phase
        + moments
        + project
        + '\n}\n#include "local_tile_geometry.cuh"\nnamespace radius_cell_mass {\n'
        + owners
        + r"""
double scratch[2000];
}
using namespace radius_cell_mass;
void close(double a, double b, double tolerance) {
    assert(std::isfinite(a) && std::isfinite(b));
    assert(fabs(a-b) <= tolerance * fmax(1., fabs(b)));
}
template<class T> void check(int batch) {
    Axis axis = parse_axis({128, 1., 1, 0., 8, 2, 1., .3, 3., 3., 1., .5, 4., 50., 1.});
    constexpr int A = 11, F = 17;
    double angles[A] = {1., .3, .15, .15, .15, .15, .15, 0., 0., -.15, .5};
    double activity[A] = {1., 1., 1., 1.8, 2.6, 3.4, 4., 1., 4., 3., 1.};
    T p[4*A], gradient[4*A], expected[4*A];
    for (int a=0;a<A;++a) {
        p[4*a] = T(angles[a]*sqrt(activity[a]));
        p[4*a+1] = T(sqrt(1-angles[a]*angles[a])*sqrt(activity[a]));
        p[4*a+2] = T(a*13.7-128); p[4*a+3] = T(256-a*12.9);
        for (int f=0;f<4;++f) gradient[4*a+f] = T(999);
    }
    std::vector<T> x(batch*128), dy(batch*128);
    for (int i=0;i<batch*128;++i) { x[i]=T(sin(i*.17)); dy[i]=T(cos(i*.31)); }
    int prefix=prefix_nodes(axis,5), J=axis.node_count-prefix;
    std::vector<double> zx(batch*J*F), zy(batch*J*F), h(batch*A), g(batch*A);
    for (int b=0;b<batch;++b) for(int node=prefix;node<axis.node_count;++node) for(int f=0;f<F;++f) {
        int index=(b*J+node-prefix)*F+f;
        zx[index]=compact_moment_element(x.data(),zx.data(),b,node,f,axis,5,false);
        zy[index]=compact_moment_element(dy.data(),zy.data(),b,node,f,axis,5,false);
    }
    double oldh[32*A]={}, oldg[32*A]={}, dci[32*A]={}, dco[32*A]={};
    std::vector<double> local(32*3*F), phi(16*F), filters(F);
    for(int a=0;a<A;++a) for(int side=0;side<2;++side) {
        Membership member=classify(radius(p,a,axis),center(p,a,axis,side),axis);
        Bank bank=axis.banks[member.bank];
        const T *signal=side?dy.data():x.data();
        const double *z=side?zy.data():zx.data();
        if(bank.level>=2 && bank.level<5) {
            for(int f=0;f<F;++f) local_tile_basis_column(phi.data(),filters.data(),f,axis,bank);
            int owner=positive_mod(member.first_tile,bank.tile_count);
            for(int b=0;b<batch;++b) for(int part=0;part<bank.count;++part) for(int f=0;f<F;++f) {
                double v=local_tile_dot(signal,b,owner,part,f,axis,bank,phi.data(),filters.data());
                int node=bank.node_offset+positive_mod(owner+part,bank.tile_count);
                double old=compact_moment_element(signal,z,b,node,f,axis,5,false);
                assert(memcmp(&v,&old,sizeof(double))==0);
                local[(b*bank.count+part)*F+f]=v;
            }
        }
        double c[17],dr[17]; profile_coefficients(radius(p,a,axis),bank.period,8,c,dr);
        for(int b=0;b<batch;++b) {
            double unused,dm,mu=wrapped_center(center(p,a,axis,side),axis);
            double v=bank.level<5 ? direct_project(signal,b,axis,member,mu,c,dr,unused,dm)
                : compact_project(z,b,axis,member,mu,c,dr,5,unused,dm);
            (side?oldg:oldh)[b*A+a]=v;
            (side?dco:dci)[b*A+a]=dm;
        }
        if(!side) local_tile_atom<T,0>(p,signal,z,local.data(),h.data(),nullptr,nullptr,nullptr,a,A,batch,axis,false);
        else {
            local_tile_atom<T,1>(p,signal,z,local.data(),g.data(),h.data(),nullptr,gradient,a,A,batch,axis,true);
            for(int f=0;f<3;++f) assert(gradient[4*a+f]==T(999));
        }
        if(!side) continue;
        Membership input=classify(radius(p,a,axis),center(p,a,axis,false),axis);
        Bank ib=axis.banks[input.bank];
        if(ib.level>=2 && ib.level<5) {
            for(int f=0;f<F;++f) local_tile_basis_column(phi.data(),filters.data(),f,axis,ib);
            int owner=positive_mod(input.first_tile,ib.tile_count);
            for(int b=0;b<batch;++b) for(int part=0;part<ib.count;++part) for(int f=0;f<F;++f)
                local[(b*ib.count+part)*F+f]=local_tile_dot(x.data(),b,owner,part,f,axis,ib,phi.data(),filters.data());
        }
        T output_column=gradient[4*a+3];
        local_tile_atom<T,2>(p,x.data(),zx.data(),local.data(),nullptr,h.data(),g.data(),gradient,a,A,batch,axis,false);
        assert(memcmp(&output_column,gradient+4*a+3,sizeof(T))==0);
        double da=0,di=0,dout=0,amp=amplitude(p,a,axis);
        for(int b=0;b<batch;++b) {
            close(h[b*A+a],oldh[b*A+a],2e-11); close(g[b*A+a],oldg[b*A+a],2e-11);
            da+=oldh[b*A+a]*oldg[b*A+a]; di+=amp*oldg[b*A+a]*dci[b*A+a]; dout+=amp*oldh[b*A+a]*dco[b*A+a];
        }
        store_gradient(p,expected,a,axis,da,0,di,dout);
        for(int f=0;f<4;++f) close(gradient[4*a+f],expected[4*a+f],std::is_same<T,float>::value?1e-5:2e-11);
    }
    // Compile every owner specialization, with no participants in this CPU
    // syntax smoke. Cooperative CUDA synchronization is a separate GPU gate.
    int64_t rows[1]={},offsets[511]={};
    local_tile_owner_kernel<T,0>(p,x.data(),zx.data(),h.data(),nullptr,nullptr,nullptr,rows,offsets,A,batch,axis,false);
    local_tile_owner_kernel<T,1>(p,dy.data(),zy.data(),g.data(),h.data(),nullptr,gradient,rows,offsets,A,batch,axis,true);
    local_tile_owner_kernel<T,2>(p,x.data(),zx.data(),nullptr,h.data(),g.data(),gradient,rows,offsets,A,batch,axis,false);
}
int main() {
    static_assert(local_tile_shared_bytes(32)==15368);
    static_assert(local_tile_shared_bytes(32)<48*1024);
    for(int b:{0,1,4,5,31,32}) { check<float>(b); check<double>(b); }
}
"""
    )
    path, binary = tmp_path / "local_tile_smoke.cpp", tmp_path / "local_tile_smoke"
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


def test_native_host_local_mode_autograd_compiles_on_cpu():
    from torch.utils.cpp_extension import include_paths

    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("host C++ compiler unavailable")
    includes = [*include_paths(), sysconfig.get_path("include")]
    command = [
        *compiler,
        "-std=c++17",
        "-fsyntax-only",
        "-DTORCH_EXTENSION_NAME=local_tile_syntax",
    ]
    for include in includes:
        command.extend(["-I", include])
    command.append(str(Path(ex.__file__).parent / "host_autograd.cpp"))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


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
def test_local_tile_saved_hook_metadata_is_rechecked_before_backward_work(fault):
    model = mixed_model(torch.float32).cuda()
    model.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    x = torch.randn(5, 128, device="cuda", requires_grad=True)

    def pack(value):
        shape = tuple(value.shape)
        kind = (
            "signal_stride"
            if shape == (5, 128)
            else "parameter_stride"
            if shape == (11, 4)
            else "moment_dtype"
            if shape == (5, 14, 17)
            else "response_shape"
            if shape == (5, 11)
            else "key_count"
            if shape == (11,) and value.dtype == torch.int64
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

    before = model.atoms.p.detach().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = model(x)
    with pytest.raises(RuntimeError, match="contiguous|compact|response|keys"):
        torch.autograd.grad(y.sum(), (x, model.atoms.p))
    assert torch.equal(
        before.view(torch.uint8), model.atoms.p.detach().view(torch.uint8)
    )


@GPU
def test_local_tile_skewed_owner_group_covers_multiple_warp_iterations():
    # All65 atoms belong to the same bank/first-tile owner on each side. The
    # owner CTA must process three i+=32 iterations, including its one-row tail.
    cpu = make_radius_cst(128, 65, dtype=torch.float64)
    with torch.no_grad():
        cpu.atoms.p.copy_(mixed_model().atoms.p[4].expand(65, -1))
        images = (torch.arange(65, dtype=torch.float64).remainder(5) - 2) * 128
        cpu.atoms.p[:, 2] = images
        cpu.atoms.p[:, 3] = -images
    cpu.selector, cpu.backend = None, "factored"
    model = copy.deepcopy(cpu).cuda()
    model.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    gen = torch.Generator().manual_seed(1917)
    x = torch.randn(5, 128, dtype=torch.float64, generator=gen).requires_grad_()
    dy = torch.randn(5, 128, dtype=torch.float64, generator=gen)
    expected_y = cpu(x)
    expected = torch.autograd.grad(expected_y, (x, cpu.atoms.p), dy)
    xx = x.detach().cuda().requires_grad_()
    actual_y = model(xx)
    actual = torch.autograd.grad(actual_y, (xx, model.atoms.p), dy.cuda())
    for got, want in zip((actual_y, *actual), (expected_y, *expected)):
        torch.testing.assert_close(got.cpu(), want, atol=2e-9, rtol=2e-8)


@GPU
@pytest.mark.parametrize("needs_p", [False, True])
def test_local_tile_public_read_lease_covers_retained_and_dx_only_calls(needs_p):
    model = mixed_model(torch.float32).cuda()
    model.atoms.p.requires_grad_(needs_p)
    model.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    x = torch.randn(5, 128, device="cuda", requires_grad=True)
    selected = (x, model.atoms.p) if needs_p else (x,)
    permutation = torch.arange(11, device="cuda").flip(0)
    y = model(x)
    with pytest.raises(RuntimeError, match="backward readers"):
        model.atom_state.relayout(permutation)
    torch.autograd.grad(y.sum(), selected, retain_graph=True)
    with pytest.raises(RuntimeError, match="backward readers"):
        model.atom_state.relayout(permutation)
    torch.autograd.grad(y.sum(), selected)
    model.atom_state.relayout(permutation)


@GPU
def test_local_tile_saved_version_and_first_order_only_contracts():
    model = mixed_model(torch.float64).cuda()
    model.selector = FixedSelector(
        LOCAL_TILE_MOMENTS, registry=make_algorithm_registry()
    )
    x = torch.randn(5, 128, dtype=torch.float64, device="cuda", requires_grad=True)
    y = model(x)
    with torch.no_grad():
        model.atoms.p.add_(0.0)
    with pytest.raises(RuntimeError, match="modified.*inplace"):
        y.sum().backward()
    y = model(x)
    gradient = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    assert gradient.requires_grad
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)
