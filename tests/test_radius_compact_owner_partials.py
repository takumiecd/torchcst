"""Mode9 packs only owner partial addresses; mode7 floating arithmetic stays exact."""

import copy
import os
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_radius_owner_response_derivatives as derivative
import torch
from test_radius_cell_mass_cuda import GPU
from test_radius_owner_response_coeffs import model

from benchmarks.cuda.linear.manifest import load_run
from torchcst import LinearInputs, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_COMPACT_PARTIALS as COMPACT,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_RESPONSE_DERIVATIVES as OLD,
)
from torchcst._backends.schema import DeviceInfo

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path(ex.__file__).parent


def test_compact_explicit_plan_protocol_limits_and_unchanged_default():
    run = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-compact-partials-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.entry(run.baseline).plan == OLD
    assert run.entry("radius-global-owner-compact-partials").plan == COMPACT
    registry = make_algorithm_registry()
    assert registry.loads_plan(registry.dumps_plan(COMPACT)) == COMPACT
    context = model().build_context(
        LinearInputs(torch.zeros(4, 1024, dtype=torch.float64))
    )
    context = replace(context, device=DeviceInfo("cuda", 0, "metadata-only"))
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(context, COMPACT.recipe).supported
    assert not algorithm.supports(
        replace(context, input_shape=(5, 1024)), COMPACT.recipe
    ).supported
    assert not algorithm.supports(
        replace(context, atom_count=65537), COMPACT.recipe
    ).supported
    for changes in (
        {"input_direct_levels": 1},
        {"output_direct_levels": 3},
        {"moment_method": "direct"},
        {"native_entry": False},
        {"output_tile_sites": 16},
    ):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(replace(COMPACT.recipe, **changes))
    assert 4 * 53675 * 8 == 1717600
    assert 4 * 4094 * 3 * 17 * 8 - 4 * 53675 * 8 == 4963808


def test_compact_checked_entry_and_custom_mode_fallback(monkeypatch):
    from torch.overrides import TorchFunctionMode

    m = model(torch.float32)
    plan = hierarchy_from_operator(m.declaration())
    x = torch.zeros(4, 1024)
    calls = []
    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            checked_autograd_linear_compact_partials=lambda *args: (
                calls.append(args) or x
            )
        ),
    )
    assert ex.radius_linear(plan, x, m.atoms.p, recipe=COMPACT.recipe) is x
    assert len(calls) == 1 and len(calls[0]) == 4

    class CustomMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    def fallback(actual_plan, actual_x, p, *, recipe):
        assert actual_plan is plan and actual_x is x and p is m.atoms.p
        assert recipe.schedule == "global_batched_cached_active_moments_grad4"
        assert not recipe.native_entry and not recipe.native_autograd
        return "callback"

    monkeypatch.setattr(ex, "radius_linear", fallback)
    with CustomMode():
        assert (
            ex.checked_owner_compact_linear(plan, x, m.atoms.p, recipe=COMPACT.recipe)
            == "callback"
        )


def test_actual_cpp_packed_indices_bijection_and_all_bank_gathers(tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("host C++ compiler unavailable")
    for name in (
        "common.cuh",
        "limits.cuh",
        "profile_values.cuh",
        "owner_partial_layout.cuh",
    ):
        (tmp_path / name).write_text(
            (DIRECTORY / name).read_text().replace("#include <torch/extension.h>", "")
        )
    hybrid = (DIRECTORY / "hybrid.cuh").read_text()
    phase = hybrid[
        hybrid.index("struct HarmonicPhase") : hybrid.index("inline int checked_start")
    ]
    old = (DIRECTORY / "owner_response_coefficients.cuh").read_text()
    old = (
        old[
            old.index("__device__ inline int64_t owner_partial_index") : old.index(
                "template<class T>\n__global__ void owner_response_coefficients_kernel"
            )
        ]
        + old[
            old.index("// Sum the K coefficient contributions") : old.index(
                "// Host orchestration:"
            )
        ]
    )
    packed = (DIRECTORY / "owner_compact_partials.cuh").read_text()
    packed = packed[
        packed.index("// Sum the K coefficient contributions") : packed.index(
            "std::vector<Tensor> owner_compact_partials_transform"
        )
    ]
    source = (
        r"""
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __noinline__
#define TORCH_CHECK(c,...) assert(c)
struct {int x=0;} blockIdx,threadIdx;
struct {int x=128;} blockDim;
inline void radius_test_sincos(double x,double*s,double*c){*s=sin(x);*c=cos(x);}
#define sincos radius_test_sincos
#include "owner_partial_layout.cuh"
namespace radius_cell_mass {
constexpr int OWNER_PARTS=3, OWNER_FEATURES=17;
"""
        + phase
        + old
        + packed
        + r"""
}
using namespace radius_cell_mass;
void check(int B,int start,int pattern) {
    Axis a=parse_axis({1024,.37,1,-1.2,8,2});
    auto layout=compact_owner_partial_layout(a,start);
    if(start==2)assert(layout.total_fields==53675);
    assert(a.bank_count==22 && a.banks[21].period>a.period);
    std::vector<unsigned char> visits(size_t(B)*layout.total_fields);
    for(int b=0;b<B;++b)for(int bi=0;bi<a.bank_count;++bi) {
        Bank bank=a.banks[bi];int fields=bank.level<start?bank.tile_sites:17;
        for(int node=bank.node_offset;node<bank.node_offset+bank.tile_count;++node)
            for(int part=0;part<bank.count;++part)for(int f=0;f<fields;++f) {
                int64_t i=compact_owner_partial_index(b,node,bi,part,f,a,layout);
                assert(i>=0 && i<int64_t(visits.size()) && visits[i]++==0);
            }
    }
    for(auto count:visits)assert(count==1);
    // Gather consumes only nonempty owners and meaningful fields. Leave every
    // inactive/padded field NaN to expose accidental reads, including K>tiles.
    std::vector<double> old(size_t(B)*a.node_count*3*17,NAN),p(visits.size(),NAN);
    std::vector<int64_t> offsets(a.node_count+1);
    for(int node=0;node<a.node_count;++node)offsets[node+1]=offsets[node]+(pattern==2 || (pattern==1 && node%3==0));
    for(int b=0;b<B;++b)for(int bi=0;bi<a.bank_count;++bi) {
        Bank bank=a.banks[bi];int fields=layout.fields[bi];
        for(int node=bank.node_offset;node<bank.node_offset+bank.tile_count;++node) {
            if(offsets[node]==offsets[node+1])continue;
            for(int part=0;part<bank.count;++part)for(int f=0;f<fields;++f) {
                double v=sin(double(b*3+node+part*7+f*13))*.01;
                old[owner_partial_index(b,node,part,f,a)]=v;
                p[compact_owner_partial_index(b,node,bi,part,f,a,layout)]=v;
            }
        }
    }
    // mode9 is cutoff2; also exercise cutoff0/1 layouts while direct fields<=17.
    for(int b=0;b<B;++b)for(int site=0;site<a.sites;++site) {
        double original=owner_partial_site(old.data(),offsets.data(),b,site,a,start);
        double compact=compact_owner_partial_site(p.data(),offsets.data(),b,site,a,start,layout);
        assert(std::isfinite(original)&&std::isfinite(compact));
        assert(memcmp(&original,&compact,sizeof(double))==0);
    }
}
int main(){for(int B:{0,1,4})for(int start:{0,1,2})for(int pattern:{0,1,2})check(B,start,pattern);}
"""
    )
    path = tmp_path / "packed.cpp"
    path.write_text(source)
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(path),
            "-o",
            str(tmp_path / "packed"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [str(tmp_path / "packed")], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def mode9(monkeypatch):
    monkeypatch.setattr(derivative, "NEW", COMPACT)
    monkeypatch.setattr(ex, "checked_radius_linear", ex.checked_owner_compact_linear)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_compact_all_independent_vjps_retained_offload(
    mode9, dtype, batch, needs_x, needs_p, skew
):
    derivative.test_all_vjps_boundary_images_skew_retained_and_offload(
        dtype, batch, needs_x, needs_p, skew
    )


@GPU
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
def test_compact_raw_saved_schema(mode9, needs_x, needs_p):
    derivative.test_raw_saved_schema_replaces_z_and_only_saves_derivatives_for_atom_edges(
        needs_x, needs_p
    )


@GPU
@pytest.mark.parametrize(
    "fault",
    ["x_stride", "p_stride", "h_shape", "dh_shape", "dh_dtype", "dh_stride", "keys"],
)
def test_compact_saved_metadata_before_reverse_launch(mode9, fault, tmp_path):
    derivative.test_saved_hook_metadata_checked_before_any_reverse_launch(
        fault, tmp_path
    )


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
def test_compact_existing_lifetime_contracts(mode9, monkeypatch, contract):
    derivative.test_existing_independent_contracts_on_explicit_derivative_plan(
        monkeypatch, contract
    )


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_mode7_vs9_mandatory_exact_y_dx_dp_parameter_adam_bytes(
    dtype, batch, needs_x, needs_p, skew
):
    initial = model(dtype, skew)
    gen = torch.Generator().manual_seed(9041)
    host_x = torch.randn(batch, 1024, dtype=dtype, generator=gen)
    dy = torch.randn(batch, 1024, dtype=dtype, generator=gen).cuda()
    results = []
    for selected in (OLD, COMPACT):
        m = copy.deepcopy(initial).cuda()
        m.atoms.p.requires_grad_(needs_p)
        x = host_x.cuda().requires_grad_(needs_x)
        axes = hierarchy_from_operator(m.declaration())
        optimizer = (
            torch.optim.AdamW([m.atoms.p], lr=1e-4, fused=True) if needs_p else None
        )
        records = []
        for step in range(3):
            m.zero_grad(set_to_none=True)
            x.grad = None
            y = ex.radius_linear(axes, x, m.atoms.p, recipe=selected.recipe)
            if selected == COMPACT:
                assert "OwnerCompactPartials" in y.grad_fn.name()
            y.backward(dy)
            records.extend(
                t.detach().cpu().clone()
                for t in (
                    y,
                    *([] if x.grad is None else [x.grad]),
                    *([] if m.atoms.p.grad is None else [m.atoms.p.grad]),
                )
            )
            if optimizer:
                optimizer.step()
                records.append(m.atoms.p.detach().cpu().clone())
                records.extend(
                    optimizer.state[m.atoms.p][name].detach().cpu().clone()
                    for name in ("step", "exp_avg", "exp_avg_sq")
                )
        results.append(records)
    for old, new in zip(*results):
        assert torch.isfinite(old).all() and torch.isfinite(new).all()
        assert torch.equal(
            old.contiguous().reshape(-1).view(torch.uint8),
            new.contiguous().reshape(-1).view(torch.uint8),
        )


def test_new_cta_instantiates_actual_header_without_emulating_barriers(
    tmp_path, monkeypatch
):
    """Reuse qualified CPU scaffold, additionally instantiate the new CTA body."""
    original_read = Path.read_text
    layout = (DIRECTORY / "owner_partial_layout.cuh").read_text()
    layout = layout[
        layout.index("struct OwnerPartialLayout") : layout.rindex("} // namespace")
    ]
    compact = (DIRECTORY / "owner_compact_partials.cuh").read_text()
    kernel = compact[
        compact.index(
            "template<class T>\n__global__ void owner_compact_partials_kernel"
        ) : compact.index("// Sum the K coefficient contributions")
    ]
    insertion = (
        layout
        + kernel
        + "\nauto force_compact_kernel_instantiation = &owner_compact_partials_kernel<float>;\n"
    )

    def read(path, *args, **kwargs):
        contents = original_read(path, *args, **kwargs)
        if path == DIRECTORY / "owner_response_derivatives.cuh":
            contents = contents.replace(
                "// Host orchestration:", insertion + "\n// Host orchestration:", 1
            )
        return contents

    monkeypatch.setattr(Path, "read_text", read)
    derivative.test_actual_complete_derivative_helpers_and_kernels_on_cpu(tmp_path)


@GPU
@pytest.mark.parametrize(
    "fault", ["batch", "sites", "polar", "modes", "spacing", "origin", "bounds"]
)
def test_compact_raw_checked_metadata_before_any_launch(fault, tmp_path):
    import json

    m = model(torch.float32).cuda()
    axes = hierarchy_from_operator(m.declaration())
    ci, co = list(axes.input.native_config()), list(axes.output.native_config())
    x = torch.randn(5 if fault == "batch" else 4, 1024, device="cuda")
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
    native = ex._extension()
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile,
        pytest.raises(RuntimeError),
    ):
        native.checked_autograd_linear_compact_partials(x, m.atoms.p, ci, co)
    trace = tmp_path / "rejected-mode9.json"
    profile.export_chrome_trace(str(trace))
    assert not any(
        event.get("cat") == "kernel"
        for event in json.loads(trace.read_text())["traceEvents"]
    )


@pytest.mark.parametrize("entry", ["checked", "radius", "algorithm"])
@pytest.mark.parametrize(
    "schedule,mode",
    [
        ("block_local", 0),
        ("block_local_parallel", 1),
        ("block_local_cached_parallel", 2),
        ("block_local_batched_cached_parallel", 3),
        ("global_batched_cached_active_moments_grad4_native", 5),
        ("global_owner_response_coefficients", 6),
        ("global_owner_response_derivatives", 7),
        ("global_owner_compact_partials", 9),
    ],
)
def test_checked_route_enums_reach_native_api_through_each_real_python_boundary(
    monkeypatch, entry, schedule, mode
):
    # The CPU tensors are metadata stand-ins. Only the native leaf is mocked:
    # Algorithm.execute and the shared checked/radius entrances remain real.
    # This catches an omitted enum/delegation before any GPU compilation.
    import torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm as algorithm_module

    layer = model(torch.float32)
    layout = hierarchy_from_operator(layer.declaration())
    x, p = torch.zeros(4, 1024), layer.atoms.p
    recipe = replace(COMPACT.recipe, schedule=schedule)
    calls = []

    def old_native(*args):
        calls.append((args[-1], args))
        return x

    def compact_native(*args):
        calls.append((9, args))
        return x

    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            checked_autograd_linear=old_native,
            checked_autograd_linear_compact_partials=compact_native,
        ),
    )
    if entry == "algorithm":
        state = SimpleNamespace(layout=layout, recipe=recipe)
        inputs = LinearInputs(x)

        def execution(actual_state, actual_inputs):
            assert actual_state is state and actual_inputs is inputs
            return x, p, None, None

        monkeypatch.setattr(algorithm_module, "linear_execution", execution)
        result = RadiusCellMassAlgorithm().execute(state, inputs)
    else:
        entrance = ex.checked_radius_linear if entry == "checked" else ex.radius_linear
        result = entrance(layout, x, p, recipe=recipe)
    assert result is x and len(calls) == 1
    actual_mode, args = calls[0]
    assert actual_mode == mode and args[0] is x and args[1] is p
    assert args[2:4] == (layout.input.native_config(), layout.output.native_config())
    assert len(args) == (4 if mode == 9 else 9)
