"""Optional H storage preserves rounded C and required backward state."""

from dataclasses import replace

import pytest
import torch
from test_cell_mass_polar import layer, oracle

from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical import (
    executor as ex,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
    SITE_SCALED,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_h_storage_flags_require_boolean(flag):
    model = layer(count=7, dtype=torch.float32, site_scaled=True)
    plan = hierarchy_from_operator(model.declaration())
    with pytest.raises(ValueError, match="store_h must be bool"):
        ex.compute_h_c(plan, model.atoms.p, None, None, store_h=flag)
    with pytest.raises(ValueError, match="omit_unused_h must be bool"):
        ex.hierarchical_linear(
            plan, torch.zeros(1, 128), model.atoms.p, omit_unused_h=flag
        )


def test_h_storage_omission_rejects_nonragged_primitive():
    model = layer(count=7, dtype=torch.float32, site_scaled=True)
    plan = hierarchy_from_operator(model.declaration())
    with pytest.raises(ValueError, match="requires ragged fused H/C"):
        ex.compute_h_c(plan, model.atoms.p, None, None, store_h=False)


@pytest.mark.parametrize(
    "polar,atom_grad,omit,ragged,expected_store",
    [
        (True, True, False, True, True),
        (True, True, True, True, False),
        (False, True, True, True, True),
        (False, False, True, True, False),
        (True, False, True, True, False),
        (False, False, False, True, True),
        (True, True, True, False, True),
    ],
)
def test_h_storage_autograd_policy_on_cpu(
    monkeypatch, polar, atom_grad, omit, ragged, expected_store
):
    model = layer(count=7, dtype=torch.float32, site_scaled=True)
    atoms = model.atoms.p.detach().clone().requires_grad_(atom_grad)
    plan = hierarchy_from_operator(model.declaration())
    if not polar:
        plan = replace(
            plan,
            input=replace(plan.input, polar=None),
            output=replace(plan.output, polar=None),
        )
    layout = ex.PreparedLayout(
        torch.arange(7),
        torch.zeros(1, 4, dtype=torch.int32),
        torch.tensor([0, 7]),
        plan,
        atoms,
        atoms._version,
        reverse_slots=torch.arange(7, dtype=torch.int32),
        reverse_offsets=torch.tensor([0] + [7] * plan.input.node_count),
    )
    seen = []

    def hc(_plan, _atoms, _layout, z, **kwargs):
        h = torch.zeros(z.shape[0], len(atoms)) if kwargs["store_h"] else torch.empty(0)
        seen.append((kwargs["store_h"], h))
        return h, torch.zeros(
            z.shape[0], plan.output.node_count, 17, dtype=torch.float64
        )

    monkeypatch.setattr(
        ex,
        "build_hierarchy",
        lambda x, axis, **kw: torch.zeros(
            x.shape[0], axis.node_count, 17, dtype=torch.float64
        ),
    )
    monkeypatch.setattr(ex, "compute_h_c", hc)
    monkeypatch.setattr(
        ex, "expand_c", lambda c, axis, like, **kw: torch.zeros_like(like)
    )
    x = torch.ones(5, 128, requires_grad=True)
    options = dict(
        SITE_SCALED.recipe.arguments(),
        window_moments=False,
        ragged_c_splits=ragged,
        release_workspaces=True,
    )
    y = ex.hierarchical_linear(
        plan, x, atoms, layout=layout, **options, omit_unused_h=omit
    )
    assert seen[0][0] is expected_store
    if not polar and atom_grad:
        saved_h = y.grad_fn.saved_tensors[-1]
        assert saved_h.shape == (5, 7)
        assert saved_h.data_ptr() == seen[0][1].data_ptr()


@CUDA
@pytest.mark.parametrize(
    "tile,batch,count,exact",
    [
        (4, 0, 7, False),
        (8, 5, 0, True),
        (4, 1, 31, False),
        (8, 5, 33, True),
        (8, 32, 65, False),
        (8, 32, 65, True),
    ],
)
def test_h_storage_primitive_c_matches_saved_h(tile, batch, count, exact):
    model = layer(
        count=max(1, count),
        dtype=torch.float32,
        device="cuda",
        site_scaled=True,
        split=True,
    )
    atoms = model.atoms.p[:count].contiguous()
    plan = hierarchy_from_operator(model.declaration())
    layout = ex.prepare_layout(plan, atoms, method="sort", reverse=True)
    x = torch.randn(batch, 128, device="cuda")
    z = ex.build_window_moments(ex.build_hierarchy(x, plan.input), plan.input)
    options = {
        "splits": 128,
        "atoms_per_split": 64,
        "windowed": True,
        "compensated": True,
        "ragged": True,
        "batch_tile": tile,
        "reuse_sigma_decode": exact,
        "known_input_bank": exact,
    }
    h, c = ex.compute_h_c(plan, atoms, layout, z, **options)
    assert h.shape == (batch, count)  # Diagnostic default still returns H.
    h2, c2 = ex.compute_h_c(plan, atoms, layout, z, **options, store_h=True)
    empty, omitted = ex.compute_h_c(plan, atoms, layout, z, **options, store_h=False)
    assert (
        empty.shape == (0,)
        and empty.dtype == atoms.dtype
        and empty.device == atoms.device
    )
    torch.testing.assert_close(h2, h, rtol=0, atol=0)
    torch.testing.assert_close(c2, c, rtol=0, atol=0)
    torch.testing.assert_close(omitted, c, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize(
    "polar,input_grad,atom_grad",
    [
        (True, True, True),
        (True, False, True),
        (True, True, False),
        (False, True, True),
        (False, True, False),
    ],
)
def test_h_storage_full_route_retained_gradients_and_oracle(
    polar, input_grad, atom_grad
):
    model = layer(
        count=33, dtype=torch.float32, device="cuda", site_scaled=True, split=True
    )
    plan = hierarchy_from_operator(model.declaration())
    atoms = model.atoms.p.detach().clone()
    if not polar:
        plan = replace(
            plan,
            input=replace(plan.input, polar=None),
            output=replace(plan.output, polar=None),
        )
        atoms[:, 1] = (
            2  # Physical rows are [amplitude,sigma,output_center,input_center].
        )
    atoms.requires_grad_(atom_grad)
    x = torch.randn(5, 128, device="cuda", requires_grad=input_grad)
    dy = torch.randn_like(x)
    targets = tuple(t for t in (x, atoms) if t.requires_grad)
    options = dict(
        SITE_SCALED.recipe.arguments(),
        hc_batch_tile=8,
        hc_atoms_per_split=64,
        hc_reuse_sigma_decode=True,
        hc_known_input_bank=True,
        release_workspaces=True,
    )
    outputs = []
    for omit in (False, True):
        y = ex.hierarchical_linear(plan, x, atoms, **options, omit_unused_h=omit)
        first = torch.autograd.grad(y, targets, dy, retain_graph=True)
        second = torch.autograd.grad(y, targets, -2 * dy)
        for a, b in zip(first, second):
            torch.testing.assert_close(b, -2 * a, rtol=3e-6, atol=2e-7)
        outputs.append((y, *first))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if polar:
        xr = x.detach().cpu().double().requires_grad_(input_grad)
        pr = atoms.detach().cpu().double().requires_grad_(atom_grad)
        yr = oracle(model, xr, pr)
        refs = tuple(t for t in (xr, pr) if t.requires_grad)
        expected = (yr, *torch.autograd.grad(yr, refs, dy.cpu().double()))
        for a, b in zip(outputs[1], expected):
            torch.testing.assert_close(a.cpu().double(), b, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize(
    "prefix", ["", "_fused", "_cached32_expand", "_cached32_expand_fused"]
)
def test_no_store_h_preset_is_explicit_and_round_trips(prefix):
    from torchcst._backends.catalog import get_registry
    from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.recipe import (
        HierarchicalCellMassRecipe,
    )
    from torchcst._backends.schema import ExecutionPlan

    base = "site_scaled_hc8_g64_cin128_slab16_exact_reuse_node_z" + prefix
    original = HierarchicalCellMassRecipe(base)
    candidate = HierarchicalCellMassRecipe(base + "_nostore_h")
    assert candidate.arguments() == dict(original.arguments(), omit_unused_h=True)
    assert "omit_unused_h" not in original.arguments()
    registry = get_registry()
    plan = ExecutionPlan("cuda_cell_mass_hierarchical", "v1", candidate)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
