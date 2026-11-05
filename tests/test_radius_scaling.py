"""Large owner cohorts: full-width keys, scope, and independent VJPs."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU

from benchmarks.cuda.linear.radius_cell_mass import fixture
from benchmarks.training.owner_window_fp32 import errors, vjp_reference
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
    _extension,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    MAX_ATOMS,
    MAX_SITES,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_RESPONSE_COEFFICIENTS,
    GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32,
    GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
)
from torchcst._backends.schema import DeviceInfo
from torchcst.operators.execution import LinearInputs

NEW = GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
BASE = GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32


@pytest.mark.parametrize("n", [2048, 4096, 8192, 16384])
def test_scaling_cohort_metadata_and_non_scalable_routes(n):
    model, x, _, _ = fixture(SimpleNamespace(size=n, atoms=4, rows=4, seed=41))
    context = replace(
        model.build_context(LinearInputs(x)),
        atom_count=n * n // 20,
        device=DeviceInfo("cuda", 0, "metadata-only"),
    )
    algorithm = RadiusCellMassAlgorithm()
    for plan in (BASE, NEW):
        assert algorithm.supports(context, plan.recipe).supported
        assert not algorithm.supports(
            replace(context, atom_count=MAX_ATOMS + 1), plan.recipe
        ).supported
        assert not algorithm.supports(
            replace(context, input_shape=(5, n)), plan.recipe
        ).supported
    assert not algorithm.supports(
        context, GLOBAL_OWNER_RESPONSE_COEFFICIENTS.recipe
    ).supported
    axis = hierarchy_from_operator(model.declaration()).input
    assert axis.node_count == 4 * n - 2 and len(axis.banks) <= 32
    assert axis.node_count**2 < 2**32
    assert (axis.node_count**2 > 2**31) is (n == 16384)
    assert 4 * MAX_ATOMS < 2**31 and MAX_SITES == 16384


def high_key_fixture(n):
    model, x, _, dy = fixture(SimpleNamespace(size=n, atoms=31, rows=4, seed=41))
    with torch.no_grad():
        # Alpha0 with small amplitude selects radius~3.125, not an extreme bank.
        model.atoms.p[:, 0] = 0.1
        model.atoms.p[:, 1] = (0.99) ** 0.5
        model.atoms.p[:, 2] = torch.linspace(-0.5001, 2 * n + 0.5001, 31)
        model.atoms.p[:, 3] = n - 0.25
        model.atoms.p[1].copy_(model.atoms.p[0])  # stable equal-key rows
    return model, x, dy


@GPU
@pytest.mark.parametrize("n", [2048, 4096, 8192, 16384])
def test_large_keys_both_sort_orientations_and_independent_vjp(n):
    cpu, x, dy = high_key_fixture(n)
    model = copy.deepcopy(cpu).cuda()
    axis = hierarchy_from_operator(cpu.declaration()).input
    ext = _extension()
    keys = ext.classify_atoms_async(
        model.atoms.p, axis.native_config(), axis.native_config()
    )
    if n == 16384:
        assert bool((keys.cpu() > 2**31 - 1).any())
    for reverse in (False, True):
        got = ext.layout_ranges_aten32(keys, axis.node_count, axis.node_count, reverse)
        want = ext.layout_ranges_aten(keys, axis.node_count, axis.node_count, reverse)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
        # Source node determines the same bank mask on both orientations.
        mask = ext.active_moment_banks_from_keys(
            keys, axis.native_config(), axis.native_config(), reverse, True, 2
        )
        assert int(mask.cpu()) > 0
    values = []
    for selected in (BASE, NEW):
        m = copy.deepcopy(model)
        m.selector = FixedSelector(selected, registry=make_algorithm_registry())
        xx = x.cuda().requires_grad_()
        y = m(xx)
        values.append((y, *torch.autograd.grad(y, (xx, m.atoms.p), dy.cuda())))
    for a, b in zip(*values):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    reference = vjp_reference(cpu, x, dy, independent=False)
    assert all(errors(a, b)["pass"] for a, b in zip(values[1], reference))


@GPU
def test_uint32_key_endpoint_mask_and_empty_owners():
    cpu, _, _ = high_key_fixture(16384)
    axis = hierarchy_from_operator(cpu.declaration()).input
    count = axis.node_count**2
    keys = torch.tensor(
        [0, 2**31 - 1, 2**31, count - 1, count - 1], device="cuda", dtype=torch.int64
    )
    ext = _extension()
    for reverse in (False, True):
        rows, offsets = ext.layout_ranges_aten32(
            keys, axis.node_count, axis.node_count, reverse
        )
        got = ext.active_moment_banks_from_keys(
            keys, axis.native_config(), axis.native_config(), reverse, True, 2
        )
        assert int(got.cpu()) > 0
        assert int(offsets[0].cpu()) == 0 and int(offsets[-1].cpu()) == len(keys)
        assert rows.cpu().tolist().index(3) < rows.cpu().tolist().index(4)
