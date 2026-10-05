"""Launch tuning must preserve independent training and sliced-gradient checks."""

import pytest
import torch

from benchmarks.cuda.linear.local_product import LocalRecipe
from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog, read_json

CATALOG = "benchmarks/cuda/linear/plans-local-tuned.json"


def test_launch_catalog_roundtrip():
    entries = decode_catalog(read_json(CATALOG)[0])
    for entry in entries:
        assert REGISTRY.load_plan(REGISTRY.dump_plan(entry.plan)) == entry.plan
    with pytest.raises(ValueError):
        LocalRecipe(route="torch_contract4")
    with pytest.raises(ValueError):
        LocalRecipe(route=[])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "route",
    [
        "persistent_supportprep_band_contract4",
        "persistent_supportprep_band_contract8",
        "persistent_supportprep_g_contract4",
        "persistent_supportprep_g_param4",
        "persistent_supportprep_band_vector",
        "persistent_supportprep_g_vector",
        "persistent_supportprep_band_vector4",
        "persistent_supportprep_g_vector4",
        "persistent_supportprep_band_unroll",
        "persistent_supportprep_g_unroll",
    ],
)
def test_launch_sliced_retained_gradients(route):
    from test_local_contraction import (
        test_contraction_slices_gradients_and_repeated_backward,
    )

    test_contraction_slices_gradients_and_repeated_backward(route, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "route",
    [
        "persistent_supportprep_band_contract4",
        "persistent_supportprep_band_contract8",
        "persistent_supportprep_g_contract4",
        "persistent_supportprep_g_param4",
        "persistent_supportprep_band_vector",
        "persistent_supportprep_g_vector",
        "persistent_supportprep_band_vector4",
        "persistent_supportprep_g_vector4",
        "persistent_supportprep_band_unroll",
        "persistent_supportprep_g_unroll",
    ],
)
def test_launch_captured_training(route):
    from test_local_product_research import (
        test_graph_training_updates_width_and_matches_public_optimizer,
    )

    entries = decode_catalog(read_json(CATALOG)[0])
    plan = next(e.plan for e in entries if e.plan.recipe.route == route)
    test_graph_training_updates_width_and_matches_public_optimizer(
        "hybrid-persistent-mid4",
        size_override=128,
        polar_update="fused",
        updates=20,
        plan_override=plan,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "route",
    ["persistent_supportprep_band_contract4", "persistent_supportprep_g_contract4"],
)
def test_atom32_graph_migration_and_retained_topology(route, monkeypatch):
    import test_local_persistent_layout as original

    from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
        PersistentLayout,
    )

    make = original.make

    def fixture(atoms=96):
        p, s, d, _recipe, _layout = make(atoms)
        recipe = LocalRecipe(
            route=route, atom_block=32, pack=False, rho_upper=(1, 4, 16)
        )
        return p, s, d, recipe, PersistentLayout(p, s, d, recipe)

    monkeypatch.setattr(original, "make", fixture)
    original.test_graph_sparse_migration_then_capacity_overflow()
    original.test_outstanding_backward_retains_its_own_topology()
