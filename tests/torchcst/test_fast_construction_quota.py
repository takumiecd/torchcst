"""cSFW is pure growth and cannot exceed the root birth grant."""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import EvenBudgetDistributor, PeriodicCadence, QuotaRegime, cSFW
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore


def _parts(seed: int = 19) -> tuple[StructuralEngine, CSTLinear, SynapseStore]:
    store = SynapseStore(
        "continuous",
        1,
        1,
        8,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.25]], dtype=torch.float64),
                torch.tensor([[0.75]], dtype=torch.float64),
                torch.tensor([0.1], dtype=torch.float64),
                torch.tensor([0]),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        3,
        mu=torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64),
        initial_live=3,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        2,
        mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.35).double())
    root = QuotaRegime(
        budget=2,
        method=cSFW(backfit=None, pool_size=32, multistart=2),
        cadence=PeriodicCadence(event_interval=2, observe_window=2),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        root,
        modules={store.site: module},
        seed=seed,
    )
    return engine, module, store


def test_csfw_obeys_event_timing_quota_and_never_pairs_a_death() -> None:
    engine, module, store = _parts()
    x = torch.tensor([[1.0, -0.4, 0.8], [-0.3, 0.5, 1.1]], dtype=torch.float64)
    upstream = torch.tensor([[0.7, -0.2], [-0.1, 0.9]], dtype=torch.float64)

    engine.begin_update()
    module(x).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    assert engine.step() == ()

    module.zero_grad(set_to_none=True)
    engine.begin_update()
    module(x).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    operations = engine.step()

    births = [op for op in operations if isinstance(op, SynapseBirth)]
    assert 0 < sum(int(op.w.numel()) for op in births) <= 2
    assert not any(isinstance(op, SynapseDeath) for op in operations)
    assert store.live_ids().numel() == 1 + sum(int(op.w.numel()) for op in births)

