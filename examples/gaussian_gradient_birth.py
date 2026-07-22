"""One Gaussian gradient-scored birth through the public policy API."""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import ContinuousGradientRequest
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    Policy,
    ScoredBirth,
    StructuralQuota,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def build_example(*, capture_mode: str = "inline_reduced"):
    inputs = NeuronStore(
        "inputs",
        4,
        mu=torch.linspace(0.0, 1.0, 4)[:, None],
        initial_live=4,
    )
    outputs = NeuronStore(
        "outputs",
        3,
        mu=torch.linspace(0.0, 1.0, 3)[:, None],
        initial_live=3,
    )
    synapses = SynapseStore(
        "layer",
        d_in=1,
        d_out=1,
        capacity=8,
        spec=RepresentationSpec.continuous(1, 1),
    )
    synapses.apply(
        [
            SynapseBirth(
                "layer",
                torch.tensor([[0.5]]),
                torch.tensor([[0.5]]),
                torch.tensor([0.1]),
                torch.tensor([0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(inputs, outputs, synapses, GaussianKernel(0.2))
    scorer = ContinuousGradientRequest(pool_size=32, decay=0.0, chunk_size=8)
    birth = ScoredBirth(scorer, initial_weight=0.0)
    policy = Policy(
        cadence=PeriodicCadence(
            event_interval=1,
            observe_window=1,
        ),
        quota=ConstantQuota(StructuralQuota(synapse_birth=1)),
        proposers=(birth,),
        distributor=EvenBudgetDistributor(),
        retention=MagnitudeCourt(drop_fraction=0.0),
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    engine = StructuralEngine(
        {"layer": synapses, "inputs": inputs, "outputs": outputs},
        policy,
        modules={"layer": layer},
        optimizer=optimizer,
        capture_mode=capture_mode,
        seed=7,
    )
    return layer, synapses, optimizer, engine


def run_one_update(*, capture_mode: str = "inline_reduced"):
    layer, synapses, optimizer, engine = build_example(capture_mode=capture_mode)
    x = torch.randn(8, 4)
    target = torch.randn(8, 3)

    engine.begin_update()
    optimizer.zero_grad()
    loss = (layer(x) - target).square().mean()
    loss.backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    optimizer.step()
    operations = engine.step()
    return synapses, operations


if __name__ == "__main__":
    store, applied = run_one_update()
    print(f"live atoms: {store.live_ids().numel()}; structural ops: {len(applied)}")
