"""Planted-target continuation check for MaturityGaussianKernel."""

from __future__ import annotations

import argparse

import torch

from torchcst import GaussianKernel, MaturityGaussianKernel, PullbackAdam
from torchcst.compute import CSTLinear
from torchcst.representation import L2NormalizedColumns, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def site(
    family: str,
    *,
    source: float,
    target: float,
    weight: float,
) -> tuple[CSTLinear, SynapseStore]:
    dtype = torch.float64
    spec = RepresentationSpec.continuous(1, 1, kernel=family)
    store = SynapseStore(family, 1, 1, 1, spec=spec, dtype=dtype)
    extras = (
        {}
        if family == "gaussian"
        else {"maturity": torch.tensor([[-2.0]], dtype=dtype)}
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[source]], dtype=dtype),
                torch.tensor([[target]], dtype=dtype),
                torch.tensor([weight], dtype=dtype),
                torch.tensor([0]),
                extras=extras,
            )
        ]
    )
    coordinates = torch.linspace(0.0, 1.0, 81, dtype=dtype).reshape(-1, 1)
    inputs = NeuronStore(
        f"{family}-in",
        81,
        mu=coordinates.clone(),
        initial_live=81,
        dtype=dtype,
    )
    outputs = NeuronStore(
        f"{family}-out",
        81,
        mu=coordinates.clone(),
        initial_live=81,
        dtype=dtype,
    )
    kernel = (
        GaussianKernel(0.07, learnable=False)
        if family == "gaussian"
        else MaturityGaussianKernel(0.07, learnable=False)
    ).double()
    return (
        CSTLinear(
            inputs,
            outputs,
            store,
            kernel,
            gauge=L2NormalizedColumns(),
        ),
        store,
    )


def run(family: str, planted: torch.Tensor, steps: int) -> dict[str, float]:
    module, store = site(family, source=0.12, target=0.18, weight=0.01)
    pullback = PullbackAdam(
        module,
        moment_space="tangent",
        metric="diag",
        cap_sigma=0.1,
        betas=(0.0, 0.99),
        target_step=0.1,
        damping=0.01,
        wall=True,
    )
    groups = [{"params": [store.w], "lr": 0.03}]
    if family == "maturity_gaussian":
        groups.append({"params": [store.maturity], "lr": 0.05})
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.99))
    loss = planted.new_tensor(float("nan"))
    for _ in range(steps):
        optimizer.zero_grad()
        pullback.zero_grad()
        loss = 0.5 * (module.dense_weight() - planted).square().sum()
        loss.backward()
        pullback.step()
        optimizer.step()
    result = {
        "loss": float(loss.detach()),
        "source": float(store.s.detach()[0, 0]),
        "target": float(store.t.detach()[0, 0]),
        "weight": float(store.w.detach()[0]),
    }
    if family == "maturity_gaussian":
        fraction = torch.sigmoid(store.maturity.detach()[0, 0])
        scale = (0.25**2 + (1.0 - 0.25**2) * fraction).sqrt()
        result["inverse_width_scale"] = float(scale)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()
    teacher, _ = site("gaussian", source=0.78, target=0.83, weight=1.0)
    planted = teacher.dense_weight().detach()
    for family in ("gaussian", "maturity_gaussian"):
        print(family, run(family, planted, args.steps))


if __name__ == "__main__":
    main()
