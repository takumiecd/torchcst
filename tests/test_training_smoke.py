from collections.abc import Callable

import pytest
import torch

from torchcst import (
    Amplitude,
    AmplitudeBandwidthSeparable,
    Chart,
    CSTLinear,
    CSTOptimizer,
    FullQuartic,
    Gaussian,
    ImplicitAdamConfig,
    Kernel,
    Separable,
)


def independent_amplitude() -> Kernel:
    return Amplitude(
        Separable(
            input_profile=Gaussian(0.7),
            output_profile=Gaussian(0.25),
        )
    )


def amplitude_bandwidth() -> Kernel:
    return AmplitudeBandwidthSeparable(
        input_profile=Gaussian(0.7),
        output_profile=Gaussian(0.25),
        sigma_explore=1.0,
        tau=0.2,
        temperature=0.5,
    )


@pytest.mark.parametrize("kernel_factory", [independent_amplitude, amplitude_bandwidth])
def test_full_optimizer_reduces_loss_for_new_amplitude_kernels(
    kernel_factory: Callable[[], Kernel],
) -> None:
    torch.manual_seed(9)
    model = CSTLinear(
        Chart.linspace(2),
        Chart.linspace(2),
        atoms=2,
        kernel=kernel_factory(),
        backend="factored",
        dtype=torch.float64,
    )
    optimizer = CSTOptimizer(
        model,
        cst=ImplicitAdamConfig(
            lr=0.03,
            betas=(0.5, 0.8),
            trust_radius=0.08,
            quartic=FullQuartic(starts=1, max_iter=20),
        ),
        dense=None,
    )
    inputs = torch.tensor(
        [[-1.0, -0.5], [-0.5, 0.25], [0.25, 0.75], [1.0, -0.25]],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [[-0.5, 0.2], [0.1, -0.3], [0.7, 0.4], [0.25, -0.6]],
        dtype=torch.float64,
    )

    def objective() -> torch.Tensor:
        return (model(inputs) - targets).square().mean()

    losses = [objective().detach()]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
        losses.append(objective().detach())

    assert all(torch.isfinite(loss) for loss in losses)
    assert losses[1] < losses[0]
    assert losses[2] < losses[1]
    assert optimizer.last_step is not None
    assert optimizer.state_dict()["cst"]["<root>"].step == 2
    assert torch.isfinite(model.atoms.p).all()
