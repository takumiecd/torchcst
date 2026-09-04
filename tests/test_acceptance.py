import pytest
import torch

from torchcst.optim import ExactLossAcceptance


def test_exact_loss_acceptance_selects_the_first_nonincreasing_candidate() -> None:
    policy = ExactLossAcceptance(backtrack_factor=0.5, max_trials=4)
    seen = []

    def evaluate(scale: float) -> torch.Tensor:
        seen.append(scale)
        return torch.tensor(0.8 if scale <= 0.25 else 1.2)

    result = policy.select(torch.tensor(1.0), evaluate)

    assert result.accepted
    assert result.scale == 0.25
    assert result.trials == 3
    assert result.loss == 0.8
    assert seen == [1.0, 0.5, 0.25]


def test_exact_loss_acceptance_skips_nonfinite_candidates() -> None:
    policy = ExactLossAcceptance(max_trials=2)

    result = policy.select(
        torch.tensor(1.0),
        lambda scale: torch.tensor(float("nan") if scale == 1.0 else 0.9),
    )

    assert result.accepted
    assert result.scale == 0.5


def test_exact_loss_acceptance_rejects_without_mutating_the_base_result() -> None:
    policy = ExactLossAcceptance(backtrack_factor=0.25, max_trials=3)
    base = torch.tensor(1.0, requires_grad=True)

    result = policy.select(base, lambda scale: torch.tensor(1.0 + scale))

    assert not result.accepted
    assert result.scale == 0.0
    assert result.trials == 3
    assert result.loss == 1.0
    assert not result.loss.requires_grad


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"backtrack_factor": 0.0}, ValueError, "backtrack_factor"),
        ({"backtrack_factor": 1.0}, ValueError, "backtrack_factor"),
        ({"max_trials": 0}, ValueError, "max_trials must be positive"),
        ({"max_trials": True}, TypeError, "max_trials must be an integer"),
    ],
)
def test_exact_loss_acceptance_validates_configuration(
    kwargs: dict[str, object], exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        ExactLossAcceptance(**kwargs)


def test_exact_loss_acceptance_requires_a_finite_scalar_base_loss() -> None:
    policy = ExactLossAcceptance()

    with pytest.raises(ValueError, match="base loss must be scalar"):
        policy.select(torch.ones(2), lambda _: torch.tensor(0.0))
    with pytest.raises(FloatingPointError, match="base loss must be finite"):
        policy.select(torch.tensor(float("inf")), lambda _: torch.tensor(0.0))
