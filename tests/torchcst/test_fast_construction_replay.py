"""Fast-construction random candidate search has deterministic replay."""

from __future__ import annotations

import torch

from torchcst.storage import SynapseBirth

from .test_fast_construction_quota import _parts


def _trajectory(seed: int) -> tuple[tuple[torch.Tensor, ...], ...]:
    engine, module, _ = _parts(seed)
    x = torch.tensor([[1.0, -0.4, 0.8], [-0.3, 0.5, 1.1]], dtype=torch.float64)
    upstream = torch.tensor([[0.7, -0.2], [-0.1, 0.9]], dtype=torch.float64)
    result: list[tuple[torch.Tensor, ...]] = []
    for _ in range(4):
        module.zero_grad(set_to_none=True)
        engine.begin_update()
        module(x).backward(upstream)
        engine.observe_microbatch()
        engine.finalize_backward()
        births = [op for op in engine.step() if isinstance(op, SynapseBirth)]
        result.append(
            tuple(value.clone() for op in births for value in (op.s, op.t, op.w))
        )
    return tuple(result)


def _equal(left: tuple[tuple[torch.Tensor, ...], ...], right: object) -> bool:
    if not isinstance(right, tuple) or len(left) != len(right):
        return False
    return all(
        len(one) == len(two) and all(torch.equal(a, b) for a, b in zip(one, two))
        for one, two in zip(left, right)
    )


def test_csfw_replay_is_bit_identical_for_one_seed() -> None:
    assert _equal(_trajectory(2026), _trajectory(2026))
    assert not _equal(_trajectory(2026), _trajectory(2027))
