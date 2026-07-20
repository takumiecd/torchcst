"""Rent and magnitude retention contract tests."""

from __future__ import annotations

import torch

from torchcst.policy import Clock, MagnitudeCourt, RentCourt
from torchcst.storage import SynapseView


def view(mass: list[float]) -> SynapseView:
    count = len(mass)
    values = torch.tensor(mass)
    return SynapseView(
        site="edge",
        version=0,
        ids=torch.arange(10, 10 + count, dtype=torch.int64),
        s=torch.arange(count, dtype=torch.int64).reshape(count, 1),
        t=torch.arange(count, dtype=torch.int64).reshape(count, 1),
        w=values,
        mass=values,
    )


def death_ids(ops) -> list[int]:
    return [] if not ops else ops[0].ids.tolist()


def test_rent_boundary_and_two_consecutive_strikes_with_recovery_reset() -> None:
    court = RentCourt(immunity_events=0, rent_ratio=0.3, strikes=2)
    ages = torch.full((3,), 9, dtype=torch.int64)
    clock = Clock(1, 1)

    assert death_ids(court.decide(view([3.0, 10.0, 10.0]), ages, clock)) == []
    assert death_ids(court.decide(view([2.0, 10.0, 10.0]), ages, clock)) == []
    assert death_ids(court.decide(view([4.0, 10.0, 10.0]), ages, clock)) == []
    assert death_ids(court.decide(view([2.0, 10.0, 10.0]), ages, clock)) == []
    assert death_ids(court.decide(view([2.0, 10.0, 10.0]), ages, clock)) == [10]


def test_rent_median_uses_all_live_rows_but_immune_rows_are_not_targets() -> None:
    court = RentCourt(immunity_events=3, rent_ratio=0.3, strikes=1)
    # All-live median is 100 (threshold 30); eligible-only median would be 20.
    # Immune ID 10 is below 30 but is not an adjudication target.
    result = court.decide(
        view([0.1, 1000.0, 1000.0, 1000.0, 10.0, 20.0, 100.0]),
        torch.tensor([0, 0, 0, 0, 3, 3, 3], dtype=torch.int64),
        Clock(1, 1),
    )

    assert death_ids(result) == [14, 15]
    assert 10 not in court.strike_state


def test_magnitude_court_selects_exactly_the_n_smallest_masses() -> None:
    court = MagnitudeCourt(drop_fraction=lambda clock: 0.4)

    result = court.decide(
        view([5.0, 1.0, 4.0, 2.0, 3.0]),
        torch.zeros(5, dtype=torch.int64),
        Clock(20, 2),
    )

    assert death_ids(result) == [11, 13]
