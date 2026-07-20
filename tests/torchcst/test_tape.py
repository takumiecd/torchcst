"""Deterministic batch tape and content hash tests."""

from __future__ import annotations

import torch

from torchcst.lab import BatchTape, RngStreams


def materialize(tape: BatchTape) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(x.clone(), y.clone()) for x, y in tape]


def assert_same_batches(
    left: list[tuple[torch.Tensor, torch.Tensor]],
    right: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    assert len(left) == len(right)
    assert all(
        torch.equal(left_x, right_x) and torch.equal(left_y, right_y)
        for (left_x, left_y), (right_x, right_y) in zip(left, right)
    )


def test_reiteration_returns_the_identical_batch_sequence() -> None:
    x = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    y = torch.arange(10, dtype=torch.int64)
    tape = BatchTape(x, y, batch_size=4, root_seed=5, epochs=3)

    first = materialize(tape)
    first[0][0].fill_(-1)
    second = materialize(tape)
    third = materialize(tape)

    assert_same_batches(second, third)
    assert len(tape) == 9
    assert not torch.equal(first[0][0], second[0][0])


def test_tape_hash_is_stable_and_sensitive_to_one_element() -> None:
    x = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    y = torch.arange(8, dtype=torch.int64)
    same_a = BatchTape(x, y, 3, root_seed=123, epochs=2)
    same_b = BatchTape(x.clone(), y.clone(), 3, root_seed=123, epochs=2)
    changed_x = x.clone()
    changed_x[2, 1] += 1
    changed = BatchTape(changed_x, y, 3, root_seed=123, epochs=2)

    assert same_a.tape_hash == same_b.tape_hash
    assert same_a.tape_hash != changed.tape_hash


def test_tape_snapshots_inputs_and_is_independent_of_proposal_consumption() -> None:
    x = torch.arange(12).reshape(6, 2)
    y = torch.arange(6)
    streams = RngStreams(44)
    torch.rand(100, generator=streams.get("proposal"))
    isolated = BatchTape(x, y, 2, streams=streams)
    reference = BatchTape(x, y, 2, root_seed=44)
    original_hash = isolated.tape_hash
    x.fill_(999)
    y.fill_(999)

    assert isolated.tape_hash == reference.tape_hash == original_hash
    assert_same_batches(materialize(isolated), materialize(reference))


def test_each_epoch_has_a_fixed_full_permutation() -> None:
    x = torch.arange(20).reshape(10, 2)
    y = torch.arange(10)
    tape = BatchTape(x, y, 10, root_seed=91, epochs=2)
    epoch_zero = next(tape.iter_epoch(0))[1]
    epoch_one = next(tape.iter_epoch(1))[1]

    assert torch.equal(torch.sort(epoch_zero).values, torch.arange(10))
    assert torch.equal(torch.sort(epoch_one).values, torch.arange(10))
    assert not torch.equal(epoch_zero, epoch_one)
