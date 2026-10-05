"""Resource limits of the optional A100 split-reduction schedule."""

import pytest

from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.schedule import (
    WORKSPACE_BYTES,
    split_count,
)


@pytest.mark.parametrize("elements", [0, 1024, 8192, 1048576, 8388608, 16777216])
@pytest.mark.parametrize("multiprocessors", [0, 42, 108])
def test_split_workspace_is_bounded_and_other_devices_keep_baseline(
    elements, multiprocessors
):
    parts = split_count(
        reduction_tiles=5,
        elements=elements,
        programs=32,
        multiprocessors=multiprocessors,
        atoms=256,
        stations=4,
    )
    assert 1 <= parts <= 5
    if parts > 1:
        assert parts * elements * 4 <= WORKSPACE_BYTES
    if multiprocessors == 0 or elements == 0:
        assert parts == 1


@pytest.mark.parametrize("programs,atoms", [(0, 256), (1024, 256), (4, 1)])
def test_empty_saturated_or_sparse_work_does_not_add_partial_buffers(programs, atoms):
    assert (
        split_count(
            reduction_tiles=8,
            elements=8192,
            programs=programs,
            multiprocessors=42,
            atoms=atoms,
            stations=4,
        )
        == 1
    )
