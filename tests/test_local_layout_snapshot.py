"""Snapshot ownership can be checked without importing or executing GPU code."""

import os
import subprocess
import sys

import pytest
import torch

from torchcst._backends.cuda.algorithms.local_product.layout import LayoutSnapshot


@pytest.mark.parametrize("persistent,slots", [(False, 204), (True, 464)])
def test_snapshot_owns_separate_storage(persistent, slots):
    metadata = torch.zeros(13, 204)
    first, second = (
        LayoutSnapshot.allocate(metadata, slots=slots, stride=9, persistent=persistent)
        for _ in range(2)
    )
    assert first.views.shape == (2, 13, slots)
    assert first.orders.shape == (2, slots)
    assert first.starts.shape == (2, 9)
    assert first.views.dtype == metadata.dtype
    assert first.orders.dtype == first.starts.dtype == torch.int32
    tensors = (
        ("views", "orders", "starts", "ends")
        if persistent
        else ("views", "orders", "starts")
    )
    for name in tensors:
        before, after = getattr(first, name), getattr(second, name)
        before.fill_(7)
        after.fill_(3)
        assert torch.all(before == 7)
        assert before.data_ptr() != metadata.data_ptr()
    metadata.fill_(5)
    assert torch.all(first.views == 7)
    if persistent:
        assert first.ends.shape == first.starts.shape
        assert first.ends.data_ptr() != first.starts.data_ptr()
    else:
        assert first.ends is None


def test_empty_snapshots_keep_configured_capacities():
    metadata = torch.empty(13, 0)
    compact = LayoutSnapshot.allocate(metadata, slots=0, stride=7, persistent=False)
    persistent = LayoutSnapshot.allocate(metadata, slots=192, stride=7, persistent=True)
    assert compact.views.shape == (2, 13, 0)
    assert persistent.views.shape == (2, 13, 192)
    assert persistent.orders.shape == (2, 192)


def test_rejects_wrong_metadata_and_insufficient_slot_capacity():
    with pytest.raises(ValueError, match="metadata"):
        LayoutSnapshot.allocate(torch.empty(9, 4), slots=4, stride=3, persistent=False)
    with pytest.raises(ValueError, match="metadata"):
        LayoutSnapshot.allocate(
            torch.empty(4, 13).T, slots=4, stride=3, persistent=False
        )
    with pytest.raises(ValueError, match="capacity"):
        LayoutSnapshot.allocate(torch.empty(13, 4), slots=3, stride=3, persistent=True)


def test_layout_imports_do_not_load_triton():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from torchcst._backends.cuda.algorithms.local_product "
                "import layout, persistent; assert 'triton' not in sys.modules"
            ),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=True,
    )
