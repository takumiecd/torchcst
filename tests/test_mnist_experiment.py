from __future__ import annotations

import struct
from pathlib import Path

import torch

from experiments.mnist_current_api import read_idx


def test_read_idx_loads_uint8_tensor_without_optional_dependencies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "images-idx3-ubyte"
    values = bytes(range(12))
    path.write_bytes(struct.pack(">IIII", 0x00000803, 2, 2, 3) + values)

    actual = read_idx(path)

    assert actual.dtype == torch.uint8
    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(actual.reshape(-1), torch.arange(12, dtype=torch.uint8))
