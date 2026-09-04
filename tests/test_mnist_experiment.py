from __future__ import annotations

import gzip
import struct
from pathlib import Path

import torch

from experiments.mnist_current_api import (
    ExperimentConfig,
    build_model,
    build_optimizer,
    read_idx,
)
from torchcst import FullQuartic, ProjectedLBFGS


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


def test_read_idx_falls_back_to_gzip_file(tmp_path: Path) -> None:
    path = tmp_path / "labels-idx1-ubyte"
    blob = struct.pack(">II", 0x00000801, 4) + bytes((3, 1, 4, 1))
    path.with_name(path.name + ".gz").write_bytes(gzip.compress(blob))

    actual = read_idx(path)

    torch.testing.assert_close(actual, torch.tensor([3, 1, 4, 1], dtype=torch.uint8))


def test_mnist_runner_selects_full_quartic() -> None:
    config = ExperimentConfig(atoms=2, solver="full", solver_max_iter=3)
    optimizer = build_optimizer(build_model(config, torch.device("cpu")), config)

    assert isinstance(optimizer.cst_config.quartic, FullQuartic)


def test_mnist_runner_selects_projected_lbfgs() -> None:
    config = ExperimentConfig(
        atoms=2,
        solver="projected",
        solver_max_iter=3,
        solver_max_evaluations=5,
    )
    optimizer = build_optimizer(build_model(config, torch.device("cpu")), config)

    assert isinstance(optimizer.cst_config.quartic, ProjectedLBFGS)
