"""CUDA eigensolve returning device info, without PyTorch's host error check."""

import platform
from functools import cache
from pathlib import Path

import torch


@cache
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    nvidia = Path(torch.__file__).resolve().parent.parent / "nvidia"
    includes = list(nvidia.glob("*/include"))
    libraries = list(nvidia.glob("*/lib"))
    if CUDA_HOME:
        target = Path(CUDA_HOME) / "targets" / (platform.machine() + "-linux")
        includes += [target / "include"]
        libraries += [target / "lib", Path(CUDA_HOME) / "lib"]
    solver = list(nvidia.glob("cusolver/lib/libcusolver.so.*"))

    return load_inline(
        name="torchcst_tangent_eigh_v1",
        cpp_sources=Path(__file__).with_suffix(".cpp").read_text(),
        functions=["eigh_unchecked"],
        with_cuda=True,
        extra_include_paths=[str(p) for p in includes if p.is_dir()],
        extra_ldflags=[
            "-ltorch_cuda_linalg",
            "-Wl,-rpath," + str(Path(torch.__file__).resolve().parent / "lib"),
        ]
        + ["-L" + str(p) for p in libraries if p.is_dir()]
        + ([str(solver[0])] if solver else ["-lcusolver"]),
        verbose=False,
    )


def eigh(matrix):
    return extension().eigh_unchecked(matrix)
