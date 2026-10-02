"""Colab request names and verified GPU families, independent of PyTorch."""

import re

ACCELERATORS = {
    "T4": (r"(?:Tesla|NVIDIA) T4", [7, 5]),
    "L4": (r"NVIDIA L4", [8, 9]),
    "A100": (r"NVIDIA A100(?:[ -].*)?", [8, 0]),
    "H100": (r"NVIDIA H100(?:[ -].*)?", [9, 0]),
    "G4": (r"NVIDIA RTX PRO 6000 Blackwell Server Edition", [12, 0]),
}


def target(accelerator):
    if type(accelerator) is not str or accelerator not in ACCELERATORS:
        raise ValueError("unsupported Colab accelerator")
    return {"provider": "colab", "accelerator": accelerator}


def validate_hardware(accelerator, name, capability):
    target(accelerator)
    pattern, expected = ACCELERATORS[accelerator]
    if (
        type(name) is not str
        or not re.fullmatch(pattern, name)
        or capability != expected
    ):
        raise ValueError("actual GPU differs from requested Colab accelerator")
