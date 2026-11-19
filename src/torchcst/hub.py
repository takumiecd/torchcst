"""Opt-in Safetensors inference loading for locally constructed PyTorch models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch
from torch import nn

__all__ = ["from_pretrained", "load_safetensors"]

_Model = TypeVar("_Model", bound=nn.Module)


def load_safetensors(
    path: str | Path,
    *,
    model_factory: Callable[[], _Model],
    device: str | torch.device = "cpu",
) -> _Model:
    """Build the model, load its tensor state, and return it in eval mode.

    The factory must recreate the declared architecture and dtypes. The file
    must contain all tensor entries from the model's state_dict. Non-tensor
    extra state is supplied by the factory.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError("Install torchcst[hub] to load Safetensors.") from error

    tensors = load_file(path, device="cpu")
    model = model_factory()
    state = model.state_dict()
    expected = {
        name: value for name, value in state.items() if isinstance(value, torch.Tensor)
    }
    if tensors.keys() != expected.keys():
        missing = sorted(expected.keys() - tensors.keys())
        unexpected = sorted(tensors.keys() - expected.keys())
        raise ValueError(
            f"Safetensors keys do not match the model. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    for name, tensor in tensors.items():
        target = expected[name]
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError(
                f"Safetensors tensor {name!r} has shape {tuple(tensor.shape)} "
                f"and dtype {tensor.dtype}, expected {tuple(target.shape)} "
                f"and {target.dtype}."
            )

    state.update(tensors)
    model.load_state_dict(state)
    return model.to(device).eval()


def from_pretrained(
    repo_id: str,
    *,
    model_factory: Callable[[], _Model],
    filename: str = "model.safetensors",
    revision: str | None = None,
    device: str | torch.device = "cpu",
    token: str | bool | None = None,
) -> _Model:
    """Download a model's Safetensors file from Hugging Face and load it."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("Install torchcst[hub] to download model weights.") from error

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
    return load_safetensors(path, model_factory=model_factory, device=device)
