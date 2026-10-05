"""Cached declarations of live Chart/Kernel state, shared by operation bindings."""

import torch


def operator_declaration(binding):
    """Cache immutable live-operator metadata; refresh configuration outside capture."""
    roots = (*binding.cst_charts(), binding.kernel)
    signature = tuple(
        (id(m), getattr(m, "spec", None), getattr(m, "binding", None))
        for root in roots
        for m in root.modules()
    )
    buffers = tuple(t for root in roots for t in root.buffers())
    reusable = not any(t.is_inference() for t in buffers)
    if reusable:
        signature += tuple((id(t), t._version, t.dtype, t.device) for t in buffers)
    previous = binding.__dict__.get("_execution_declaration")
    if reusable and previous is not None and previous[0] == signature:
        return previous[1]
    if (
        binding.execution_parameters().is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "chart/kernel metadata must be refreshed outside CUDA graph capture"
        )
    declaration = binding.declaration()
    binding.__dict__["_execution_declaration"] = (signature, declaration)
    return declaration
