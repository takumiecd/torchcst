"""Cached declarations of live Chart/Kernel state, shared by operation bindings."""

import torch


def operator_declaration(binding):
    """Cache immutable live-operator metadata; refresh configuration outside capture."""
    roots = (*binding.cst_charts(), binding.kernel)
    modules = tuple(m for root in roots for m in root.modules())
    signature = tuple(
        (id(m), getattr(m, "spec", None), getattr(m, "binding", None)) for m in modules
    )
    # We already visited every submodule. Reading its registered buffers directly
    # avoids another recursive named-buffer traversal at each state check.
    buffers = tuple(t for m in modules for t in m._buffers.values() if t is not None)
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
