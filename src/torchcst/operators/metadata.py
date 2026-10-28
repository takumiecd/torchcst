"""Cached declarations of live Chart/Kernel state, shared by operation bindings."""

import torch


def fixed_state_signature(roots):
    """Track configuration ownership and Tensor mutations without reading values.

    Atom Parameters are intentionally excluded: normal learning must not cause
    frozen chart/profile/bandwidth declarations to be read back from the GPU.
    Inference tensors have no version counter, so their values cannot be cached.
    """
    modules = tuple(m for root in roots for m in root.modules())
    signature = tuple(
        (id(m), getattr(m, "spec", None), getattr(m, "binding", None)) for m in modules
    )
    # We already visited every submodule. Reading its registered buffers directly
    # avoids another recursive named-buffer traversal at each state check.
    buffers = tuple(t for m in modules for t in m._buffers.values() if t is not None)
    if any(t.is_inference() for t in buffers):
        return None
    return signature + tuple((id(t), t._version, t.dtype, t.device) for t in buffers)


def operator_declaration(binding):
    """Cache immutable live-operator metadata; refresh configuration outside capture."""
    signature = fixed_state_signature((*binding.cst_charts(), binding.kernel))
    previous = binding.__dict__.get("_execution_declaration")
    if signature is not None and previous is not None and previous[0] == signature:
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
