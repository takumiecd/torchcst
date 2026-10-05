"""Bind regular normalized radial operators to existing CPU/CUDA algorithms."""

import torch

from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.contract import (
    NormalizedStripGeometry,
)
from torchcst.kernels.presets import NORMALIZED_RADIAL_TRIWEIGHT


def binding(site):
    """Refresh only configuration metadata, never atom values or gradients."""
    if site.kernel.spec != NORMALIZED_RADIAL_TRIWEIGHT:
        return None
    modules = (*site.cst_charts(), site.kernel)
    signature = tuple(
        (id(m), getattr(m, "spec", None), getattr(m, "binding", None))
        for root in modules
        for m in root.modules()
    ) + tuple(
        (id(t), t._version, t.dtype, t.device)
        for root in modules
        for t in root.buffers()
    )
    previous = site.__dict__.get("_normalized_binding")
    if previous is not None and previous[0] == signature:
        return previous[1]
    if site.atoms.p.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "chart/kernel metadata must be refreshed outside CUDA graph capture"
        )
    operator = site.declaration()
    try:
        plan = NormalizedStripGeometry.from_declaration(operator)
    except ValueError:
        result = None
    else:
        result = (operator, plan)
    site.__dict__["_normalized_binding"] = (signature, result)
    return result


def supports(site):
    bound = binding(site)
    if bound is None:
        return False
    p = site.atoms.p
    if p.device.type == "cpu":
        return p.dtype in (torch.float32, torch.float64)
    if p.device.type == "cuda" and p.dtype == torch.float32:
        from torchcst._backends.cuda.algorithms.normalized_euclidean_strip.constraints import (
            routing_reasons,
        )

        return not routing_reasons(bound[1])
    return False


def forward(site, inputs, p):
    operator, plan = binding(site)
    if inputs.device != p.device or inputs.dtype != p.dtype:
        raise ValueError("input and parameters must have the same device and dtype")
    if p.dtype not in (torch.float32, torch.float64):
        raise TypeError("normalized radial algorithms require float32 or float64")
    flat = inputs.reshape(-1, site.in_features).contiguous()
    if len(p) == 0 or len(flat) == 0:
        result = (
            flat.new_zeros((len(flat), site.out_features))
            + flat.reshape(-1)[:1].mul(0).sum()
            + p.reshape(-1)[:1].mul(0).sum()
        )
    elif flat.device.type == "cpu":
        from torchcst._backends.torch.operators.normalized_radial import apply

        result = apply(flat, p, plan)
    elif flat.device.type == "cuda":
        from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import (
            BOOTSTRAP_SELECTOR,
            REGISTRY,
        )
        from torchcst._backends.cuda.context import context_from_tensors

        # Triton kernels use packed rows. Keep shared Parameters intact and let
        # autograd propagate through a temporary copy for strided atom tables.
        parameters = p.contiguous()
        context = context_from_tensors(operator, flat, parameters)
        selector = site.selector if site.selector is not None else BOOTSTRAP_SELECTOR
        decision = selector.select(context)
        algorithm = REGISTRY.get(
            decision.plan.algorithm_id, revision=decision.plan.algorithm_revision
        )
        result = REGISTRY.execute(
            decision.plan,
            context,
            x=flat,
            parameters=parameters,
            operator=operator,
            state=site.algorithm_state(
                algorithm, recipe=decision.plan.recipe, operator=operator
            ),
        )
    else:
        raise ValueError("normalized radial algorithms require CPU or CUDA")
    return result.reshape(*inputs.shape[:-1], site.out_features)
