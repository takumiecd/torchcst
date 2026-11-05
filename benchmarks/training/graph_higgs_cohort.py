"""Explicit HIGGS density cohort for the private B4 whole-step experiment.

The tag is a declaration, not an exemption from the stock callback, native
execution, optimizer, storage or replay checks in ``PreparedStep``.
"""

import torch
from torch import nn

from benchmarks.training.graph_stock_qualification import QualificationError
from torchcst import CSTLinear

HIGGS_GRAPH_COHORT = "higgs_middle_density_v1"
HIGGS_GRAPH_WIDTHS = (1024, 2048, 4096, 8192)
HIGGS_ATOM_DIVISORS = (200, 100, 50, 20, 10)
HIGGS_CONFIRMATION_COHORT = "higgs_alpha_confirmation_v1"
HIGGS_CONFIRMATION_WIDTHS = (1024, 2048)
HIGGS_CONFIRMATION_DIVISORS = (1000, 500, 200, 100)


def higgs_graph_signature(model, tx, ty, kind):
    """Return primitive cohort metadata, rejecting any changed declaration.

    Untagged existing MNIST cohorts retain their original qualification path.
    The caller must run stock qualification before examining tagged modules.
    """
    namespace = object.__getattribute__(model, "__dict__")
    tag = namespace.get("_cst_research_cohort")
    if tag is None:
        children = namespace.get("_modules")
        if type(children) is dict:
            first, last = children.get("0"), children.get("4")
            if (
                type(first) is nn.Linear
                and type(last) is nn.Linear
                and first.in_features == 28
                and last.out_features == 2
            ):
                raise QualificationError(
                    "HIGGS models require their explicit research cohort tag"
                )
        return None
    if type(tag) is not str or tag not in (
        HIGGS_GRAPH_COHORT,
        HIGGS_CONFIRMATION_COHORT,
    ):
        raise QualificationError("unknown whole-step research cohort")
    widths, divisors = (
        (HIGGS_GRAPH_WIDTHS, HIGGS_ATOM_DIVISORS)
        if tag == HIGGS_GRAPH_COHORT
        else (HIGGS_CONFIRMATION_WIDTHS, HIGGS_CONFIRMATION_DIVISORS)
    )
    if (
        kind not in ("cst", "dense")
        or type(model) is not nn.Sequential
        or tuple(model._modules) != ("0", "1", "2", "3", "4")
        or tuple(type(child) for child in model._modules.values())
        != (
            nn.Linear,
            nn.ReLU,
            CSTLinear if kind == "cst" else nn.Linear,
            nn.ReLU,
            nn.Linear,
        )
    ):
        raise QualificationError("HIGGS requires the canonical five-element model")
    width = model[0].out_features
    if (
        type(width) is not int
        or width not in widths
        or model[0].in_features != 28
        or (model[2].in_features, model[2].out_features) != (width, width)
        or (model[4].in_features, model[4].out_features) != (width, 2)
        or tuple(model[0].weight.shape) != (width, 28)
        or tuple(model[4].weight.shape) != (2, width)
        or model[0].bias is None
        or model[4].bias is None
        or tuple(model[0].bias.shape) != (width,)
        or tuple(model[4].bias.shape) != (2,)
        or not all(parameter.requires_grad for parameter in model.parameters())
    ):
        raise QualificationError(
            "HIGGS requires trainable 28/N/N/2 layers at an explicit width"
        )
    if (
        type(tx) is not torch.Tensor
        or type(ty) is not torch.Tensor
        or tx.dtype != torch.float32
        or ty.dtype != torch.int64
        or tx.ndim != 2
        or tuple(tx.shape) != (ty.numel(), 28)
        or ty.ndim != 1
        or len(ty) < 4
        or tx.device != ty.device
        or tx.requires_grad
        or ty.requires_grad
    ):
        raise QualificationError(
            "HIGGS requires FP32 rows/INT64 labels for true B4 updates"
        )
    count = None
    if kind == "cst":
        shape = tuple(model[2].atoms.p.shape)
        count = shape[0] if len(shape) == 2 else None
        if shape != (count, 4) or count not in tuple(
            width * width // divisor for divisor in divisors
        ):
            raise QualificationError(
                "HIGGS graph atom count is outside the declared density grid"
            )
    elif tuple(model[2].weight.shape) != (width, width):
        raise QualificationError(
            "HIGGS dense middle weight differs from the declared width"
        )
    return tag, width, count, kind
