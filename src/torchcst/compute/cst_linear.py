"""General CST linear map over continuous coordinates."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.representation import (
    GaussianKernel,
    RepresentationSpec,
    TriangularKernel,
    propose_chart,
)
from torchcst.storage import NeuronStore, SynapseStore

from .cst_map import _ContinuousCSTMap

_KERNELS = {"gaussian": GaussianKernel, "triangular": TriangularKernel}


class CSTLinear(_ContinuousCSTMap):
    """Apply a continuous CST measure to feature rows.

    Neuron coordinates are fixed floating buffers. Synapse source and target
    coordinates, atom weights, and global kernel bandwidths remain learnable.
    """

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_rows(x)

    @classmethod
    def propose(
        cls,
        site: str,
        in_features: int,
        out_features: int,
        sigma: float,
        *,
        capacity: int | None = None,
        kernel: str = "gaussian",
        generator: torch.Generator | None = None,
        axis_extent: float | None = None,
    ) -> "CSTLinear":
        """Build a hidden-site CSTLinear on lawful, sampled charts.

        The one-call form of the propose → sample → wire flow for a site
        whose populations carry no data-pinned geometry: both endpoint charts
        come from :func:`torchcst.representation.propose_chart` (lawful by
        construction), neuron coordinates are uniform samples from the
        proposed boxes (the measured winning placement), and the boxes become
        the synapse store's birth domains, so every future structural birth
        also lands inside the envelope.

        ``capacity`` defaults to the larger of the two proposals'
        ``recommended_atoms`` — one atom per resolvable cell, a floor with
        growth headroom, not a ceiling.  ``axis_extent`` (σ units per axis)
        is forwarded to the proposals when given.  The returned module's
        neuron stores are named ``f"{site}.in"`` / ``f"{site}.out"``;
        :meth:`stores` hands the engine-wiring dict straight to
        ``StructuralEngine``.

        Sites with data-pinned charts (pixels, taps) should not use this —
        build their stores from the data's own coordinates and, if in doubt,
        run :func:`torchcst.representation.survey_chart` on them.
        """
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS)}, got {kernel!r}")
        extent = {} if axis_extent is None else {"axis_extent": axis_extent}
        proposal_in = propose_chart(in_features, sigma, **extent)
        proposal_out = propose_chart(out_features, sigma, **extent)
        rng = generator if generator is not None else torch.Generator()
        inputs = NeuronStore(
            f"{site}.in",
            in_features,
            mu=proposal_in.box.sample(in_features, rng),
            initial_live=in_features,
        )
        outputs = NeuronStore(
            f"{site}.out",
            out_features,
            mu=proposal_out.box.sample(out_features, rng),
            initial_live=out_features,
        )
        rows = capacity if capacity is not None else max(
            proposal_in.recommended_atoms, proposal_out.recommended_atoms
        )
        synapses = SynapseStore(
            site,
            d_in=proposal_in.dim,
            d_out=proposal_out.dim,
            capacity=rows,
            spec=RepresentationSpec.continuous(
                proposal_in.dim,
                proposal_out.dim,
                bounds=proposal_in.box.bounds,
                bounds_out=proposal_out.box.bounds,
                kernel=kernel,
            ),
        )
        return cls(inputs, outputs, synapses, _KERNELS[kernel](sigma))
