"""Depthwise spatial conv + CST channel mixing: the measured winning conv form.

The 2026-08 conv arc ended with an arbitration (reports in the sibling
``cst`` repository, ``docs/research/cst-native-convolution.md``): a 3×3
spatial axis is quasi-discrete — its taps sit ~5σ apart, so continuous
coordinates there move through data-free vacuum — while the channel axis
supports a lawful 2-D chart on which every CST part (coordinate learning,
scored birth, growth) works.  The honest conv is therefore a composition:

* **spatial** — a plain per-channel (depthwise) convolution, the
  quasi-discrete axis handled discretely;
* **channel** — :class:`~torchcst.compute.CSTLinear` applied as a 1×1
  across channel populations proposed on lawful charts.

Measured on the arc's CIFAR testbed, this composition (dense stem +
two such blocks) reached 0.6400 mean over 3 seeds at ~10k total parameters —
above every anchor of the arc, including the standalone factorized winner
(0.6351) and a parameter-matched dense net (0.6243).

Two composition-level rules travel with the module as documented defaults,
not hard gates:

1. **The stem rule** — populations that carry data geometry (e.g. the RGB
   input) must *not* be charted: :func:`propose_chart` would hand 3 channels
   a 1-D line and starve the mixing (the measured v1 failure, −4.7pt).  Keep
   the stem a plain conv and start the CST blocks from the first hidden
   population.
2. **The capacity rule** — the atom budget defaults to twice the resolvable
   cells of the larger endpoint chart (``capacity_scale=2.0``): the measured
   capacity ladder was still unsaturated at ~1.3 atoms per cell.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst.representation import (
    GaussianFactor,
    TriangularFactor,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

from .cst_linear import CSTLinear

__all__ = ["DepthwiseCSTConv2d"]

_KERNELS = {"gaussian": GaussianFactor, "triangular": TriangularFactor}


class DepthwiseCSTConv2d(nn.Module):
    """Per-channel spatial convolution followed by CST channel mixing.

    Composition-first, like every CST layer: the caller builds the channel
    populations and the synapse store (neurons first, synapses derived), and
    this module owns only the depthwise weights and the shape plumbing.  Use
    :meth:`propose` for the one-call lawful construction.

    Engine wiring: the CST half is an ordinary :class:`CSTLinear` at
    ``self.mix`` — pass ``self.stores()`` to ``StructuralEngine`` and
    register ``modules={self.capture_site: self.mix}`` for observation
    capture.  The depthwise weights are plain parameters and take part in
    the ordinary optimizer only.
    """

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        factor: GaussianFactor | TriangularFactor,
        kernel_size: int | tuple[int, int],
        *,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        size = (
            (int(kernel_size), int(kernel_size))
            if isinstance(kernel_size, int)
            else (int(kernel_size[0]), int(kernel_size[1]))
        )
        channels = in_neurons.n_max
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            size,
            stride=stride,
            padding=(size[0] // 2, size[1] // 2) if padding is None else padding,
            groups=channels,
        )
        self.mix = CSTLinear(in_neurons, out_neurons, synapses, factor)

    # -- delegation to the CST half ------------------------------------------

    @property
    def in_neurons(self) -> NeuronStore:
        return self.mix.in_neurons

    @property
    def out_neurons(self) -> NeuronStore:
        return self.mix.out_neurons

    @property
    def synapses(self) -> SynapseStore:
        return self.mix.synapses

    @property
    def capture_site(self) -> str:
        return self.mix.capture_site

    def stores(self):
        return self.mix.stores()

    # -- forward --------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        h = self.depthwise(x)
        batch, channels, height, width = h.shape
        rows = h.permute(0, 2, 3, 1).contiguous().reshape(-1, channels)
        mixed = self.mix(rows)
        return (
            mixed.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        )

    # -- lawful one-call construction -----------------------------------------

    @classmethod
    def propose(
        cls,
        site: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        sigma: float,
        *,
        capacity_scale: float = 2.0,
        seed_atoms: bool = True,
        seed_weight_scale: float = 0.05,
        factor: str = "gaussian",
        stride: int = 1,
        padding: int | None = None,
        generator: torch.Generator | None = None,
    ) -> "DepthwiseCSTConv2d":
        """Build the block on lawful charts, atoms seeded uniform-in-box.

        Both channel populations come from :meth:`NeuronStore.propose`, the
        synapse store from :meth:`SynapseStore.between` with its capacity
        scaled by ``capacity_scale`` (default: the measured 2× cells), and —
        unless ``seed_atoms=False`` — the store starts with ``capacity``
        live atoms drawn uniformly from the birth domains with amplitudes
        ``U(−seed_weight_scale, seed_weight_scale)``, the arc's convention.

        Remember the stem rule (module docstring): do not call this for a
        population that carries data geometry, such as the RGB input.
        """
        if factor not in _KERNELS:
            raise ValueError(f"factor must be one of {sorted(_KERNELS)}, got {factor!r}")
        if not capacity_scale > 0.0:
            raise ValueError(f"capacity_scale must be positive, got {capacity_scale}")
        rng = generator if generator is not None else torch.Generator()
        inputs = NeuronStore.propose(f"{site}.in", in_channels, sigma, generator=rng)
        outputs = NeuronStore.propose(f"{site}.out", out_channels, sigma, generator=rng)
        base = SynapseStore.between(site, inputs, outputs, sigma, factor=factor)
        synapses = SynapseStore.between(
            site,
            inputs,
            outputs,
            sigma,
            factor=factor,
            capacity=max(1, int(round(capacity_scale * base.capacity))),
        )
        if seed_atoms:
            _seed_uniform(synapses, synapses.capacity, seed_weight_scale, rng)
        return cls(
            inputs,
            outputs,
            synapses,
            _KERNELS[factor](sigma),
            kernel_size,
            stride=stride,
            padding=padding,
        )


def _seed_uniform(
    store: SynapseStore, count: int, weight_scale: float, rng: torch.Generator
) -> None:
    """Uniform-in-box birth of ``count`` live atoms (the arc's convention)."""

    def sample(bounds: tuple, dim: int) -> Tensor:
        lo = torch.as_tensor(bounds[0], dtype=torch.get_default_dtype()).reshape(1, -1)
        hi = torch.as_tensor(bounds[1], dtype=torch.get_default_dtype()).reshape(1, -1)
        return lo + (hi - lo) * torch.rand(count, dim, generator=rng)

    store.apply(
        [
            SynapseBirth(
                store.site,
                sample(store.spec.domain_in.bounds, store.d_in),
                sample(store.spec.domain_out.bounds, store.d_out),
                (torch.rand(count, generator=rng) * 2.0 - 1.0) * weight_scale,
                torch.arange(count, dtype=torch.int64),
            )
        ]
    )
