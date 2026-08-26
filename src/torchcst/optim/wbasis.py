"""Whitened amplitude basis -- the cross-atom block of the full pullback.

:class:`~torchcst.optim.pullback.PullbackAdam` owns the within-atom
coordinate geometry and :class:`~torchcst.optim.chart.ChartPullbackAdam` the
chart geometry.  The amplitude *cross-atom* Gram ``G_kl = <psi_k, psi_l>``
is owned by neither: under :class:`~torchcst.representation.gauges
.L2NormalizedColumns` the within-atom amplitude row of the metric is
identically zero, which is a statement about ``diag(G)``, not about the
atom-overlap off-diagonal, and every shipped optimizer steps ``w`` in the
Euclidean metric of that residual coupling.

:func:`install_whitened_basis` closes the gap by reparametrising one site's
live amplitudes as ``w = L^-T c`` with ``G + ridge * mean(diag(G)) * I =
L L^T``, where ``G`` is assembled from the site's *gauge-delivered* factor
columns on the actual neuron sets (so the same code is exact in both the
amplitude and the L2-normalised gauge, and for every factor family).  Any
diagonal optimizer stepping ``c`` then walks in the ``G^-1`` amplitude
geometry for the whole run; weight decay on ``c`` prices ``w`` along the
same geometry.  Drawing ``c`` i.i.d. is the whitened *initialisation*; this
module is its permanent, training-time form.

Deliberately unsupported, with explicit errors rather than silent
degradation (the repository-wide contract):

- **Structural events.** The factorisation is a snapshot of the live set,
  so the parametrization pins the ``SynapseStore`` version it was built
  against and every later access raises once the store has mutated.
  Re-fit with :func:`refresh_whitened_basis`.
- **Coordinate drift** is *tolerated* (the basis merely goes stale, the
  represented map stays exact); :func:`refresh_whitened_basis` re-expresses
  ``c`` against the current geometry, preserving the map.  Optimizer state
  attached to the amplitude leaf is expressed in the old basis and is the
  caller's to reset or transport.
- The bandwidth ``sigma`` block of the full pullback is future work; it is
  not touched here.
"""

from __future__ import annotations

import weakref

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from torchcst.compute import CSTLinear
from torchcst.storage import SynapseStore

__all__ = [
    "WhitenedAmplitudeBasis",
    "amplitude_leaf",
    "install_whitened_basis",
    "refresh_whitened_basis",
]


def _live_gram(site: CSTLinear, ridge: float) -> tuple[Tensor, Tensor]:
    """(cholesky factor, live slots) of the ridged live-atom Gram.

    Columns come from the site's amplitude gauge so ``G`` is the Gram of
    exactly the dictionary the backend multiplies ``w`` against.
    """
    if ridge <= 0.0:
        raise ValueError(f"ridge must be positive, got {ridge!r}")
    store = site.synapses
    live = store.live_slots().to(store.s.device)
    if live.numel() == 0:
        raise ValueError(f"{store.site!r}: no live atoms to whiten")
    with torch.no_grad():
        columns_out = site.gauge.columns(
            site.factor_out, site.out_neurons.mu.detach(),
            store.t.detach().index_select(0, live),
        ).double()
        columns_in = site.gauge.columns(
            site.factor_in, site.in_neurons.mu.detach(),
            store.s.detach().index_select(0, live),
        ).double()
        gram = (columns_out.T @ columns_out) * (columns_in.T @ columns_in)
        gram.diagonal().add_(ridge * float(gram.diagonal().mean()))
        chol = torch.linalg.cholesky(gram)
    return chol, live


class WhitenedAmplitudeBasis(nn.Module):
    """Parametrization ``w = L^-T c`` over one store's live amplitude rows.

    Dead slots pass through untouched (they are zero and must stay zero).
    The module pins the store version it was built against and raises on
    any access after a structural mutation -- a stale ``L`` would silently
    re-couple the atoms it claims to decorrelate.
    """

    upper: Tensor
    live: Tensor

    def __init__(self, store: SynapseStore, chol: Tensor, live: Tensor):
        super().__init__()
        parameter = store.w
        self.register_buffer(
            "upper",
            chol.T.contiguous().to(parameter.device, parameter.dtype),
        )
        self.register_buffer("live", live.to(parameter.device))
        self._store = weakref.ref(store)
        self._version = store.version

    def _check_version(self) -> None:
        store = self._store()
        if store is not None and store.version != self._version:
            raise RuntimeError(
                f"{store.site!r}: the whitened amplitude basis was built "
                f"against store version {self._version} but the store is now "
                f"at {store.version}; structural events under an installed "
                "basis are unsupported -- refresh_whitened_basis() re-fits "
                "the factorisation (optimizer state for the amplitude leaf "
                "is the caller's to reset)"
            )

    def forward(self, c: Tensor) -> Tensor:
        self._check_version()
        solved = torch.linalg.solve_triangular(
            self.upper, c.index_select(0, self.live).unsqueeze(1), upper=True
        ).squeeze(1)
        return c.index_copy(0, self.live, solved)

    def right_inverse(self, w: Tensor) -> Tensor:
        self._check_version()
        packed = (self.upper @ w.index_select(0, self.live).unsqueeze(1))
        return w.index_copy(0, self.live, packed.squeeze(1))


def amplitude_leaf(store: SynapseStore) -> nn.Parameter:
    """The trainable leaf behind ``store.w``, parametrized or not."""
    if parametrize.is_parametrized(store, "w"):
        return store.parametrizations.w.original
    return store.w


def install_whitened_basis(
    site: CSTLinear, *, ridge: float = 1e-2
) -> WhitenedAmplitudeBasis:
    """Reparametrise ``site``'s amplitudes into the whitened basis.

    The represented map is preserved exactly (``c0 = L^T w0``) up to solve
    round-off.  Install before constructing the optimizer that will own the
    amplitude leaf.
    """
    store = site.synapses
    if parametrize.is_parametrized(store, "w"):
        raise RuntimeError(
            f"{store.site!r}: amplitudes are already parametrized; "
            "refresh_whitened_basis() replaces an installed basis"
        )
    chol, live = _live_gram(site, ridge)
    basis = WhitenedAmplitudeBasis(store, chol, live)
    parametrize.register_parametrization(store, "w", basis)
    return basis


def refresh_whitened_basis(
    site: CSTLinear, *, ridge: float = 1e-2
) -> WhitenedAmplitudeBasis:
    """Re-fit an installed basis against the current *coordinate* geometry.

    Materialises the current ``w``, drops the stale factorisation, and
    installs a fresh one, so the represented map is preserved while the
    basis tracks coordinate drift.  Structural events remain unsupported
    end-to-end: once the store version has moved, ``c`` no longer has a
    consistent meaning and the pinned-version guard fires here too.
    Optimizer state attached to the old amplitude leaf is expressed in the
    old basis; resetting or transporting it is the caller's contract,
    exactly as with the optimizer-state follower on structural events.
    """
    store = site.synapses
    if not parametrize.is_parametrized(store, "w"):
        raise RuntimeError(
            f"{store.site!r}: no whitened basis installed; use "
            "install_whitened_basis()"
        )
    parametrize.remove_parametrizations(store, "w", leave_parametrized=True)
    return install_whitened_basis(site, ridge=ridge)
