"""Deterministic amplitude-to-bandwidth controllers for CST atoms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..representation import MaturityGaussianFactor
from ..storage import SynapseStore

__all__ = ["InverseAmplitudeBandwidth"]


@dataclass
class _ControlledSite:
    name: str
    module: nn.Module
    store: SynapseStore
    slots: Tensor
    reference_amplitude: Tensor


def _pearson(left: Tensor, right: Tensor) -> float:
    left = left.float()
    right = right.float()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float((left * right).sum() / denominator)


class InverseAmplitudeBandwidth:
    """Keep each atom's bandwidth inversely tied to its amplitude.

    For the amplitude magnitude captured at construction, ``w_ref``, the
    controller applies the detached law

    ``sigma_k / sigma_0 = clip((|w_ref| + eps) / (|w_k| + eps), lo, hi)``.

    Every atom therefore starts at the ordinary factor bandwidth exactly,
    while weakening broadens it and strengthening narrows it.  The controller
    writes the existing :class:`MaturityGaussianFactor` atom column; it adds
    no trainable parameter and intentionally supplies no chain-rule gradient
    from bandwidth back to amplitude.

    This first contract is structure-frozen.  A changed live-slot set raises
    instead of silently assigning a newborn atom somebody else's reference.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        min_ratio: float = 0.5,
        max_ratio: float = 4.0,
        eps: float = 1e-12,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        for value, name in (
            (min_ratio, "min_ratio"),
            (max_ratio, "max_ratio"),
            (eps, "eps"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not float(min_ratio) < 1.0 < float(max_ratio):
            raise ValueError("bandwidth ratios must bracket the initial ratio 1")

        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.eps = float(eps)
        self._sites: list[_ControlledSite] = []
        claimed: set[int] = set()

        expected_min_scale = 1.0 / self.max_ratio
        expected_max_scale = 1.0 / self.min_ratio
        for name, module in model.named_modules():
            store = getattr(module, "synapses", None)
            factor_in = getattr(module, "factor_in", None)
            factor_out = getattr(module, "factor_out", None)
            if not isinstance(store, SynapseStore):
                continue
            if not isinstance(factor_in, MaturityGaussianFactor) or not isinstance(
                factor_out, MaturityGaussianFactor
            ):
                continue
            if id(store) in claimed:
                raise ValueError(
                    f"synapse store {store.site!r} is controlled more than once"
                )
            claimed.add(id(store))
            for factor, side in ((factor_in, "input"), (factor_out, "output")):
                if not math.isclose(
                    factor.min_scale, expected_min_scale, rel_tol=0.0, abs_tol=1e-12
                ) or not math.isclose(
                    factor.max_scale, expected_max_scale, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"{name!r} {side} maturity scale range must be "
                        f"[{expected_min_scale:g}, {expected_max_scale:g}] "
                        "for the requested bandwidth clip"
                    )
            maturity = getattr(store, "maturity", None)
            if not isinstance(maturity, nn.Parameter):
                raise TypeError(f"{store.site!r} has no maturity parameter")
            maturity.requires_grad_(False)
            slots = store.live_slots().to(store.w.device)
            self._sites.append(
                _ControlledSite(
                    name=name,
                    module=module,
                    store=store,
                    slots=slots.clone(),
                    reference_amplitude=store.w.detach()
                    .index_select(0, slots)
                    .abs()
                    .clone(),
                )
            )

        if not self._sites:
            raise ValueError("model contains no maturity-Gaussian CST sites")
        self.sync()

    @staticmethod
    def _maturity_for_ratio(factor: MaturityGaussianFactor, ratio: Tensor) -> Tensor:
        scale_sq = ratio.reciprocal().square()
        low = factor.min_scale**2
        span = factor.max_scale**2 - low
        fraction = ((scale_sq - low) / span).clamp(0.0, 1.0)
        # A finite maturity cannot attain sigmoid endpoints.  nextafter keeps
        # the physical bandwidth at the dtype-nearest lawful clip value.
        zero = torch.zeros((), device=fraction.device, dtype=fraction.dtype)
        one = torch.ones((), device=fraction.device, dtype=fraction.dtype)
        fraction = fraction.clamp(
            min=torch.nextafter(zero, one),
            max=torch.nextafter(one, zero),
        )
        return torch.logit(fraction)

    def _validate_slots(self, site: _ControlledSite) -> None:
        current = site.store.live_slots().to(site.slots.device)
        if not torch.equal(current, site.slots):
            raise RuntimeError(
                f"{site.store.site!r}: inverse bandwidth currently requires "
                "a fixed live atom set"
            )

    def ratios(self) -> tuple[Tensor, ...]:
        """Current effective bandwidth divided by its initial value."""
        values = []
        for site in self._sites:
            self._validate_slots(site)
            amplitude = site.store.w.detach().index_select(0, site.slots).abs()
            reference = site.reference_amplitude.to(amplitude)
            values.append(
                ((reference + self.eps) / (amplitude + self.eps)).clamp(
                    self.min_ratio, self.max_ratio
                )
            )
        return tuple(values)

    @torch.no_grad()
    def sync(self) -> None:
        """Apply the detached clipped law to every controlled atom."""
        for site, ratio in zip(self._sites, self.ratios()):
            factor = site.module.factor_in
            maturity = self._maturity_for_ratio(factor, ratio).reshape(-1, 1)
            site.store.maturity.index_copy_(0, site.slots, maturity)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "torchcst-inverse-amplitude-bandwidth-v1",
            "min_ratio": self.min_ratio,
            "max_ratio": self.max_ratio,
            "eps": self.eps,
            "sites": {
                site.name: {
                    "slots": site.slots.detach().cpu().clone(),
                    "reference_amplitude": (
                        site.reference_amplitude.detach().cpu().clone()
                    ),
                }
                for site in self._sites
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "torchcst-inverse-amplitude-bandwidth-v1":
            raise ValueError("unsupported inverse-bandwidth state")
        for name in ("min_ratio", "max_ratio", "eps"):
            if float(state[name]) != getattr(self, name):
                raise ValueError(f"inverse-bandwidth {name} mismatch")
        saved = state.get("sites")
        if not isinstance(saved, dict) or set(saved) != {
            site.name for site in self._sites
        }:
            raise ValueError("inverse-bandwidth site set mismatch")
        for site in self._sites:
            entry = saved[site.name]
            slots = entry["slots"].to(site.slots)
            reference = entry["reference_amplitude"].to(
                site.reference_amplitude
            )
            if slots.shape != site.slots.shape or not torch.equal(slots, site.slots):
                raise ValueError(f"{site.name!r} inverse-bandwidth slots mismatch")
            if reference.shape != site.reference_amplitude.shape:
                raise ValueError(
                    f"{site.name!r} inverse-bandwidth reference shape mismatch"
                )
            site.reference_amplitude.copy_(reference)
        self.sync()

    @torch.no_grad()
    def summary(self) -> dict[str, float | int]:
        ratios = torch.cat([value.flatten() for value in self.ratios()])
        amplitudes = torch.cat(
            [
                site.store.w.detach().index_select(0, site.slots).abs().flatten()
                for site in self._sites
            ]
        )
        tolerance = 8.0 * torch.finfo(ratios.dtype).eps
        quantiles = torch.quantile(
            ratios, ratios.new_tensor([0.1, 0.5, 0.9])
        )
        return {
            "atoms": ratios.numel(),
            "ratio_min": float(ratios.min()),
            "ratio_p10": float(quantiles[0]),
            "ratio_p50": float(quantiles[1]),
            "ratio_p90": float(quantiles[2]),
            "ratio_max": float(ratios.max()),
            "lower_clip_fraction": float(
                (ratios <= self.min_ratio + tolerance).float().mean()
            ),
            "upper_clip_fraction": float(
                (ratios >= self.max_ratio - tolerance).float().mean()
            ),
            "abs_w_ratio_correlation": _pearson(amplitudes, ratios),
        }
