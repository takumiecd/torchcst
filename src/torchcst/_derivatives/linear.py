"""Matrix-free local derivatives of a CSTLinear represented weight."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

from torchcst.nn import CSTLinear

from .layout import ParameterLayout


class _RepresentedWeight(nn.Module):
    """Expose ``dense_weight`` as a functional-call-compatible forward."""

    def __init__(self, site: CSTLinear) -> None:
        super().__init__()
        self.site = site

    def forward(self) -> Tensor:
        return self.site.dense_weight()


class LinearDerivatives:
    """Evaluate the second-order local CST map without materializing ``J`` or ``H``."""

    def __init__(self, site: CSTLinear) -> None:
        if not isinstance(site, CSTLinear):
            raise TypeError("site must be a CSTLinear")
        if site.input_chart.trainable or site.output_chart.trainable:
            raise ValueError("the first derivative engine supports frozen charts only")

        self.site = site
        self.layout = ParameterLayout(site)
        self._functional = _RepresentedWeight(site)

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters explicitly owned by this CST site."""

        return self.layout.parameters

    @property
    def numel(self) -> int:
        return self.layout.numel

    def point(self) -> Tensor:
        """Return the current parameter point in local vector coordinates."""

        return self.layout.current()

    def represented(self, point: Tensor | None = None) -> Tensor:
        """Evaluate the represented matrix at a local parameter point."""

        point = self._point_or_current(point)
        replacements = {
            f"site.{name}": value for name, value in self.layout.unpack(point).items()
        }
        return functional_call(
            self._functional,
            replacements,
            (),
            tie_weights=True,
            strict=False,
        )

    def jvp(self, direction: Tensor, *, point: Tensor | None = None) -> Tensor:
        """Evaluate ``J(point) direction`` in represented-weight space."""

        point = self._point_or_current(point)
        self.layout.validate(direction, name="direction")
        return jvp(self.represented, (point,), (direction,))[1]

    def second(
        self,
        left: Tensor,
        right: Tensor,
        *,
        point: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the bilinear Hessian contraction ``H(point)[left, right]``."""

        point = self._point_or_current(point)
        self.layout.validate(left, name="left direction")
        self.layout.validate(right, name="right direction")

        def right_jvp(candidate: Tensor) -> Tensor:
            return jvp(self.represented, (candidate,), (right,))[1]

        return jvp(right_jvp, (point,), (left,))[1]

    def displacement(
        self, direction: Tensor, *, point: Tensor | None = None
    ) -> Tensor:
        """Evaluate ``J d + 1/2 H[d, d]`` at the supplied parameter point."""

        point = self._point_or_current(point)
        return self.jvp(direction, point=point) + 0.5 * self.second(
            direction, direction, point=point
        )

    def pushforward(
        self,
        vector: Tensor,
        *,
        at: Tensor | None = None,
        point: Tensor | None = None,
    ) -> Tensor:
        """Evaluate ``V(at) vector = J vector + H[at, vector]``."""

        point = self._point_or_current(point)
        self.layout.validate(vector, name="vector")
        if at is None:
            at = torch.zeros_like(point)
        else:
            self.layout.validate(at, name="linearization direction")
        return self.jvp(vector, point=point) + self.second(at, vector, point=point)

    def pullback(
        self,
        cotangent: Tensor,
        *,
        at: Tensor | None = None,
        point: Tensor | None = None,
    ) -> Tensor:
        """Evaluate ``V(at).T cotangent`` without constructing ``V``."""

        point = self._point_or_current(point)
        expected_shape = (self.site.out_features, self.site.in_features)
        if cotangent.shape != expected_shape:
            raise ValueError(f"cotangent must have shape {list(expected_shape)}")
        if cotangent.device != point.device or cotangent.dtype != point.dtype:
            raise ValueError("cotangent must match the CST site device and dtype")
        if at is None:
            at = torch.zeros_like(point)
        else:
            self.layout.validate(at, name="linearization direction")

        def local_displacement(direction: Tensor) -> Tensor:
            return self.displacement(direction, point=point)

        _, apply_pullback = vjp(local_displacement, at)
        return apply_pullback(cotangent)[0]

    def _point_or_current(self, point: Tensor | None) -> Tensor:
        if point is None:
            return self.point()
        self.layout.validate(point, name="point")
        return point
