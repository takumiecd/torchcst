"""Independent continuous Gaussian CST two-dimensional convolution."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Literal

from torchcst.representation import ContinuousKernel
from torchcst.storage import NeuronStore, SynapseStore

from .cst_map import _ContinuousCSTMap


def conv2d_neuron_coordinates(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    *,
    channel_scale: float = 1.0,
    spatial_scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Return Box-compatible coordinates for conv taps and output channels.

    Input rows follow PyTorch's ``(channel, kernel_y, kernel_x)`` flattening
    order and have three coordinates. Output rows have one channel coordinate.
    Both grids lie in ``[0, 1]`` so they pair directly with
    ``RepresentationSpec.continuous(3, 1)``.
    """
    if isinstance(in_channels, bool) or not isinstance(in_channels, int):
        raise TypeError("in_channels must be an int")
    if isinstance(out_channels, bool) or not isinstance(out_channels, int):
        raise TypeError("out_channels must be an int")
    if in_channels <= 0 or out_channels <= 0:
        raise ValueError("channel counts must be positive")
    size = CSTConv2d._positive_pair(kernel_size, "kernel_size")
    for value, name in (
        (channel_scale, "channel_scale"),
        (spatial_scale, "spatial_scale"),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    coordinate_dtype = dtype if dtype is not None else torch.get_default_dtype()
    if not coordinate_dtype.is_floating_point:
        raise TypeError("coordinate dtype must be floating point")

    def grid(count: int, scale: float) -> Tensor:
        if count == 1:
            return torch.full((1,), 0.5, device=device, dtype=coordinate_dtype)
        values = torch.linspace(0.0, 1.0, count, device=device, dtype=coordinate_dtype)
        return 0.5 + (values - 0.5) * float(scale)

    channels = grid(in_channels, channel_scale)
    ys = grid(size[0], spatial_scale)
    xs = grid(size[1], spatial_scale)
    cc, yy, xx = torch.meshgrid(channels, ys, xs, indexing="ij")
    input_mu = torch.stack((cc.reshape(-1), yy.reshape(-1), xx.reshape(-1)), dim=1)
    output_mu = grid(out_channels, channel_scale)[:, None]
    return input_mu, output_mu


class CSTConv2d(_ContinuousCSTMap):
    """Apply one continuous CST measure as a shared convolutional filter.

    The module directly owns its endpoint neuron charts, synapse measure, and
    Gaussian kernels. Its input chart has
    ``in_channels * kernel_height * kernel_width`` neurons and its output chart
    has one neuron per output channel. Spatial output positions share the same
    represented map, giving the translation-equivariant contract of
    :class:`torch.nn.Conv2d` without depending on :class:`CSTLinear`.

    Grouped convolution and non-zero padding modes are intentionally outside
    this first compute contract. Bias is a conventional per-output-channel
    parameter and is not part of the synapse store.
    """

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel: ContinuousKernel,
        in_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        kernel_out: ContinuousKernel | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        implementation: Literal["unfold", "materialized"] = "unfold",
        track_mass: bool = True,
    ) -> None:
        super().__init__(in_neurons, out_neurons, synapses, kernel, kernel_out, track_mass=track_mass)
        if isinstance(in_channels, bool) or not isinstance(in_channels, int):
            raise TypeError("in_channels must be an int")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if implementation not in ("unfold", "materialized"):
            raise ValueError("implementation must be 'unfold' or 'materialized'")
        self.kernel_size = self._positive_pair(kernel_size, "kernel_size")
        self.stride = self._positive_pair(stride, "stride")
        self.dilation = self._positive_pair(dilation, "dilation")
        self.padding = self._nonnegative_pair(padding, "padding")
        expected = in_channels * self.kernel_size[0] * self.kernel_size[1]
        if self.in_features != expected:
            raise ValueError(
                "in_neurons width must equal in_channels * kernel_size area"
            )

        self.in_channels = in_channels
        self.out_channels = self.out_features
        self.implementation = implementation
        self.bias = (
            nn.Parameter(synapses.w.new_zeros(self.out_channels)) if bias else None
        )

    @staticmethod
    def _positive_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
        pair = _pair(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
            raise TypeError(f"{name} must contain ints")
        if any(item <= 0 for item in pair):
            raise ValueError(f"{name} values must be positive")
        return pair

    @staticmethod
    def _nonnegative_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
        pair = _pair(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
            raise TypeError(f"{name} must contain ints")
        if any(item < 0 for item in pair):
            raise ValueError(f"{name} values must be non-negative")
        return pair

    def dense_weight(self) -> Tensor:
        """Materialize the effective represented filter.

        This includes neuron gates, so the returned kernel is exactly the one
        used by the materialized convolution path.
        """
        weight = super().dense_weight()
        in_gate = self.in_neurons.gate_vector().to(weight)
        out_gate = self.out_neurons.gate_vector().to(weight)
        weight = out_gate[:, None] * weight * in_gate[None, :]
        return weight.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"x must have shape (batch, {self.in_channels}, height, width)"
            )
        # Materializing the current CST filter avoids an enormous unfolded
        # activation in normal convolutional training while preserving
        # gradients to s, t, w, sigma, and gates. The capture contract needs
        # patch-level observations, so it deliberately retains the reference
        # unfold implementation whenever capture is enabled.
        if self.implementation == "materialized" and not self.capture_enabled:
            return F.conv2d(
                x,
                self.dense_weight(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
            )
        patches = F.unfold(
            x,
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        batch, _, locations = patches.shape
        patch_rows = patches.transpose(1, 2).reshape(-1, self.in_features)
        output = self._forward_rows(patch_rows)
        if self.bias is not None:
            output = output + self.bias
        height, width = self._output_shape(x.shape[-2], x.shape[-1])
        if height * width != locations:
            raise RuntimeError(
                "unfold output shape does not match convolution geometry"
            )
        return (
            output.reshape(batch, locations, self.out_channels)
            .transpose(1, 2)
            .reshape(batch, self.out_channels, height, width)
        )

    def _output_shape(self, height: int, width: int) -> tuple[int, int]:
        result = []
        for size, kernel, stride, padding, dilation in zip(
            (height, width),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        ):
            result.append(
                (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            )
        if any(value <= 0 for value in result):
            raise ValueError("convolution geometry produces an empty spatial output")
        return result[0], result[1]

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}, implementation={self.implementation!r}"
        )
