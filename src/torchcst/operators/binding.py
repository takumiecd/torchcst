"""Non-owning views of the state held by a Module."""

from dataclasses import dataclass

from torch import Tensor

from torchcst.atoms import Atoms
from torchcst.geometry import Chart
from torchcst.kernels.state import KernelState

from .spec import ChartPairSpec, OperatorSpec, SingleChartSpec


@dataclass(frozen=True, kw_only=True, eq=False)
class Operator:
    """Bind current chart/kernel Modules and an Atoms owner without registration.

    This is not an nn.Module and never clones Parameter/buffer state. A view
    observes in-place updates and replacement of atoms.p during Module.to().
    declaration() is an explicit configuration snapshot, not a forward lookup.
    """

    charts: tuple[Chart, ...]
    kernel: KernelState
    atoms: Atoms

    def __post_init__(self):
        from torchcst._backends.torch.kernels import execution as _kernel

        if (
            not isinstance(self.charts, tuple)
            or len(self.charts) not in (1, 2)
            or not all(isinstance(c, Chart) for c in self.charts)
        ):
            raise TypeError("charts must be a tuple containing one or two Charts")
        if not isinstance(self.kernel, KernelState) or not isinstance(
            self.atoms, Atoms
        ):
            raise TypeError("kernel and atoms must implement their Module contracts")
        if len(self.charts) == 1 and len(getattr(self.charts[0], "shape", ())) != 2:
            raise ValueError("a single chart must describe an [out, in] operator")
        if tuple(self.kernel.parameters()):
            raise ValueError("a Kernel cannot own trainable state; put it in atom p")
        if _kernel.parameter_dim(self.kernel, *self.charts) != self.atoms.parameter_dim:
            raise ValueError("atom parameter width differs from the kernel contract")

    @property
    def shape(self):
        if len(self.charts) == 1:
            return self.charts[0].shape
        return (self.charts[1].features, self.charts[0].features)

    @property
    def in_features(self):
        return self.shape[1]

    @property
    def out_features(self):
        return self.shape[0]

    @property
    def p(self):
        return self.atoms.p

    def declaration(self) -> OperatorSpec:
        """Snapshot settings explicitly; scalar/point reads may synchronize."""
        charts = tuple(c.declaration() for c in self.charts)
        layout = (
            SingleChartSpec(chart=charts[0])
            if len(charts) == 1
            else ChartPairSpec(input_chart=charts[0], output_chart=charts[1])
        )
        return OperatorSpec(layout=layout, kernel=self.kernel.declaration())

    def _parameters(self, p):
        p = self.p if p is None else p
        if not isinstance(p, Tensor) or p.ndim != 2:
            raise ValueError("p must be an [atoms, parameters] Tensor")
        if p.shape[1] != self.atoms.parameter_dim:
            raise ValueError("atom parameter width differs from the bound operator")
        return p

    def materialize_atoms(self, p=None):
        from torchcst._backends.torch.operators.linear import materialize_atoms

        return materialize_atoms(self, self._parameters(p))

    def weight(self, p=None):
        from torchcst._backends.torch.operators.linear import weight

        return weight(self, self._parameters(p))

    def factors(self, p=None):
        from torchcst._backends.torch.operators.linear import factors

        return factors(self, self._parameters(p))

    def apply(self, inputs, p=None, *, algorithm="materialized"):
        """Execute a Torch reference algorithm using live state and autograd."""
        from torchcst._backends.torch.operators.linear import apply

        return apply(self, inputs, self._parameters(p), algorithm=algorithm)
