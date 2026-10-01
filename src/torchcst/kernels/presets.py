"""Fixed mathematical kernel declarations used by public modules."""

from torchcst.kernels.normalization import NormalizationSpec
from torchcst.kernels.parameterizations.log_width import LogWidthSpec
from torchcst.kernels.profiles import TriweightSpec
from torchcst.kernels.spec import KernelSpec, ProfileBinding, StatePolicySpec

NORMALIZED_RADIAL_TRIWEIGHT = KernelSpec(
    composition="radial",
    profiles=(
        ProfileBinding(
            profile=TriweightSpec(),
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=1e-6
            ),
        ),
    ),
    parameterization=LogWidthSpec(sigma_min=0.03, sigma_max=3.25),
    initialization=StatePolicySpec(id="provided_atoms"),
    update=StatePolicySpec(id="euclidean"),
)
