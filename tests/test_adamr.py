import copy

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torchcst import (
    AdamWConfig,
    Amplitude,
    AmpWidth,
    Atoms,
    Chart,
    CSTAdamR,
    CSTLinear,
    CSTModule,
    CSTNormalizedAdam,
    Gaussian,
    Kernel,
    Separable,
)


def make_site(*, atoms=3, inputs=5, outputs=4, backend="factored"):
    return CSTLinear(
        Chart.linspace(inputs, low=-1.0, high=1.0),
        Chart.linspace(outputs, low=-1.0, high=1.0),
        atoms=atoms,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.4), output_profile=Gaussian(0.3))
        ),
        backend=backend,
        dtype=torch.float64,
    )


class ExpandingKernel(Kernel):
    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return 1

    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart,
        atoms: int,
        *,
        mode: str,
    ) -> Tensor:
        return torch.linspace(
            0.5,
            1.5,
            atoms,
            device=input_chart.coordinates.device,
            dtype=input_chart.coordinates.dtype,
        ).unsqueeze(-1)

    def materialize_atoms(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        shape = (p.shape[0], output_chart.features, input_chart.features)
        return p[:, :1, None].square().expand(shape)


def pairwise_energy(atoms: Tensor, *, kind: str) -> Tensor:
    if kind == "cosine":
        norms = torch.linalg.vector_norm(atoms, dim=(1, 2), keepdim=True)
        scale = torch.where(norms > 0, norms.reciprocal(), torch.zeros_like(norms))
        atoms = atoms * scale
    elif kind == "abs":
        atoms = atoms.abs()
    elif kind != "raw":
        raise ValueError("kind must be 'cosine', 'raw', or 'abs'")
    flat = atoms.flatten(1)
    gram = flat @ flat.transpose(-2, -1)
    return gram.sum() - gram.diagonal().sum()


@pytest.mark.parametrize("kind", ["cosine", "raw", "abs"])
@pytest.mark.parametrize("backend", ["factored", "materialized"])
def test_repulsion_terms_match_the_pairwise_identity(kind, backend):
    site = make_site(backend=backend)
    summed, kappa = site.repulsion_terms(kind=kind)

    assert summed.shape == (site.out_features, site.in_features)
    assert kappa.shape == ()
    torch.testing.assert_close(
        site.repulsion_energy(kind=kind),
        pairwise_energy(site.materialized_atoms(), kind=kind),
    )


def test_raw_sum_is_the_dense_weight():
    site = make_site()
    summed, kappa = site.repulsion_terms(kind="raw")

    torch.testing.assert_close(summed, site.dense_weight())
    torch.testing.assert_close(
        kappa, site.materialized_atoms().square().sum()
    )


def test_abs_sum_is_the_elementwise_absolute_atoms():
    site = make_site()
    atoms = site.materialized_atoms()
    summed, kappa = site.repulsion_terms(kind="abs")

    torch.testing.assert_close(summed, atoms.abs().sum(dim=0))
    torch.testing.assert_close(kappa, atoms.square().sum())
    torch.testing.assert_close(kappa, site.repulsion_terms(kind="raw")[1])


def test_one_atom_has_zero_repulsion_energy():
    site = make_site(atoms=1)

    torch.testing.assert_close(
        site.repulsion_energy(kind="cosine"),
        torch.zeros((), dtype=torch.float64),
    )
    torch.testing.assert_close(
        site.repulsion_energy(kind="raw"),
        torch.zeros((), dtype=torch.float64),
    )
    torch.testing.assert_close(
        site.repulsion_energy(kind="abs"),
        torch.zeros((), dtype=torch.float64),
    )


def test_identical_cosine_atoms_have_known_pair_energy():
    site = CSTLinear(
        Chart.linspace(3, low=-1.0, high=1.0),
        Chart.linspace(2, low=-1.0, high=1.0),
        atoms=3,
        kernel=ExpandingKernel(),
        dtype=torch.float64,
    )
    site.atoms.p.data.copy_(site.atoms.p.data[:1].expand_as(site.atoms.p))

    torch.testing.assert_close(
        site.repulsion_energy(kind="cosine"),
        torch.tensor(6.0, dtype=torch.float64),
    )


def test_zero_norm_cosine_atom_is_omitted():
    site = CSTLinear(
        Chart.linspace(3, low=-1.0, high=1.0),
        Chart.linspace(2, low=-1.0, high=1.0),
        atoms=2,
        kernel=ExpandingKernel(),
        dtype=torch.float64,
    )
    site.atoms.p.data[1] = 0

    torch.testing.assert_close(
        site.repulsion_energy(kind="cosine"),
        torch.zeros((), dtype=torch.float64),
    )


@pytest.mark.parametrize("kind", ["cosine", "raw", "abs"])
def test_repulsion_energy_gradients_match_the_pairwise_formula(kind):
    site = make_site()
    oracle = copy.deepcopy(site)

    site.repulsion_energy(kind=kind).backward()
    pairwise_energy(oracle.materialized_atoms(), kind=kind).backward()

    torch.testing.assert_close(site.atoms.p.grad, oracle.atoms.p.grad)


def test_amplitude_bandwidth_kernel_uses_the_same_oracle():
    site = CSTLinear(
        Chart.linspace(6, low=-1.0, high=1.0),
        Chart.linspace(4, low=-1.0, high=1.0),
        atoms=5,
        kernel=AmpWidth(
            sigma_min=0.10,
            sigma_max=1.0,
            tau=0.005,
            temperature=0.25,
        ),
        dtype=torch.float64,
    )

    for kind in ("cosine", "raw", "abs"):
        torch.testing.assert_close(
            site.repulsion_energy(kind=kind),
            pairwise_energy(site.materialized_atoms(), kind=kind),
        )


def test_opposite_sign_copies_keep_positive_abs_energy():
    site = CSTLinear(
        Chart.linspace(6, low=-1.0, high=1.0),
        Chart.linspace(4, low=-1.0, high=1.0),
        atoms=2,
        kernel=AmpWidth(
            sigma_min=0.10,
            sigma_max=1.0,
            tau=0.005,
            temperature=0.25,
        ),
        dtype=torch.float64,
    )
    site.atoms.p.data[1] = site.atoms.p.data[0]
    site.atoms.p.data[0, 0] = 0.4
    site.atoms.p.data[1, 0] = -0.4

    raw = site.repulsion_energy(kind="raw")
    absolute = site.repulsion_energy(kind="abs")
    assert raw < 0
    torch.testing.assert_close(absolute, -raw)
    torch.testing.assert_close(
        absolute,
        pairwise_energy(site.materialized_atoms(), kind="abs"),
    )


def test_linear_is_a_cst_module_site():
    site = make_site()

    assert isinstance(site, CSTModule)
    assert site.cst_charts() == (site.input_chart, site.output_chart)
    assert site.cst_parameters() == (site.atoms.p,)


class FakeSite(CSTModule):
    def __init__(self) -> None:
        super().__init__()
        self.atoms = Atoms(torch.tensor([[1.0], [-0.5]], dtype=torch.float64))
        self._charts = (Chart.linspace(2, low=-1.0, high=1.0),)

    def cst_parameters(self):
        return (self.atoms.p,)

    def cst_charts(self):
        return self._charts

    def repulsion_terms(self, *, kind="cosine"):
        atoms = self.atoms.p.unsqueeze(-1)
        if kind == "cosine":
            norms = torch.linalg.vector_norm(atoms, dim=(1, 2), keepdim=True)
            scale = torch.where(norms > 0, norms.reciprocal(), torch.zeros_like(norms))
            atoms = atoms * scale
        return atoms.sum(dim=0), atoms.square().sum()


def test_normalized_adamr_still_requires_linear_sites():
    with pytest.raises(ValueError, match="CSTLinear"):
        CSTAdamR(FakeSite(), repulsion=0.1)


def test_unknown_repulsion_kind_is_rejected():
    site = make_site()

    with pytest.raises(ValueError, match="kind must be"):
        site.repulsion_terms(kind="l2")


def _take_task_step(optimizer, model, inputs, targets) -> None:
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(model(inputs), targets).backward()
    optimizer.step()


def test_zero_repulsion_matches_normalized_adam():
    torch.manual_seed(7)
    site = make_site()
    reference = copy.deepcopy(site)
    opt = CSTAdamR(site, lr=0.01, repulsion=0.0, trust_radius=0.25)
    ref = CSTNormalizedAdam(reference, lr=0.01, trust_radius=0.25)
    inputs = torch.randn(8, site.in_features, dtype=torch.float64)
    targets = torch.randn(8, site.out_features, dtype=torch.float64)

    _take_task_step(opt, site, inputs, targets)
    _take_task_step(ref, reference, inputs, targets)

    torch.testing.assert_close(site.atoms.p, reference.atoms.p)
    torch.testing.assert_close(
        opt.state_dict()["cst"]["<root>"].numerator.m,
        ref.state_dict()["cst"]["<root>"].numerator.m,
    )


def test_decoupled_repulsion_leaves_normalized_moments_unchanged():
    torch.manual_seed(7)
    site = make_site()
    reference = copy.deepcopy(site)
    opt = CSTAdamR(site, lr=0.01, repulsion=0.1, kind="cosine", trust_radius=0.25)
    ref = CSTNormalizedAdam(reference, lr=0.01, trust_radius=0.25)
    inputs = torch.randn(8, site.in_features, dtype=torch.float64)
    targets = torch.randn(8, site.out_features, dtype=torch.float64)

    opt.zero_grad(set_to_none=True)
    ref.zero_grad(set_to_none=True)
    F.mse_loss(site(inputs), targets).backward()
    F.mse_loss(reference(inputs), targets).backward()
    with torch.enable_grad():
        repulsion_grad = torch.autograd.grad(
            site.repulsion_energy(kind="cosine"), site.atoms.p
        )[0].detach()
    opt.step()
    ref.step()

    torch.testing.assert_close(
        opt.state_dict()["cst"]["<root>"].numerator.m,
        ref.state_dict()["cst"]["<root>"].numerator.m,
    )
    torch.testing.assert_close(
        site.atoms.p,
        reference.atoms.p - 0.01 * 0.1 * repulsion_grad,
    )
    assert not torch.equal(site.atoms.p, reference.atoms.p)


def test_mixed_model_requires_dense_config_and_sums_site_energies():
    first = make_site(atoms=2, inputs=5, outputs=4)
    second = make_site(atoms=3, inputs=4, outputs=2)
    mixed = nn.Sequential(first, second, nn.Linear(2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="dense=None cannot own"):
        CSTAdamR(mixed, repulsion=0.2)

    opt = CSTAdamR(
        mixed,
        repulsion=0.2,
        kind="raw",
        dense=AdamWConfig(lr=0.002, weight_decay=0.0),
    )
    expected = first.repulsion_energy(kind="raw") + second.repulsion_energy(kind="raw")

    torch.testing.assert_close(opt.repulsion_energy(), expected)


def test_rejects_models_without_cst_sites():
    with pytest.raises(ValueError, match="does not contain a CSTLinear"):
        CSTAdamR(nn.Linear(3, 2), repulsion=0.1)
