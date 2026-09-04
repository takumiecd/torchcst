import torch

from torchcst import Chart, Gaussian, Separable


def test_separable_kernel_alone_interprets_the_opaque_p_split() -> None:
    input_chart = Chart.linspace(3)
    output_chart = Chart.grid((2, 2))
    kernel = Separable(
        input_profile=Gaussian(0.4),
        output_profile=Gaussian(0.7),
    )
    p = kernel.initialize(input_chart, output_chart, 2, mode="balanced")

    phi_input, phi_output = kernel.factors(input_chart, output_chart, p)
    represented = kernel.materialize_atoms(input_chart, output_chart, p)

    assert kernel.parameter_dim(input_chart, output_chart) == 4
    assert p.shape == (2, 4)
    assert phi_input.shape == (3, 2)
    assert phi_output.shape == (4, 2)
    assert represented.shape == (2, 4, 3)
    torch.testing.assert_close(
        represented,
        torch.einsum("oa,ia->aoi", phi_output, phi_input),
    )

    unit_amplitude = p.detach().clone()
    unit_amplitude[:, 0] = 1
    unit_atoms = kernel.materialize_atoms(input_chart, output_chart, unit_amplitude)
    torch.testing.assert_close(
        represented,
        p[:, 0, None, None] * unit_atoms,
    )
    assert tuple(kernel.parameters()) == ()
