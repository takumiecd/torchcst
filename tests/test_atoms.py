import pytest
import torch
from torch import nn

from torchcst import Atoms


def test_atoms_owns_copies_without_interpreting_p() -> None:
    weight = torch.tensor([1.0, -2.0])
    p = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    atoms = Atoms(weight, p)

    weight.zero_()
    p.zero_()

    assert atoms.count == 2
    assert atoms.parameter_dim == 3
    assert atoms.local_parameter_dim == 4
    assert isinstance(atoms.weight, nn.Parameter)
    assert isinstance(atoms.p, nn.Parameter)
    assert torch.equal(atoms.weight, torch.tensor([1.0, -2.0]))
    assert torch.equal(atoms.p, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_local_parameters_preserves_atom_correspondence() -> None:
    atoms = Atoms(
        torch.tensor([10.0, 20.0]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert torch.equal(
        atoms.local_parameters(),
        torch.tensor([[10.0, 1.0, 2.0], [20.0, 3.0, 4.0]]),
    )


def test_atoms_state_dict_rejects_a_different_fixed_shape() -> None:
    source = Atoms(torch.ones(2), torch.ones(2, 3))
    target = Atoms(torch.ones(3), torch.ones(3, 3))

    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    ("weight", "p", "message"),
    [
        (torch.ones(2, 1), torch.ones(2, 3), "weight must have shape"),
        (torch.ones(2), torch.ones(3, 3), "same number of atoms"),
        (torch.ones(2), torch.ones(2, 0), "at least one coordinate"),
    ],
)
def test_atoms_validates_its_two_fixed_shapes(
    weight: torch.Tensor, p: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Atoms(weight, p)
