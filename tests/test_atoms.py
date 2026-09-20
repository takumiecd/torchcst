import pytest
import torch
from torch import nn

from torchcst import Atoms


def test_atoms_owns_one_opaque_parameter_table() -> None:
    p = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    atoms = Atoms(p)

    p.zero_()

    assert atoms.count == 2
    assert atoms.parameter_dim == 3
    assert isinstance(atoms.p, nn.Parameter)
    assert torch.equal(atoms.p, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert not hasattr(atoms, "weight")


def test_atoms_state_dict_rejects_a_different_fixed_shape() -> None:
    source = Atoms(torch.ones(2, 3))
    target = Atoms(torch.ones(3, 3))

    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    ("p", "error_type", "message"),
    [
        (torch.ones(2), ValueError, "p must have shape"),
        (torch.ones(0, 3), ValueError, "at least one atom"),
        (torch.ones(2, 0), ValueError, "at least one coordinate"),
        (torch.ones(2, 3, dtype=torch.int64), TypeError, "floating-point"),
        (torch.tensor([[float("nan")]]), ValueError, "finite"),
    ],
)
def test_atoms_validates_its_fixed_shape(
    p: torch.Tensor, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        Atoms(p)
