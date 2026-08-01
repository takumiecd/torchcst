"""Pure cSFW growth installs no same-event retention path."""

from torchcst.policy import cSFW


def test_csfw_has_no_prune_court_that_can_bypass_birth_immunity() -> None:
    built = cSFW(backfit=None, pool_size=8, multistart=1).build(None)
    assert built.birth is not None
    assert built.prune is None
    assert built.absorb is None
