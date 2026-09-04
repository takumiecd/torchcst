"""Model-level CST optimization.

The public optimizer is introduced after the derivative operator contract is
validated against dense autograd oracles.
"""

from .atom_grad import ImplicitLinearAtomGrad

__all__ = ["ImplicitLinearAtomGrad"]
