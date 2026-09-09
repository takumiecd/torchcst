"""Experimental parameter-square EMA; no public optimizer API changes.

alpha_diagonal squares a diagonally recompressed instantaneous gradient, then
maps its denominator back to the original parameter covector coordinates.
It is not the square of the smoothed first moment.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from torchcst.optim._local_tangent import solve
from torchcst.optim.atom_grad import AtomGradRequest
from torchcst.optim.moments import MomentSystem
from torchcst.optim.moments.atom_rms import AtomBlockMetric
from torchcst.optim.moments.base import ExpandedSecondMoment, SecondMomentComponent
from torchcst.optim.moments.tangent import TangentFirstMoment


@dataclass(frozen=True)
class ParameterSquareState:
    value: torch.Tensor
    beta_power: float


class ParameterSquareEMA(SecondMomentComponent):
    def __init__(self, beta, eps, mode, damping):
        self.beta, self.eps, self.mode, self.damping = beta, eps, mode, damping

    @property
    def observation_request(self):
        return AtomGradRequest(jg=True)

    def initialize(self, context):
        return ParameterSquareState(torch.zeros_like(context.current_point), 1.0)

    def expand(self, state, observation, context, *, next_step):
        observation.require(self.observation_request)
        if self.mode == "raw":
            scale = torch.ones_like(observation.jg)
        else:
            scale = (
                context.geometry.prepared(context.current_point)
                .gram_blocks()
                .diagonal(dim1=-2, dim2=-1)
            )
            scale = scale.clamp_min(0) + self.damping
        incoming = observation.jg / scale
        value = self.beta * state.value + (1 - self.beta) * incoming.square()
        power = self.beta**next_step
        denominator = scale * ((value / (1 - power)).sqrt() + self.eps)
        return ExpandedSecondMoment(
            AtomBlockMetric(
                torch.diag_embed(denominator), context.geometry.visible_shape
            ),
            ParameterSquareState(value.detach(), power),
        )

    def compress(self, expanded, accepted_displacement, context):
        return expanded.pending_state


def optimizer_class(base, mode):
    class ParameterAdam(base):
        def _make_moments(self):
            c = self.cst_config
            if mode == "transported_block":
                from experiments.transported_parameter_rms import (
                    TransportedParameterRMS,
                )

                second = TransportedParameterRMS(
                    c.betas[1], eps=c.eps, rtol=c.tangent_rtol
                )
            else:
                second = ParameterSquareEMA(
                    c.betas[1], c.eps, mode, c.first_moment_damping
                )
            return MomentSystem(
                first=TangentFirstMoment(c.betas[0], damping=c.first_moment_damping),
                second=second,
            )

        def _solve(self, context, expanded):
            self.solve_start.record()
            if mode == "transported_block":
                blocks = expanded.second.metric.blocks / self.cst_config.lr
                problem = SimpleNamespace(
                    linear=expanded.first.corrected.constant,
                    operator=SimpleNamespace(blocks=lambda: blocks),
                )
                result = solve(
                    problem,
                    approximation="atom_block",
                    radius=self.cst_config.trust_radius,
                    rtol=self.cst_config.update_rtol,
                )
                self.solve_end.record()
                return result
            diagonal = (
                expanded.second.metric.blocks.diagonal(dim1=-2, dim2=-1).double()
                / self.cst_config.lr
            )
            problem = SimpleNamespace(
                linear=expanded.first.corrected.constant,
                operator=SimpleNamespace(diagonal=lambda: diagonal),
            )
            result = solve(
                problem,
                approximation="diagonal",
                radius=self.cst_config.trust_radius,
                rtol=self.cst_config.update_rtol,
            )
            self.solve_end.record()
            return result

    return ParameterAdam
