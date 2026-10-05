"""Backend-independent processing/state lifecycle; no operator-family hierarchy."""

from abc import ABC, abstractmethod

from torch import Tensor

from .state import AlgorithmState


class Algorithm(ABC):
    """Stateless processing; the caller owns each AlgorithmState.

    Concrete algorithms define operation inputs, preparation and execution.
    Registry instances never retain input tensors or mutable learning state.
    """

    def create_state(self, atom_state, **configuration):
        return AlgorithmState(atom_state, self, **configuration)

    def prepare(self, state):
        state.invalidate()
        state.mark_current()

    def run(self, state, **inputs):
        if not isinstance(state, AlgorithmState) or state.algorithm is not self:
            raise ValueError("state belongs to a different algorithm")
        if not state.is_current():
            self.prepare(state)
        if not state.is_current():
            raise RuntimeError("algorithm did not prepare the current AtomState layout")
        result = self._execute_state(state, **inputs)
        return (
            state.atom_state.protect_tensor(result)
            if isinstance(result, Tensor)
            else result
        )

    def _execute_state(self, state, **inputs):
        return self.execute(state=state, **inputs)

    @abstractmethod
    def execute(self, **inputs):
        """Execute this algorithm's declared operation."""
