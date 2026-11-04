"""G1 changes only three model columns of the qualified research signature."""

from benchmarks.training.full_step_graph import PreparedStep
from benchmarks.training.graph_signature_leaves import (
    GraphSignatureLeaves,
    LeavesUnavailable,
)

_INVALID_COLUMNS = (None, None, None)


class FlatPreparedStep(PreparedStep):
    """Share qualification, optimizer checks and all numerical code with G0."""

    def __init__(self, *args, **kwargs):
        self._use_flat = False
        self._leaves = None
        self._column_tokens = (object(), object(), object())
        super().__init__(*args, **kwargs)
        if self.invalidated:
            return
        try:
            leaves = GraphSignatureLeaves(self.model)
        except LeavesUnavailable as error:
            self.invalidated = True
            self.qualification_error = str(error)
            return
        # The complete recursive signature still checks the original expected
        # values here. Building a certificate never licenses a new baseline.
        if not self.eligible() or not leaves.check():
            self.invalidated = True
            self.qualification_error = "signature changed during flat preparation"
            return
        self._leaves = leaves
        self._use_flat = True
        self.signature = (
            self.signature[:3]
            + self._column_tokens[:2]
            + self.signature[5:7]
            + self._column_tokens[2:]
            + self.signature[8:]
        )
        if not self.eligible():
            self.invalidated = True
            self.qualification_error = "flat signature qualification refused cohort"

    def _model_signature(self):
        if not self._use_flat:
            return super()._model_signature()
        if self._leaves.model_ref() is not self.model or not self._leaves.check():
            return _INVALID_COLUMNS
        return self._column_tokens
