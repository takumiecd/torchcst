"""Center initialization, projection and retraction for integrated grid profiles.

These operations are profile-independent geometry operations. Reuse precisely
the same center policy as the existing Polar separable kernel. Factor values
are evaluated by polar_cell_mass, not by the point-sampled evaluator.
"""

from torchcst._builtin_callbacks import seal_functions as _seal_functions

from .compact import (  # noqa: F401
    apply_parameter_update,
    initialize,
    project_gradient,
    transport_state,
)

# Optional prepared updates may specialize only these original definitions.

_seal_functions(globals())
