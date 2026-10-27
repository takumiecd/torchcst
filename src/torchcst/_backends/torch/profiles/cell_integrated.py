"""Center initialization, projection and retraction for integrated grid profiles.

These operations are profile-independent geometry operations. Reuse precisely
the same center policy as the existing Polar separable kernel. Factor values
are evaluated by polar_cell_mass, not by the point-sampled evaluator.
"""

from .compact import (  # noqa: F401
    apply_parameter_update,
    initialize,
    project_gradient,
    transport_state,
)
