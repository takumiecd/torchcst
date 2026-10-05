"""Backend-independent Plan selectors and versioned offline artifacts."""

from .base import Match, Selector
from .exact import ExactEntry, ExactSelector, load_selector, validate_selector_artifact
from .runtime import Dispatcher, ExecutionBinding
from .select import FixedSelector, OrderedSelector
from .validation import UnsupportedPlan, validate_plan_context

__all__ = [
    "Dispatcher",
    "ExactEntry",
    "ExactSelector",
    "ExecutionBinding",
    "FixedSelector",
    "Match",
    "OrderedSelector",
    "Selector",
    "UnsupportedPlan",
    "load_selector",
    "validate_plan_context",
    "validate_selector_artifact",
]
