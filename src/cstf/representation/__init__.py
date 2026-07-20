"""座標domainとrepresentation仕様の公開面。"""

from .domains import CoordinateDomain, IntegerGrid, ParameterRole, Role, Sphere
from .spec import RepresentationSpec

__all__ = [
    "CoordinateDomain",
    "IntegerGrid",
    "ParameterRole",
    "RepresentationSpec",
    "Role",
    "Sphere",
]
