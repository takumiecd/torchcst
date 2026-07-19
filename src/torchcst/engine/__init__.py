from .plan import MutationPlan, SiteBatch

__all__ = ["CSTEngine", "MutationPlan", "SiteBatch"]


def __getattr__(name: str):
    if name == "CSTEngine":
        from .engine import CSTEngine

        return CSTEngine
    raise AttributeError(name)
