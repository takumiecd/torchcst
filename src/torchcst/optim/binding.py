"""Connect Torch's lazy parameter-state lookup to atom-owned dictionaries."""

from collections import defaultdict

import torch

from torchcst.atoms import OptimizerFieldSpec


class _AtomStateMap(defaultdict):
    def __init__(self, source):
        super().__init__(getattr(source, "default_factory", dict), source)
        self.owners = {}

    def __missing__(self, parameter):
        owner = self.owners.get(parameter)
        if owner is None:
            return super().__missing__(parameter)
        self[parameter] = owner.fields
        return owner.fields


def field_specs(optimizer, vector_keys, overrides=None):
    """Exact supported optimizer schemas; no Tensor-shape inference."""
    schemas = {
        torch.optim.SGD: ("momentum_buffer",),
        torch.optim.Adam: ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        torch.optim.AdamW: ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        torch.optim.Adamax: ("exp_avg", "exp_inf"),
        torch.optim.RMSprop: ("square_avg", "momentum_buffer", "grad_avg"),
        torch.optim.Adagrad: ("sum",),
        torch.optim.Adadelta: ("square_avg", "acc_delta"),
        torch.optim.Rprop: ("prev", "step_size"),
    }
    keys = schemas.get(type(optimizer), vector_keys)
    specs = {key: OptimizerFieldSpec(0) for key in keys}
    specs["step"] = OptimizerFieldSpec()
    if overrides is not None:
        specs.update(overrides)
    if any(
        not isinstance(k, str) or not k or not isinstance(v, OptimizerFieldSpec)
        for k, v in specs.items()
    ):
        raise TypeError("state_specs must contain named OptimizerFieldSpecs")
    return specs


def validate_binding(optimizer, sites, specs, *, groups=None):
    points = {
        p
        for group in (optimizer.param_groups if groups is None else groups)
        for p in group["params"]
    }
    owned = [
        (site.atoms.p, site.atom_state.optimizer_state)
        for site in sites
        if site.atoms.p in points
    ]
    contract = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    # Validate all contracts before changing any connection.
    for _, owner in owned:
        previous = owner._updater() if owner._updater is not None else None
        if previous is not None and previous is not optimizer:
            raise ValueError("AtomState already has an optimizer updater")
        if owner.contract is not None and (
            owner.contract != contract or owner.field_specs != specs
        ):
            raise ValueError("atom optimizer state contract differs")
    return contract, owned


def bind_atom_states(optimizer, sites, specs, *, restored=False):
    contract, owned = validate_binding(optimizer, sites, specs)
    state = optimizer.state
    if not isinstance(state, _AtomStateMap):
        state = _AtomStateMap(state)
    for point, owner in owned:
        owner.bind(contract, specs, updater=optimizer)
        existing = state.get(point)
        if restored or not owner.fields:
            owner._adopt(existing if existing is not None else {})
        state.owners[point] = owner
        # Preserve lazy initialization: empty owned cells do not create entries.
        if point in state or owner.fields:
            state[point] = owner.fields
    optimizer.state = state
