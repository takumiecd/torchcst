"""Semantic proposal rates; coordinate layout is declared by parameterizations."""

import math
from collections.abc import Mapping

from torchcst.kernels.parameterizations import ParameterFieldSpec

REFERENCE = "_cst_field_lr_reference"


def rates(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("field_lrs must be a mapping of field names to absolute rates")
    result = {}
    for name, rate in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("field_lrs names must be nonempty strings")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0:
            raise ValueError(
                "field_lrs values must be finite nonnegative Python scalars"
            )
        result[name] = float(rate)
    return result


def configure_group(optimizer, group, *, default, default_lr):
    """Copy user mappings and record the scheduler anchor at a group boundary."""
    atom_ids = {id(site.atoms.p) for site in optimizer._sites}
    has_atoms = any(
        id(p[1] if isinstance(p, tuple) else p) in atom_ids for p in group["params"]
    )
    if "field_lrs" not in group and default and has_atoms:
        group["field_lrs"] = dict(default)
    if "field_lrs" in group:
        group["field_lrs"] = rates(group["field_lrs"])
        if group["field_lrs"]:
            group.setdefault(REFERENCE, group.get("lr", default_lr))


def layout(site):
    dimensions = tuple(c.geometry.spec.center_parameter_dim for c in site.cst_charts())
    declare = getattr(site.kernel.spec.parameterization, "optimizer_fields", None)
    if not callable(declare):
        raise ValueError("kernel parameterization does not declare optimizer fields")
    fields = declare(dimensions)
    if not isinstance(fields, tuple) or not fields:
        raise ValueError(
            "parameterization must declare a nonempty tuple of optimizer fields"
        )
    names, columns = set(), set()
    for field in fields:
        if not isinstance(field, ParameterFieldSpec):
            raise TypeError("optimizer fields must be ParameterFieldSpec declarations")
        if field.stop > site.atoms.p.shape[1]:
            raise ValueError("optimizer field extends past the Atom coordinate shape")
        if field.name in names or columns.intersection(range(field.start, field.stop)):
            raise ValueError(
                "optimizer fields must have unique names and disjoint blocks"
            )
        names.add(field.name)
        columns.update(range(field.start, field.stop))
    if max(columns) + 1 != site.atoms.p.shape[1]:
        raise ValueError(
            "optimizer field layout differs from the Atom coordinate shape"
        )
    return fields


def validate_groups(optimizer, groups):
    if not any("field_lrs" in group for group in groups):
        return
    sites = {id(site.atoms.p): site for site in optimizer._sites}
    defaults = getattr(optimizer, "defaults", {})
    for group in groups:
        configured = rates(group.get("field_lrs"))
        if not configured:
            continue
        managed = [sites[id(p)] for p in group["params"] if id(p) in sites]
        if not managed:
            raise ValueError("field_lrs requires a managed Atom group")
        reference = group.get(REFERENCE)
        if (
            type(reference) not in (int, float)
            or not math.isfinite(reference)
            or reference <= 0
        ):
            raise ValueError(
                "field_lrs requires a finite positive initial group lr anchor"
            )
        current = group.get("lr", defaults.get("lr"))
        if (
            type(current) not in (int, float)
            or not math.isfinite(current)
            or current < 0
        ):
            raise ValueError("field_lrs requires a finite nonnegative Python group lr")
        if group.get("weight_decay", defaults.get("weight_decay", 0)) != 0:
            raise ValueError("field_lrs requires raw weight_decay=0 for Atom groups")
        for rate in configured.values():
            if not math.isfinite(rate / reference) or not math.isfinite(
                (rate / reference) * current
            ):
                raise ValueError("effective field learning rate overflow")
        for site in managed:
            known = {field.name for field in layout(site)}
            unknown = configured.keys() - known
            if unknown:
                raise ValueError(
                    f"unknown field_lrs {sorted(unknown)}; declared fields: {sorted(known)}"
                )


def proposal_ratios(site, group):
    """Capture the recipe before Adam or user post-hooks can change settings."""
    configured = group.get("field_lrs")
    if not configured:
        return ()
    reference = group[REFERENCE]
    return tuple(
        (field.start, field.stop, configured[field.name] / reference)
        for field in layout(site)
        if field.name in configured and configured[field.name] != reference
    )


def apply(point, previous, ratios):
    # The caller runs under no_grad. Scale normalized proposals, not gradients;
    # retain standard moment accumulation and one step counter per Parameter.
    for first, last, ratio in ratios:
        old = previous[:, first:last]
        proposal = point[:, first:last]
        proposal.copy_(old + (proposal - old) * ratio)


def current_rates(optimizer):
    validate_groups(optimizer, optimizer.param_groups)
    return [
        {
            name: rate * (group["lr"] / group[REFERENCE])
            for name, rate in rates(group.get("field_lrs")).items()
        }
        for group in optimizer.param_groups
    ]


def manifest(optimizer, groups):
    if not any(group.get("field_lrs") for group in groups):
        return None
    sites = {id(site.atoms.p): site for site in optimizer._sites}
    names = {id(p): name for name, p in optimizer.model.named_parameters()}
    return {
        "version": 1,
        "groups": [
            [
                (
                    names[id(p)],
                    tuple((f.name, f.start, f.stop) for f in layout(sites[id(p)])),
                )
                for p in group["params"]
                if id(p) in sites
            ]
            if group.get("field_lrs")
            else None
            for group in groups
        ],
    }


def validate_checkpoint(optimizer, state_dict, prospective):
    validate_groups(optimizer, prospective)
    if state_dict.get("cst_fields") != manifest(optimizer, prospective):
        raise ValueError("CST optimizer field declarations differ from checkpoint")
