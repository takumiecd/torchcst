"""CUDA preparation only: group atom indices and build varying-period Z.

There is deliberately no Linear/forward/backward facade here. H/C need the
new profile contract; using the old radius-floor profile would change the
meaning of the newly defined width. See cell-mass-kernel-flow.ja.md.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from .plan import AxisPlan, HierarchyPlan


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    sources = [directory / "bindings.cpp", directory / "preparation.cu"]
    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in sources)
    ).hexdigest()[:12]
    return load(
        name=f"torchcst_cell_mass_hierarchy_{digest}",
        sources=[str(path) for path in sources],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


@dataclass(frozen=True)
class PreparedLayout:
    """Forward-scoped cache: indices/ranges only, never packed atom values.

    groups[g] = (output_level, input_level, output_owner, input_owner).
    rows[offsets[g]:offsets[g+1]] lists original atom rows in that group.
    Key/sort/count scratch is released before this object is returned.
    """

    rows: torch.Tensor
    groups: torch.Tensor
    offsets: torch.Tensor
    plan: HierarchyPlan
    _source: torch.Tensor = field(repr=False)
    _source_version: int = field(repr=False)

    def validate_for(self, atoms, plan):
        if (
            atoms is not self._source
            or atoms._version != self._source_version
            or plan != self.plan
        ):
            raise RuntimeError("atom values or plan changed; rebuild the layout cache")

    @property
    def tensor_bytes(self):
        """Retained cache payload, not an allocator peak or whole-step budget."""
        return sum(
            t.numel() * t.element_size() for t in (self.rows, self.groups, self.offsets)
        )


def prepare_layout(plan, atoms):
    """Classify, stable-sort indices, and retain sparse nonempty group ranges.

    Values remain in the original Parameter. A library sort is the initial
    implementation, not a permanent requirement to materialize all atom keys.
    Width/finite rejection reads one error word. unique_consecutive also has a
    dynamic output shape, so this version does not claim CUDA Graph support.
    """
    if type(plan) is not HierarchyPlan:
        raise TypeError("plan must be HierarchyPlan")
    if atoms.device.type != "cuda" or atoms.dtype not in (torch.float32, torch.float64):
        raise ValueError("atoms must be CUDA float32/float64")
    if atoms.ndim != 2 or atoms.shape[1] != 4 or not atoms.is_contiguous():
        raise ValueError("atoms must be contiguous [atom_count,4]")
    if atoms.shape[0] > 65536:
        raise ValueError("preparation supports at most 65536 atoms")

    native = _extension()
    input_config = plan.input.native_config()
    output_config = plan.output.native_config()
    keys, errors = native.classify_atoms(atoms, input_config, output_config)
    error_bits = errors.item()
    if error_bits & 1:
        raise ValueError("all atom fields must be finite")
    if error_bits & 2:
        raise ValueError("sigma must lie in (0, min(input period,output period))")

    sorted_keys, rows = torch.sort(keys, stable=True)
    group_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    offsets = torch.empty(counts.numel() + 1, dtype=torch.int64, device=atoms.device)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    groups = native.decode_groups(group_keys, input_config, output_config)
    return PreparedLayout(rows, groups, offsets, plan, atoms, atoms._version)


def build_hierarchy(signal, axis, *, method="merge", team_size=4):
    """Return double Z[batch,node,1+2*M] for the banks declared by axis.

    One team cooperates on a single (node,batch,mode) sum. The merge route
    directly creates leaves, odd parent harmonics and the full-chart bank;
    matching even harmonics are merged level by level. The direct route is a
    comparison that generates every element from signal independently.

    This low-level stage has no autograd binding. It is for a future explicit
    forward/backward implementation, not a differentiable public operator.
    """
    if type(axis) is not AxisPlan:
        raise TypeError("axis must be AxisPlan")
    if method not in ("direct", "merge"):
        raise ValueError("method must be direct or merge")
    if type(team_size) is not int or team_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError("team_size must be 1,2,4,8,16 or 32")
    if signal.device.type != "cuda" or signal.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("signal must be CUDA float32/float64")
    if (
        signal.ndim != 2
        or signal.shape[1] != axis.site_count
        or not signal.is_contiguous()
    ):
        raise ValueError("signal must be contiguous [batch,site_count]")
    return _extension().build_hierarchy(
        signal, axis.native_config(), method == "merge", team_size
    )
