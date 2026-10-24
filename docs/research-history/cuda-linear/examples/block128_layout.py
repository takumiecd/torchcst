"""Payload layout for the phased, single-team CUDA design example.

This does not build live atom geometry, launch CUDA or measure GPU memory.
Run from any directory with Python 3.10+. Counts describe the documented
N128/M8 bank envelope; live node pruning and multiple teams are separate plans.
"""

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class View:
    name: str
    offset: int
    elements: int
    element_bytes: int
    first_phase: int
    end_phase: int

    @property
    def end(self):
        return self.offset + self.elements * self.element_bytes


def layout(stage, *, node_count=28, c_count=12):
    """Return disjoint allocations with lifetime-checked Z/C aliases.

    Phase 0 loads sources, 1 builds moments, 2 contracts all atoms,
    3 writes C, 4 expands C and stores the owned output tile.
    The block barrier between phases 2 and 3 is mandatory.
    """
    if stage not in ("forward", "backward_param", "backward_dx"):
        raise ValueError("unknown stage")
    if type(node_count) is not int or not 0 <= node_count <= 28:
        raise ValueError("node_count must be an integer in [0,28]")
    if type(c_count) is not int or not 0 <= c_count <= 12:
        raise ValueError("c_count must be an integer in [0,12]")
    # Sources retain their FP32 input dtype. Moments/accumulations are FP64.
    batch, sites, atoms, basis = 4, 128, 128, 17
    cursor, views = 0, []

    def allocate(name, elements, size, begin, end):
        nonlocal cursor
        cursor = (cursor + 15) // 16 * 16
        view = View(name, cursor, elements, size, begin, end)
        cursor = view.end
        views.append(view)
        return view

    if stage != "backward_dx":
        allocate("X", batch * sites, 4, 0, 3)
    if stage != "forward":
        allocate("dY", batch * sites, 4, 0, 3)

    # Reserve enough space for either view; C need not fit in the actual Z
    # payload when node_count is small. This also covers a direct-only case.
    arena_count = max(node_count, c_count) * batch * basis
    first = allocate(
        "Z_X" if stage != "backward_dx" else "Z_dY", node_count * batch * basis, 8, 1, 3
    )
    cursor = first.offset + arena_count * 8
    if stage == "backward_param":
        allocate("Z_dY", node_count * batch * basis, 8, 1, 3)
    allocate("H_value" if stage == "forward" else "G_value", batch * atoms, 8, 2, 5)
    views.append(
        View(
            "C_output" if stage == "forward" else "C_input",
            first.offset,
            c_count * batch * basis,
            8,
            3,
            5,
        )
    )
    total = (cursor + 15) // 16 * 16

    for v in views:
        assert v.offset % 16 == 0 and v.end <= total
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            address_overlap = max(a.offset, b.offset) < min(a.end, b.end)
            lifetime_overlap = max(a.first_phase, b.first_phase) < min(
                a.end_phase, b.end_phase
            )
            assert not (address_overlap and lifetime_overlap), (a, b)
    return {
        "stage": stage,
        "shared_payload_bytes": total,
        "views": [asdict(v) for v in views],
    }


def check_thread_coverage():
    """Check unique owners, including a partial batch, for the shown mapping."""
    for batch in (1, 3, 4):
        for count in (batch * 128, batch * 17, batch * 12 * 17, batch * 32):
            assigned = [q for tid in range(128) for q in range(tid, count, 128)]
            assert sorted(assigned) == list(range(count))
    for batch_groups in (1, 8):
        writers = [
            (bg, atom)
            for bg in range(batch_groups)
            for input_tile in range(4)
            if input_tile == 0
            for atom in range(128)
        ]
        assert len(writers) == len(set(writers)) == batch_groups * 128


if __name__ == "__main__":
    check_thread_coverage()
    for stage in ("forward", "backward_param", "backward_dx"):
        # Exercise both all-bank capacity and a direct-only payload.
        layout(stage, node_count=0, c_count=0)
        layout(stage, node_count=0, c_count=12)
    print(
        json.dumps(
            [layout(stage) for stage in ("forward", "backward_param", "backward_dx")],
            indent=2,
        )
    )
