"""Fresh per-forward preparation and launches for window execution."""

import math

import torch
import triton as tr

from .._shared.common import _norm_options
from .kernels import (
    COMPILED_KERNELS,
    _hist,
    _packed_singletons,
    _persistent_atoms,
    _scatter,
    atlas_tables,
    narrow_tables,
)


class PackedWindowProvider:
    """CPU-orchestrated GPU provider; globalnorm, freshbins, rolling8 halo.

    Validated CUDA provider. The owner
    wrapper must serialize all windows and finish before the optimizer update.
    """

    def __init__(self, plan, window=512, enable_fp_fusion=True):
        if window < 32 or window % 32:
            raise ValueError("window must be a multiple of32 and at least32")
        if any(
            abs(o) + n * s > 16384
            for o, n, s in zip(plan.origin, plan.sizes, plan.spacing)
        ):
            raise ValueError(
                "window halo requires bounded chart coordinates; use validated broad model otherwise"
            )
        self.plan = plan
        self.window = window
        self.fp_fusion = enable_fp_fusion

    def prepare(self, p):
        from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip._shared.common import (
            _owners,
            ball_offsets,
            validate_norm_plan,
        )

        validate_norm_plan(self.plan)
        options = _norm_options(self.plan, 0.03, 3.25)
        options.pop("A")
        offsets = ball_offsets(self.plan, p.device)
        options["B"] = tr.next_power_of_2(len(offsets))
        options.update(
            BALL=True,
            COUNT=len(offsets),
            SAVED_FLAGS=True,
            TUPLE_GRADS=False,
            ATLAS_GEOMETRY=all(
                o * 4 == round(o * 4) and abs(o) + n * s <= 16384
                for o, n, s in zip(self.plan.origin, self.plan.sizes, self.plan.spacing)
            ),
        )
        atlas, counts = atlas_tables(p.device)
        narrow, narrow_counts = narrow_tables(p.device)
        groups = tr.cdiv(self.plan.sizes[0], 32)
        state = {
            "norm": p.new_empty(len(p)),
            "flags": p.new_empty(len(p), dtype=torch.uint8),
            "order": p.new_empty(len(p), dtype=torch.int32),
            "total": torch.zeros(1, dtype=torch.int32, device=p.device),
            "prefix": torch.zeros(groups + 1, dtype=torch.int32, device=p.device),
            "offsets": offsets,
            "atlas": atlas,
            "counts": counts,
            "narrow": narrow,
            "narrow_counts": narrow_counts,
            "options": options,
            "zero_halo": p.new_zeros((8, math.prod(self.plan.sizes[1:]))),
        }
        empty = p.new_empty(0)
        self._launch(p, state, 0, empty, None, state["zero_halo"], 0)
        # Normprepare's compactfallback IDs are finished; sameOrder can nowbe
        # overwritten with currentrowbucket permutation. NoSortedKeys tensor.
        keys = p.new_empty(len(p), dtype=torch.int32)
        hist = torch.zeros(groups, dtype=torch.int32, device=p.device)
        _owners[(tr.cdiv(len(p), 256),)](
            p,
            keys,
            A=len(p),
            O=self.plan.origin[0],
            S=self.plan.spacing[0],
            G=groups,
            B=256,
            enable_fp_fusion=False,
        )
        _hist[(tr.cdiv(len(p), 256),)](keys, hist, A=len(p), num_warps=4)
        torch.cumsum(hist, dim=0, dtype=torch.int32, out=state["prefix"][1:])
        cursor = state["prefix"][:-1].clone()
        _scatter[(tr.cdiv(len(p), 256),)](
            keys, cursor, state["order"], A=len(p), num_warps=4
        )
        return state

    def _launch(self, p, state, start, w, dp, halo, mode):
        args = (
            p,
            state["norm"],
            state["flags"],
            w,
            dp,
            state["order"],
            state["offsets"],
            state["atlas"],
            state["counts"],
            state["narrow"],
            state["narrow_counts"],
            state["total"],
        )
        opts = dict(
            state["options"],
            BACK=mode == 2,
            SORTED=True,
            RowStart=start,
            Halo=halo,
            MODE=mode,
            R=self.window,
            Prefix=state["prefix"],
            enable_fp_fusion=self.fp_fusion,
        )
        grid = tr.cdiv(len(p), 256) if mode == 0 else min(tr.cdiv(len(p), 256), 256)
        packed = _packed_singletons[(grid,)](*args, A=len(p), **opts, num_warps=4)
        fallback = _persistent_atoms[(min(len(p), 1024),)](*args, **opts, num_warps=1)
        COMPILED_KERNELS.setdefault(f"packed_mode{mode}", packed)
        COMPILED_KERNELS.setdefault(f"fallback_mode{mode}", fallback)

    def build(self, p, state, start, count, out):
        out.zero_()
        self._launch(p, state, start, out, None, state["zero_halo"], 1)

    def vjp(self, p, state, start, dw, previous8, dp):
        halo = state["zero_halo"] if previous8 is None else previous8
        self._launch(p, state, start, dw, dp, halo, 2)
