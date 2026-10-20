"""Independent bank CTAs and scratch-zero CTAs in one launch; no grid barriers."""

import triton as tr
import triton.language as tl

from . import bundled_bank_kernels as bk
from . import local_kernels as lk


@tr.jit
def bank_and_zero(
    X,
    Bank,
    N: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    LOW: tl.constexpr,
    LEVELS: tl.constexpr,
    SS: tl.constexpr,
    BB: tl.constexpr,
    Active,
    ML: tl.constexpr,
    DIM: tl.constexpr,
    ACTIVE: tl.constexpr,
    PAD: tl.constexpr,
    MIN_LOG: tl.constexpr,
    TB: tl.constexpr,
    MASK_MODE: tl.constexpr,
    Delta,
    Border,
    GRID0: tl.constexpr,
    GRID1: tl.constexpr,
    DELTA_ELEMENTS: tl.constexpr,
    BORDER_ELEMENTS: tl.constexpr,
    ZERO_BLOCK: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    bank_programs: tl.constexpr = GRID0 * GRID1
    if pid < bank_programs:
        # Preserve the original x-fast ordering of the two-dimensional grid.
        pid0, pid1 = pid % GRID0, pid // GRID0
        if TB > 1:
            bk.bank(
                X,
                Bank,
                N,
                B,
                POINTS,
                LOW,
                LEVELS,
                SS,
                BB,
                Active,
                ML,
                DIM,
                ACTIVE,
                PAD,
                MIN_LOG,
                TB,
                MASK_MODE,
                PID0=pid0,
                PID1=pid1,
            )
        else:
            lk.bank(
                X,
                Bank,
                N,
                B,
                POINTS,
                LOW,
                LEVELS,
                SS,
                BB,
                Active,
                ML,
                DIM,
                ACTIVE,
                PAD,
                MIN_LOG,
                MASK_MODE,
                PID0=pid0,
                PID1=pid1,
            )
    else:
        j = (pid - bank_programs) * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
        if BORDER_ELEMENTS:
            target = tl.where(
                j < DELTA_ELEMENTS, Delta + j, Border + j - DELTA_ELEMENTS
            )
        else:
            target = Delta + j
        tl.store(target, 0, j < DELTA_ELEMENTS + BORDER_ELEMENTS)
