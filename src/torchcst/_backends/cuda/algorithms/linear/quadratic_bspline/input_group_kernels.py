"""Exact packed input-event keys, hash membership and bounded query chunks."""

import triton as tr
import triton.language as tl


@tr.jit
def members(
    Pos,
    Count,
    Keys,
    Hist,
    Bucket,
    A: tl.constexpr,
    CAP: tl.constexpr,
    BITS: tl.constexpr,
):
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    valid = (a < A) & (a >= tl.load(Count))
    key = tl.full((128,), 0, tl.uint64)
    for event in tl.static_range(4):
        point = tl.load(Pos + (6 + event) * A + a, valid, 0).to(tl.uint64)
        key |= point << (event * BITS)
    hashed = key ^ (key >> 30)
    hashed *= 0xBF58476D1CE4E5B9
    hashed ^= hashed >> 27
    hashed *= 0x94D049BB133111EB
    hashed ^= hashed >> 31
    bucket = (hashed & (CAP - 1)).to(tl.int32)
    pending = valid
    while tl.sum(pending.to(tl.int32), 0) > 0:
        # Predication avoids CAS traffic from completed lanes to a shared dummy.
        previous = tl.inline_asm_elementwise(
            "{ .reg .pred p; mov.b64 $0, -1; setp.ne.u32 p, $4, 0; "
            "@p atom.global.cas.b64 $0, [$1], $2, $3; }",
            constraints="=&l,l,l,l,r",
            args=[
                Keys + bucket,
                tl.full((128,), -1, tl.int64),
                key.to(tl.int64),
                pending.to(tl.int32),
            ],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        found = (previous == -1) | (previous == key.to(tl.int64))
        pending &= ~found
        bucket = tl.where(pending, (bucket + 1) & (CAP - 1), bucket)
    tl.store(Bucket + a, tl.where(valid, bucket, -1), a < A)
    tl.atomic_add(Hist + bucket, 1, valid, sem="relaxed")


@tr.jit
def place(Bucket, Hist, Ends, Cursor, IDs, Count, A: tl.constexpr):
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    valid = (a < A) & (a >= tl.load(Count))
    bucket = tl.load(Bucket + a, valid, 0)
    base = tl.load(Ends + bucket) - tl.load(Hist + bucket)
    rank = tl.atomic_add(Cursor + bucket, 1, valid, sem="relaxed")
    tl.store(IDs + base + rank, a, valid)


@tr.jit
def chunk_counts(Hist, Chunks, CAP: tl.constexpr, ATOMS: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    count = tl.load(Hist + i, i < CAP, 0)
    tl.store(Chunks + i, tl.cdiv(count, ATOMS), i < CAP)


@tr.jit
def chunk_map(Ends, Mapping, A: tl.constexpr, CAP: tl.constexpr, STEPS: tl.constexpr):
    base = tl.program_id(0) * 128
    total = tl.load(Ends + CAP - 1)
    chunk = base + tl.arange(0, 128)
    if base < total:
        valid = (chunk < A) & (chunk < total)
        low, high = tl.full((128,), 0, tl.int32), tl.full((128,), CAP, tl.int32)
        for _ in tl.static_range(STEPS):
            mid = (low + high) // 2
            end = tl.load(Ends + mid, valid & (mid < CAP), 2147483647)
            right = end <= chunk
            low = tl.where(right, mid + 1, low)
            high = tl.where(right, high, mid)
        tl.store(Mapping + chunk, low, valid)
