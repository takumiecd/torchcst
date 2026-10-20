"""SM90+ packed reductions, with per-element scalar fallback for every tail."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl


@tr.jit
def add4(ptr, value, mask):
    # The caller gives each thread four distinct logical elements. Packing order
    # is compiler-owned, so check addresses rather than assuming lane order.
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .pred vector, aligned, contiguous, live, scalar, p;
        .reg .b64 tmp;
        .reg .b32 masks;
        and.b64 tmp, $4, 15;
        setp.eq.u64 aligned, tmp, 0;
        add.u64 tmp, $4, 4;
        setp.eq.u64 contiguous, tmp, $5;
        and.pred vector, aligned, contiguous;
        add.u64 tmp, $4, 8;
        setp.eq.u64 contiguous, tmp, $6;
        and.pred vector, vector, contiguous;
        add.u64 tmp, $4, 12;
        setp.eq.u64 contiguous, tmp, $7;
        and.pred vector, vector, contiguous;
        and.b32 masks, $12, $13;
        and.b32 masks, masks, $14;
        and.b32 masks, masks, $15;
        setp.ne.u32 live, masks, 0;
        and.pred vector, vector, live;
        @vector red.relaxed.gpu.global.add.v4.f32 [$4], {$8, $9, $10, $11};
        not.pred scalar, vector;
        setp.ne.u32 p, $12, 0;
        and.pred p, p, scalar;
        @p red.relaxed.gpu.global.add.f32 [$4], $8;
        setp.ne.u32 p, $13, 0;
        and.pred p, p, scalar;
        @p red.relaxed.gpu.global.add.f32 [$5], $9;
        setp.ne.u32 p, $14, 0;
        and.pred p, p, scalar;
        @p red.relaxed.gpu.global.add.f32 [$6], $10;
        setp.ne.u32 p, $15, 0;
        and.pred p, p, scalar;
        @p red.relaxed.gpu.global.add.f32 [$7], $11;
        selp.u32 $0, 1, 0, vector;
        mov.u32 $1, $0;
        mov.u32 $2, $0;
        mov.u32 $3, $0;
        }""",
        constraints="=r,=r,=r,=r,l,l,l,l,f,f,f,f,r,r,r,r",
        args=[ptr.to(tl.uint64), value.to(tl.float32), mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=4,
    )
