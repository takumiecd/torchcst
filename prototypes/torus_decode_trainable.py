"""Single-launch intrinsic 4D torus decode with an analytic center gradient."""

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice


@tr.jit
def _decode(Center, Major, Minor, Out, A: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    arc = tl.load(Center + atom * S0, valid, 0.0)
    u = tl.load(Center + atom * S0 + S1, valid, 0.0)
    v = tl.load(Center + atom * S0 + 2 * S1, valid, 0.0)
    major = tl.load(Major)
    minor = tl.load(Minor)
    angle = tl.sqrt(u * u + v * v) / minor
    sinc = tl.where(angle == 0.0, 1.0, libdevice.sin(angle) / angle)
    radial = major + minor * libdevice.cos(angle)
    theta = arc / major
    tl.store(Out + atom * 4, radial * libdevice.cos(theta), valid)
    tl.store(Out + atom * 4 + 1, radial * libdevice.sin(theta), valid)
    tl.store(Out + atom * 4 + 2, sinc * u, valid)
    tl.store(Out + atom * 4 + 3, sinc * v, valid)


@tr.jit
def _decode_grad(Center, Major, Minor, GradOut, GradCenter,
                 A: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                 G0: tl.constexpr, G1: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    arc = tl.load(Center + atom * S0, valid, 0.0)
    u = tl.load(Center + atom * S0 + S1, valid, 0.0)
    v = tl.load(Center + atom * S0 + 2 * S1, valid, 0.0)
    gx = tl.load(GradOut + atom * G0, valid, 0.0)
    gy = tl.load(GradOut + atom * G0 + G1, valid, 0.0)
    gz = tl.load(GradOut + atom * G0 + 2 * G1, valid, 0.0)
    gw = tl.load(GradOut + atom * G0 + 3 * G1, valid, 0.0)
    major = tl.load(Major)
    minor = tl.load(Minor)
    radius2 = u * u + v * v
    angle = tl.sqrt(radius2) / minor
    sine = libdevice.sin(angle)
    cosine = libdevice.cos(angle)
    sinc = tl.where(angle == 0.0, 1.0, sine / angle)
    theta = arc / major
    circle_cos = libdevice.cos(theta)
    circle_sin = libdevice.sin(theta)
    radial = major + minor * cosine
    ga = radial * (-circle_sin * gx + circle_cos * gy) / major
    radial_grad = -sinc * (circle_cos * gx + circle_sin * gy) / minor
    angle2 = angle * angle
    series = (-1.0 / 3.0 + angle2 / 30.0 - angle2 * angle2 / 840.0) / (minor * minor)
    curvature = tl.where(angle < 1.0e-3, series, (cosine - sinc) / radius2)
    section_dot = u * gz + v * gw
    gu = u * radial_grad + sinc * gz + curvature * u * section_dot
    gv = v * radial_grad + sinc * gw + curvature * v * section_dot
    tl.store(GradCenter + atom * 3, ga, valid)
    tl.store(GradCenter + atom * 3 + 1, gu, valid)
    tl.store(GradCenter + atom * 3 + 2, gv, valid)


class _TrainableDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, center, major, minor):
        count = center.shape[0]
        decoded = torch.empty((count, 4), device=center.device, dtype=center.dtype)
        ctx.save_for_backward(center, major, minor)
        if count:
            _decode[(tr.cdiv(count, 256),)](
                center, major, minor, decoded, count, *center.stride(), 256,
                num_warps=4, enable_fp_fusion=False,
            )
        return decoded

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        center, major, minor = ctx.saved_tensors
        count = center.shape[0]
        grad_center = torch.empty((count, 3), device=center.device, dtype=center.dtype)
        if count:
            _decode_grad[(tr.cdiv(count, 256),)](
                center, major, minor, grad_out, grad_center, count,
                *center.stride(), *grad_out.stride(), 256,
                num_warps=4, enable_fp_fusion=False,
            )
        return grad_center, None, None


def trainable_decode_intrinsic_torus(geometry, center):
    if (
        geometry.representation != "intrinsic"
        or center.ndim != 2
        or center.shape[1] != 3
        or center.device.type != "cuda"
        or center.dtype != torch.float32
    ):
        raise ValueError("trainable decoder requires intrinsic 4D CUDA FP32 centers")
    return _TrainableDecode.apply(center, geometry.major_radius, geometry.minor_radius)
