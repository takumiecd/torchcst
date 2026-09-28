"""Screen a stable station-local Triweight polynomial before a GPU kernel.

This synthetic geometry probe covers the radii of 1024- and 8192-square
BlockStripLinear layers. It does not benchmark training or replace validation
against prepared, trained atoms.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def prepared_geometry(size):
    from prototypes.block_strip_linear import BlockStripLinear
    from torchcst.nn._backends._preparation import execution_plan

    layer = BlockStripLinear((size, size), (64, 64), 1)
    plan = execution_plan(layer.strip)
    return plan.circle.numpy(), plan.section.numpy()


def sample_radius(size, trials, tile_rows, rng, geometry=None):
    groups = size // 64
    radius = max(2, groups * groups) * 4.1 / (2 * math.pi)
    max_abs_f = 0.0
    max_abs_w = 0.0
    bad_support = 0
    noncontiguous = 0
    tested_columns = 0
    active_pairs = 0
    max_poly_direct = 0.0
    max_rounded_weight = 0.0
    max_rounded_poly_weight = 0.0
    max_rounded_poly_weight_f64 = 0.0
    rounded_support_disagreements = 0
    for _ in range(trials):
        if geometry is None:
            station_angle = rng.uniform(-math.pi, math.pi)
            row_offset = np.arange(64, dtype=np.float64) * 0.1 / radius
            angles = station_angle + row_offset
            circle = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(
                np.float32
            )
            section = rng.uniform(-0.25, 0.25, size=(64, 3)).astype(np.float32)
            section[:, 0] += np.float32(radius)
        else:
            all_circle, section = geometry
            station = rng.integers(0, len(all_circle) // 64)
            circle = all_circle[station * 64 : (station + 1) * 64]
            angles = np.unwrap(np.arctan2(circle[:, 1], circle[:, 0]))
            row_offset = angles - angles[0]
            station_angle = float(angles[0])
        center_row = rng.integers(0, 64)
        center_col = rng.integers(0, 64)
        center_xy = circle[center_row] * section[center_col, 0]
        center = np.array(
            (
                center_xy[0],
                center_xy[1],
                section[center_col, 1],
                section[center_col, 2],
            ),
            dtype=np.float32,
        )
        center += rng.uniform(-0.12, 0.12, size=4).astype(np.float32)
        precision = np.float32(rng.uniform(2.0, 5.0))
        amplitude = np.float32(rng.uniform(-0.1, 0.1))
        site_xy = circle[:, None, :] * section[None, :, :1]
        delta_xy = site_xy - center[None, None, :2]
        delta_zw = section[None, :, 1:] - center[None, None, 2:]
        squared = np.sum(delta_xy * delta_xy, axis=-1) + np.sum(
            delta_zw * delta_zw, axis=-1
        )
        direct_f = np.float32(1) - precision * squared
        direct_w = amplitude * np.maximum(direct_f, np.float32(0)) ** 3

        # Anchor each row tile to its actually rounded FP32 site coordinates.
        # The resulting cubic has 20 monomials in u, v, and h=u²+v².
        for row_start in range(0, 64, tile_rows):
            xy = site_xy[row_start : row_start + tile_rows]
            dx0 = xy[0, :, 0] - center[0]
            dy0 = xy[0, :, 1] - center[1]
            u = xy[:, :, 0] - xy[0:1, :, 0]
            v = xy[:, :, 1] - xy[0:1, :, 1]
            h = u * u + v * v
            d0 = (
                dx0 * dx0
                + dy0 * dy0
                + (section[:, 1] - center[2]) ** 2
                + (section[:, 2] - center[3]) ** 2
            )
            h0 = np.float32(1) - precision * d0
            au = -np.float32(2) * precision * dx0
            bv = -np.float32(2) * precision * dy0
            ch = -precision
            local_f = h0[None, :] + au[None, :] * u + bv[None, :] * v + ch * h
            target = direct_w[row_start : row_start + tile_rows]
            local_w = amplitude * np.maximum(local_f, np.float32(0)) ** 3
            max_rounded_weight = max(
                max_rounded_weight, float(np.max(np.abs(local_w - target)))
            )
            rounded_support_disagreements += int(
                np.count_nonzero(
                    (local_f > 0) != (direct_f[row_start : row_start + tile_rows] > 0)
                )
            )
            expanded = np.zeros_like(local_f)
            expanded64 = np.zeros_like(local_f, dtype=np.float64)
            for i in range(4):
                for j in range(4 - i):
                    for k in range(4 - i - j):
                        l = 3 - i - j - k
                        scale = math.factorial(3) // (
                            math.factorial(i)
                            * math.factorial(j)
                            * math.factorial(k)
                            * math.factorial(l)
                        )
                        expanded += (
                            np.float32(scale)
                            * h0[None, :] ** l
                            * au[None, :] ** i
                            * bv[None, :] ** j
                            * ch**k
                            * u**i
                            * v**j
                            * h**k
                        )
                        expanded64 += (
                            float(scale)
                            * h0[None, :].astype(np.float64) ** l
                            * au[None, :].astype(np.float64) ** i
                            * bv[None, :].astype(np.float64) ** j
                            * float(ch) ** k
                            * u.astype(np.float64) ** i
                            * v.astype(np.float64) ** j
                            * h.astype(np.float64) ** k
                        )
            expanded_w = amplitude * np.maximum(expanded, np.float32(0))
            max_rounded_poly_weight = max(
                max_rounded_poly_weight,
                float(np.max(np.abs(expanded_w - target))),
            )
            expanded_w64 = float(amplitude) * np.maximum(expanded64, 0)
            max_rounded_poly_weight_f64 = max(
                max_rounded_poly_weight_f64,
                float(np.max(np.abs(expanded_w64 - target))),
            )

        c0, s0 = (
            np.float32(math.cos(station_angle)),
            np.float32(math.sin(station_angle)),
        )
        cr = center[0] * c0 + center[1] * s0
        ct = -center[0] * s0 + center[1] * c0
        theta = row_offset.astype(np.float32)
        q = np.float32(2 * radius * radius) * np.sin(theta / 2) ** 2
        s = np.float32(radius) * np.sin(theta)
        rho = section[:, 0]
        h = np.float32(1) - precision * (
            (rho - cr) ** 2
            + ct**2
            + (section[:, 1] - center[2]) ** 2
            + (section[:, 2] - center[3]) ** 2
        )
        a = -np.float32(2) * precision * rho * cr / np.float32(radius * radius)
        b = np.float32(2) * precision * rho * ct / np.float32(radius)
        factored_f = h[None, :] + a[None, :] * q[:, None] + b[None, :] * s[:, None]
        factored_w = amplitude * np.maximum(factored_f, np.float32(0)) ** 3
        max_abs_f = max(max_abs_f, float(np.max(np.abs(direct_f - factored_f))))
        max_abs_w = max(max_abs_w, float(np.max(np.abs(direct_w - factored_w))))
        bad_support += int(np.count_nonzero((direct_f > 0) != (factored_f > 0)))
        active_pairs += int(np.count_nonzero(direct_f > 0))
        for col in range(64):
            indices = np.flatnonzero(direct_f[:, col] > 0)
            if len(indices) and indices[-1] - indices[0] + 1 != len(indices):
                noncontiguous += 1
            tested_columns += 1
        polynomial = np.zeros_like(factored_f)
        for i in range(4):
            for j in range(4 - i):
                k = 3 - i - j
                scale = math.factorial(3) // (
                    math.factorial(i) * math.factorial(j) * math.factorial(k)
                )
                polynomial += (
                    np.float32(scale)
                    * h[None, :] ** k
                    * a[None, :] ** i
                    * b[None, :] ** j
                    * q[:, None] ** i
                    * s[:, None] ** j
                )
        active = direct_f > 0
        if np.any(active):
            max_poly_direct = max(
                max_poly_direct,
                float(np.max(np.abs(polynomial[active] - direct_f[active] ** 3))),
            )
    return {
        "size": size,
        "radius": radius,
        "trials": trials,
        "tile_rows": tile_rows,
        "active_pairs": active_pairs,
        "max_abs_f": max_abs_f,
        "max_abs_w": max_abs_w,
        "support_disagreements": bad_support,
        "noncontiguous_columns": noncontiguous,
        "tested_columns": tested_columns,
        "max_poly_vs_factored_cube_active": max_poly_direct,
        "max_rounded_weight": max_rounded_weight,
        "max_rounded_poly_weight": max_rounded_poly_weight,
        "max_rounded_poly_weight_f64": max_rounded_poly_weight_f64,
        "rounded_support_disagreements": rounded_support_disagreements,
    }


def aggregate_radius(size, rng, atoms=205, repeats=10, geometry=None, tile_rows=4):
    groups = size // 64
    radius = max(2, groups * groups) * 4.1 / (2 * math.pi)
    max_weight_error = 0.0
    max_output_error = 0.0
    rows = max(16, tile_rows)
    for _ in range(repeats):
        if geometry is None:
            theta0 = rng.uniform(-math.pi, math.pi)
            angles = theta0 + np.arange(rows, dtype=np.float64) * 0.1 / radius
            circle = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(
                np.float32
            )
            section = rng.uniform(-0.25, 0.25, size=(64, 3)).astype(np.float32)
            section[:, 0] += np.float32(radius)
        else:
            all_circle, section = geometry
            station = rng.integers(0, len(all_circle) // 64)
            circle = all_circle[station * 64 : station * 64 + rows]
        xy = circle[:, None, :] * section[None, :, :1]
        direct = np.zeros((rows, 64), dtype=np.float32)
        expanded_sum = np.zeros_like(direct)
        for _ in range(atoms):
            row = rng.integers(0, rows)
            col = rng.integers(0, 64)
            center = np.array(
                (xy[row, col, 0], xy[row, col, 1], section[col, 1], section[col, 2]),
                dtype=np.float32,
            )
            center += rng.uniform(-0.12, 0.12, size=4).astype(np.float32)
            precision = np.float32(rng.uniform(2.0, 5.0))
            amp = np.float32(rng.uniform(-0.1, 0.1))
            delta = xy - center[None, None, :2]
            cross = section[None, :, 1:] - center[None, None, 2:]
            f = np.float32(1) - precision * (
                np.sum(delta * delta, axis=-1) + np.sum(cross * cross, axis=-1)
            )
            direct += amp * np.maximum(f, np.float32(0)) ** 3
            for row_start in range(0, rows, tile_rows):
                local = xy[row_start : row_start + tile_rows]
                dx = local[0, :, 0] - center[0]
                dy = local[0, :, 1] - center[1]
                u = local[:, :, 0] - local[0:1, :, 0]
                v = local[:, :, 1] - local[0:1, :, 1]
                h = u * u + v * v
                h0 = np.float32(1) - precision * (
                    dx * dx
                    + dy * dy
                    + (section[:, 1] - center[2]) ** 2
                    + (section[:, 2] - center[3]) ** 2
                )
                au, bv, ch = -2 * precision * dx, -2 * precision * dy, -precision
                poly = np.zeros((tile_rows, 64), dtype=np.float32)
                for i in range(4):
                    for j in range(4 - i):
                        for k in range(4 - i - j):
                            l = 3 - i - j - k
                            scale = math.factorial(3) // (
                                math.factorial(i)
                                * math.factorial(j)
                                * math.factorial(k)
                                * math.factorial(l)
                            )
                            poly += (
                                np.float32(scale)
                                * h0[None, :] ** l
                                * au[None, :] ** i
                                * bv[None, :] ** j
                                * ch**k
                                * u**i
                                * v**j
                                * h**k
                            )
                expanded_sum[row_start : row_start + tile_rows] += amp * np.maximum(
                    poly, np.float32(0)
                )
        max_weight_error = max(
            max_weight_error, float(np.max(np.abs(direct - expanded_sum)))
        )
        x = rng.standard_normal((128, 64)).astype(np.float32)
        max_output_error = max(
            max_output_error,
            float(np.max(np.abs(x @ direct.T - x @ expanded_sum.T))),
        )
    return {
        "size": size,
        "atoms_per_tile": atoms,
        "repeats": repeats,
        "rows": rows,
        "tile_rows": tile_rows,
        "max_weight_error": max_weight_error,
        "max_output_error_m128": max_output_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tile-rows", type=int, choices=(4, 8, 16, 64), default=16)
    parser.add_argument("--real-geometry", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(21)
    geometries = {
        n: prepared_geometry(n) if args.real_geometry else None for n in (1024, 8192)
    }
    result = {
        "cases": [
            sample_radius(n, args.trials, args.tile_rows, rng, geometries[n])
            for n in (1024, 8192)
        ],
        "aggregate": [
            aggregate_radius(n, rng, geometry=geometries[n], tile_rows=args.tile_rows)
            for n in (1024, 8192)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
