"""CPU precision of arbitrary real powers and integer repeated squaring."""

import argparse
import json
from pathlib import Path

import torch


def study():
    r = torch.linspace(0, 51.2, 200001, dtype=torch.float64)
    rows = []
    powers = [1, 2, 4, 8, 16, 32, 64, 100, 128, 10 / 0.13, 10 / 0.3]
    for power in powers:
        b = 10 / power
        truth = (1 + r / b) * torch.exp(-r / b)
        for dtype in [torch.float32, torch.float64]:
            rr = r.to(dtype)
            u = torch.exp(-rr / 10)
            profile = (1 + rr / b) * u.pow(power)
            repeated_error = None
            if isinstance(power, int) and power & (power - 1) == 0:
                z = u
                for _ in range(power.bit_length() - 1):
                    z = z * z
                repeated = (1 + rr / b) * z
                repeated_error = float((repeated.double() - truth).abs().max())
            u2 = torch.exp2(-(rr / 10) * 1.4426950408889634)
            profile2 = (1 + rr / b) * u2.pow(power)
            rows.append(
                {
                    "power": power,
                    "decay_length": b,
                    "dtype": str(dtype),
                    "pow_profile_max_abs": float(
                        (profile.double() - truth).abs().max()
                    ),
                    "repeated_square_profile_max_abs": repeated_error,
                    "base2_pow_profile_max_abs": float(
                        (profile2.double() - truth).abs().max()
                    ),
                }
            )
    return {
        "scope": "CPU rounded-line profile peak1; r0..51.2; FP64 direct-exponential control; no GPU or VJP claim",
        "points": len(r),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = study()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
