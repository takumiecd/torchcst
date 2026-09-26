"""Compare both FP32 paths against the same FP64 dense/autograd oracle."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.atom_prefix_linear import atom_prefix_linear
from prototypes.benchmark_atom_prefix import error
from prototypes.benchmark_triton_linear import model


def evaluate(layer, x, grad, *, prefix):
    p = layer.atoms.p
    y = (
        atom_prefix_linear(layer, x, p, atom_chunk=64)
        if prefix
        else F.linear(x, layer.dense_weight())
    )
    dx, dp = torch.autograd.grad(y, (x, p), grad)
    return tuple(t.detach().double() for t in (y, dx, dp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for batch in (32, 128):
        torch.manual_seed(21)
        layer = model(512, rows=256, columns=512)
        x = torch.randn(batch, 512, device="cuda", requires_grad=True)
        grad = torch.randn(batch, 256, device="cuda")
        double_layer = copy.deepcopy(layer).double()
        double_x = x.detach().double().requires_grad_()
        reference = evaluate(double_layer, double_x, grad.double(), prefix=False)
        results.append({"batch": batch, "comparisons": {}})
        for label, selected_layer, selected_x, selected_grad, prefix in (
            ("dense_fp32", layer, x, grad, False),
            ("prefix_fp32", layer, x, grad, True),
            ("prefix_fp64", double_layer, double_x, grad.double(), True),
        ):
            actual = evaluate(selected_layer, selected_x, selected_grad, prefix=prefix)
            results[-1]["comparisons"][label] = {
                name: error(a, b)
                for name, a, b in zip(
                    ("output", "dx", "dp"), actual, reference, strict=True
                )
            }
            if prefix and selected_x.dtype == torch.float64:
                for a, b in zip(actual, reference, strict=True):
                    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
