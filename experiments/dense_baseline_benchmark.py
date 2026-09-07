"""Queued Dense/ordinary-CST Adam baselines for the implicit-update benchmark."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments import mnist_current_api as runner


def run(data, method, seed, steps=128):
    config = replace(runner.ExperimentConfig(), seed=seed)
    torch.manual_seed(seed)
    model = (
        torch.nn.Linear(784, 10, bias=False).cuda()
        if method == "dense_adam"
        else runner.build_model(config, torch.device("cuda"))
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, foreach=True
    )
    inputs, labels, test_inputs, test_labels = data
    permutation = torch.randperm(
        8192, generator=torch.Generator().manual_seed(seed + 1)
    ).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        offset = step * 128 % 8192
        indices = permutation[offset : offset + 128]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(inputs.index_select(0, indices)), labels.index_select(0, indices)
        )
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "method": method,
        "seed": seed,
        "steps": steps,
        "seconds": elapsed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "final": runner.evaluate(model, test_inputs, test_labels),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    data = tuple(
        x.cuda() for x in runner.load_mnist(args.data, train_size=8192, test_size=2000)
    )
    for method in ("dense_adam", "cst_adam"):
        run(data, method, 17, steps=8)
    results = []
    for seed in (17, 29, 43):
        for method in ("dense_adam", "cst_adam"):
            result = run(data, method, seed)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
