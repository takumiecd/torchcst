"""Actual CUDA gates; CPU collection never counts skips as qualification."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA"
)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("strategy", ["direct", "shared"])
@pytest.mark.parametrize(
    "cached, table, grouped, reuse, moment_policy",
    [
        (False, False, False, "none", "all"),
        (True, False, False, "none", "all"),
        (True, True, False, "none", "all"),
        (True, True, True, "none", "all"),
        (True, True, True, "h", "all"),
        (True, True, True, "all", "all"),
        (True, True, True, "chunk", "all"),
        (True, True, True, "none", "demand"),
        (True, True, True, "h", "hierarchical"),
    ],
)
def test_centered_support_dynamic_width_input_and_all_gradients(
    modes,
    strategy,
    cached,
    table,
    grouped,
    reuse,
    moment_policy,
    sample_cache=0,
    sort_atoms=False,
    local_blocks=0,
    local_overlap=False,
):
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
        LocalFourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        spec = FourierLinearSpec(256, modes)
        recipe_type = LocalFourierRecipe if local_blocks else FourierRecipe
        recipe = recipe_type(
            strategy=strategy,
            atom_chunk=256,
            minimum_sites=32 if modes == 4 else 64,
            native_cache=cached,
            table_coefficients=table,
            atom_block=4 if grouped else 1,
            batch_block=4 if grouped else 1,
            reuse=reuse,
            moment_policy=moment_policy,
            sample_cache=sample_cache,
            sort_atoms=sort_atoms,
            **(
                {"local_blocks": local_blocks, "local_overlap": local_overlap}
                if local_blocks
                else {}
            ),
        )
        p = torch.tensor(
            [
                [0.4, 12.713, 10.137, 0.0],
                [0.0, 3.25, 11.3, -4.0],
                [-0.2, -30.0, 17.0, -4.0],
                [0.7, -0.02, 25.31, -6.0],
                [0.5, 12.011, 12.713, math.log((6.5 - 0.1) / (10 - 6.5))],
                # Only the output axis is nonempty and near the floor. Keep
                # the other axis in its core, including the width pullback.
                [
                    0.3,
                    -0.999 * (0.1375 if modes == 4 else 0.25),
                    12.713,
                    math.log(0.005 / 9.895),
                ],
            ],
            device="cuda",
            requires_grad=True,
        )
        x = torch.randn(3, 256, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        for step in range(2):
            if step:
                with torch.no_grad():
                    p[:, 3].add_(0.13)
                    if sort_atoms:
                        p[0, 3].add_(5.0)
                    p[:, 1:3].add_(0.013)
                    x.add_(0.007)
            y = apply(x, p, spec, recipe)
            dx, dp = torch.autograd.grad(y, (x, p), dy, retain_graph=True)
            repeated = torch.autograd.grad(y, (x, p), dy)
            target, tx, tp = independent_gate(x, p, dy, spec)
            for a, b in ((y, target), (dx, tx), (dp, tp), *zip((dx, dp), repeated)):
                torch.testing.assert_close(a.double(), b.double(), atol=4e-4, rtol=4e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("reuse", ["none", "h", "all", "chunk"])
@pytest.mark.parametrize("sort_atoms", [False, True])
def test_saved_forward_parameter_version_is_checked(reuse, sort_atoms):
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        p = torch.tensor([[0.5, 3.0, 2.0, -3.0]], device="cuda", requires_grad=True)
        x = torch.randn(2, 64, device="cuda", requires_grad=True)
        y = apply(
            x, p, FourierLinearSpec(64), FourierRecipe(strategy="direct", reuse=reuse)
        )
        with torch.no_grad():
            p.add_(0.001)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("sample_cache", [128, 512])
def test_streamed_sample_cache_matches_independent_oracle(modes, sample_cache):
    test_centered_support_dynamic_width_input_and_all_gradients(
        modes, "direct", True, True, True, "h", "all", sample_cache
    )


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("minimum_sharing", [1, 8, 1000])
def test_live_sharing_routes_counts_and_all_gradients(modes, minimum_sharing):
    from benchmarks.cuda.linear.fourier_sharing import routing_reference
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        _routes,
        apply,
        tables,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    spec = FourierLinearSpec(256, modes)
    recipe = FourierRecipe(
        atom_chunk=256,
        minimum_sites=32,
        width_cutoff=0.1,
        native_cache=True,
        table_coefficients=True,
        atom_block=4,
        batch_block=4,
        reuse="h",
        sample_cache=512,
        moment_policy="hierarchical",
        minimum_sharing=minimum_sharing,
    )
    gen = torch.Generator().manual_seed(413)
    p = torch.randn(64, 4, generator=gen)
    p[:, 1:3] = torch.rand(64, 2, generator=gen) * 25.5
    p[:, 3] = torch.linspace(-6, 5, 64)
    p[0, 0] = 0
    p[1, 1:3] = -300
    p = p.cuda().requires_grad_()
    x = torch.randn(5, 256, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    tab = tables(spec, recipe, p.device)
    for step in range(2):
        if step:
            with torch.no_grad():
                p[:, 3].add_(0.37)
                p[:, 1:3].add_(0.137)
                x.add_(0.003)
        expected = routing_reference(p, spec, recipe)[1]
        actual = _routes(p, spec, recipe, tab)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        y = apply(x, p, spec, recipe)
        dx, dp = torch.autograd.grad(y, (x, p), dy)
        target, tx, tp = independent_gate(x, p, dy, spec)
        for a, b in ((y, target), (dx, tx), (dp, tp)):
            torch.testing.assert_close(a.double(), b.double(), atol=4e-4, rtol=4e-4)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("minimum_sharing", [0, 8, 1000])
def test_boundary_cache_reuses_only_partial_component_values(
    modes, minimum_sharing, coefficient_norm=False, sample_cache=512, norm_warps=4
):
    from dataclasses import replace

    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    spec = FourierLinearSpec(256, modes)
    base = FourierRecipe(
        atom_chunk=256,
        minimum_sites=32,
        width_cutoff=0.1,
        native_cache=True,
        table_coefficients=True,
        atom_block=4,
        batch_block=4,
        reuse="h",
        sample_cache=sample_cache,
        moment_policy="hierarchical",
        minimum_sharing=minimum_sharing,
    )
    gen = torch.Generator().manual_seed(414)
    p = torch.randn(64, 4, generator=gen)
    p[:, 1:3] = torch.rand(64, 2, generator=gen) * 25.5
    p[:, 3] = torch.linspace(-6, 5, 64)
    p[0, 0] = 0
    p[1, 1:3] = -300
    p[2, 1:3] = -0.99 * (0.1375 if modes == 4 else 0.25)
    p[2, 3] = math.log(0.005 / 9.895)
    p[-1, 3] = math.log((6.35 - 0.1) / (10 - 6.35))
    p[-2, 3] = math.log((9.4 - 0.1) / (10 - 9.4))
    p = p.cuda().requires_grad_()
    x = torch.randn(5, 256, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    for step in range(2):
        if step:
            with torch.no_grad():
                p[:, 3].add_(0.37)
                p[:, 1:3].add_(0.137)
        truth, tx, tp = independent_gate(x, p, dy, spec)
        for recipe in (
            base,
            replace(
                base,
                boundary_cache=True,
                coefficient_norm=coefficient_norm,
                coefficient_norm_warps=norm_warps,
            ),
        ):
            y = apply(x, p, spec, recipe)
            dx, dp = torch.autograd.grad(y, (x, p), dy)
            for a, b in ((y, truth), (dx, tx), (dp, tp)):
                torch.testing.assert_close(a.double(), b.double(), atol=4e-4, rtol=4e-4)


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("minimum_sharing", [0, 8])
@pytest.mark.parametrize("sample_cache", [128, 512])
@pytest.mark.parametrize("norm_warps", [1, 4])
def test_coefficient_norm_preserves_discrete_l2_and_all_gradients(
    modes, minimum_sharing, sample_cache, norm_warps
):
    test_boundary_cache_reuses_only_partial_component_values(
        modes,
        minimum_sharing,
        coefficient_norm=True,
        sample_cache=sample_cache,
        norm_warps=norm_warps,
    )


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("strategy", ["direct", "shared"])
def test_width_permutation_preserves_input_and_all_atom_gradients(modes, strategy):
    test_centered_support_dynamic_width_input_and_all_gradients(
        modes,
        strategy,
        True,
        True,
        True,
        "h",
        "hierarchical",
        sample_cache=512,
        sort_atoms=True,
    )


@pytest.mark.parametrize("modes", [4, 8])
def test_sorted_graph_replays_a_changed_width_permutation(modes):
    from benchmarks.cuda.linear.hierarchical_fourier import independent_gate
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
        FourierLinearSpec,
        FourierRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        apply,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    spec = FourierLinearSpec(256, modes)
    recipe = FourierRecipe(
        sort_atoms=True,
        atom_chunk=256,
        minimum_sites=32,
        width_cutoff=0.1,
        native_cache=True,
        table_coefficients=True,
        atom_block=4,
        batch_block=4,
        reuse="h",
        sample_cache=512,
        boundary_cache=True,
        coefficient_norm=True,
        coefficient_norm_warps=1,
        moment_policy="hierarchical",
    )
    gen = torch.Generator().manual_seed(415)
    p = torch.randn(16, 4, generator=gen)
    p[:, 1:3] = torch.rand(16, 2, generator=gen) * 25.5
    p[:, 3] = torch.linspace(-6, 5, 16).roll(5)
    p = p.cuda().requires_grad_()
    x = torch.randn(3, 256, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            p.grad = None
            x.grad = None
            (apply(x, p, spec, recipe) * dy).sum().backward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        p.grad = None
        x.grad = None
        y = apply(x, p, spec, recipe)
        (y * dy).sum().backward()
    with torch.no_grad():
        p[:, 3].copy_(p[:, 3].flip(0))
        p[:, 1:3].add_(0.137)
        x.add_(0.017)
    graph.replay()
    torch.cuda.synchronize()
    truth, tx, tp = independent_gate(x, p, dy, spec)
    for a, b in ((y, truth), (x.grad, tx), (p.grad, tp)):
        torch.testing.assert_close(a.double(), b.double(), atol=4e-4, rtol=4e-4)
