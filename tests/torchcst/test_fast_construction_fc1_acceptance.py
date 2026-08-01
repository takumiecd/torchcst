"""FC-1 oracle acceptance at the preregistered MNIST tensor scale."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import TangentStatisticsRequest
from torchcst.policy import (
    EvenBudgetDistributor,
    PeriodicCadence,
    ProfitCourt,
    QuotaRegime,
    cSFW,
    cVP,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseRefit, SynapseStore

_SIBLING = Path(__file__).resolve().parents[3] / "cst"
_ORACLE_PATH = _SIBLING / "scripts" / "run_fastcon_fc1.py"
_MNIST_RAW = _SIBLING / "data" / "MNIST" / "raw"


def _load_oracle() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fastcon_fc1_oracle", _ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FC-1 oracle at {_ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _idx(path: Path, *, offset: int, width: int) -> torch.Tensor:
    payload = bytearray(path.read_bytes())
    values = torch.frombuffer(payload, dtype=torch.uint8, offset=offset)
    return values.reshape(-1, width) if width > 1 else values


@pytest.fixture(scope="module")
def fc1_problem() -> tuple[ModuleType, object]:
    """Build exactly FC-1's seed-777 12k subset without torchvision."""
    if not _ORACLE_PATH.is_file() or not _MNIST_RAW.is_dir():
        pytest.skip("FC-1 sibling oracle or its preregistered MNIST data is absent")
    fc1 = _load_oracle()
    cfg = fc1.Config()
    images = _idx(
        _MNIST_RAW / "train-images-idx3-ubyte", offset=16, width=784
    ).to(torch.float64)
    labels = _idx(
        _MNIST_RAW / "train-labels-idx1-ubyte", offset=8, width=1
    ).to(torch.int64)
    generator = torch.Generator().manual_seed(777)
    selected = torch.randperm(images.shape[0], generator=generator)[: cfg.n_train]
    x = images.index_select(0, selected) / 255.0
    y = labels.index_select(0, selected)
    mu_xy = fc1.pixel_grid()
    mu_h = torch.linspace(0.0, 1.0, cfg.h_hidden, dtype=torch.float64)
    covariance = x.t() @ x / x.shape[0]
    kin_gram = torch.exp(
        -torch.cdist(mu_xy, mu_xy).square() / (2.0 * cfg.sigma**2)
    )
    kout_gram = torch.exp(
        -(mu_h[:, None] - mu_h[None, :]).square() / (2.0 * cfg.sigma**2)
    )
    problem = fc1.Problem(
        cfg,
        x,
        y,
        x[: cfg.n_eval],
        y[: cfg.n_eval],
        x[: cfg.n_eval],
        y[: cfg.n_eval],
        mu_xy,
        mu_h,
        covariance,
        kin_gram,
        kout_gram,
    )
    return fc1, problem


def _network(
    problem: object,
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    method: object,
    *,
    budget: int,
    seed: int,
) -> tuple[StructuralEngine, CSTLinear, SynapseStore]:
    cfg = problem.cfg
    store = SynapseStore(
        "fc1",
        2,
        1,
        cfg.k_ref,
        max_capacity=cfg.k_ref,
        spec=RepresentationSpec.continuous(2, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    if weight.numel():
        store.apply(
            [
                SynapseBirth(
                    store.site,
                    source,
                    target[:, None],
                    weight,
                    torch.arange(weight.numel()),
                )
            ]
        )
    inputs = NeuronStore(
        "fc1.in",
        784,
        mu=problem.mu_xy,
        initial_live=784,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "fc1.out",
        cfg.h_hidden,
        mu=problem.mu_h[:, None],
        initial_live=cfg.h_hidden,
        dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianKernel(cfg.sigma).double())
    root = QuotaRegime(
        budget=budget,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
        profit=ProfitCourt(),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        root,
        modules={store.site: module},
        seed=seed,
    )
    return engine, module, store


def _objective(
    fc1: ModuleType,
    problem: object,
    module: CSTLinear,
    head_w: torch.Tensor,
    head_b: torch.Tensor,
) -> float:
    with torch.no_grad():
        logits = fc1.head_logits(torch.nn.functional.gelu(module(problem.x)), head_w, head_b)
        return float(fc1.ce_loss(logits, problem.y))


def _capture(
    fc1: ModuleType,
    problem: object,
    engine: StructuralEngine,
    module: CSTLinear,
    head_w: torch.Tensor,
    head_b: torch.Tensor,
) -> torch.Tensor:
    module.zero_grad(set_to_none=True)
    engine.begin_update()
    logits = fc1.head_logits(
        torch.nn.functional.gelu(module(problem.x)), head_w, head_b
    )
    fc1.ce_loss(logits, problem.y).backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    request = next(
        req for req in engine._tree.requires if isinstance(req, TangentStatisticsRequest)
    )
    return engine.instrument("fc1", request.name).snapshot().cross


def test_cvp_matches_fc1_same_scale_gram_trajectory(fc1_problem) -> None:
    fc1, problem = fc1_problem
    generator = torch.Generator().manual_seed(7000)
    oracle_model = fc1._DeepModel(problem, generator)
    source = oracle_model.s.detach().clone()
    target = oracle_model.t.detach().clone()
    weight = oracle_model.w.detach().clone()
    head_w = oracle_model.head_w.detach().clone()
    head_b = oracle_model.head_b.detach().clone()
    engine, module, store = _network(
        problem, source, target, weight, cVP(), budget=1, seed=7000
    )
    torch.testing.assert_close(
        module(problem.x[:32]),
        fc1.hidden_pre(problem, problem.x[:32], source, target, weight),
    )

    accepted = 0
    for event in range(3):
        # A deterministic head perturbation makes this an event trajectory,
        # rather than three reads of one stationary solve.
        event_head = head_w + event * 0.002
        cross = _capture(fc1, problem, engine, module, event_head, head_b)
        before_w = store.view().w.clone()
        delta = fc1.tangent_gram_solve(
            problem,
            store.view().s,
            store.view().t[:, 0],
            cross,
            ridge=1.0e-4,
        )
        before_loss = _objective(fc1, problem, module, event_head, head_b)
        operations = engine.step(
            lambda: _objective(fc1, problem, module, event_head, head_b)
        )
        after_loss = _objective(fc1, problem, module, event_head, head_b)
        assert after_loss <= before_loss + 1.0e-12
        refits = [op for op in operations if isinstance(op, SynapseRefit)]
        if refits:
            accepted += 1
            torch.testing.assert_close(
                refits[0].w,
                before_w + delta,
                rtol=1.0e-6,
                atol=2.0e-10,
            )
    assert accepted >= 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_csfw_grow_matches_fc1_birth_statistics(fc1_problem, seed: int) -> None:
    fc1, problem = fc1_problem
    generator = torch.Generator().manual_seed(6000 + seed)
    oracle_model = fc1._DeepModel(problem, generator)
    # The shipped family requires a non-empty nucleus; retain one FC-1 atom.
    source = oracle_model.s[:1].detach().clone()
    target = oracle_model.t[:1].detach().clone()
    weight = oracle_model.w[:1].detach().clone()
    head_w = oracle_model.head_w.detach().clone()
    head_b = oracle_model.head_b.detach().clone()
    engine, module, store = _network(
        problem,
        source,
        target,
        weight,
        cSFW(
            polish_iters=0,
            backfit=None,
            pool_size=4096,
            multistart=4,
        ),
        budget=problem.cfg.births_per_event,
        seed=6000 + seed,
    )
    cross = _capture(fc1, problem, engine, module, head_w, head_b)
    before_loss = _objective(fc1, problem, module, head_w, head_b)
    operations = engine.step(lambda: _objective(fc1, problem, module, head_w, head_b))
    after_loss = _objective(fc1, problem, module, head_w, head_b)
    births = [op for op in operations if isinstance(op, SynapseBirth)]
    assert len(births) == 1
    birth = births[0]
    assert birth.w.numel() == problem.cfg.births_per_event
    assert after_loss <= before_loss + 1.0e-12

    # FC-1's grid peak is the exact reference search.  The generic tree
    # family searches a 4096-point continuous pool, so compare achieved
    # profile gain statistically while requiring every solved amplitude and
    # every sequential deflation update to match the oracle formula exactly.
    best_theta = torch.tensor(
        fc1.seed_peaks(cross, problem, 1)[0], dtype=torch.float64
    )
    best_g, best_a = fc1._g_a(best_theta, cross, problem)
    selected_theta = torch.cat((birth.s[0], birth.t[0]))
    selected_g, selected_a = fc1._g_a(selected_theta, cross, problem)
    gain_ratio = float(
        (selected_g.square() / selected_a)
        / (best_g.square() / best_a).clamp_min(1.0e-30)
    )
    assert gain_ratio >= 0.70

    residual = cross.clone()
    live_scale = float(weight.abs().max())
    for index in range(birth.w.numel()):
        theta = torch.cat((birth.s[index], birth.t[index]))
        gradient, curvature = fc1._g_a(theta, residual, problem)
        expected = (gradient / curvature).clamp(-live_scale, live_scale)
        torch.testing.assert_close(
            birth.w[index], expected, rtol=1.0e-6, atol=2.0e-10
        )
        kin = fc1.kin_of(problem, birth.s[index : index + 1])[0]
        kout = fc1.kout_of(problem, birth.t[index : index + 1, 0])[0]
        residual -= birth.w[index] * torch.outer(kout, problem.sx @ kin)
