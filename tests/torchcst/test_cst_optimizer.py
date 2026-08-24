"""CSTOptimizer: one owner per parameter across many sites in many layers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch import nn

from torchcst import CSTOptimizer, PullbackConfig, continuous_sites
from torchcst.compute import CSTLinear, Materialized
from torchcst.optim import ChartPullbackAdam, PullbackAdam, SmoothRent
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.35
D = 2


def _birth(store, source, target, weights, lineage_start=0):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
    )


def _synapses(name, atoms, generator):
    store = SynapseStore(
        name,
        D,
        D,
        atoms,
        spec=RepresentationSpec.continuous(D, D),
        dtype=torch.float64,
    )
    source = torch.rand(atoms, D, generator=generator, dtype=torch.float64)
    target = torch.rand(atoms, D, generator=generator, dtype=torch.float64)
    weights = 0.5 + torch.rand(atoms, generator=generator, dtype=torch.float64)
    store.apply([_birth(store, source, target, weights)])
    return store


def _chart(name, count, generator):
    mu = nn.Parameter(
        torch.rand(count, D, generator=generator, dtype=torch.float64)
    )
    return NeuronStore(name, count, mu=mu, initial_live=count, dtype=torch.float64)


def _map(in_chart, out_chart, synapses):
    return CSTLinear(
        in_chart,
        out_chart,
        synapses,
        GaussianFactor(SIGMA).double(),
        gauge=L2NormalizedColumns(),
        backend=Materialized(lean=False),
    )


class _Layer(nn.Module):
    """One transformer-ish block: FFN up, FFN down, and a dense norm."""

    def __init__(self, index, residual, generator, atoms=2):
        super().__init__()
        self.hidden = _chart(f"chart-hidden-{index}", 5, generator)
        self.ffn_up = _map(
            residual, self.hidden, _synapses(f"up-{index}", atoms, generator)
        )
        self.ffn_down = _map(
            self.hidden, residual, _synapses(f"down-{index}", atoms, generator)
        )
        self.norm = nn.LayerNorm(residual.n_max, dtype=torch.float64)

    def forward(self, x):
        return self.norm(x + self.ffn_down(self.ffn_up(x)))


class _Model(nn.Module):
    """Two layers times FFN up/down: four continuous sites, one shared chart."""

    def __init__(self, seed=5, layers=2, atoms=2):
        super().__init__()
        # The CST halves draw from a local generator; nn.Linear/LayerNorm draw
        # from the global one, so seed that too or two "identical" models are
        # only identical within a single process.
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        self.residual = _chart("chart-residual", 4, generator)
        self.layers = nn.ModuleList(
            [_Layer(index, self.residual, generator) for index in range(layers)]
        )
        self.head = nn.Linear(self.residual.n_max, 3, dtype=torch.float64)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _config(**overrides):
    return replace(
        PullbackConfig(moment_space="tangent", cap_sigma=0.1), **overrides
    )


def _adamw(lr=1e-3):
    return lambda params: torch.optim.AdamW(params, lr=lr)


def _loss(model, seed=0):
    inputs = torch.rand(
        3,
        model.residual.n_max,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )
    return model(inputs).square().sum()


# ---- discovery -------------------------------------------------------------


def test_every_site_in_every_layer_is_discovered_in_registration_order():
    model = _Model()

    names = [name for name, _ in continuous_sites(model)]

    assert names == [
        "layers.0.ffn_up",
        "layers.0.ffn_down",
        "layers.1.ffn_up",
        "layers.1.ffn_down",
    ]


def test_coordinator_builds_one_pullback_adam_per_site():
    model = _Model()

    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())

    assert optimizer.site_names == (
        "layers.0.ffn_up",
        "layers.0.ffn_down",
        "layers.1.ffn_up",
        "layers.1.ffn_down",
    )
    assert all(
        isinstance(site, PullbackAdam) for site in optimizer.sites.values()
    )
    # Each bound to *its own* store, not four views of one.
    assert len({id(site.store) for site in optimizer.sites.values()}) == 4


def test_one_store_reachable_under_two_names_is_rejected():
    class Aliased(nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(3)
            chart = _chart("chart-alias", 4, generator)
            self.first = _map(chart, chart, _synapses("alias", 2, generator))
            # A second module over the identical store: two PullbackAdam
            # instances would step the same rows twice.
            self.second = _map(
                chart, chart, self.first.synapses
            )

    with pytest.raises(ValueError, match="one store must have one owning module"):
        continuous_sites(Aliased())


# ---- ownership -------------------------------------------------------------


def test_dense_amplitude_and_coordinate_ownership_is_correct():
    model = _Model()

    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())

    ownership = optimizer.ownership()
    assert ownership["layers.0.ffn_up.synapses.s"] == "site:layers.0.ffn_up"
    assert ownership["layers.0.ffn_up.synapses.t"] == "site:layers.0.ffn_up"
    # No rent and no decay: amplitudes stay ordinary parameters.
    assert ownership["layers.0.ffn_up.synapses.w"] == "base"
    assert ownership["layers.1.ffn_down.synapses.s"] == "site:layers.1.ffn_down"
    assert ownership["head.weight"] == "base"
    assert ownership["layers.0.norm.weight"] == "base"
    # A learnable chart with no chart optimizer is the base optimizer's.
    assert ownership["residual.mu"] == "base"
    assert set(ownership.values()) == {
        "base",
        "site:layers.0.ffn_up",
        "site:layers.0.ffn_down",
        "site:layers.1.ffn_up",
        "site:layers.1.ffn_down",
    }


def test_rent_moves_amplitudes_from_the_base_optimizer_to_the_site():
    model = _Model()

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        overrides={
            "layers.0.*": _config(rent=SmoothRent(1e-3), lr_w=1e-3),
        },
        base=_adamw(),
    )

    ownership = optimizer.ownership()
    assert ownership["layers.0.ffn_up.synapses.w"] == "site:layers.0.ffn_up"
    assert ownership["layers.0.ffn_down.synapses.w"] == "site:layers.0.ffn_down"
    assert ownership["layers.1.ffn_up.synapses.w"] == "base"
    assert optimizer.sites["layers.0.ffn_up"].owns_amplitudes
    assert not optimizer.sites["layers.1.ffn_up"].owns_amplitudes


def test_decay_alone_also_claims_amplitudes():
    model = _Model()

    optimizer = CSTOptimizer(
        model, coordinates=_config(decay=0.01, lr_w=1e-3), base=_adamw()
    )

    assert optimizer.ownership()["layers.0.ffn_up.synapses.w"] == (
        "site:layers.0.ffn_up"
    )


def test_no_parameter_is_stepped_twice_and_none_is_left_behind():
    model = _Model()

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(rent=SmoothRent(1e-3), lr_w=1e-3),
        charts=[
            ChartPullbackAdam(
                model.residual,
                [model.layers[0].ffn_up, model.layers[0].ffn_down,
                 model.layers[1].ffn_up, model.layers[1].ffn_down],
            )
        ],
        base=_adamw(),
    )

    structural = [
        id(parameter)
        for site in optimizer.sites.values()
        for parameter in (site.store.s, site.store.t, site.store.w)
    ]
    structural += [id(chart.store.mu) for chart in optimizer.charts]
    held = [
        id(parameter)
        for group in optimizer.base.param_groups
        for parameter in group["params"]
    ]

    assert len(set(structural)) == len(structural)
    assert not set(structural) & set(held)
    trainable = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert set(structural) | set(held) == trainable
    assert len(structural) + len(held) == len(trainable)


def test_a_base_optimizer_holding_owned_coordinates_is_rejected():
    model = _Model()

    with pytest.raises(ValueError, match="structurally owned"):
        CSTOptimizer(
            model,
            coordinates=_config(),
            base=torch.optim.AdamW(model.parameters(), lr=1e-3),
        )


def test_a_factory_that_ignores_the_partition_is_rejected():
    model = _Model()

    with pytest.raises(ValueError, match="structurally owned"):
        CSTOptimizer(
            model,
            coordinates=_config(),
            base=lambda params: torch.optim.AdamW(model.parameters(), lr=1e-3),
        )


def test_a_base_optimizer_cannot_step_a_parameter_outside_the_model():
    model = _Model()
    external = nn.Parameter(torch.ones((), dtype=torch.float64))

    with pytest.raises(ValueError, match="outside its assigned partition"):
        CSTOptimizer(
            model,
            coordinates=_config(),
            base=lambda params: torch.optim.AdamW([*params, external], lr=1e-3),
        )


def test_an_explicit_base_optimizer_that_covers_the_partition_is_accepted():
    model = _Model()
    # subscribe=False: this probe only computes the partition, so it must not
    # leave a second follower on every store.
    probe = CSTOptimizer(
        model,
        coordinates=_config(),
        base=None,
        allow_unowned=True,
        subscribe=False,
    )

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        base=torch.optim.AdamW(list(probe.expected_base_parameters()), lr=1e-3),
    )

    held = [
        id(parameter)
        for group in optimizer.base.param_groups
        for parameter in group["params"]
    ]
    assert held == [
        id(parameter) for parameter in probe.expected_base_parameters()
    ]
    assert "unowned" not in optimizer.ownership().values()


def test_parameters_left_without_an_owner_are_rejected():
    model = _Model()

    with pytest.raises(ValueError, match="have no owner"):
        CSTOptimizer(model, coordinates=_config(), base=None)

    # ... and the escape hatch is explicit, not silent.
    lenient = CSTOptimizer(
        model, coordinates=_config(), base=None, allow_unowned=True
    )
    assert lenient.ownership()["head.weight"] == "unowned"


def test_a_partial_base_optimizer_reports_its_omissions_as_unowned():
    model = _Model()
    probe = CSTOptimizer(
        model,
        coordinates=_config(),
        base=None,
        allow_unowned=True,
        subscribe=False,
    )
    partial = [
        parameter
        for parameter in probe.expected_base_parameters()
        if parameter is not model.head.weight
    ]

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        base=torch.optim.AdamW(partial, lr=1e-2),
        allow_unowned=True,
    )

    ownership = optimizer.ownership()
    assert ownership["head.weight"] == "unowned"
    assert ownership["head.bias"] == "base"
    assert "UNOWNED" in optimizer.summary()
    # expected_base_parameters() stays the expectation; ownership() reports.
    assert any(
        parameter is model.head.weight
        for parameter in optimizer.expected_base_parameters()
    )

    before = model.head.weight.detach().clone()
    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()

    assert torch.equal(model.head.weight, before)


def test_frozen_parameters_are_neither_owned_nor_reported_missing():
    model = _Model()
    model.head.weight.requires_grad_(False)

    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())

    assert optimizer.ownership()["head.weight"] == "frozen"
    held = {
        id(parameter)
        for group in optimizer.base.param_groups
        for parameter in group["params"]
    }
    assert id(model.head.weight) not in held


# ---- charts ----------------------------------------------------------------


def test_a_chart_optimizer_takes_mu_from_the_base_optimizer():
    model = _Model()
    chart = ChartPullbackAdam(
        model.residual,
        [
            model.layers[0].ffn_up,
            model.layers[0].ffn_down,
            model.layers[1].ffn_up,
            model.layers[1].ffn_down,
        ],
    )

    optimizer = CSTOptimizer(
        model, coordinates=_config(), charts=[chart], base=_adamw()
    )

    assert optimizer.ownership()["residual.mu"] == "chart:chart-residual"
    assert optimizer.ownership()["layers.0.hidden.mu"] == "base"


def test_a_chart_outside_the_model_is_rejected():
    model = _Model()
    generator = torch.Generator().manual_seed(9)
    outside = _chart("chart-outside", 4, generator)
    outside_site = _map(outside, outside, _synapses("outside", 2, generator))

    # Nothing in `model` registers this chart, so stepping it would move a
    # parameter the coordinator cannot even name in ownership().
    with pytest.raises(ValueError, match="chart 'chart-outside' mu"):
        CSTOptimizer(
            model,
            coordinates=_config(),
            charts=[ChartPullbackAdam(outside, [outside_site])],
            base=_adamw(),
        )


def test_a_site_whose_store_is_not_registered_under_the_model_is_rejected():
    class DetachedStoreSite(nn.Module):
        """Site-shaped, but keeping its store out of the module tree."""

        def __init__(self, site):
            super().__init__()
            self.in_neurons = site.in_neurons
            self.out_neurons = site.out_neurons
            self.factor_in = site.factor_in
            self.factor_out = site.factor_out
            self.gauge = site.gauge
            self.__dict__["synapses"] = site.synapses

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(4)
            chart = _chart("chart-host", 4, generator)
            self.site = DetachedStoreSite(
                _map(chart, chart, _synapses("detached", 2, generator))
            )

    host = Host()
    assert "site.synapses.s" not in dict(host.named_parameters())

    with pytest.raises(ValueError, match="not among model.named_parameters"):
        CSTOptimizer(host, coordinates=_config(), base=None, allow_unowned=True)


def test_two_optimizers_for_one_shared_chart_are_rejected():
    model = _Model()
    sites = [model.layers[0].ffn_up, model.layers[0].ffn_down]

    with pytest.raises(ValueError, match="two chart optimizers own chart"):
        CSTOptimizer(
            model,
            coordinates=_config(),
            charts=[
                ChartPullbackAdam(model.residual, sites),
                ChartPullbackAdam(model.residual, sites),
            ],
            base=_adamw(),
        )


# ---- overrides -------------------------------------------------------------


def test_per_site_overrides_match_by_module_name_glob():
    model = _Model()

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(target_step=0.01),
        overrides={"layers.*.ffn_down": _config(target_step=0.005, metric="block")},
        base=_adamw(),
    )

    assert optimizer.sites["layers.0.ffn_down"].target_step == 0.005
    assert optimizer.sites["layers.1.ffn_down"].metric == "block"
    assert optimizer.sites["layers.0.ffn_up"].target_step == 0.01
    assert optimizer.sites["layers.1.ffn_up"].metric == "diag"


def test_the_first_matching_override_pattern_wins():
    model = _Model()

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        overrides={
            "layers.0.ffn_up": _config(target_step=0.002),
            "layers.*": _config(target_step=0.004),
        },
        base=_adamw(),
    )

    assert optimizer.sites["layers.0.ffn_up"].target_step == 0.002
    assert optimizer.sites["layers.0.ffn_down"].target_step == 0.004
    assert optimizer.sites["layers.1.ffn_up"].target_step == 0.004


def test_an_override_pattern_matching_no_site_raises():
    model = _Model()

    with pytest.raises(ValueError, match="match no discovered site"):
        CSTOptimizer(
            model,
            coordinates=_config(),
            overrides={"layers.7.ffn_up": _config()},
            base=_adamw(),
        )


def test_an_override_of_none_hands_that_site_to_the_base_optimizer():
    model = _Model()

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        overrides={"layers.1.*": None},
        base=_adamw(),
    )

    assert optimizer.site_names == ("layers.0.ffn_up", "layers.0.ffn_down")
    assert optimizer.delegated_sites == ("layers.1.ffn_up", "layers.1.ffn_down")
    assert optimizer.ownership()["layers.1.ffn_up.synapses.s"] == "base"


# ---- the config ------------------------------------------------------------


def test_config_forwards_moment_distance_independently_of_betas():
    model = _Model()

    optimizer = CSTOptimizer(
        model,
        coordinates=_config(betas=(0.5, 0.7), moment_distance=(0.25, 1.0)),
        base=_adamw(),
    )

    site = optimizer.sites["layers.0.ffn_up"]
    assert (site.beta1, site.beta2) == (0.5, 0.7)
    assert site.moment_distance == (0.25, 1.0)


def test_the_framework_default_leaves_the_distance_clock_off():
    model = _Model()

    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())

    assert PullbackConfig(moment_space="tangent", cap_sigma=0.1).moment_distance is None
    assert optimizer.sites["layers.0.ffn_up"].moment_distance is None


def test_config_build_reproduces_a_hand_written_pullback_adam():
    model = _Model()
    site = model.layers[0].ffn_up
    config = _config(
        metric="block",
        betas=(0.8, 0.95),
        amplitude_betas=(0.7, 0.9),
        eps=1e-7,
        damping=2e-2,
        target_step=0.02,
        decay=0.01,
        lr_w=3e-4,
        wall=True,
        seed=11,
        chunk_elements=1 << 20,
        moment_distance=(0.25, 1.0),
    )

    built = config.build(site, subscribe=False)
    manual = PullbackAdam(
        site,
        moment_space="tangent",
        metric="block",
        cap_sigma=0.1,
        betas=(0.8, 0.95),
        amplitude_betas=(0.7, 0.9),
        eps=1e-7,
        damping=2e-2,
        target_step=0.02,
        decay=0.01,
        lr_w=3e-4,
        wall=True,
        seed=11,
        subscribe=False,
        chunk_elements=1 << 20,
        moment_distance=(0.25, 1.0),
    )

    knobs = (
        "moment_space", "metric", "cap_sigma", "beta1", "beta2",
        "amplitude_beta1", "amplitude_beta2", "eps", "damping", "target_step",
        "decay", "lr_w", "wall", "seed", "chunk_elements", "moment_distance",
        "owns_amplitudes",
    )
    assert [getattr(built, knob) for knob in knobs] == [
        getattr(manual, knob) for knob in knobs
    ]


def test_config_is_immutable_and_derives_variants():
    config = _config()

    with pytest.raises(FrozenInstanceError):
        config.target_step = 0.5

    assert replace(config, target_step=0.5).target_step == 0.5
    assert config.target_step == 0.01


def test_config_validation_is_pullback_adams_own():
    model = _Model()

    with pytest.raises(ValueError, match="lr_w comes with amplitude ownership"):
        CSTOptimizer(
            model, coordinates=_config(rent=SmoothRent(1e-3)), base=_adamw()
        )


# ---- the update ------------------------------------------------------------


def test_one_update_moves_dense_weights_and_every_site_coordinate():
    model = _Model()
    chart = ChartPullbackAdam(
        model.residual,
        [
            model.layers[0].ffn_up,
            model.layers[0].ffn_down,
            model.layers[1].ffn_up,
            model.layers[1].ffn_down,
        ],
    )
    optimizer = CSTOptimizer(
        model, coordinates=_config(), charts=[chart], base=_adamw(lr=1e-2)
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()

    for name in (
        "layers.0.ffn_up.synapses.s",
        "layers.0.ffn_down.synapses.t",
        "layers.1.ffn_up.synapses.s",
        "layers.1.ffn_down.synapses.t",
        "layers.0.ffn_up.synapses.w",
        "head.weight",
        "residual.mu",
    ):
        current = dict(model.named_parameters())[name]
        assert not torch.equal(current, before[name]), name


def test_zero_grad_clears_dense_and_structural_gradients():
    model = _Model()
    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())

    _loss(model).backward()
    assert model.layers[0].ffn_up.synapses.s.grad is not None
    optimizer.zero_grad()

    assert model.layers[0].ffn_up.synapses.s.grad is None
    assert model.head.weight.grad is None


def test_lr_scale_zero_freezes_the_coordinates_but_not_the_base():
    model = _Model()
    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw(lr=1e-2))
    coordinates = model.layers[0].ffn_up.synapses.s.detach().clone()
    dense = model.head.weight.detach().clone()

    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step(lr_scale=0.0)

    assert torch.equal(model.layers[0].ffn_up.synapses.s, coordinates)
    assert not torch.equal(model.head.weight, dense)


def test_step_order_is_base_then_every_site_then_every_chart():
    """The order is a contract, not a detail.

    Disjoint ownership does not make it neutral: the base step moves ``w``,
    ``mu`` and ``sigma``, which are exactly the quantities a site's pullback
    metric is built from, so a later optimizer whitens with a ``G`` the
    earlier one changed. This records the whole sequence.
    """
    model = _Model()
    charts = [
        ChartPullbackAdam(
            model.residual,
            [model.layers[0].ffn_up, model.layers[0].ffn_down,
             model.layers[1].ffn_up, model.layers[1].ffn_down],
        ),
        ChartPullbackAdam(
            model.layers[0].hidden,
            [model.layers[0].ffn_up, model.layers[0].ffn_down],
        ),
    ]
    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        overrides={"layers.1.ffn_down": _config(target_step=0.02)},
        charts=charts,
        base=_adamw(),
    )
    stepped: list[str] = []

    def record(label, original):
        def wrapper(*args, **kwargs):
            stepped.append(label)
            return original(*args, **kwargs)

        return wrapper

    optimizer.base.step = record("base", optimizer.base.step)
    for name, site in optimizer.sites.items():
        site.step = record(f"site:{name}", site.step)  # type: ignore[method-assign]
    for chart in optimizer.charts:
        chart.step = record(  # type: ignore[method-assign]
            f"chart:{chart.store.site}", chart.step
        )

    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()

    assert stepped == [
        "base",
        "site:layers.0.ffn_up",
        "site:layers.0.ffn_down",
        "site:layers.1.ffn_up",
        "site:layers.1.ffn_down",
        "chart:chart-residual",
        "chart:chart-hidden-0",
    ]


def test_built_sites_follow_their_stores_lifecycle():
    model = _Model()
    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())
    site = optimizer.sites["layers.0.ffn_up"]
    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()
    store = site.store
    before = store.capacity

    store.apply(
        [
            _birth(
                store,
                torch.full((2, D), 0.3, dtype=torch.float64),
                torch.full((2, D), 0.6, dtype=torch.float64),
                torch.full((2,), 0.2, dtype=torch.float64),
                lineage_start=100,
            )
        ]
    )

    assert store.capacity > before
    assert site.m_s.shape == store.s.shape


def test_subscribe_false_keeps_the_coordinator_out_of_the_follower_hub():
    model = _Model()
    optimizer = CSTOptimizer(
        model, coordinates=_config(), base=_adamw(), subscribe=False
    )
    site = optimizer.sites["layers.0.ffn_up"]

    assert site not in site.store.followers()._followers


# ---- persistence -----------------------------------------------------------


def test_state_roundtrips_across_the_whole_coordinator():
    def build(seed=5):
        model = _Model(seed=seed)
        chart = ChartPullbackAdam(
            model.residual,
            [model.layers[0].ffn_up, model.layers[0].ffn_down,
             model.layers[1].ffn_up, model.layers[1].ffn_down],
        )
        return model, CSTOptimizer(
            model, coordinates=_config(), charts=[chart], base=_adamw(lr=1e-2)
        )

    model, optimizer = build()
    for _ in range(2):
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()

    restored_model, restored = build()
    # deepcopy is what a checkpoint does: torch optimizers hand out live
    # references in state_dict(), so loading one in-process would otherwise
    # alias the base optimizer's moments between the two coordinators.
    restored.load_state_dict(deepcopy(optimizer.state_dict()))

    for name in optimizer.site_names:
        assert restored.sites[name].step_count == optimizer.sites[name].step_count
        assert torch.allclose(
            restored.sites[name].m_s, optimizer.sites[name].m_s
        )
        assert restored.sites[name].eta == optimizer.sites[name].eta
    assert torch.allclose(restored.charts[0].m, optimizer.charts[0].m)
    assert restored.state_dict()["base"]["param_groups"] == (
        optimizer.state_dict()["base"]["param_groups"]
    )
    # The restored coordinator continues the same trajectory: same parameters
    # in, same third step out -- which a fresh (unloaded) coordinator would
    # not produce, since its eta recalibrates and its moments start at zero.
    restored_model.load_state_dict(model.state_dict())
    optimizer.zero_grad()
    restored.zero_grad()
    _loss(model).backward()
    _loss(restored_model).backward()
    optimizer.step()
    restored.step()
    assert torch.allclose(
        restored_model.layers[0].ffn_up.synapses.s,
        model.layers[0].ffn_up.synapses.s,
    )


def test_state_from_a_differently_shaped_coordinator_is_rejected():
    model = _Model()
    optimizer = CSTOptimizer(model, coordinates=_config(), base=_adamw())
    partial = CSTOptimizer(
        _Model(), coordinates=_config(), overrides={"layers.1.*": None},
        base=_adamw(),
    )

    with pytest.raises(ValueError, match="covers sites"):
        partial.load_state_dict(optimizer.state_dict())
    with pytest.raises(ValueError, match="unsupported CSTOptimizer state schema"):
        optimizer.load_state_dict({"schema": "nope"})


def test_summary_names_every_site_chart_and_the_base():
    model = _Model()
    chart = ChartPullbackAdam(model.residual, [model.layers[0].ffn_up])
    optimizer = CSTOptimizer(
        model,
        coordinates=_config(),
        overrides={"layers.1.ffn_down": None},
        charts=[chart],
        base=_adamw(),
    )

    text = optimizer.summary()

    assert "layers.0.ffn_up" in text
    assert "layers.1.ffn_down" in text
    assert "chart-residual" in text
    assert "AdamW" in text
