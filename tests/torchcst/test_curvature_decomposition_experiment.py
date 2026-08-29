"""Checks for the isolated CST curvature experiment."""

from __future__ import annotations

import torch

from experiments.cst_curvature.decomposition import (
    TrialConfig,
    charts,
    gaussian_cst_weight,
    initial_theta,
    run_trial,
)
from torchcst.compute import CSTLinear, Materialized
from torchcst.representation import GaussianFactor, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def test_functional_weight_is_the_cst_linear_weight():
    config = TrialConfig(atoms=2, amplitude=0.2)
    theta = initial_theta(config)
    mu_in, mu_out = charts(config)
    factor = GaussianFactor(config.sigma, learnable=False).double()

    synapses = SynapseStore(
        "curvature",
        1,
        1,
        config.atoms,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    synapses.apply(
        [
            SynapseBirth(
                synapses.site,
                theta[:, 1:2],
                theta[:, 2:3],
                theta[:, 0],
                torch.arange(config.atoms, dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            "curvature-in",
            config.n_in,
            mu=mu_in,
            initial_live=config.n_in,
            dtype=torch.float64,
        ),
        NeuronStore(
            "curvature-out",
            config.n_out,
            mu=mu_out,
            initial_live=config.n_out,
            dtype=torch.float64,
        ),
        synapses,
        factor,
        track_mass=False,
        backend=Materialized(lean=False),
    )

    torch.testing.assert_close(
        gaussian_cst_weight(theta, mu_in, mu_out, factor),
        layer.dense_weight(),
    )


def test_mse_weight_space_remainder_closes_the_actual_change():
    result = run_trial(TrialConfig(atoms=2, amplitude=1e-4, learning_rate=0.03))

    torch.testing.assert_close(
        torch.tensor(result.p_actual),
        torch.tensor(result.p_weight_exact + result.loss_quadratic_exact),
        rtol=1e-10,
        atol=1e-12,
    )


def test_contracted_map_hessian_has_no_cross_atom_entries():
    result = run_trial(TrialConfig(atoms=2, amplitude=0.4, learning_rate=0.02))

    assert result.cross_atom_map_hessian_max < 1e-14


def test_same_atom_field_blocks_close_the_cst_quadratic():
    result = run_trial(TrialConfig(atoms=2, amplitude=0.2, learning_rate=0.02))

    parts = (
        result.cst_quadratic_amplitude_amplitude
        + result.cst_quadratic_amplitude_position
        + result.cst_quadratic_position_position
    )
    torch.testing.assert_close(
        torch.tensor(result.cst_quadratic),
        torch.tensor(parts),
        rtol=1e-10,
        atol=1e-12,
    )


def test_zero_amplitude_has_no_position_step_or_directional_map_curvature():
    result = run_trial(TrialConfig(atoms=1, amplitude=0.0, learning_rate=0.02))

    assert result.position_step_norm == 0.0
    assert result.cst_quadratic_amplitude_position == 0.0
    assert result.cst_quadratic_position_position == 0.0


def test_small_step_full_second_order_beats_first_order():
    result = run_trial(TrialConfig(atoms=2, amplitude=0.1, learning_rate=1e-4))

    assert result.relative_error("p2_full") < result.relative_error("p1")


def test_controlled_and_unit_directions_share_the_requested_radius():
    for direction in (
        "sgd_unit",
        "amplitude",
        "position",
        "mixed",
        "mixed_flip",
        "random",
        "pullback_inverse",
        "pullback_bounded",
    ):
        result = run_trial(
            TrialConfig(
                atoms=2,
                amplitude=0.1,
                learning_rate=0.013,
                direction=direction,
            )
        )
        torch.testing.assert_close(
            torch.tensor(result.dimensionless_step_norm),
            torch.tensor(0.013),
        )


def test_mixed_sign_pair_flips_only_the_amplitude_position_quadratic():
    common = dict(atoms=2, amplitude=0.1, learning_rate=0.01)
    positive = run_trial(TrialConfig(direction="mixed", **common))
    negative = run_trial(TrialConfig(direction="mixed_flip", **common))

    torch.testing.assert_close(
        torch.tensor(positive.cst_quadratic_amplitude_position),
        -torch.tensor(negative.cst_quadratic_amplitude_position),
    )
    torch.testing.assert_close(
        torch.tensor(positive.cst_quadratic_position_position),
        torch.tensor(negative.cst_quadratic_position_position),
    )


def test_random_directions_are_seeded_and_distinct():
    common = dict(
        atoms=2,
        amplitude=0.1,
        learning_rate=0.01,
        direction="random",
    )
    first = run_trial(TrialConfig(direction_seed=5, **common))
    repeated = run_trial(TrialConfig(direction_seed=5, **common))
    different = run_trial(TrialConfig(direction_seed=6, **common))

    assert first == repeated
    assert first.p1 != different.p1


def test_inverse_pullback_near_zero_exposes_cst_map_curvature():
    result = run_trial(
        TrialConfig(
            atoms=1,
            amplitude=1e-3,
            learning_rate=0.01,
            direction="pullback_inverse",
        )
    )

    assert abs(result.cst_quadratic) > 100 * result.loss_quadratic_jacobian
    assert result.relative_error("p2_cst") < result.relative_error("p1") / 100
