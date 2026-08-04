"""Rise-time measurement, immunity conversion, and scripted collection."""

from __future__ import annotations

import pytest

from torchcst.engine import StructuralEngine
from torchcst.lab import (
    CalibrationRun,
    TauRiseResult,
    immunity_from_tau,
    tau_rise,
)
from tests.torchcst._recipes import LC
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseStore


def test_tau_rise_matches_scripted_analytic_arrival() -> None:
    trajectory = [0.0, 0.2, 0.6, 0.9, 1.0, 1.0]

    assert tau_rise(trajectory) == TauRiseResult(tau=3, censored=False)
    assert tau_rise(dict(enumerate(trajectory, start=10))).tau == 3
    assert tau_rise([]).censored is True


def test_immunity_uses_cross_birth_max_and_reports_censoring() -> None:
    taus = [
        TauRiseResult(2, False),
        TauRiseResult(4, False),
        TauRiseResult(3, True),
    ]
    result = immunity_from_tau(taus)

    assert result.immunity_events == 5
    assert result.censored_rate == pytest.approx(1 / 3)
    assert result.immunity_events > max(row.tau for row in taus) - 1


def test_calibration_run_collects_newborn_mass_by_event() -> None:
    store = SynapseStore(
        "entry",
        1,
        1,
        1,
        spec=RepresentationSpec.entry(bounds_in=1, bounds_out=1),
    )
    engine = StructuralEngine(
        {"entry": store},
        LC(
            event_interval=1,
            birth_end_event=1,
            birth_budget=1,
            freeze_event=2,
            bounds_in=1,
            bounds_out=1,
            initial_weight=1.0,
        ),
    )
    run = CalibrationRun(engine, lambda _engine, _step: None)

    trajectories = run.run(3)

    assert trajectories == {("entry", 0): (1.0, 1.0, 1.0)}
    assert run.event_trajectories[("entry", 0)] == {1: 1.0, 2: 1.0, 3: 1.0}
