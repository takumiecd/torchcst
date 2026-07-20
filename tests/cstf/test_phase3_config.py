"""Frozen Phase 3-A2 lifecycle constants as canonical Arm configs."""

from __future__ import annotations

import json

import pytest

from cstf.lab import phase3_a2_arms


def test_phase3_a2_frozen_constants_are_directly_expressed() -> None:
    by_name = {arm.name: arm for arm in phase3_a2_arms()}
    entry = by_name["A-entry"]
    atom = by_name["A-atom"]

    assert entry.policy_factory == atom.policy_factory == "LC"
    assert dict(entry.constants["immunity_events"]) == {
        "stage0": 8,
        "stage1": 7,
        "stage2": 5,
    }
    assert dict(atom.constants["immunity_events"]) == {
        "stage0": 5,
        "stage1": 8,
        "stage2": 9,
    }
    for arm in (entry, atom):
        rent = arm.constants["rent"]
        window = arm.constants["birth_window"]
        assert rent["statistic"] == "median"
        assert rent["multiplier"] == 0.3
        assert rent["consecutive_failures"] == 2
        assert rent["hysteresis"] is True
        assert rent["immune_entities_eligible"] is False
        assert (window["start_step"], window["end_step"]) == (200, 8000)
        assert window["event_count"] == 40
        assert window["event_interval_steps"] == 200
        assert window["vacancy_refill_within_window"] is True
        assert window["vacancy_refill_after_window"] is False
        assert arm.constants["loss_triggers"] is False
        assert arm.seeds == (0, 1, 2)


def test_arm_json_and_sha_are_stable_and_constants_are_frozen() -> None:
    first = phase3_a2_arms()
    second = phase3_a2_arms()

    assert [arm.sha256 for arm in first] == [arm.sha256 for arm in second]
    assert [arm.to_json() for arm in first] == [arm.to_json() for arm in second]
    for arm in first:
        document = json.loads(arm.to_json())
        assert document["sha256"] == arm.sha256
        assert arm.to_json().endswith("\n")
        with pytest.raises(TypeError):
            arm.constants["loss_triggers"] = True
