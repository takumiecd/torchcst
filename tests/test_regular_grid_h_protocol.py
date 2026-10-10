"""Frozen matched fixtures and dense memory gates for the existing research runner."""

import copy
import json

import pytest
import torch

from benchmarks.cuda.linear import periodic_comparison as protocol
from benchmarks.cuda.linear import periodic_profile_product as fixture


def test_regular_grid_preserves_every_initial_tensor_and_full_operator():
    legacy = fixture.fixture(17, 3, atoms=19)
    regular = fixture.fixture(17, 3, atoms=19, regular_grid=True)
    torch.testing.assert_close(legacy.atoms.p, regular.atoms.p, rtol=0, atol=0)
    x = torch.randn(3, 17, generator=torch.Generator().manual_seed(531))
    dy = torch.randn(3, 17, generator=torch.Generator().manual_seed(532))
    expected = fixture.oracle_vjp(legacy, x, dy)
    actual = fixture.oracle_vjp(regular, x, dy)
    for aa, bb in zip(actual, expected, strict=True):
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    torch.testing.assert_close(legacy(x), regular(x), rtol=0, atol=0)


@pytest.mark.parametrize("regular", [False, True])
def test_frozen_case_roundtrip_and_tamper_refusal(tmp_path, regular):
    value = protocol.case_definition(1024, 3, regular_grid=regular)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(value))
    assert protocol.load_case(path) == value
    value["batch"] = 8
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        protocol.load_case(path)


def record(kind, time, allocated, reserved):
    return {
        "kind": kind,
        "status": "PASS",
        "runtime": {"gpu": "L4"},
        "timing": {"median_ms": time},
        "peak_capture_replay": {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
        },
    }


@pytest.mark.parametrize(
    "allocated,reserved,fits",
    [(80, 90, True), (100, 100, True), (101, 90, False), (90, 101, False)],
)
def test_both_dense_peaks_required_even_for_fast_candidate(allocated, reserved, fits):
    values = [
        record("dense", 1, 100, 100),
        record("factor", 3, 120, 120),
        record("matrix-torch", 2, 110, 110),
        record("onchip-h", 0.5, allocated, reserved),
    ]
    result = protocol.assess_regular({r["kind"]: r for r in values}, values)
    assert result["candidates"]["onchip-h"]["fits_dense_allocated_and_reserved"] == fits
    assert result["fastest_fitting_cst"] == ("onchip-h" if fits else None)
    assert result["adopted"] is False
    mismatch = copy.deepcopy(values)
    mismatch[-1]["runtime"]["gpu"] = "A100"
    with pytest.raises(AssertionError, match="runtime"):
        protocol.assess_regular({r["kind"]: r for r in mismatch}, mismatch)


def test_isolated_oracle_uses_saved_live_state_and_all_atom_cotangents(
    tmp_path, monkeypatch
):
    layer = fixture.fixture(17, 3, atoms=19, regular_grid=True)
    with torch.no_grad():
        layer.atoms.p[:, 2:] += 0.1
        layer.kernel.amplitude_max.mul_(0.8)
        layer.kernel.sigma_max_input.mul_(0.9)
    x = torch.randn(3, 17, generator=torch.Generator().manual_seed(643))
    dy = torch.randn(3, 17, generator=torch.Generator().manual_seed(644))
    snapshot = {
        "size": 17,
        "rho": 3,
        "model": layer.state_dict(),
        "x": x,
        "dy": dy,
        "actual": list(fixture.oracle_vjp(layer, x, dy)),
    }
    # Exercise snapshot reconstruction and the independent checker on CPU;
    # the real subprocess CUDA route is exercised by the L4 cohort.
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    path = tmp_path / "snapshot.pt"
    torch.save(snapshot, path)
    assert protocol.oracle_snapshot(path)["status"] == "PASS"
    snapshot["actual"][2][18, 3] += 1.0  # Last atom, centre cotangent: must be checked.
    torch.save(snapshot, path)
    with pytest.raises(AssertionError, match="all_dP"):
        protocol.oracle_snapshot(path)
