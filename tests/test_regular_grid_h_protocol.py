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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_state_initialization_does_not_update_and_matches_lazy(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    if device == "cpu":
        # The benchmark uses capturable CUDA AdamW; the CPU check isolates init.
        capturable = False
    else:
        capturable = True
    import copy
    from types import SimpleNamespace

    model = torch.nn.Linear(13, 17, bias=False, device=device)
    lazy_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, fused=True, capturable=capturable
    )
    lazy = torch.optim.AdamW(
        lazy_model.parameters(), lr=1e-4, fused=True, capturable=capturable
    )
    previous = model.weight.detach().clone()
    protocol.initialize_optimizer_state(SimpleNamespace(model=model, opt=optimizer))
    assert torch.equal(model.weight, previous) and model.weight.grad is None
    state = optimizer.state[model.weight]
    assert int(state["step"]) == 0
    assert not state["exp_avg"].any() and not state["exp_avg_sq"].any()
    for _ in range(3):
        gradient = (
            torch.arange(221, device=device, dtype=torch.float32).reshape(17, 13) / 221
        )
        model.weight.grad = gradient.clone()
        lazy_model.weight.grad = gradient.clone()
        optimizer.step()
        lazy.step()
    torch.testing.assert_close(model.weight, lazy_model.weight, rtol=0, atol=0)
    for name in state:
        torch.testing.assert_close(
            state[name], lazy.state[lazy_model.weight][name], rtol=0, atol=0
        )
