"""Actual installed AdamW API, moments and serialization across supported Torch."""

import copy
import pytest
import torch
from torchcst import CSTAdamW


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_ordinary_parameters_and_resume_match_pytorch(amsgrad, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("actual CUDA required")
    with torch.random.fork_rng():
        torch.manual_seed(912)
        ref = torch.nn.Linear(7, 3).to(device)
    live = copy.deepcopy(ref)
    options = dict(
        lr=0.003, betas=(0.8, 0.95), weight_decay=0.01, amsgrad=amsgrad, foreach=False
    )
    expected = torch.optim.AdamW(ref.parameters(), **options)
    actual = CSTAdamW(live.parameters(), model=live, **options)
    for step in range(3):
        rng = torch.Generator().manual_seed(151 + step)
        for a, b in zip(ref.parameters(), live.parameters()):
            g = torch.randn(a.shape, generator=rng).to(device)
            a.grad = g.clone()
            b.grad = g.clone()
        expected.step()
        actual.step()
        for a, b in zip(ref.parameters(), live.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            for k, v in expected.state[a].items():
                torch.testing.assert_close(v, actual.state[b][k], atol=0, rtol=0)
    restored = copy.deepcopy(live)
    resumed = CSTAdamW(restored.parameters(), model=restored, **options)
    resumed.load_state_dict(copy.deepcopy(actual.state_dict()))
    for a, b in zip(live.parameters(), restored.parameters()):
        g = torch.ones_like(a) * 0.002
        a.grad = g.clone()
        b.grad = g.clone()
    actual.step()
    resumed.step()
    for a, b in zip(live.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for k, v in actual.state[a].items():
            torch.testing.assert_close(v, resumed.state[b][k], atol=0, rtol=0)
