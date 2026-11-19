import pytest
import torch
from torch import nn

from torchcst import CSTLinear, TriweightSpec, chart_presets, kernels
from torchcst.hub import from_pretrained, load_safetensors

save_file = pytest.importorskip("safetensors.torch").save_file


def build_model():
    chart = chart_presets.regular_grid((2, 2))
    layer = CSTLinear(
        charts=(chart, chart),
        atoms=5,
        kernel=kernels.product(kernels.point(TriweightSpec(), sigma=1.5)),
        backend="factored",
    )
    return nn.Sequential(layer, nn.Linear(4, 2))


@pytest.fixture
def weights(tmp_path):
    model = build_model().eval()
    state = model.state_dict()
    assert any(not isinstance(value, torch.Tensor) for value in state.values())
    tensors = {
        name: value.contiguous()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    return model, path, tensors


def test_load_safetensors_rebuilds_cst_model_for_inference(weights):
    expected, path, _ = weights
    model = load_safetensors(
        path, model_factory=build_model, device=torch.device("cpu")
    )
    inputs = torch.rand(3, 4)
    torch.testing.assert_close(model(inputs), expected(inputs))
    assert all(not layer.training for layer in model.modules())
    assert model[0].backend == "factored"


@pytest.mark.parametrize("mismatch", ["missing", "extra", "shape", "dtype"])
def test_load_safetensors_rejects_incompatible_tensor_state(weights, mismatch):
    _, path, tensors = weights
    if mismatch == "missing":
        del tensors["1.weight"]
    elif mismatch == "extra":
        tensors["extra"] = torch.ones(1)
    elif mismatch == "shape":
        tensors["1.weight"] = torch.ones(2, 3)
    else:
        tensors["1.weight"] = tensors["1.weight"].double()
    save_file(tensors, path)
    with pytest.raises(ValueError, match="Safetensors"):
        load_safetensors(path, model_factory=build_model)


def test_from_pretrained_downloads_weights_for_the_factory(weights, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    expected, path, _ = weights

    def download(**kwargs):
        assert kwargs == {
            "repo_id": "owner/model",
            "filename": "variant.safetensors",
            "revision": "commit",
            "token": False,
        }
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    model = from_pretrained(
        "owner/model",
        model_factory=build_model,
        filename="variant.safetensors",
        revision="commit",
        token=False,
    )
    torch.testing.assert_close(model[0].atoms.p, expected[0].atoms.p)
