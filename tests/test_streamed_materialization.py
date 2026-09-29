"""Streamed logical weight windows preserve mapped output after atom updates."""

import pytest
import torch
import torch.nn.functional as F

from experiments.cuda.linear.block_strip_linear import BlockStripLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_windows_and_graph_update():
    torch.manual_seed(547)
    layer = BlockStripLinear((128, 128), (64, 64), 819, device="cuda")
    x = torch.randn(128, 7, device="cuda").T
    with torch.no_grad():
        expected = F.linear(x, layer.dense_weight())
        for chunk, tile in ((64, (32, 32)), (64, (64, 32)), (128, (64, 64))):
            actual = layer(
                x,
                backend="triton_streamed",
                weight_chunk_rows=chunk,
                materialize_tile=tile,
            )
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert layer(x[:0], backend="triton_streamed").shape == (0, 128)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer(x, backend="triton_streamed", weight_chunk_rows=64)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x, backend="triton_streamed", weight_chunk_rows=64)
        layer.strip.atoms.p[:, 0] *= 0.7
        layer.strip.atoms.p[:, 2] += 0.2
        graph.replay()
        torch.testing.assert_close(
            captured, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
        )
    with pytest.raises(NotImplementedError, match="forward only"):
        layer(x, backend="triton_streamed")
