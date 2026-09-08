"""Keep measurement-boundary syncs out of transfer attribution."""

import json

import pytest

from experiments.tangent_transfer_profile import summarize_trace, union_us


def event(name, category, start, duration, **args):
    return {
        "name": name,
        "cat": category,
        "ts": start,
        "dur": duration,
        "ph": "X",
        "args": args,
    }


def test_trace_uses_launch_correlation_and_excludes_measurement_drain(tmp_path):
    events = [
        event("torchcst::pcg", "user_annotation", 10, 10),
        event("aten::_local_scalar_dense", "cpu_op", 11, 3, **{"External id": 7}),
        event(
            "cudaMemcpyAsync",
            "cuda_runtime",
            12,
            1,
            correlation=8,
            **{"External id": 7},
        ),
        event("cudaStreamSynchronize", "cuda_runtime", 13, 1),
        event("cudaLaunchKernel", "cuda_runtime", 15, 1, correlation=9),
        # GPU completion occurs after the CPU scope closes, but belongs to it.
        event(
            "Memcpy DtoH (Device -> Pinned)",
            "gpu_memcpy",
            21,
            1,
            correlation=8,
            bytes=4,
        ),
        event("test_kernel", "kernel", 22, 3, correlation=9),
        event("measurement::drain", "user_annotation", 20, 7),
        event("cudaDeviceSynchronize", "cuda_runtime", 20, 7),
        event("unrelated_kernel", "kernel", 10, 3, correlation=100),
        event("Memcpy HtoD", "gpu_memcpy", 11, 1, correlation=101, bytes=1024),
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    result = summarize_trace(trace, "pcg")
    assert result["gpu_kernel_count"] == 1
    assert result["gpu_kernel_union_us"] == 3
    assert "cudaDeviceSynchronize" not in result["cuda_runtime"]
    assert result["cuda_runtime"]["cudaStreamSynchronize"]["count"] == 1
    assert len(result["gpu_transfer"]) == 1
    assert result["transfer_attribution"] == [
        {
            "transfer": "Memcpy DtoH (Device -> Pinned)",
            "cpu_op": "aten::_local_scalar_dense",
            "bytes_each": 4,
            "count": 1,
        }
    ]


def test_missing_cuda_trace_does_not_claim_zero_transfers(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"traceEvents": [event("torchcst::pcg", "user_annotation", 0, 10)]})
    )
    with pytest.raises(RuntimeError, match="No correlated CUDA kernels"):
        summarize_trace(trace, "pcg")


def test_kernel_busy_time_merges_overlaps():
    assert (
        union_us(
            [
                event("a", "kernel", 1, 4),
                event("b", "kernel", 2, 1),
                event("c", "kernel", 4, 3),
                event("d", "kernel", 9, 2),
            ]
        )
        == 8
    )
