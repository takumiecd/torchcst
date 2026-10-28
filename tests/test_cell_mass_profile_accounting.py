"""Allocator evidence distinguishes requested bytes, live blocks and reservation."""

import pytest

from benchmarks.training.profile_cell_mass_mnist import replay_peak, trace_stage_costs


def test_allocator_replay_handles_address_reuse_rounding_and_pending_gpu_free():
    before = {
        "segments": [
            {
                "address": 1000,
                "segment_type": "small",
                "blocks": [
                    {"address": 1000, "size": 512, "state": "active_allocated"},
                    {"address": 1512, "size": 512, "state": "inactive"},
                ],
            },
            {
                "address": 1000000,
                "segment_type": "large",
                "blocks": [
                    {"address": 1000000, "size": 4 * 1048576, "state": "inactive"},
                ],
            },
        ]
    }
    events = [
        {"action": "snapshot"},
        {"action": "alloc", "addr": 1000000, "size": 1048576},
        {"action": "free_requested", "addr": 1000},
        {"action": "free_completed", "addr": 1000},
        {"action": "alloc", "addr": 1000, "size": 600},
        {"action": "segment_alloc", "addr": 10000000, "size": 2097152},
        {"action": "free_requested", "addr": 1000000},
    ]
    report = replay_peak(before, {"device_traces": [events]}, 1)
    assert report["bytes"] == 1048576 + 1024
    assert report["index"] == 4
    assert report["end_bytes"] == 1024
    assert [(v["address"], v["born"]) for v in report["blocks"]] == [
        (1000000, 1),
        (1000, 4),
    ]
    assert report["intervals"][0]["end"] == 2


def test_large_block_keeps_small_remainder_in_allocated_counter():
    before = {
        "segments": [
            {
                "address": 1000000,
                "segment_type": "large",
                "blocks": [
                    {"address": 1000000, "size": 3 * 1048576, "state": "inactive"},
                ],
            }
        ]
    }
    events = [{"action": "alloc", "addr": 1000000, "size": 2621440}]
    report = replay_peak(before, {"device_traces": [events]}, 0)
    assert report["bytes"] == 3 * 1048576
    assert report["blocks"][0]["logical_bytes"] == 2621440


@pytest.mark.parametrize(
    ("category", "launch"),
    [("cuda_runtime", "cudaLaunchKernel"), ("cuda_driver", "cuLaunchKernelEx")],
)
def test_active_gpu_cost_uses_launch_thread_and_avoids_nested_range_double_count(
    category, launch
):
    events = [
        {
            "cat": "user_annotation",
            "name": "stage::backward_total",
            "pid": 1,
            "tid": 2,
            "ts": 0,
            "dur": 1000,
        },
        {
            "cat": "user_annotation",
            "name": "stage::C_in",
            "pid": 1,
            "tid": 2,
            "ts": 100,
            "dur": 100,
        },
        {
            "cat": category,
            "name": launch,
            "pid": 1,
            "tid": 2,
            "ts": 110,
            "dur": 1,
            "args": {"correlation": 9},
        },
        {"cat": "gpu_user_annotation", "name": "stage::C_in", "ts": 2000, "dur": 9000},
        {
            "cat": "kernel",
            "name": "reduce",
            "ts": 2000,
            "dur": 500,
            "args": {"correlation": 9},
        },
    ]
    costs = trace_stage_costs(events, 1)
    assert costs == {"C_in": {"kernel_ms_per_step": 0.5, "kernel_calls_per_step": 1.0}}
