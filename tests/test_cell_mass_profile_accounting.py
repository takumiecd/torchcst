"""Allocator diagnostic accounting must separate frees from reserved segments."""

from benchmarks.training.profile_cell_mass_mnist import replay_peak


def test_allocator_replay_handles_reused_addresses_and_pending_gpu_free():
    before = {
        "segments": [
            {
                "blocks": [
                    {"address": 100, "size": 512, "state": "active_allocated"},
                    {"address": 200, "size": 1024, "state": "inactive"},
                ]
            }
        ]
    }
    events = [
        {"action": "snapshot"},
        {"action": "alloc", "addr": 200, "size": 1024},
        {"action": "free_requested", "addr": 100},
        {"action": "free_completed", "addr": 100},
        {"action": "alloc", "addr": 100, "size": 2048},
        {"action": "segment_alloc", "addr": 10000, "size": 2000000},
        {"action": "free_requested", "addr": 200},
    ]
    report = replay_peak(before, {"device_traces": [events]}, 1)
    assert report["bytes"] == 3072
    assert report["index"] == 4
    assert report["end_bytes"] == 2048
    assert [(v["address"], v["born"]) for v in report["blocks"]] == [(200, 1), (100, 4)]
    assert report["intervals"][0]["end"] == 2
