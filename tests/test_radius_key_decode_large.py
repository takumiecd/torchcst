"""Exercise the actual native owner-key decoder at its unsigned32 boundary."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_actual_native_owner_key_decode_matches_independent_int64(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    native = (
        Path(__file__).resolve().parents[1]
        / "src/torchcst/_backends/cuda/algorithms/linear/radius_cell_mass"
    )
    text = (native / "owner_key_masks.cuh").read_text()
    begin = text.index("__device__ inline uint32_t key_moment_dependencies(")
    end = text.index("\n__global__ void key_moment_banks_kernel", begin)
    actual_decoder = text[begin:end]
    source = r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
#include "limits.cuh"
#define __device__

namespace radius_cell_mass {
struct Axis { int node_count, marker; };
int decode_calls = 0;
int bank_for_node(int node, const Axis &axis) {
    ++decode_calls;
    assert(node >= 0 && node < axis.node_count);
    return node + axis.marker;
}
uint32_t moment_bank_dependencies(int decoded_node, int, bool) {
    // Expose the selected axis and decoded node instead of forming bank bits.
    return uint32_t(decoded_node);
}
""" + actual_decoder + r"""
} // namespace radius_cell_mass

int main() {
    using namespace radius_cell_mass;
    int valid_checks = 0, rejected_checks = 0, valid_unsigned_checks = 0;
    for (int sites : {1024, 2048, 8192, 16384}) {
        Axis input{4 * sites - 2, 100000};
        Axis output{4 * sites - 2, 200000};
        const int64_t count = int64_t(input.node_count) * output.node_count;
        std::vector<int64_t> keys{
            0, 1, int64_t(input.node_count) - 1, input.node_count,
            int64_t(std::numeric_limits<int32_t>::max()) - 1,
            std::numeric_limits<int32_t>::max(),
            int64_t(std::numeric_limits<int32_t>::max()) + 1,
            count - 1, -1, count, count + 1,
            std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max()
        };
        for (int64_t key : keys) {
            for (bool output_side : {false, true}) {
                bool valid = true;
                decode_calls = 0;
                const uint32_t got = key_moment_dependencies(
                    key, input, output, output_side, 2, true, valid);
                const bool expected_valid = key >= 0 && key < count;
                assert(valid == expected_valid);
                if (!expected_valid) {
                    // Invalid 64-bit input must not reach node decoding.
                    assert(got == 0 && decode_calls == 0);
                    ++rejected_checks;
                    continue;
                }
                const int64_t expected_node = output_side
                    ? key / int64_t(input.node_count)
                    : key % int64_t(input.node_count);
                const Axis &axis = output_side ? output : input;
                assert(got == uint32_t(expected_node + axis.marker));
                assert(decode_calls == 1);
                ++valid_checks;
                if (key > std::numeric_limits<int32_t>::max())
                    ++valid_unsigned_checks;
            }
        }
    }
    assert(valid_checks > 0 && rejected_checks > 0 && valid_unsigned_checks > 0);
}
"""
    executable = tmp_path / "owner_key_decode"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-x",
            "c++",
            "-I",
            str(native),
            "-o",
            str(executable),
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run([str(executable)], capture_output=True, text=True, check=True)
