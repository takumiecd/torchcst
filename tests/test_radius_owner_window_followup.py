"""Isolated fixed-DFT and raw hierarchy candidates retain the operator contract."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import test_radius_owner_window as window
import test_radius_owner_window_c_batch as c_batch
import torch
from test_radius_cell_mass_cuda import GPU

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW_C_BATCH,
    GLOBAL_OWNER_WINDOW_COMBINED,
    GLOBAL_OWNER_WINDOW_DFT_RAW,
    GLOBAL_OWNER_WINDOW_FIXED_DFT,
    GLOBAL_OWNER_WINDOW_TIGHT_RAW,
)

PLANS = [
    GLOBAL_OWNER_WINDOW_FIXED_DFT,
    GLOBAL_OWNER_WINDOW_TIGHT_RAW,
    GLOBAL_OWNER_WINDOW_COMBINED,
    GLOBAL_OWNER_WINDOW_DFT_RAW,
]


@pytest.mark.parametrize("plan", PLANS)
def test_followup_scope(plan, monkeypatch):
    monkeypatch.setattr(
        window, "PLANS", [*window.PLANS, GLOBAL_OWNER_WINDOW_C_BATCH, *PLANS]
    )
    window.test_window_scope_and_explicit_native_leaf(plan, monkeypatch)


@GPU
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_followup_vjp(plan, dtype, batch, needs_x, needs_p, skew):
    window.test_window_vjp_and_retained_offload(
        plan, dtype, batch, needs_x, needs_p, skew
    )


@GPU
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize(
    "contract",
    [
        "public_retained_hooks",
        "raw_version_once_diff",
        "no_grad",
        "inference",
        "frozen",
        "empty_atoms",
        "empty_batch",
    ],
)
def test_followup_lifetime(monkeypatch, plan, contract):
    window.test_window_live_contracts(monkeypatch, plan, contract)


@GPU
@pytest.mark.parametrize("plan", PLANS)
def test_followup_saved_payload(monkeypatch, plan):
    monkeypatch.setattr(window, "GLOBAL_OWNER_WINDOW_RECOMPUTE", plan)
    window.test_window_saved_payload(plan)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "needs_x,needs_p", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("skew", [False, True])
def test_raw_exact_optimizer_state(monkeypatch, dtype, batch, needs_x, needs_p, skew):
    monkeypatch.setattr(c_batch, "NEW", GLOBAL_OWNER_WINDOW_TIGHT_RAW)
    c_batch.test_c_batch_exact_optimizer_state(dtype, batch, needs_x, needs_p, skew)


def test_actual_raw_level_indexing_and_moments(tmp_path):
    """Execute the actual independent-thread level kernel on CPU, every bank."""
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not shutil.which(compiler[0]):
        pytest.skip("C++ compiler unavailable")
    directory = Path(ex.__file__).parent
    for name in ["common.cuh", "limits.cuh", "profile_values.cuh"]:
        (tmp_path / name).write_text(
            (directory / name).read_text().replace("#include <torch/extension.h>", "")
        )
    hybrid = (directory / "hybrid.cuh").read_text()
    phase = hybrid[
        hybrid.index("struct HarmonicPhase") : hybrid.index("inline int checked_start")
    ]
    moments = hybrid[
        hybrid.index(
            "template<class T>\n__device__ inline double compact_moment_element"
        ) : hybrid.index("template<class T>\n__global__ void compact_moments_kernel")
    ]
    new = (directory / "owner_window.cuh").read_text()
    kernel = new[
        new.index(
            "template <class T>\n__global__ void window_raw_level_kernel"
        ) : new.index("inline Tensor build_window_raw_moments")
    ]
    src = (
        r"""
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __constant__
#define __noinline__
#define TORCH_CHECK(c,...) assert(c)
inline void radius_test_sincos(double x,double*s,double*c) {*s=sin(x);*c=cos(x);}
#define sincos radius_test_sincos
struct { int x=0; } blockIdx,threadIdx;
struct { int x=128; } blockDim;
#include "common.cuh"
namespace radius_cell_mass {
inline int bank_for_node(int node,const Axis&a) {int b=0;while(b+1<a.bank_count&&node>=a.banks[b+1].node_offset)++b;return b;}
"""
        + phase
        + moments
        + kernel
        + r"""
}
using namespace radius_cell_mass;
int main() {
 Axis a=parse_axis({1024.,.37,1.,-1.2,8.,2.});
 int B=4,F=17,J=a.node_count-prefix_nodes(a,2);uint32_t required=(1u<<a.bank_count)-1;
 std::vector<double> x(B*a.sites),old(B*J*F,NAN),now=old;
 for(int i=0;i<int(x.size());++i)x[i]=sin(i*.13);
 int blocks=0;
 for(int level=2;level<a.bank_count/2;++level) {
  int first=prefix_nodes(a,level),count=prefix_nodes(a,level+1)-first;
  for(int b=0;b<B;++b)for(int node=first;node<first+count;++node)for(int f=0;f<F;++f)
   old[(b*J+node-prefix_nodes(a,2))*F+f]=compact_moment_element(x.data(),old.data(),b,node,f,a,2,true);
  int work=B*count*F;blocks+=(work+127)/128;
  for(int index=0;index<((work+127)/128)*128;++index) {
   blockIdx.x=index/128;threadIdx.x=index%128;
   window_raw_level_kernel(x.data(),now.data(),&required,B,a,first,count);
  }
 }
 assert(blocks==546);
 for(int i=0;i<int(now.size());++i) {assert(std::isfinite(now[i]));assert(memcmp(&now[i],&old[i],sizeof(double))==0);}
}
"""
    )
    path = tmp_path / "raw.cpp"
    path.write_text(src)
    binary = tmp_path / "raw"
    r = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            str(path),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
