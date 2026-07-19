"""torchcst — Continuous Sparse Training in PyTorch.

構造 = Engine × Forward × Backward × Policy × Storage:

               synapse column          neuron column
  Storage   │ SynapseStore + ops   │ NeuronStore + ops   │ ← 縦割り(自由)
  観測状態  │ synapse observers    │ neuron observers    │ ← Policy所有
  ──────────┼──────────────────────┴─────────────────────┤
  Policy    │   prepare/capture/step → MutationPlan          │
  Compute   │   横断: CSTLinear = synapse × neuron の合成   │

  backward: PyTorch hook ─ModuleGradRecord→ Policy-owned observer
  step:     Schedule ─UpdateRequest→ Policy ─MutationPlan→ Storage

ストレージ第一原理:
  P1. 置換不変 — slot 順に意味なし。同一性は id のみ。
  P2. id は int64・never-reuse・単調増加・上位ビット rank。
  P3. slot-indexed付随状態 (optimizer moment等) はFollowerとしてmutationに追従。
  P4. mutation は store へのトランザクション。version 単調増加 + op ログ。
      分散は op ログの broadcast 同一適用 (テンソル同期なし)。
  P5. checkpoint は正準形 (compact → id ソート) でバイト決定的。
"""

from . import policy as policies
from .forward import CSTConv2d, CSTLinear, GaussianKernel, TriangularKernel
from .policy import CandidateScores, IdScores, MassEMA, Policy
from .policy import PolicyBinding, ReadPort
from .policy import cRigL, cSET
from .policy import Clock, PeriodicSchedule, UpdateRequest, UpdateSchedule
from .storage.base import EntityStore, Follower, Op, View
from .engine import CSTEngine
from .storage import (
    BalancedSlotPool,
    FollowerHub,
    IdAllocator,
    NeuronBirth,
    NeuronDeath,
    NeuronKick,
    NeuronStore,
    NeuronView,
    SlotPool,
    SlotBackend,
    SlotChange,
    SynapseBirth,
    SynapseDeath,
    SynapseKick,
    SynapseMerge,
    SynapseStore,
    SynapseView,
)

__version__ = "0.0.1"

__all__ = [
    # contracts / policy lifecycle
    "View", "Op", "EntityStore", "Follower", "Policy",
    "PolicyBinding", "ReadPort", "Clock", "UpdateRequest",
    "UpdateSchedule", "PeriodicSchedule",
    # storage
    "IdAllocator", "SlotBackend", "SlotChange",
    "SlotPool", "BalancedSlotPool", "FollowerHub",
    "SynapseStore", "SynapseView",
    "SynapseBirth", "SynapseDeath", "SynapseMerge", "SynapseKick",
    "NeuronStore", "NeuronView", "NeuronBirth", "NeuronDeath", "NeuronKick",
    # compute
    "CSTLinear", "CSTConv2d", "GaussianKernel", "TriangularKernel",
    # decision / engine
    "policies", "MassEMA", "IdScores", "CandidateScores", "cSET", "cRigL",
    "CSTEngine",
]
