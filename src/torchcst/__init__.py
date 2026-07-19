"""torchcst — Continuous Sparse Training in PyTorch.

構造 = 三角形 × entity 縦割り:

               synapse column          neuron column
  Storage   │ SynapseStore + ops   │ NeuronStore + ops   │ ← 縦割り(自由)
  計器      │ synapse Instruments  │ neuron Instruments  │ ← 縦割り
  ──────────┼──────────────────────┴─────────────────────┤
  Policy    │   横断: 全計器の読み値 → 協調 op バッチ       │
  Compute   │   横断: CSTLinear = synapse × neuron の合成   │

  Storage ──View──→ Compute ──Observation──→ Decision ──Op──→ Storage

辺の型は View / Observation / Op の 3 つだけ (契約面は contracts.py)。

ストレージ第一原理:
  P1. 置換不変 — slot 順に意味なし。同一性は id のみ。
  P2. id は int64・never-reuse・単調増加・上位ビット rank。
  P3. per-atom 付随状態 (moment/計器) は Follower として mutation に自動追従。
  P4. mutation は store へのトランザクション。version 単調増加 + op ログ。
      分散は op ログの broadcast 同一適用 (テンソル同期なし)。
  P5. checkpoint は正準形 (compact → id ソート) でバイト決定的。
"""

from . import decision as policies
from .compute import CSTConv2d, CSTLinear, GaussianKernel, TriangularKernel
from .decision import CandidateReading, MassEMA
from .contracts import (
    DecisionContext,
    DecisionStage,
    EntityStore,
    Follower,
    Instrument,
    Kernel,
    KernelPort,
    Observation,
    Op,
    Policy,
    Reading,
    View,
)
from .engine import CSTEngine
from .storage import (
    FollowerHub,
    IdAllocator,
    NeuronBirth,
    NeuronDeath,
    NeuronKick,
    NeuronStore,
    NeuronView,
    SlotPool,
    SynapseBirth,
    SynapseDeath,
    SynapseKick,
    SynapseMerge,
    SynapseStore,
    SynapseView,
)

__version__ = "0.0.1"

__all__ = [
    # contracts
    "View", "Observation", "Op",
    "EntityStore", "Follower", "Kernel", "KernelPort", "Instrument", "Reading",
    "Policy", "DecisionContext", "DecisionStage",
    # storage
    "IdAllocator", "SlotPool", "FollowerHub",
    "SynapseStore", "SynapseView",
    "SynapseBirth", "SynapseDeath", "SynapseMerge", "SynapseKick",
    "NeuronStore", "NeuronView", "NeuronBirth", "NeuronDeath", "NeuronKick",
    # compute
    "CSTLinear", "CSTConv2d", "GaussianKernel", "TriangularKernel",
    # decision / engine
    "policies", "MassEMA", "CandidateReading", "CSTEngine",
]
