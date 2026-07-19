"""torchcst.compute.linear — 横断合成: synapse と neuron が出会う場所。"""

from __future__ import annotations

from typing import Callable

from torch import Tensor, nn

from ..contracts import EntityStore, Kernel, Observation
from ..storage.neuron import NeuronStore
from ..storage.synapse import SynapseStore


class CSTLinear(nn.Module):
    """in/out の NeuronStore × SynapseStore × Kernel の合成。

    forward は View のみから:
      y = gate_out ⊙ (((x ⊙ gate_in) @ K_in) * w) @ K_out.T
      K_in = κ_in(μ_in ⊖ s) [N_in,K], K_out = κ_out(μ_out ⊖ t) [N_out,K]
    (N_in×N_out 非実体化)。View は store.version が動いた時のみ再取得。

    backward hook は Observation を publish する「だけ」— この module は
    Op を発行できない (三角形の辺制約)。hook 内は per-item 同期禁止・
    バッチ一括ベクトル演算のみ。
    """

    def __init__(self, in_neurons: NeuronStore, out_neurons: NeuronStore,
                 synapses: SynapseStore, kernel_in: Kernel,
                 kernel_out: Kernel | None = None):
        """kernel_out=None なら kernel_in を両側共有。in/out に同じ
        NeuronStore を渡せば座標共有、別 store なら分離。"""
        super().__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.synapses = synapses

        shared = kernel_out is None
        self.kernel_in = kernel_in
        # kernel_out=None: kernel_in を両側共有。nn.Module の attribute には
        # 実体を一度だけ持たせれば十分だが、名前を分けて代入しても
        # parameters() は id ベースの memo で重複排除されるので二重計上
        # の心配はない。
        self.kernel_out = kernel_in if shared else kernel_out

        self.kernel_in.install(self.synapses)
        if not shared:
            self.kernel_out.install(self.synapses)

        # Decision へ Observation を publish する口。Engine 未実装の v0 では
        # 誰も繋がないので None のままで良い。この module 自身は Op を
        # 発行できない (三角形の辺制約)。
        self.on_observation: Callable[[Observation], None] | None = None

    def forward(self, x: Tensor) -> Tensor:
        in_view = self.in_neurons.view()
        out_view = self.out_neurons.view()
        syn_view = self.synapses.view()

        k_in = self.kernel_in(in_view.mu, syn_view.s, syn_view.extras)    # [N_in, K]
        k_out = self.kernel_out(out_view.mu, syn_view.t, syn_view.extras)  # [N_out, K]

        gate_in = in_view.gate
        x_gated = x * gate_in if gate_in is not None else x

        h = (x_gated @ k_in) * syn_view.w   # [B, K]  (N_in×N_out 非実体化)
        y = h @ k_out.t()                    # [B, N_out]

        gate_out = out_view.gate
        if gate_out is not None:
            y = y * gate_out

        if self.on_observation is not None:
            site = self.synapses.site
            version = syn_view.version
            x_detached = x.detach()

            def _publish(grad_output: Tensor,
                         _x: Tensor = x_detached, _v: int = version) -> None:
                # バッチ一括のみ・per-item 同期禁止。detach して渡すだけ。
                self.on_observation(
                    Observation(site=site, x=_x, g_out=grad_output.detach(),
                                version=_v)
                )

            y.register_hook(_publish)

        return y

    def stores(self) -> list[EntityStore]:
        return [self.in_neurons, self.out_neurons, self.synapses]
