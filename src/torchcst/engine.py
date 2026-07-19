"""torchcst.engine — 三角形の配線と Executor (ユーザーが触る唯一の入口)。"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

from .compute.linear import CSTLinear
from .contracts import (
    DecisionContext,
    EntityStore,
    Instrument,
    KernelPort,
    Op,
    Policy,
    Reading,
    View,
)
from .storage.synapse import SynapseStore


class _LayerKernelPort:
    """CSTLinear 1 つ分の kernel_in/kernel_out・neuron 座標・gate を閉じ込め
    た読み取り専用 KernelPort。bind 時の静的配線でのみ作られる — 三角形の
    per-step の辺 (View/Observation/Op) には現れない (contracts.KernelPort
    の docstring 参照)。

    gate は毎回 view() から取り直す (in_neurons/out_neurons.view().gate)。
    これは detach された現在値でよい: 計器は「今の構造が入出力をどう伝えて
    いるか」を観測するためのものであって、gate 自体を学習する経路ではない。

    評価はすべて torch.no_grad() の中で行う: ここで autograd グラフに乗せて
    しまうと、毎 step 呼ばれる計器がグラフを蓄積し続けてメモリリークする
    (計器の統計量そのものに勾配は要らない)。
    """

    def __init__(self, module: CSTLinear):
        self._module = module

    def in_features(self, x, coords):
        with torch.no_grad():
            in_view = self._module.in_neurons.view()
            k = self._module.kernel_in(in_view.mu, coords, {})
            gate = in_view.gate
            x_gated = x * gate if gate is not None else x
            return x_gated @ k

    def out_features(self, g, coords):
        with torch.no_grad():
            out_view = self._module.out_neurons.view()
            k = self._module.kernel_out(out_view.mu, coords, {})
            gate = out_view.gate
            g_gated = g * gate if gate is not None else g
            return g_gated @ k


class _AdamStateFollower:
    """P3 追従: mutation で再利用される slot に残る optimizer momentum を
    明示的にゼロ化する Follower。

    前回 (v0 初回実装) は「param object 自体が capacity 一括確保で不変・
    行の書き換えのみだから Adam state と自動整合する」としていたが、それは
    「optimizer.state の keyed 対応が壊れない」ことの保証でしかなく、
    再利用 slot に *前任原子の momentum が残留する* 問題は別に残っていた
    (新生原子の exp_avg/exp_avg_sq が stale な値から再開してしまう)。
    これがその本修正: birth/death の両方で対象 slot 行を明示的に 0 埋めする。

    lazy init 前 (その param がまだ 1 度も optimizer.step() されておらず
    optimizer.state にエントリが無い) 場合は単にスキップする — ゼロ初期化
    される前の状態を無理に作る必要はない。
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 params: Iterable[nn.Parameter]):
        self._optimizer = optimizer
        self._params = list(params)

    def _zero_rows(self, slots: torch.Tensor) -> None:
        state = self._optimizer.state
        for p in self._params:
            st = state.get(p)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                buf = st.get(key)
                if buf is not None:
                    buf.index_fill_(0, slots, 0.0)

    def grow(self, new_capacity: int) -> None:
        raise NotImplementedError("v0: AdamStateFollower.grow not implemented")

    def on_birth(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_death(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_merge(self, src_slots: torch.Tensor, dst_slots: torch.Tensor,
                 mass: torch.Tensor) -> None:
        raise NotImplementedError("v0: AdamStateFollower.on_merge not implemented")

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        raise NotImplementedError("v0: AdamStateFollower.on_remap not implemented")


class CSTEngine:
    """配線:
      - model から stores を収集 (site 重複は即エラー)
      - policy.instruments() を bind し、backward hook で Observation を配る
      - store.parameters() を optimizer に接続し、moment 影列を Follower
        として followers() に subscribe (P3)
      - step(): DecisionStageの対象siteを反復し、site-local op batchを
        対応Storeへ一括適用 (P4: version++, opログ, 派生キャッシュ無効化)
      - 分散: rank0 で DecisionStage.run → op broadcast → 全 rank 同一適用
    Engine は op の中身も store の内部レイアウトも知らない。
    """

    def __init__(self, model: nn.Module,
                 optimizer: "torch.optim.Optimizer | Callable[[list[nn.Parameter]], torch.optim.Optimizer]",
                 policy: Policy, *, rank: int = 0, world_size: int = 1,
                 seed: int = 0):
        if world_size != 1:
            raise NotImplementedError("v0: single rank only")

        self.model = model
        self.policy = policy
        self.rank = rank
        self.world_size = world_size

        # ── store 収集 (site 重複: 異なるオブジェクトなら ValueError。
        #    同一オブジェクトの共有 — 例: 中間 NeuronStore を2層で使う — は
        #    正常なので許容する) ──────────────────────────────────────
        self._stores: dict[str, EntityStore] = {}
        # site (synapse site) → CSTLinear module。KernelPort 構築用
        # (bind 時の静的配線でのみ使う — per-step の三角形には現れない)。
        site_to_module: dict[str, CSTLinear] = {}
        for module in model.modules():
            stores_fn = getattr(module, "stores", None)
            if stores_fn is None:
                continue
            for store in stores_fn():
                site = store.site
                if site in self._stores and self._stores[site] is not store:
                    raise ValueError(
                        f"site {site!r} is bound to two different store objects"
                    )
                self._stores[site] = store
            if isinstance(module, CSTLinear):
                site_to_module[module.synapses.site] = module

        # ── param 収集: 全 store.parameters() + 全 kernel.global_params()
        #    (kernel は CSTLinear の submodule として model.modules() の
        #    走査で自然に見つかる。id ベースで重複排除するので、store 由来か
        #    module 由来かに関わらず二重登録の心配はない) ────────────────
        seen: set[int] = set()
        collected: list[nn.Parameter] = []
        for store in self._stores.values():
            for p in store.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    collected.append(p)
        for module in model.modules():
            global_params_fn = getattr(module, "global_params", None)
            if global_params_fn is None:
                continue
            for p in global_params_fn():
                if id(p) not in seen:
                    seen.add(id(p))
                    collected.append(p)

        # ── optimizer: インスタンス or factory callable の両対応
        #    (torch.optim.Adam([]) は空リストで即死するため、factory の形を
        #    正式にサポートする) ─────────────────────────────────────
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
            if collected:
                existing = {
                    id(p) for group in self.optimizer.param_groups for p in group["params"]
                }
                missing = [p for p in collected if id(p) not in existing]
                if missing:
                    self.optimizer.add_param_group({"params": missing})
        elif callable(optimizer):
            self.optimizer = optimizer(collected)
        else:
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer instance or a "
                "callable factory (params -> Optimizer)"
            )

        # ── instruments 配線: policy.instruments() を bind。site が
        #    CSTLinear の synapse site なら KernelPort を渡す (kernel アクセス
        #    が要る計器 — GradEMA/CandidateProbe — はここで受け取る)。
        #    port を構成できない site (neuron site 等) には None を渡す。──
        self._instruments: dict[str, Instrument] = {}
        self._instruments_by_site: dict[str, list[str]] = {}
        for name, (inst, site) in policy.instruments().items():
            if site not in self._stores:
                raise ValueError(
                    f"policy instrument {name!r} is bound to unknown site {site!r}"
                )
            module = site_to_module.get(site)
            port: KernelPort | None = _LayerKernelPort(module) if module is not None else None
            inst.bind(self._stores[site], port)
            self._instruments[name] = inst
            self._instruments_by_site.setdefault(site, []).append(name)

        # ── Observation 配線: 各 CSTLinear.on_observation にハンドラを設定。
        #    handler は該当 site に bind された instrument 群に
        #    instrument.update(obs, store.view()) を配る (バッチ一括;
        #    per-item 同期はしない) ────────────────────────────────────
        for module in model.modules():
            if isinstance(module, CSTLinear):
                site = module.synapses.site
                store = self._stores[site]
                names = self._instruments_by_site.get(site, [])

                def _make_handler(_names=names, _store=store):
                    def _handler(obs) -> None:
                        view = _store.view()
                        for name in _names:
                            self._instruments[name].update(obs, view)
                    return _handler

                module.on_observation = _make_handler()

        # ── AdamStateFollower を各 SynapseStore に subscribe (B の本修正) ──
        for store in self._stores.values():
            if isinstance(store, SynapseStore):
                follower = _AdamStateFollower(self.optimizer, list(store.parameters()))
                store.followers().subscribe(follower)

        self._step = 0
        self._op_log: list[tuple[int, Op]] = []
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def step(self) -> list[Op]:
        """optimizer.step() の後・次の backward の前に呼ぶ。適用 op を返す。

        順序不変条件: backward → optimizer.step() → engine.step()。
        backward と optimizer.step() の間で呼んではいけない — mutation で
        行が入れ替わった後に前住人の stale grad が適用されてしまう
        (新生原子が死んだ原子の勾配で初手更新される)。"""
        stages = self.policy.schedule(self._step)
        applied: list[Op] = []
        for stage in stages:
            readings: dict[str, Reading] = {
                name: inst.read() for name, inst in self._instruments.items()
            }
            views: dict[str, View] = {
                site: store.view() for site, store in self._stores.items()
            }
            ctx = DecisionContext(
                step=self._step,
                readings=readings,
                views=views,
                rng=self._rng,
            )

            for site in stage.sites:
                store = self._stores[site]
                site_ops = stage.run(site, ctx)
                store.apply(site_ops)  # Engine は op の中身を見ない
                applied.extend(site_ops)
                self._op_log.extend((store.version, op) for op in site_ops)
        self._step += 1
        return applied

    def op_log(self) -> list[tuple[int, Op]]:
        return list(self._op_log)

    def save(self, path: str) -> None:   # P5 正準形
        raise NotImplementedError("v0: CSTEngine.save not implemented")

    def load(self, path: str) -> None:
        raise NotImplementedError("v0: CSTEngine.load not implemented")
