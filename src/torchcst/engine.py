"""torchcst.engine — 三角形の配線と Executor (ユーザーが触る唯一の入口)。"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

from .compute.linear import CSTLinear
from .contracts import EntityStore, Instrument, Op, Policy, Reading, View
from .storage.synapse import SynapseStore


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
      - step(): schedule 発火 → decide → op を site でルーティングし
        store.apply (P4: version++, op ログ, 派生キャッシュ無効化)
      - 分散: rank0 で decide → op broadcast → 全 rank 同一適用
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

        # ── instruments 配線: policy.instruments() を bind ─────────────
        self._instruments: dict[str, Instrument] = {}
        self._instruments_by_site: dict[str, list[str]] = {}
        for name, (inst, site) in policy.instruments().items():
            if site not in self._stores:
                raise ValueError(
                    f"policy instrument {name!r} is bound to unknown site {site!r}"
                )
            inst.bind(self._stores[site])
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
        """optimizer.step() の後に呼ぶ。適用 op を返す (ロギング用)。"""
        phases = self.policy.schedule(self._step)
        applied: list[Op] = []
        for phase in phases:
            readings: dict[str, Reading] = {
                name: inst.read() for name, inst in self._instruments.items()
            }
            views: dict[str, View] = {
                site: store.view() for site, store in self._stores.items()
            }
            ops = self.policy.decide(phase, readings, views, self._rng)
            for op in ops:
                store = self._stores.get(op.site)
                if store is None:
                    raise KeyError(f"op targets unknown site {op.site!r}")
                store.apply(op)  # P4: version++, engine は op の中身を見ない
                applied.append(op)
                self._op_log.append((store.version, op))
        self._step += 1
        return applied

    def op_log(self) -> list[tuple[int, Op]]:
        return list(self._op_log)

    def save(self, path: str) -> None:   # P5 正準形
        raise NotImplementedError("v0: CSTEngine.save not implemented")

    def load(self, path: str) -> None:
        raise NotImplementedError("v0: CSTEngine.load not implemented")
