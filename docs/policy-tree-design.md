# Policy Tree — v2 API 設計ノート（草案）

起草: 2026-07-29（ユーザーとの設計議論より）。ステータス: **草案・実装前**。
現行 API（composed `Policy` + site 文字列 + `StructuralEngine`）はベンチ走行中のため凍結。
本ノートは次期 API の合意形成用。

## 動機（観測された問題）

1. **可読性**: `engine.py` は capture・計器束縛・site 辞書引き・原子的ユニット組立てが
   絡み、「1つの層で何が起きるか」を読むのに複数箇所を往復する。参照の局所性がない。
2. **site 文字列**: op と store の対応付けが実行時の文字列住所＋型検証で、
   構築時に間違えられない設計になっていない。
3. **密結合の危険**: 調整機構（rent 経済 / quota 分配）を engine に持たせると、
   それを使わない構成にとって恒久的な死荷重になる（設計議論でユーザーが指摘）。

## 設計: ポリシーの木（composite）

```
engine = StructuralEngine(model, root_policy)   # engine は根しか知らない

root = RentEconomy(lam=0.05, cadence=Periodic(25), children={
    layer1:      SynapseLifecycle(birth=ScoredBirth(...), absorb=AbsorbCourt(...)),
    layer2:      SynapseLifecycle(...),
    iface_1_2:   NeuronLifecycle(...),          # neuron は「層の間」の子ノード
})
# 差し替え可能な根の例:
#   QuotaRegime(budget=64, children=...)   — DST 風の中央分配
#   Independent(children=...)              — 調整なし・各層素通し
```

### 原則

1. **engine は機構のみ**: clock を刻む → 根に「このイベントの op は？」と聞く →
   返った op 列をカスケード展開・原子的 commit・optimizer follower・監査。
   economy も quota も site も知らない。
2. **調整の意味論は根ノードの種類が決める**: RentEconomy は値札つき提案を λ で
   フィルタ、QuotaRegime は枠で切る、Independent は素通し。
   **使わない機構は木に存在しない**（死荷重ゼロ）。
3. **提案は上に昇り、裁定は下に降りる**: 子は自分の store だけを見て
   priced proposal（op ＋ 期待利得/費用）を親に出す。親が跨層裁定して確定 op 列を
   engine へ。op は今までどおり**データ**（関数ではない）——原子性・監査・
   optimizer 手術は op が検査可能なデータであることに依存する。この不変量は維持。
4. **対応付けは構築時所有**: 子ポリシーは自分の store への参照を持つ。
   site 文字列はユーザー API から消える（内部実装で使うのは可）。
5. **理論との同型**: J_λ = L + λK の λK 項 = RentEconomy ノード、
   固定予算制約 = QuotaRegime ノード。目的汎関数の構造 = 木の構造。
   実験の構造軸の水準とコードが1対1対応する。

### neuron の扱い

neuron は隣接2層の synapse に跨る「界面」の実体なので、その policy も界面の
子ノードとして木に住む。NeuronRetire → 接続 synapse 死のカスケードは現行どおり
engine が原子的プランに展開する（`_expand_retirements` 相当は機構側に残る）。

### 経済の教訓（toy 実証済み）

RENT-T2 で「quota 無圧力でも rent だけで K が自己調整する」ことを実証済み。
つまり**価格（λ 共有）だけで層間調整が済む**ので、rent 構成では中央分配者が
不要になり、per-layer 分権と大域整合が両立する。quota 方式が中央 distributor を
要するのと対照的（このアーキテクチャ選択は理論の主張そのもの）。

## 移行路（compile-down）

storage 層（SlotPool / follower / op 型 / Neuron・SynapseStore）は無変更。
書き換わるのは policy 合成と engine の中間層のみ。第1段は**木 → 現行 composed
Policy への畳み込み adapter** として実装し、257 本の既存テストとベンチを
壊さずに新 API の書き味だけ先に提供する。第2段で engine 内部を木ネイティブに。

## 未決の設計論点（ユーザー裁定待ち）

1. **cadence の所有**: 根だけが持つ（子は間引きのみ許可）か、子も独自 cadence を
   持てるか。起草者の推奨は前者（規律・再現性・監査の単純さ）。
2. **異種の根の混在**: 半分 rent・半分 quota のような混成を許すか。
   許すなら「調整しない親（Independent）の下に異種の子の木」をぶら下げる形が自然。
3. **計器の宣言**: 現行の `requires`（前宣言・engine が束縛）を木でどう表現するか。
   子が自分の requires を持ち、根が合算して engine に渡す、が素直。

## 現行実装との対応表

| 現行 | v2 |
|---|---|
| composed `Policy`（cadence+actions+quota） | 深さ1の木の特殊形 |
| `op.site` 文字列 + 実行時型検証 | 構築時所有（子→store 参照） |
| `EvenBudgetDistributor` / `QuotaShare` | `QuotaRegime` 根ノードの内部 |
| `AbsorbCourt(rent=λ)` を action ごとに設定 | λ は `RentEconomy` 根が一元保持し子に降ろす |
| `neuron_retention_rule` | 界面子ノードの court |
