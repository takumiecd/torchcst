# Fast-Construction family（cSFW / cVP / cELM）— policy 契約ノート

起草: 2026-07-30。ステータス: **実装済み（2026-08-01）**。以下は起草時の契約で、
実装で確定した差分は各節の「実装（2026-08-01）」注記に記す。実装が正であり、
凍結既定値とその根拠は [`family-implementation-brief.md`](family-implementation-brief.md)
（§2 の設計入力表と §6 の実装判断）が持つ。
実験側の事前登録: `cst/docs/roadmap/FAST_CONSTRUCTION.md`（FASTCON アーク）。
前提: `policy-tree-phase2.md`（一直線 protocol stack）・`policy-tree-design.md`
（family 密結合裁定・語彙3段）。

実装の所在: `policy/fast_construction.py`（birth/refit 規則）・
`policy/families.py`（`cVP` / `cSFW` / `gamma_ungate`）・
`policy/neuron_growth.py`・`instruments/tangent.py`・`instruments/gate.py`・
`storage/synapse.py`（`SynapseRefit`）・`policy/recipes.py`（`FastConstruction`）。
機構テストは `tests/torchcst/test_fast_construction_*.py` と
`test_gamma_ungate.py` / `test_refit_*.py`。

## 実装ゲート（着手条件・両方必要 — 両方とも解除済み）

1. FASTCON FC-0（standalone 1層実験）で勝者ダイヤルが確定していること。
   → 2026-07-31 完了（勝者 cSFW(polish=0, backfit=K/10)、FC-1 の出荷ゲートは cVP 経路）。
2. policy-tree Phase 2 の **S2（子の実行時化）完了**。
   削除予定の composed `Policy` 経路にはこの family を書かない。
   → Phase 2 は S0..S6b まで完了・main マージ済み。実装は木の経路のみ。

## 一直線スタック上の座席

```
engine   変更なし。capture サービスに「イベント時 probe batch」の供給が載るだけ
   │     （requires 宣言経由。engine は用途を知らない）。
root     RentEconomy（既存 family）。cSFW の子は Newton フィット後の残差減少
   │     という解析的値札を持つので priced proposal を出せる＝ λ フィルタに
   │     そのまま乗る。backfit の cadence（何 birth ごとに Refit を許すか）は
   │     root の cadence 所有に含める（phase2 裁定 3 と同じ原理）。
children FastConstruction 子（SynapseLifecycle の新しい方法合成）。
   │     誕生: 残差ピークシード → 低次元 GN（≤6 次元、trust region）→
   │     値札付き birth proposal（パラメータ確定済み: 位置・σ・w）。
   │     返済: Refit proposal（全 w 一括再 solve ＋ 位置 round-robin GN の結果表）。
storage  op 種別 **Refit を 1 個追加**（下記）。他は無変更。
```

**実装（2026-08-01）**: engine は *完全に* 無変更で済んだ。イベント時 probe batch
の供給は不要で、backward 窓に貯めた `(C, Σx)` 十分統計量（`instruments/tangent.py`）
だけで誕生も backfit も回る。root は `RentEconomy`（`rent=λ` を値札閾に使う）でも
`QuotaRegime`（+ opt-in `ProfitCourt`）でも動く。

## 方法コンストラクタ（語彙3）

```python
cSFW(polish_iters=3, backfit="K/10", seed="residual_peak", trust=...)
    # → SynapseLifecycle(birth=NewtonBirth(...), refit=Backfit(...))
cVP(head="irls", trust=...)
    # → w はイベント時 solve・位置/σ は SGD 続行（Refit の w 部分のみ使用）
cELM(head="irls")
    # → 対照。置くだけ＋ヘッド solve
```

引数スキーマは family 所有（フレームワークは固定しない）。λ と cadence は
root 所有で方法には渡さない（引数規律どおり）。

**実装（2026-08-01）— 上の擬似シグネチャは起草時のもの。実物は:**

```python
cVP(ridge=1e-4, timing="after_backward")
    # 毎ルートイベントで全振幅を tangent Gram solve → 1 個の SynapseRefit。
    # 位置は通常の SGD パラメータのまま（head="irls" は未実装）。
cSFW(polish_iters=0, backfit="K/10", pool_size=4096, multistart=4,
     trust=0.01, ridge=1e-4, ...)
    # 純成長 birth（イベント内 rank-1 deflation 付き逐次選択）＋周期 backfit。
    # seed= 引数はなく、シードは残差プロファイルのピーク（多点スタート）で固定。
gamma_ungate(curvature_floor=1e-12)   # NeuronLifecycle。独立 γ 誕生（bundle なし）
```

- `polish_iters` の既定は起草時の `3` ではなく **`0`**（FC-0/FC-1 の壁時計勝者。
  縮約 Newton は `polish_iters>0` と `backfit_position_iters>0` で使える部品として残る）。
  下の戦死者台帳 ONESHOT-1 の項も参照。
- **`cELM` は未実装**。FC-0/1 で K=128–192 を要する対照アームであり、出荷経路に
  入らなかった。必要になったら standalone 側（`cst`）で先に測ること。

## op 語彙の追加: Refit（唯一の storage 変更）

- 内容: 既存原子群のパラメータ一括更新。`{atom_id: (w', pos', σ')}` の**表を運ぶ
  データ**。関数は運ばない — 「op はデータ」の不動軸を維持し、監査・replay・
  follower は commit 前に全容を見られる。
- 実行: storage の版つき二相 mutation にそのまま乗る（振幅を書き換える absorb と
  同カテゴリのパラメータ mutation）。optimizer follower には位置・σ の状態
  リセット規則を定義する（w は solve 値なので moment リセット、位置は保持、が
  起草者案 — FC-0 の勝者設定を見て確定）。
- 裁定は root: Refit も PricedProposal（返済で減る損失 − 計算コスト）として昇り、
  λ フィルタを通る。無料自動適用にはしない（rent 経済との同型を保つ）。

**実装（2026-08-01）**:

- 表は `SynapseRefit(site, ids, w, s=None, t=None)`。`w` は必須、`s`/`t` は
  省略時に既存座標を保存する任意列。**`σ'` 列は実装しない** — 現行 representation
  の σ は原子行ではなく compute 側 factor が持つ共有 parameter で、storage op から
  書くと一直線依存と「唯一の storage 変更」に反する。原子別 bandwidth を
  representation/storage 契約として導入するまで保留（brief §6）。
- follower 規則は起草者案どおり確定: **w は moment リセット・座標は保持**
  （`OptimizerStateFollower.on_refit`、`test_refit_optimizer_following.py`）。
- 同一チケット内で死ぬ原子を Refit の対象にはできない（prepare で拒否）。
  二相 atomic と rollback は `test_refit_atomicity.py`。

## capture / 計器要件（family 私有）

- イベント時 probe batch 上の残差 r（CE では per-sample (g, h)）。
- 残差の**増分更新** r ← r − w·φ_new は子の私有状態（誕生の連打を full forward
  なしで回すための帳簿）。commit 確定通知（follower 経路）で同期する。
- 既存の R_{t-1}=0 窓 ΣG accumulator（実装済み計器）は cVP の tangent LS 側で使う。
- これらは profit trial と同族の「実測による確定」＝ root 裁定内・family 私有。
  engine に持ち上げない（「演算ごとに処理が異なるものは共通層に上げない」裁定）。

**実装（2026-08-01）**: probe batch は**不要だった**。誕生も backfit も、backward 窓に
貯めた符号付き `(C, Σx)`（`TangentStatisticsRequest`、既存 dual-timing 契約に素直に乗る）
だけで回る。連打の残差 deflate は follower 通知ではなく**イベント内**の rank-1 更新
（`TangentBirth` 私有・commit をまたがない）。ニューロン成長は `∂L/∂γ` と対角曲率を
休眠行でも厳密に取る別計器（`GateTangentRequest`）。両計器とも `abs()` は計測内で
取らず、微小バッチ重み付き和の**後**に非線形化する（勾配累積の正しさ）。

## 型レベルの表現

- cSFW / cVP は値札を出せる → RentEconomy の子になれる。
- cELM は値札なし（無条件 birth）→ QuotaRegime の子としてのみ許す。
  「cSET は RentEconomy の子になれない」と同じ形の、不正な組合せが書けない設計。

**実装（2026-08-01）**: `cVP` / `cSFW` はともに `priceable=True` で、根が渡す λ が
birth/refit の解析的 gain 閾になる（`rent=lam`）。cELM 未実装のため「値札なし方法」の
型は現状ぶら下がっていない。

## 戦死者台帳との整合（レビュー時チェックリスト)

- solve はイベント時のみ（DEEP-3: 毎 step solve 禁止）。
- ~~磨きゼロの構成は出荷しない（ONESHOT-1: `polish_iters=0` は実験の負対照専用）~~
  → **2026-07-31 の FC-0/FC-1 で反証**。位置の2階 GN は必要条件だが、支払いは
  **誕生時ではなく backfit 後払い一括**が最適（裁定1の実証）。したがって出荷既定は
  `polish_iters=0` ＋ 周期 backfit であり、これは「磨かない」ではなく
  「磨きを後払いにする」。誕生時磨きゼロ**かつ** backfit なし（`backfit=None`）の
  構成が ONESHOT-1 の負対照であり、それは今も出荷しない。
- 選択は逐次＋残差 deflate（cRigL: batch top-K のコピー病理）。
  → 実装は `TangentBirth` のイベント内逐次 deflation。
- GN/Newton は必ず trust region 付き。Taylor 2 次で打ち止め。
  → 実装の `_polish` は Levenberg 減衰＋trust 半径＋座標 6 次元上限。
