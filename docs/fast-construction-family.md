# Fast-Construction family（cSFW / cVP / cELM）— policy 契約ノート

起草: 2026-07-30。ステータス: **設計のみ（実装ゲート付き）**。
実験側の事前登録: `cst/docs/roadmap/FAST_CONSTRUCTION.md`（FASTCON アーク）。
前提: `policy-tree-phase2.md`（一直線 protocol stack）・`policy-tree-design.md`
（family 密結合裁定・語彙3段）。

## 実装ゲート（着手条件・両方必要）

1. FASTCON FC-0（standalone 1層実験）で勝者ダイヤルが確定していること。
2. policy-tree Phase 2 の **S2（子の実行時化）完了**。
   削除予定の composed `Policy` 経路にはこの family を書かない。

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

## capture / 計器要件（family 私有）

- イベント時 probe batch 上の残差 r（CE では per-sample (g, h)）。
- 残差の**増分更新** r ← r − w·φ_new は子の私有状態（誕生の連打を full forward
  なしで回すための帳簿）。commit 確定通知（follower 経路）で同期する。
- 既存の R_{t-1}=0 窓 ΣG accumulator（実装済み計器）は cVP の tangent LS 側で使う。
- これらは profit trial と同族の「実測による確定」＝ root 裁定内・family 私有。
  engine に持ち上げない（「演算ごとに処理が異なるものは共通層に上げない」裁定）。

## 型レベルの表現

- cSFW / cVP は値札を出せる → RentEconomy の子になれる。
- cELM は値札なし（無条件 birth）→ QuotaRegime の子としてのみ許す。
  「cSET は RentEconomy の子になれない」と同じ形の、不正な組合せが書けない設計。

## 戦死者台帳との整合（レビュー時チェックリスト)

- solve はイベント時のみ（DEEP-3: 毎 step solve 禁止）。
- 磨きゼロの構成は出荷しない（ONESHOT-1: `polish_iters=0` は実験の負対照専用）。
- 選択は逐次＋残差 deflate（cRigL: batch top-K のコピー病理）。
- GN/Newton は必ず trust region 付き。Taylor 2 次で打ち止め。
