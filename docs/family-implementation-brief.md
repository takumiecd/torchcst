# Fast-Construction Family 実装ブリーフ（外部エージェント向けハンドオフ）

作成: 2026-07-31。対象: この repo（torchcst）に fast-construction family を
実装するエージェント。**このファイルだけで作業が始められる**ように書いてある。

## 0. まず読むもの（この順で）

1. [`CLAUDE.md`](../CLAUDE.md) — repo の不変量。**違反禁止**。
2. [`docs/fast-construction-family.md`](fast-construction-family.md) — family の契約仕様（座席・op・capture 要件）。
3. [`docs/policy-tree-phase2.md`](policy-tree-phase2.md) — 一直線 protocol stack（実装済み・S0..S6b 完了）。
4. `../cst/docs/roadmap/FAST_CONSTRUCTION.md` — 事前登録＋**付録 A〜H**（数式と実験結果）。
5. 参照実装（数理のオラクル）: `../cst/scripts/run_fastcon_fc0.py` / `fc1` / `fc2`。
   各ファイルの `selftest` が「接線モデルの微分 = autograd 厳密一致」を検証する。
   結果報告: `../cst/output/fastcon_fc{0,1,2}_report.md`。

## 1. ミッション

standalone 実験（FC-0/1/2、2026-07-31 完走）で検証済みの高速構成機構を、
**policy tree の正規の住人**として実装する。実験で確定した勝者形：

- **cVP 経路**（最優先）: 振幅 w はイベント毎に solve、位置は SGD。
- **cSFW-grow**: 少数原子から、SGD を止めずに接線 birth で成長（置換ではなく成長）。
- **ニューロン成長**: gate 振幅 γ の solve 付き ungate。**独立誕生で十分**
  （束ね ProposalBundle は FC-2 で不要と実証 — conv 系まで保留）。

## 2. 実験で確定した設計入力（変更禁止の既定値）

| 項目 | 値 / 規則 | 根拠 |
|---|---|---|
| 接線 Gram solve の ridge | **1e-4**（相対） | 弱 ridge は巨大相殺 w → 位置摂動 σ/10 で崩壊（FC-1） |
| 挿入の受理 | **実測受理必須**（profit trial または減衰受理 1→0.5→0.25→0.1→0） | 学習後期の solve 済み挿入は爆弾（FC-1 追補） |
| 成長 vs 置換 | **純成長**（birth と prune を同一イベントで対にしない） | 置換＋solve 挿入は不安定（cSET-tb 崩壊） |
| 初期状態 | **小さな核から**（空の層は構成勾配が完全停止する） | 3層デッドロック（FC-2） |
| 誕生時 GN polish | 既定 **0**（縮約 Newton は部品として提供、既定オフ） | polish 0 が壁時計勝者（FC-0/FC-1） |
| ニューロン誕生の結合 | **独立誕生が既定**（bundle は書かない） | random ≒ smart（FC-2） |
| w の moment | 誕生/Refit 時にリセット、位置は保持 | family ノート起草者案どおり |

数式は FASTCON 付録 A（(C, Σx) 十分統計量・シード場・VarPro 縮約・deflation・
Gram backfit）と付録 G/H（深さの接線版・γ 場）。**式は付録が正、実装は fc スクリプトが正。**

## 3. 成果物（この順で1つずつ commit）

1. **接線計器** `instruments/` に新規: 既存の observation 契約
   （`prepare` / `measure_after_backward` または `reduce_backward` /
   `finalize_update`）で (x, g_out) から C = −G_hᵀX/N 型の統計を貯める
   frozen request + instrument。**engine への登録手続きは不要**（CLAUDE.md
   「新しい backward 統計」の項どおり）。イベント時 probe batch の供給は
   family 私有（profit trial と同族の root 裁定内サブプロトコル）とし、
   engine には持ち上げない。
2. **Refit op** `storage/` に追加（唯一の storage 変更）: `{atom_id: (w', pos'?, σ'?)}`
   の**表データ**。既存の版つき二相 mutation に乗せる。follower 規則: w は
   moment リセット、位置は保持。監査・replay が commit 前に全容を見られること。
3. **方法コンストラクタ** `policy/families.py` に `cVP()`（最優先）と
   `cSFW()`: SynapseLifecycle を返す。priceable=True（Newton/solve 後の残差減少
   が解析的値札 = RentEconomy の子になれる）。QuotaRegime 下でも動くこと。
4. **NeuronLifecycle の γ ungate**: 誕生場（∂L/∂γ、休眠でも厳密）で選び、
   γ を solve して `NeuronUngate(gate=...)` を出す。独立誕生（bundle なし）。
5. **recipe** `policy/recipes.py` に1つ: 「grow 序盤 → cVP 運転」の検証済み
   全体構成（recipe は既定1＋対照2の上限規律に従う）。
6. **機構テスト** `tests/torchcst/` に repo 慣習どおり1機構1ファイル:
   quota 上限・イベントタイミング・immunity・atomic 失敗 rollback・
   optimizer-state following・決定的 replay・**dual-timing 等価**
   （`test_scored_birth.py` が型見本）。
7. **受け入れテスト（最重要）**: standalone ハーネスをオラクルに使う —
   同一データ・同一規模で、family 実装の cVP/grow が fc1 ハーネスの
   対応アームと**統計的に等価**な軌跡を出すこと（bit 一致は不要）。

## 4. 違反禁止の不変量（CLAUDE.md の要約 — 全文を必ず読むこと）

- backward は証拠を作るだけ。構造変更・top-K 選択をしない。
- 通常 policy は loss-blind。目的関数を読むのは opt-in の profit court のみ。
- **木が唯一の実行経路**。composed Policy / StructuralPolicy 経路は存在しない。
- 二相 cross-store atomic（prepare 全員完了 → commit 一斉）。監査は一方向。
- **engine は変更しない**（capture への requires 宣言以外）。
- 微分次数・polish 手法の概念を共通層に持ち込まない（family 私有）。
- 実験の runner・結果・レシピ組み立ては cst 側。torchcst は再利用実装と
  実行可能契約のみ。

## 5. Definition of Done

- `.venv/bin/pytest` 全緑（既存 ~260 本 + 新規機構テスト）。
- `ruff check .` クリーン。
- `tools/phase2_equiv_harness.py` → `phase2_equiv_compare.py` が
  `tools/phase2_baseline.json` に対し **OVERALL: PASS**（既存挙動の非破壊証明。
  許容値の緩和は禁止）。
- 成果物3の受け入れテスト PASS。
- 各段階を独立の commit にし、設計判断は commit メッセージに記録。
  仕様の曖昧さに当たったら**勝手に契約を変えず**、この brief か
  fast-construction-family.md への追記として判断を残すこと。

## 6. 実装判断の追記（2026-08-01）

- `Refit` の表は、現行 `SynapseStore` が原子ごとに所有する `w` と任意の
  `s` / `t` を更新対象とする。現行 representation の σ は原子行ではなく
  compute 側 kernel が所有する共有 parameter であり、storage op から更新すると
  一直線依存と「唯一の storage 変更」に反する。このため原子別 `σ'` 列は、原子別
  bandwidth を representation/storage 契約として別途導入するまで実装しない。
- `cSFW(polish_iters=...)` の既定は、旧 family 契約ノート起草時の `3` ではなく、
  FC-0/1 完走後に凍結された本 brief の `0` を採用する。縮約 Newton は
  `polish_iters>0` と周期 backfit の位置更新で利用できる部品として残す。
- 標準 recipe の「grow 序盤 → cVP 運転」は、最初の `growth_events` 回を
  **純 cSFW-grow**（birth のみ）、次の root event から **毎イベント cVP-only**
  （位置更新なしの全振幅 Refit）と解釈する。成長中に cVP を同時実行しない。
  移行は受理 birth 数ではなく root 発行 event 数で決めるため、profit rejection
  があっても位相がずれない。深いネットワークの小核は store 初期化の責務であり、
  recipe は各層が非空であることを呼び出し側の前提として明記する。
