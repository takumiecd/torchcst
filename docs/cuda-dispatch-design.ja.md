# CUDA 実行ポリシー設計（提案）

2026-09-29。対象はまず `CSTLinear` の CUDA 最適化経路と、
`BlockStripLinear` で試作した正確な mapped 学習経路。現在の公開 API と
`backend="auto"` の挙動は、この文書を作るだけでは変更しない。

## 境界と責務

```text
利用者: CSTLinear / 将来の CST module
    ↓ 演算の意味、パラメータと chart の所有
PyTorch: Tensor、device、autograd、演算子登録、compile との接続
    ↓ CUDA 演算の入口
torchcst CUDA policy (Python): 対応判定と検証済みレシピの選択
    ↓ 不変の ExecutionPlan
torchcst CUDA implementation: 準備、局所 W、GEMM、backward
    ↓
Triton / 将来の CUDA C++ kernel と PyTorch 演算
```

Python は小さな制御面とする。選択時に Tensor の**値**を CPU に読み戻さない。
大きなループ、支持域の判定、atom の並べ替えなど、値に依存する処理は GPU に置く。
PyTorch の dispatcher は device や autograd などの演算子レベルの接続に使い、
同じ CUDA 上の CST 計算方式の選択は torchcst が担当する。`torch.library` で
包む単位は、演算の意味と backward に必要な中間値が確定してから決める。
`nn.Module` や chart オブジェクトを登録演算子の隠れた入力にしない。

## 演算の意味を先に分ける

1. **Exact CST**: 正準演算 `sum(kernel(p[a], charts))` を許容誤差内で実現する。
   `materialized`、`factored`、`tiled`、融合、局所 W は、その対応範囲内で
   同じ演算の実装候補となる。forward、入力勾配、atom 勾配を同じ契約で照合する。
2. **Approximate CST**: アンカー補間など、異なる演算子を定義する方式。
   補間基底とアンカー配置も演算の意味の一部。明示的な選択として開始し、
   Exact CST の `auto` 候補には混ぜない。将来 `auto` で近似を許す場合は、
   利用者が別途指定する品質・近似ポリシーを先に定義する。

## CUDA policy の入力

最初の selector は副作用のない Python 関数とし、次の**メタデータ**だけを受け取る。

| 区分 | 入力 |
| --- | --- |
| 演算 | 演算 ID と意味の版、chart / geometry / profile / kernel capability、tile shape、論理 N/K、atom 数 |
| 呼び出し | 実効入力行数 `M = inputs.numel() // K`、shape、stride / contiguous 要件、dtype、autocast、必要な勾配 |
| 制約 | deterministic algorithms、許可する数値誤差、dispatch 管理下の workspace 上限、明示 backend 指定 |
| CUDA | `inputs.device` の実デバイス番号、compute capability、SM 数、共有メモリなどの公開 device properties、GPU 名、PyTorch / Triton 版 |

`M` は microbatch size ではなく、線形層に入る総行数。GPU 名は性能測定の
一致条件に使い、kernel の実行可能性は capability・dtype・必要機能で判定する。
世代名だけで未測定機種へ速さを一般化しない。device properties は選択された
実デバイスから取得してキャッシュし、複数 GPU の環境で device 0 を仮定しない。
空き VRAM や atom 値から推測した支持密度には依存しない。全メモリ使用量の
厳密な上限と、dispatch 管理下の scratch 上限は区別する。

## 登録単位と選択結果

**アルゴリズム**と**設定**を別々に扱う。アルゴリズムは計算の流れ
（全 W、局所 W、atom 直接など）を表し、設定はその中の候補リスト方式、
窓幅、GEMM 精度、tile、warp 数などを表す。アルゴリズムには安定した ID、
対応判定、workspace の上界式、正確性の区分、実行関数を持たせる。
設定の任意の直積を実行時に探索せず、整合を検証した組を
**名前付きレシピ**として登録する。レシピを選ぶデータ構造は、後述する
version 付きの探索木とする。

```python
@dataclass(frozen=True)
class CudaRecipe:
    id: str
    semantics: str              # 例: "exact-cst-v1"
    algorithm: str              # 例: "mapped-streamed"
    preparation: str            # 例: "listed-bounded"
    materialization: str        # 例: "windowed"
    gemm: str                   # 例: "ieee" / 検証済みの別精度
    backward: str
    schedule: str
    window_rows: int
    cache_windows: int


@dataclass(frozen=True)
class DispatchDecision:
    recipe: CudaRecipe
    reason: str
    workspace_upper_bound: int
    evidence_id: str | None
    matched_path: tuple[str, ...]
```

これは設計上の形で、公開 Python API の確定案ではない。`supports(context)` は
幾何、dtype、device、勾配、決定性、Graph 適格性を判定する。明示指定で
非対応なら理由付きエラーを出す。`auto` は探索木で見つけた対応 plan を使い、
通過した枝と fallback した節点を診断 API で確認できるようにする。
途中の CUDA OOM を捕まえて別方式で再実行することは、状態更新や Graph capture
を壊し得るため選択方法にしない。

初回導入では現行 `auto` の選択結果を保つ。mapped 経路を本体へ移し、
正確性と対象環境での完全ステップ測定が揃ってから、限定した条件を昇格する。
GPU 型番や compute capability だけで既定を一斉に変更しない。

### 最初に扱うレシピ

| ID の例 | 意味 | 最初の位置付け |
| --- | --- | --- |
| `linear.materialized.torch` | Exact | 現行の単一 chart の `auto` 基準。全 W の容量を見積もれる。 |
| `linear.factored.torch` | Exact | 現行の二 chart の選択基準を維持。 |
| `linear.tiled.torch` | Exact | 既存の明示指定を維持。 |
| `linear.strip_fused.triton` | Exact | 既存の明示指定を維持。FP32、幾何、一次勾配などの制限を記録。 |
| `linear.mapped.listed_bounded` | Exact | prototype から移植後に明示指定。対応 GPU と完全ステップ測定が揃うまで `auto` に入れない。 |
| `linear.mapped.listed_csr` | Exact | Graph 対応候補。`8 × atom 数 × 4` byte の候補配列上限を workspace 見積りに含める。 |
| `linear.anchor.*` | Approximate | Exact の選択表から独立させ、明示指定で始める。 |

この表の ID は設計上の仮名。公開 `backend` の値を一度に増やす意味ではない。
明示 `backend` は演算方式を固定し、その方式内の実行設定だけを選べるようにする。
選択には `torch.is_grad_enabled()` と入力・atom の `requires_grad`、active な
`AtomGrad` 経路も含める。forward 専用試作を学習経路に選ばない。

## version 付き探索木

CUDA の `auto` は**root から一つの枝をたどる木**で選ぶ。各節点は次に調べる
key と、互いに重ならない枝の条件を持つ。節点には任意で完全な plan
（kernel の登録 ID と設定、根拠となる測定 ID）を置く。枝がなければそこで
止まり、通過した節点のうち**最も深い有効な plan**を返す。子節点は設定だけを
変えても、アルゴリズムと kernel ID ごと変えてもよい。

```python
@dataclass(frozen=True)
class DispatchNode:
    split_key: str | None
    branches: tuple[Branch, ...]  # Branch は条件と子節点を持つ
    plan_id: str | None           # ここまで到達したときの fallback
```

```text
root                         [現行の保守的な plan]
└─ exact linear              [演算の意味]
   └─ Strip + Torus + Triweight
      └─ FP32 + 一次学習
         └─ 対象の N/K/M/atom 範囲       [局所 W の共通 plan]
            └─ 対象の CUDA 世代          [同方式の世代共通設定]
               └─ 特定 GPU 構成         [必要なら kernel または設定を上書き]
```

これは枝の形を示す例であり、角括弧の局所 W plan の採用は未決定。たとえば
特定 GPU の子節点がなければ CUDA 世代の plan を使い、世代の子もなければ
workload 節点の plan を使う。子があっても、その plan が dtype、勾配、
workspace などの制約に合わなければ、次に深い有効な祖先 plan を使う。
祖先にも適格な plan がなければ理由付きエラーにする。近似演算は Exact CST
とは別の root 配下に置き、両者の間では fallback しない。

`dispatch_schema_version` が、使える key、key の型・意味、枝の条件形式、
探索手順を固定する。v1 の key 候補は演算 ID / 意味の版、geometry と profile、
dtype と学習形態、N/K/M/atom 数・密度の範囲、Graph 要件、CUDA の機能、
アーキテクチャ世代、GPU 構成。節点ごとにこれらから次の分岐 key を選べる。
同じ path で同じ key を再利用する場合は、子の条件が祖先条件の真部分集合に
なることを求める。兄弟の条件は重ならないよう検証する。
これにより一つの context が同時に二つの子へ進む曖昧さをなくす。
演算の数学的意味の版、探索スキーマの版、kernel / recipe ID は別々に管理する。
未対応の `dispatch_schema_version` は推測して読まずに拒否し、旧版の木は
明示的な変換で移行する。設定木自体にも revision と source commit を記録する。

作成時には親 plan への「設定差分」を書けるようにしてよい。ただしロード時に
親から子へ解決し、実行時の各節点は**完全な plan**を持つ。kernel は文字列 ID
からコード内の登録表で引き、保存データに Python 関数や任意コードを入れない。
この仕組みなら世代共通の方式を継承し、機種の窓幅だけ変えることもできる。
性能差が明確な機種では子が別 kernel を指せる。木の branch 定義と plan の
整合は起動前または CI で検証する。

実験記録は、各レシピの完全ステップ時間の中央値とばらつき、ピーク割当、
forward・勾配誤差、測定環境を**候補ごとに**保存する。木の plan は採用した
測定 ID を参照する。実験 JSON を起動時に読んで順位付けしない。
速度差が測定の揺れと同程度なら workspace が小さく単純な設定を採る。
新 GPU は到達できる最深の検証済み祖先 plan から始め、実測により子節点を足す。
世代共通 plan を置くには複数実機で速度・精度・ピークを確認する。
細かい `num_warps` などは内部設定であり、利用者向け backend 名に増やさない。
木の変更と既定への昇格はコードレビューを通す。

## Autograd、Graph、キャッシュ

- 1 回の forward は 1 つの計画を確定し、その backward でも同じ計画を使う。
  atom が更新されても、候補リストと支持域は必要に応じて GPU 上で毎回作り直す。
- 構造定数だけをキャッシュする。chart / kernel 設定、checkpoint load、`.to()`、
  device / dtype 変更で無効化する。学習中の atom 値から得た動的結果を
  不変データとしてキャッシュしない。
- CUDA Graph は学習ループが capture する。dispatch は capture 前に warmup と
  適格性確認ができるようにし、replay 中は同じ計画を使う。`.item()` による同期、
  データ依存の Python 分岐、候補数に応じた可変割当を Graph 経路へ持ち込まない。
- atomic backward は deterministic algorithms 有効時に選ばない。高階微分、
  autocast、非連続入力などは各レシピが対応を明示し、未対応の明示指定は失敗、
  `auto` は対応する経路へ戻す。forward 時点で判定できない要件は、PyTorch の
  backward 側で正しく拒否する契約を別途検証する。

## オープンな実装の昇格手順

1. `experiments/cuda/linear/` で新方式を明示指定して試す。既定の dispatch は変えない。
2. 独立した Torch / dense oracle と forward、`dX`、atom 勾配を照合する。
   支持域境界、Torus seam、atom 移動、overflow、非連続入力も含める。
3. 対象 GPU ごとに完全学習ステップとピーク割当を測る。速度は同一環境の
   交互測定、メモリは必要なら別プロセス。eager と Graph は分けて報告する。
4. 実装を本体に入れて明示指定可能にし、対応条件・制限・測定データを添える。
5. 複数 seed / 対象形状で回帰を確認し、限定した `auto` ルールを追加する。
   selector の純粋な単体試験は架空の device facts を注入して GPU なしでも行う。

測定記録には source commit、GPU 名と capability、SM 数、CUDA / PyTorch /
Triton 版、shape、M、atom 数、dtype、演算 ID、recipe ID、warmup / capture /
replay 条件、完全ステップ時間、forward・勾配誤差、ピーク割当を保存する。
kernel 単体の勝ちと完全ステップの勝ちは区別する。メモリや精度の失格を
速度で相殺しない。

## 実装順

実装・レシピ・説明の配置と、開発用 `experiments/` からの移行単位は
[リポジトリ配置](repository-layout.ja.md)に従う。

1. `nn/_backends/` に `CudaContext`、`CudaRecipe`、`DispatchNode`、
   `DispatchDecision` と純粋な木の traversal を追加し、現行の選択を再現する。
   既存の明示 backend を維持する。
2. 木の重複枝・版・plan ID を検証し、通過 path と fallback 理由を表示する
   内部診断、GPU を使わない traversal テストを追加する。
3. `BlockStripLinear` の**正確な**学習経路を本体へ移す。その際、準備・
   forward・backward の責務と Tensor 入出力を固定する。
4. NVIDIA CUDA / FP32 / Triweight / 64×64 の検証済み条件だけで
   レシピを追加する。アンカーは近似演算として別に統合する。
5. 演算子境界を確定し、必要な `torch.library` 登録、fake、autograd と
   `opcheck` を追加する。現在の `torch>=2.0` という最小依存版との整合もここで決める。

## この設計の根拠になる既存記録

実験の採否と追加時の記録形式は[dispatch の実験台帳](dispatch-evidence.ja.md)に集約する。

- [`five-percent-breakthrough-research.ja.md`](five-percent-breakthrough-research.ja.md):
  小さな内部ポリシーを先に作る方針、CSR・Graph・メモリの測定。
- [`small-shape-ada-ampere-20260929.ja.md`](small-shape-ada-ampere-20260929.ja.md):
  同じ形状でも GPU と M により利点が変わる測定。
- [`blackwell-bounded-cst-20260929.ja.md`](blackwell-bounded-cst-20260929.ja.md):
  絶対時間と dense 比率が GPU 世代間で同じ方向に動くとは限らない例。
- [`l4-anchor-8192-dispatch-handoff-20260929.ja.md`](l4-anchor-8192-dispatch-handoff-20260929.ja.md):
  近似演算を Exact CST の `auto` に混ぜない根拠。

## 実装前に固定する詳細

探索木の方式、深い plan の優先、祖先への fallback、Exact / Approximate の
分離、Python 制御面と GPU 計算面は設計済み。実装に入る最初の変更では、
`dispatch_schema_version=1` の key と条件表現、初期 root plan、
保存形式、plan ID と kernel 登録表、診断 API を具体的に固定する。
`torch.library` で公開する演算子の粒度と最低 PyTorch 版は、mapped 学習経路の
Tensor 入出力が本体で固まる段階で別に決める。性能の木を追加する際は
[実験台帳](dispatch-evidence.ja.md)の evidence ID を必須にする。
