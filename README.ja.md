# torchcst

[English](README.md) | [日本語](README.ja.md)

PyTorch 向けの連続演算子。CST 層は全ての重みを保存する代わりに、
atom kernel の和として重み演算子を表す。

```math
W = \sum_a K(p_a), \qquad Y = XW^\mathsf{T}.
```

学習中の atom 数と Parameter の shape は固定する。Kernel の宣言は数式を、
Chart は観測サイトと geometry を、backend の Algorithm は評価と微分の実装を定める。

## インストール

リポジトリのルートで実行する。

```bash
python -m pip install -e .
```

Python 3.10 以上と PyTorch 2.0 以上が必要。以下の `CSTAdamW` の例は PyTorch 2.6 以上を
必要とする。追加依存は開発用の `'.[dev]'` と CUDA / Triton 実装用の `'.[cuda]'`。
CUDA 実装には device と精度の個別制約もある。

## 標準 optimizer で学習する

以下は16入力を8出力へ写し、CPU で1回の学習 step を実行する完全な例。

```python
import torch
from torchcst import (
    BandwidthBounds, CSTAdamW, CSTLinear, GaussianSpec, chart_presets, kernels,
)

model = CSTLinear(
    charts=(
        chart_presets.regular_grid(8, spacing=0.3),   # output sites
        chart_presets.regular_grid(16, spacing=0.2),  # input sites
    ),
    atoms=12,
    kernel=kernels.product(
        kernels.l2(GaussianSpec()),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.1,
            bounds=BandwidthBounds(
                minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1,
            ),
            radial_regularization=0.2,
        ),
    ),
    backend="factored",
)
optimizer = CSTAdamW(model.parameters(), model=model, lr=1e-3)

x, target = torch.randn(4, 16), torch.randn(4, 8)
optimizer.zero_grad(set_to_none=True)
y = model(x)
loss = (y - target).square().mean()
loss.backward()
optimizer.step()
assert y.shape == (4, 8)
```

`CSTAdamW` は scheduler、hook、AMP の step skip を含む通常の PyTorch optimizer の
ライフサイクルを所有し、AdamW の proposal の後で Kernel の座標更新則を適用する。
管理対象 atom group の raw `weight_decay` は既定で0。論理的な振幅の減衰には
`amplitude_weight_decay` を使う。通常の Parameter group は AdamW の既定の減衰を保つ。
`check_finite=True` は厳密な有限値検査を追加する。native の既定値は `False`。

別の optimizer を使う場合は `CSTOptimizer(base_optimizer, model=model)` を作り、
wrapper を1回 step する。SGD、Adam、AdamW、RMSprop、Adamax、Adagrad、Adadelta、Rprop の
組み込み adapter がある。group、更新則、独自 state、checkpoint 移行は
[optimizer ガイド](docs/cst-optimizer.ja.md)を参照する。

Atomの宣言済みフィールドに絶対値の初期学習率を指定するには`field_lrs`を使う。

```python
optimizer = CSTAdamW(
    model.parameters(), model=model, lr=4.8e-3,
    field_lrs={"amplitude": 2.4e-3, "input_center": 4.8e-3,
               "output_center": 4.8e-3},
)
```

Polarの`amplitude`は`(s,t)`をまとめた提案更新量を調整する。
未指定フィールドと通常Parameterはgroupの`lr`を使い、この`lr`は座標更新則の外部時間も決める。
標準schedulerの共通係数はフィールドの学習率にも反映される。
`optimizer.get_field_lrs()`で現在の有効な上書き値を確認できる。
通常ParameterとAtomで基本学習率を分ける場合は別のparam groupを使う。
[フィールド別学習率とcheckpointの契約](docs/cst-optimizer.ja.md#field-learning-rates)を参照する。

## He・Xavier 初期化

2-chart Polar radius cell-mass kernelを持つCPUの`CSTLinear`を、CUDAへ移動したり
optimizerを作成したりする前に、明示的に初期化できます。

```python
from torchcst import init as cst_init

report = cst_init.kaiming_normal_(layer, nonlinearity="relu", q=0.0)
# または cst_init.xavier_normal_(layer, gain=1.0, q=0.25)
```

`q`は振幅依存の半径上下限の間で初期の対数的な位置を選びます。
Denseと二乗平均目標を揃え、Tensor/Moduleではなくクリップ数などの診断reportを返します。
constructorの既定は保持します。対応範囲・数式・新preset用の`variance_scaling_`は
[初期化とDense–CSTの橋渡し](docs/initialization.ja.md)を参照してください。

実験的な`radius_envelope="closed_polynomial"`と
`closed_lower_power` / `closed_upper_power`の設定は
[両端で合流する半径境界](docs/closed-radius-envelope.ja.md)を参照してください。
この研究用opt-inは、それぞれの初期化とnative実行条件の検証が必要です。

## 公開 API で推奨する CUDA 構成

現在検証済みの **FP32・小バッチの半径セル質量** workload では、Linear の明示的 Plan
`GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C` と、独立した座標更新 Plan
`RADIUS_FLAT_TORUS_FUSED` を指定した `CSTAdamW` を出発点とする。
これらの opt-in Plan は `main` にあり、自動選択の既定は変更していない。
推奨の対象と制約は以下の通り。

| 設定 | この構成の対応範囲 |
| --- | --- |
| 入力・出力サイト | 同一の固定1次元 FlatTorus regular grid 二つ。N∈{1024, 2048, 4096, 8192, 16384}、spacing=1 |
| バッチ | 入力の総行数が4以下。例では真の B4 |
| atom | 2²⁴（16,777,216）個以下。各 atom は Polar 半径座標4列。native の work-index・メモリ制約も適用 |
| Kernel | Product cell mass、`ChebyshevSpec(modes=8)`、base tile 1、merge factor 2 |
| 精度 | 入出力・Parameter・owner-window 演算・融合 optimizer は FP32。共有 geometry と Polar pullback は FP64 を維持 |
| 実行 | eager、autocast 無効、1階微分のみ。決定的 reduction と CUDA Graph capture は未対応 |

この Plan は FP32 Tensor、amplitude_max≤1、[0.5, 50] 内の宣言された半径境界を必要とする。
FP16/BF16、FP64、一般的な mixed precision の推奨構成ではない。
geometry・有限値・shape の guard は保持する。半径は物理単位であり、sigma の単位とは異なる。
大きい幅の証拠は短期の正しさ・scaling 検証で、一般的な長期学習の品質を保証しない。
以下の例は N1024、測定済みの atom 密度5%の構成を使う。

CUDA 対応 PyTorch 環境に CUDA extras を導入し、autocast / Graph の外で以下の
完全な eager 学習 step を実行する。

```bash
python -m pip install -e '.[cuda]'
```

```python
import torch
from torchcst import (
    CSTAdamW, CSTLinear, ChebyshevSpec, FixedSelector, RadiusBounds,
    chart_presets, kernels, make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C,
)
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(False)

n = 1024
registry = make_algorithm_registry()
grid = chart_presets.regular_grid(n, spacing=1.0, geometry="flat_torus")
layer = CSTLinear(
    charts=(grid, grid),
    atoms=n * n // 20,  # Illustrative 5% atom density, not an optimal density claim.
    dtype=torch.float32,
    kernel=kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8), base_tile_sites=1, merge_factor=2),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.3,
            bounds=RadiusBounds(minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0),
        ),
    ),
    selector=FixedSelector(GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C, registry=registry),
)
layer = layer.to("cuda")  # Set placement before constructing the optimizer.
optimizer = CSTAdamW(
    layer.parameters(), model=layer, lr=1e-4,
    amplitude_weight_decay=0.01, fused=True, capturable=False, check_finite=False,
    update_selector=FixedSelector(RADIUS_FLAT_TORUS_FUSED, registry=registry),
)
x = torch.randn(4, n, device="cuda", dtype=torch.float32, requires_grad=True)
target = torch.randn(4, n, device="cuda", dtype=torch.float32)
optimizer.zero_grad(set_to_none=True)
y = layer(x)
loss = (y - target).square().mean()
loss.backward()
optimizer.step()
torch.cuda.synchronize()
assert y.shape == (4, n)
```

学習率と密度は構成例であり、タスクごとに調整する。管理対象 atom の raw decay は0、
論理的な振幅の decay は0.01。`check_finite=False` は通常の AdamW と同じ非有限値の
伝播を選ぶもので、宣言・device・shape の guard を除去しない。
optimizer の厳密な有限値検査が必要なら `check_finite=True` を指定する。

FP32 owner-window と Flat-C の研究は NVIDIA L4、PyTorch 2.11.0+cu130 / CUDA 13.0 を使用した。
これは検証環境の記録であり、対応 runtime version の一覧ではない。
完全 step Graph での高速化がこの eager 例へ自動的に適用されるわけではなく、dense に対する
速度優位も意味しない。研究 runner は比較用に CPU math を1 thread に固定するが、
パッケージ全体の要件ではない。
[FP32 の数値検証](docs/research-history/cuda-linear/owner-window-fp32-body-20261103.ja.md)、
[Flat-C 研究](docs/research-history/cuda-linear/owner-window-fp32-c-batch-20261104.ja.md)、
[大きい幅の制約と短期 scaling 検証](docs/research-history/cuda-linear/radius-large-scale-20261105.ja.md)を参照する。
`GLOBAL_OWNER_WINDOW_RECOMPUTE` は元の N1024 範囲に対する FP64 内部演算の比較用 Plan として残る。
[以前の eager 測定](docs/research-history/cuda-linear/owner-window-fold-20261103.ja.md)と
[半径セル質量ガイド](docs/radius-cell-mass.ja.md)も参照できる。

## Kernel を組み立てる

Kernel の宣言には独立した三つの要素がある。

| 要素 | コンストラクタ | 意味 |
| --- | --- | --- |
| Profile | `GaussianSpec`, `TriweightSpec`, `ChebyshevSpec` | 関数の形 |
| Factor | `kernels.point`, `kernels.l2`, `kernels.cell_mass` | 点評価・離散 L2 正規化・セル積分 |
| Coordinates | `kernels.polar`, `kernels.direct`, `kernels.physical` | atom の座標・初期化・更新則 |

これらを Tensor を持たない `KernelSpec` へ組み立てる。

```python
from torchcst import BandwidthBounds, TriweightSpec, kernels

coordinates = kernels.polar(
    amplitude_max=1.0, w_c=0.1,
    bounds=BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1),
)
product = kernels.product(kernels.l2(TriweightSpec()), coordinates=coordinates)
radial = kernels.radial(kernels.l2(TriweightSpec()), coordinates=coordinates)
```

`product(output_factor, input_factor, ...)` は Chart ごとに factor を評価・正規化してから
積を取る。factor 引数が一つなら両 Chart で再利用する。多次元 Chart でも、それぞれの
Chart の全距離と全サイトを使う。

`radial(factor, ...)` はスケールされた Chart 距離の二乗を足してから一つの profile を
適用する。L2 正規化の範囲は演算子全体。atom $a$ に対し次を定める。

```math
\begin{aligned}
W_{oi}^{\mathrm{product}}
  &= \sum_a A_a F_{\mathrm{out},a}(o)F_{\mathrm{in},a}(i), \\
q_{oi,a}
  &= \frac{d_{\mathrm{out}}(o,c_{a,\mathrm{out}})^2}{\sigma_{a,\mathrm{out}}^2}
   + \frac{d_{\mathrm{in}}(i,c_{a,\mathrm{in}})^2}{\sigma_{a,\mathrm{in}}^2}, \\
W_{oi}^{\mathrm{radial}}
  &= \sum_a A_a\,\mathcal{N}_{oi}[\phi(q_{oi,a})].
\end{aligned}
```

$\mathcal{N}$ は `point` では恒等写像、`l2` では離散 L2 正規化。
コンパクトな profile は正のノルム下限（既定 `floor=1e-6`）を使う。
Gaussian L2 は下限を設けず、安定な対数正規化を使う。

- 点評価 profile の単位は sigma（`BandwidthBounds`）、セル積分は物理半径
  （`RadiusBounds`）。これらの単位を混ぜない。
- 共通の `bounds=`、または明示的な `input_bounds=` / `output_bounds=` を指定する。
  動的座標が幅を所有するため、factor の `sigma=` との併用はエラー。
- 固定幅 factor では動的座標則を使わず `sigma=` を指定する。符号付き振幅を加えるには
  composition を `kernels.amplitude(...)` で包む。
- `cell_mass(ChebyshevSpec(modes=M), ...)` は M=1..512 に対応し、固定1次元 FlatTorus
  regular grid 二つの上で Polar 半径または physical 座標の product 結合を必要とする。
  完全な例、階層の数式、CUDA Plan は[半径セル質量ガイド](docs/radius-cell-mass.ja.md)を参照する。

座標則・正規化・結合の詳細は [Kernel reference](src/torchcst/kernels/README.md)を参照する。

## Chart と重みの軸

`charts=(output_chart, input_chart)` による重みの shape は
`[output_chart.ncs, input_chart.ncs]`。atom の中心列も同じ順序に従う。
例えば `chart_presets.regular_grid((16, 16), spacing=0.1)` は256サイトを持つ一つの2次元 Chart。
128サイトの出力 Chart と組み合わせると `[128, 256]` を得る。
`geometry="flat_torus"` は各周期を格子数×spacing から導出する。

Regular grid は index からサイトを生成する。格子の origin / spacing と geometry の
半径・周期は Tensor buffer を持たない固定の Python metadata で、cast 後も宣言の精度を保つ。
他の配置には explicit points と Product chart を使える。
[Charts](src/torchcst/charts/README.md) と [regular-grid ガイド](docs/regular-grid.ja.md)を参照する。

別形式の `chart=` は論理 shape `[out, in]` の一つの operator Chart を表す。
例えば `kernels.NORMALIZED_RADIAL_TRIWEIGHT` は単一の Euclidean operator Chart、
全体の L2 正規化、振幅 / log-width / 中心の座標を使う。二つの軸 Chart とは別の契約。

`atoms=` は正整数による表の初期化、detach してコピーする Tensor、再利用する
`nn.Parameter`、共有する `Atoms` owner を受け付ける。明示的な空の表はゼロ演算子を表す。
共有 Parameter は指定する device / dtype と既に一致している必要がある。
[atom の所有と配置](docs/atom-state.ja.md)を参照する。

## Checkpoint

最初の学習例に続けて保存・復元する。

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, "checkpoint.pt")

# Recreate the same model declaration and optimizer before resuming.
checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

optimizer を作る前に model を目的の device へ移し、復元時は model、optimizer の順に読む。
checkpoint は Kernel / 座標の契約、Chart の順序、固定の格子・geometry metadata、atom state を
検証する。互換性のない旧形式は拒否し、読み替えない。Algorithm の workspace は再構築する。
atom Parameter を置き換えた場合は optimizer も作り直す。
state の所有は [checkpoint 契約](docs/atom-state.ja.md#checkpoint)を参照する。

## 実行と制約

Kernel の宣言は実装と独立して演算子を定める。`backend=` は組み込み Plan の alias、
`selector=` は明示的な Plan を選び、後者が優先する。汎用 Torch 実装には factored と
materialized の評価がある。二つの Chart を使う radial kernel は Torch materialized 参照で
評価し、factored や専用 CUDA Plan は持たない。

CUDA Algorithm ごとに shape・dtype・精度・微分の制約がある。半径セル質量 Plan は
opt-in かつ eager 専用。Linear Plan を選んでも optimizer の更新 Plan は選ばれない。
管理対象 CST optimizer の step は CUDA Graph capture と微分可能な step に対応しない。
学習可能な非 Euclidean 観測 Chart は別の更新則が必要なため、現在は拒否する。
対応範囲は[実行と selector ガイド](docs/algorithm-dispatch-boundaries.ja.md)と
[optimizer ガイド](docs/cst-optimizer.ja.md)を参照する。

本パッケージは研究用 API。過去の測定は記録された source・workload・hardware の結果であり、
異なる revision や実装の性能を示すものではない。

## CST の研究

研究計画、モデル単位の実験 protocol、結果一覧は
[cst-research](https://github.com/takumiecd/cst-research)で扱う。atom 密度と幅・深さの scaling、
dense / dynamic sparse training との比較、最適化 step 数と経過時間に対する学習曲線を含む。

`torchcst` は演算子 API、数学的契約、実装、および正しさ・性能の検証を所有する。
研究 repo は revision を固定した実装を利用し、各実験に source と証拠を記録する。
過去の実験・kernel 検証記録は[文書一覧](docs/README.md)から参照できる。
実験用の実行経路は、このパッケージの公開 API の対応範囲を広げるものではない。

### 最適化された HIGGS 研究構成

HIGGS 研究は共有 FP64 geometry を保持する FP32 owner-window 演算、sorted owner window、
Flat-C 保存、guard 付きの完全学習 step CUDA Graph を使う。上記の明示的 Linear Plan は
`main` で利用できるが、完全 step Graph と非公開の HIGGS cohort は **benchmark 専用の実行経路**。
`backend=` の alias や公開 `CSTAdamW` の Graph 対応ではない。
固定済み実験の再現には研究 source
[`97580582ed28c4e494bf5c2656bbe94a85c544c3`](https://github.com/takumiecd/torchcst/tree/97580582ed28c4e494bf5c2656bbe94a85c544c3)、
[HIGGS protocol](https://github.com/takumiecd/torchcst/blob/97580582ed28c4e494bf5c2656bbe94a85c544c3/docs/research-history/cuda-linear/higgs-alpha-confirmation-20261105.ja.md)、
[qualification / campaign runner](https://github.com/takumiecd/torchcst/blob/97580582ed28c4e494bf5c2656bbe94a85c544c3/benchmarks/training/higgs_confirmation_campaign.py)
を使う。

構成は28→N→N→2、N∈{1024, 2048}、中間層だけ CST、全3層を学習、真の B4、FP32、
TF32 off、NVIDIA L4、PyTorch 2.11 / CUDA 13。atom の探索点は
A=floor(αN²)、α∈{0.001, 0.002, 0.005, 0.01}。
学習率と checkpoint は validation の先頭50k例で選び、後半50k例は全選択を凍結した後の
3 seed 確認用に保持する。精度を保つ最小の α は調査中であり、seed17 の development 曲線だけで
α=0.001 を一般的な推奨値とはしない。進捗と結果一覧は
[cst-research](https://github.com/takumiecd/cst-research)を参照し、この実験経路を実行する前に
固定 protocol の qualification gate を通す。

例えば、その研究 checkout と protocol に従って準備した HIGGS `screen.npz` がある場合、
GPU qualification gate は次のように実行する。

```bash
python -m benchmarks.training.higgs_confirmation_campaign \
  --stage qualify --data /path/to/screen.npz \
  --output output/higgs-qualification --campaign-state output/higgs-state
```

出力先には新しいディレクトリを使う。その後の tuning / confirmation stage と source・data・
recovery-state の identity は固定 protocol に従う。このコマンドは公開 `torchcst` API ではない。

## 文書

- [文書一覧：現在の契約・ガイド・研究履歴](docs/README.md)
- [参加方法：開発・検証・GPU benchmark の提出](CONTRIBUTING.ja.md)（[English](CONTRIBUTING.md)）
- [CUDA kernel 開発](docs/kernel-development.ja.md)
- [宣言と backend の配置規約](src/torchcst/_backends/README.md)

## ライセンス

Copyright 2026 takumiecd and contributors.
[Apache License, Version 2.0](LICENSE) により提供する。
帰属情報は [NOTICE](NOTICE) を参照する。
