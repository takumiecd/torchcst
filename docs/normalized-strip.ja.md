> 以下の source commit・コマンド・測定値は検証当時の記録。現在の実行入口は
> [benchmark README](../benchmarks/cuda/linear/README.md)を参照。

# 正規化 Strip 線形層

`CSTLinear` に通常の3次元 Euclidean の regular Product / 連続 Strip chart と
`presets.NORMALIZED_RADIAL_TRIWEIGHT` を渡す。各 atom の離散 L2 ノルムを
演算子全体から計算する。公開 Linear は共通入口へ統一し、旧専用クラスは削除した。

## インストール

```bash
python -m pip install -e '.[cuda]'
```

CUDA 用追加依存には PyTorch、Linux 用 Triton が含まれます。CUDA の実行環境は
別途必要です。基本パッケージのインストールだけでは CUDA 高速経路の依存は
入りません。

## モデル

パラメータ表は `[atoms, 5]` で、列は `[amplitude, log_sigma, c0, c1, c2]` です。
振幅は符号付きで、ノルムの外に置きます。

```text
sigma_a = exp(clamp(log_sigma_a, log(.03), log(3.25)))
q_a(s) = ||s - center_a||² / sigma_a²
k_a(s) = max(0, 1 - q_a(s))³
norm_a = max(sqrt(sum_{s in whole operator} k_a(s)²), 1e-6)
W[i,j] = sum_a amplitude_a * k_a(site[i,j]) / norm_a
y = x @ W.T
```

距離は共同3次元の距離です。軸ごとのカーネル積への置き換えや固定 anchor への
補間を行いません。ノルムはタイルや窓ごとのノルムではありません。現在の
パラメータに対して再評価し、振幅、幅、3つの中心座標すべてに勾配が流れます。
空の支持はゼロで、床に達していない単一サイトの atom は符号付き振幅だけを
そのサイトへ与えます。ノルム床が有効な極小支持は床の規則に従います。

## 使用例

次の chart はベンチマークと同じ行間隔1、列の2軸の間隔0.5です。列は最後の
軸を最速で平坦化します。例のサイズは小さくしてあります。

```python
from torchcst import pattern_presets as sites
from torchcst import geometry_presets as spaces
import math
import torch
from torchcst import chart_presets as layout
from torchcst import CSTLinear, LinearOptions, presets

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

rows, h, j = 64, 8, 8
chart = layout.strip(
    shape=(rows, h * j),
    tile_shape=(32, h * j),
    axis=0,
    tile_pitch=32.0,
    axes=(
        sites.line(rows, spacing=1.0, center=(rows - 1) / 2),
        sites.grid((h, j), spacing=0.5,
                    center=((h - 1) * 0.25, (j - 1) * 0.25)),
    ),
    geometry=spaces.euclidean(3),
)

torch.manual_seed(21)
p = torch.zeros(128, 5, device='cuda', dtype=torch.float32)
p[:, 0].uniform_(-0.5, 0.5)
p[:, 1] = math.log(3.0)
p[:, 2] = torch.rand(128, device='cuda') * (rows - 1)
p[:, 3] = torch.rand(128, device='cuda') * (h - 1) * 0.5
p[:, 4] = torch.rand(128, device='cuda') * (j - 1) * 0.5

layer = CSTLinear(chart=chart, atoms=p, kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
                  linear_options=LinearOptions(memory='full'))
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(128, h * j, device='cuda')
optimizer.zero_grad(set_to_none=True)
loss = layer(x).square().mean()
loss.backward()
optimizer.step()
```

入力の最終次元は `h * j` で、その前のバッチ次元は任意です。出力の最終次元は
`rows` になります。`atoms=p` の `p` が Tensor なら detach して複製し、`layer.atoms.p` が所有する。
`nn.Parameter` なら同じオブジェクトを登録し、既に作った optimizer の参照と状態を
維持する。`Atoms` を渡すと owner ごと再利用する。共用パラメーターの constructor 内
での dtype/device 変換は拒否する。移動は optimizer を作る前に済ませる。

```python
shared_p = torch.nn.Parameter(p)
base_optimizer = torch.optim.AdamW([shared_p], lr=1e-3)
shared_layer = CSTLinear(chart=chart, atoms=shared_p,
                         kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT)
assert shared_layer.atoms.p is shared_p
```

通常の torch.optim と CSTOptimizer のどちらも使用できる。log幅と中心の更新は
Euclidean の通常更新となる。CSTOptimizer の同一モデルに複数の CST policy owner を
持つ共用 atom table は現時点で拒否する。

対応する高速 algorithm は LinePattern と2次元 GridPattern を持つ固定の3次元
Euclidean chart を扱う。Product または axis0・連続 tile pitch の Strip に対応する。
共通 ChartState / KernelState / Atoms をそのまま checkpoint に保存する。
LinearOptions も checkpoint の契約に含むので、読み込み先に同じ設定を渡す。
旧専用クラスの checkpoint は読み込まない。

CPU では FP32/FP64 の局所支持を列挙する PyTorch 参照経路を使います。これは
CPU 用の機能経路で、CUDA の性能測定には含めません。CUDA 高速経路は FP32 と
間隔 `(1, 0.5, 0.5)` に対応し、Triton を必要になった時点で import します。
基本パッケージや CPU 利用の import に Triton は必要ありません。CUDA 経路は
一次微分を対象とし、autocast と TF32 を拒否します。CUDA の
window 経路は行窓512を用い、対応する標準メタデータの条件を満たさない場合は
full 経路へ戻ります。その場合に同じメモリ削減を保証するものではありません。

高速 algorithm は固定 chart が対象です。対応しない配置は CSTLinear の一般
Torch 参照経路を使います。中心の3座標は atom パラメーターとして学習できます。

## メモリ経路と測定の範囲

以下の既存測定は各記録の revision に対する履歴であり、入口統一後の性能を
示すものではありません。統一後の検証は [Linear 統一記録](linear-unification.ja.md) に記録します。

既定の `LinearOptions(memory='full')` は通常幅の検証済み経路です。`LinearOptions(memory='window')` は
メモリを優先する選択です。W と dW を行窓の作業領域で再利用し、境界の勾配に
必要な8行の halo を保持します。全体ノルムと全5パラメータの勾配は維持します。
逆伝播で W を再生成するため、メモリ削減には追加計算が伴います。

以下は公開クラスを計測する前の研究実装の結果です。公開クラスの性能として
読み替えないでください。[履歴と対象ソース](../benchmarks/cuda/linear/results/normalized-strip-history-20261001.json)
に記録した L4、FP32、TF32 無効、M128、atom 数 N² の5%、AdamW を含む完全な
ステップの測定では、N8192 の通常幅 sigma3 が main Atlas 経路で約59ms、
窓経路で約131msでした。通常幅の速度向上を示す窓経路の結果ではありません。
別の鋭い単一サイト支持 fixture では、研究用窓経路の Graph 最大 allocated が
約354MiB、研究用 packed 経路が約850MiBでした。この packed 経路は公開クラスの
既定 full 経路と同一のクラスではありません。窓経路の時間は同 GPU の dense
ステップの約1.85倍でした。公開クラスの比較は
[公開 API 統合測定](../benchmarks/cuda/linear/results/normalized-strip-public-integration.json)
を確認してください。統合測定の記録がない場合、公開クラスでの性能は未確認です。
任意の幅、サイズ、学習状態で dense の2倍以内になる保証はありません。
鋭い fixture の結果を sigma3 の結果として扱わないでください。

`max_memory_allocated` は PyTorch の生きたテンソルに関する値です。
`max_memory_reserved`、Graph private pool、CUDA コンテキスト、ライブラリの
作業領域を含むプロセス全体の使用量とは異なります。Graph 測定では capture
前から最大値を記録し、初期 oracle の一時割り当てや allocator の保持分も
区別する必要があります。

## 実行上の注意

CUDA Graph を使う場合は、同じ形状と経路で通常の forward、backward、optimizer
step を先に実行してカーネルと optimizer state を準備してください。capture
中に初回コンパイルを行わないでください。Graph replay ごとに現在のパラメータ
から支持と全体ノルムを再評価します。

CUDA の atom 集約には atomic 加算を用います。浮動小数点の加算順が変わり得る
ため、実行間のビット単位の決定性を保証しません。決定的なアルゴリズムを
要求する設定では CUDA atomic 経路を拒否します。性能と精度の検証は上記の
FP32/TF32 無効の条件で行っています。別の dtype、chart、GPU、入力形状の結果を
同じ測定と見なさないでください。


## 旧専用クラスの配布版検証（2026-10-01）

wheel を展開し、`experiments/` に依存しない実際の配布コードから実行しました。
同じ L4（GPU-639ce143）、FP32/TF32 無効、M128、5% atom、forward・dX・全5勾配・
fused capturable AdamW を含む step の同期 wall time 中央値です。各ケースを別の
Python プロセスで測り、以前のケースの cuBLAS ストリーム作業領域を除外しました。

| N | 条件・経路 | Graph ms | peak allocated MiB |
|---|---|---:|---:|
|1024|通常 sigma3 / full|0.63983|51.85|
|1024|通常 sigma3 / window|1.52735|46.16|
|1024|構成した鋭い支持 / window|0.30756|46.16|
|1024|dense|0.12816|50.50|
|8192|通常 sigma3 / full|59.97223|849.32|
|8192|通常 sigma3 / window|134.59388|354.20|
|8192|構成した鋭い支持 / window|23.62340|354.20|
|8192|dense|12.80303|1072.50|

window の鋭い条件では N8192 が dense 比1.845倍、N1024 が2.400倍です。
通常幅の2倍目標は両サイズとも未達です。既定 full は通常幅向け、window は
明示的なメモリ優先の選択として公開します。8192 window の max reserved は540 MiBで、
354.20 MiBの allocated と区別してください。各ケースは20回の実 optimizer 更新を
行い、capture/replayを含む peak と全7サンプルを記録しています。

配布版の L4 テストは67件通過。小さい混合 fixture の独立 FP64 oracle で
W（2e-5）、y/dX/全5勾配（3e-4）を照合し、512行境界、床、空支持、clip、
Graph capture後の中心/幅の変更とAdamW状態を検証しました。大きいサイズの
記録は完全stepの時間とpeakであり、全atomの独立勾配照合とは扱いません。
[機械記録](../benchmarks/cuda/linear/results/normalized-strip-public-integration.json)
にはGPU/runtime、source/result/wheel SHA、全サンプル、誤差、既知の測定失敗を保存しています。

同じ検証を再実行するには、リポジトリのルートで共有 pool を使います。

```bash
python3 tools/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_normalized_strip_wheel.py" --label normalized-strip-wheel --timeout 1800
python3 tools/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

runner は job 内で wheel をビルドし、公開 API のテストと各ケースの独立プロセス測定を
実行します。結果は `CST_JOB_OUTPUT` へ保存され、pool がSHA検証して回収します。
