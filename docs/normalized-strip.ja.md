# 正規化 Strip 線形層

`torchcst.nn.NormalizedStripLinear` は、固定された通常の3次元 Euclidean
`StripChart` 上で、各 atom の離散 L2 ノルムを演算子全体から計算する線形層です。
既存の `CSTLinear` の API や既定のモデルを変更しません。

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
import math
import torch
from torchcst import EuclideanGeometry, GridPattern, LinePattern, StripChart
from torchcst.nn import NormalizedStripLinear

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

rows, h, j = 64, 8, 8
chart = StripChart(
    shape=(rows, h * j),
    tile_shape=(32, h * j),
    axis=0,
    tile_pitch=32.0,
    axes=(
        LinePattern(rows, spacing=1.0, center=(rows - 1) / 2),
        GridPattern((h, j), spacing=0.5,
                    center=((h - 1) * 0.25, (j - 1) * 0.25)),
    ),
    geometry=EuclideanGeometry(3),
).to('cuda')

torch.manual_seed(21)
p = torch.zeros(128, 5, device='cuda', dtype=torch.float32)
p[:, 0].uniform_(-0.5, 0.5)
p[:, 1] = math.log(3.0)
p[:, 2] = torch.rand(128, device='cuda') * (rows - 1)
p[:, 3] = torch.rand(128, device='cuda') * (h - 1) * 0.5
p[:, 4] = torch.rand(128, device='cuda') * (j - 1) * 0.5

layer = NormalizedStripLinear(chart, p, memory='full')
optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
x = torch.randn(128, h * j, device='cuda')
optimizer.zero_grad(set_to_none=True)
loss = layer(x).square().mean()
loss.backward()
optimizer.step()
```

入力の最終次元は `h * j` で、その前のバッチ次元は任意です。出力の最終次元は
`rows` になります。渡した `p` は detach して複製され、層の `layer.p` が独立した
`nn.Parameter` になります。元の入力テンソルを optimizer に登録せず、例のように
`layer.parameters()` を登録してください。通常の `torch.optim.AdamW` を使用できます。
この層は既存 CST optimizer の更新規則を自動適用するものではありません。

対応する chart は axis0、連続した tile pitch、LinePattern と2次元 GridPattern を
持つ通常の3次元 Euclidean Strip です。固定の chart メタデータを registered buffer
へ保存し、学習可能な chart パラメータを拒否します。層の `.to(device)` は atom と
保存した descriptor を移動します。`state_dict` にはパラメータと descriptor に加え、
形式のバージョン、メモリ経路、固定の幅範囲とノルム床を保存します。

CPU では FP32/FP64 の局所支持を列挙する PyTorch 参照経路を使います。これは
CPU 用の機能経路で、CUDA の性能測定には含めません。CUDA 高速経路は FP32 と
間隔 `(1, 0.5, 0.5)` に対応し、Triton を必要になった時点で import します。
基本パッケージや CPU 利用の import に Triton は必要ありません。CUDA 経路は
一次微分を対象とし、autocast と TF32 を拒否します。CUDA の
window 経路は行窓512を用い、対応する標準メタデータの条件を満たさない場合は
full 経路へ戻ります。その場合に同じメモリ削減を保証するものではありません。

chart は固定してください。学習する chart はこの層の対象外です。中心の3座標は
atom パラメータとして学習できます。

## メモリ経路と測定の範囲

既定の `memory='full'` は通常幅の検証済み経路です。`memory='window'` は
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
