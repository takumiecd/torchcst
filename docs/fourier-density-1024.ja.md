# 1024×1024・5% atoms の有限 Fourier 比較

2026-10-14。前の [N300・A1024 の実測](fourier-shared-tiles-triton.ja.md)と区別する。
今回は dense の自由な重み1048576個に対し、CST は
`int(0.05 * 1024 * 1024) = 52428` atoms、209712個の学習可能な atom coordinates。
batch は32、TF32/autocast は使わない。

## 合意した格子

ユーザーが計算・数学の整合性を優先して選んだ等幅tile配置を使う。

| 項目 | 値 |
| --- | ---: |
| 入力／出力 sites | 1024／1024 |
| spacing | 0.1 |
| global Torus period | 102.4 |
| tile 数 | 8 |
| sites / tile | 128 |
| tile 幅 | 12.8 |
| profile の局所周期 | 38.4 = 3 tile |
| live sigma 範囲 | 0.1〜10 |
| K=H² の最高正周波数 M | 32、64 |

tile=10 / localP=30 / spacing=0.1 は1 tile100 sitesなので、1024 sitesを等分できない。
端数tileの特殊処理を避けるため、tile幅と局所周期を明示的に変更した。
sigma 最大10はtile幅12.8以下で、ownerと左右の計3 tilesという支持定義を保つ。
全3-tile局所周期は384 sitesで、`384 > 2M` の離散直交条件を満たす。
M64の実特徴数129は一つのtileの128 sitesより多いが、この条件には反しない。
再利用の対象は全atomの入力読み取りと出力集約である。

immutable geometry の周期・原点は元の宣言値からFP64で保存する。
atoms、入力、Triton contractionsはFP32。係数・L2準備はGPU上のTorch FP64。
`.cuda()`等のdevice-only移動はこの配置を保つ。
`.float()`でgeometryまで丸めると、正確な一様周期を要求する研究routeの対応判定から外れる。
整数tile番号にPython floatを掛ける際の既定FP32への丸めは、FP64 castで修正した。
元の整数tile幅10の測定条件はその修正で変わらない。

## 比較と検証

既存 `benchmarks.cuda.linear.periodic_comparison` の明示的rev5 caseを使う。
必須cohortは shared Triton、native direct Triton、dense。
巨大なsite×atom×harmonicのautograd scratchを作るTorch-factored経路はこのcaseでは使わない。
directは全global sitesのfactorsを保存する現実装であり、最適compact-support baselineではない。

新しい独立CPU FP64 oracleは、NumPyで宣言した次数M/2のnative Hを全sitesで評価し、
H²と実際の観測siteノルムを微分する。GPUのK係数、tile moments、中心非依存L2式は使わない。
atom chunk64で全52428 atomsを確認し、サンプリングしない。
同一X/target/dyと初期atoms、更新前／24実更新後のY・dX・全atom勾配、
固定max_abs／relative_L2 `4e-4` gate、live幅、Graph/public eager更新のmoments／clockを確認する。

未計装の完全stepは係数・L2準備、forward、MSE loss、backward、fused AdamW、
Polar projection、周期中心更新を含む。各routeを隔離subprocessで実行し、
正逆のworker順で二つの独立cohortを測る。21 replay samplesは一つのrunのサンプル。
allocator allocated／reserved peakはwarmup・Graph capture・replayを含む。
独立CPU oracleとGPUの照合scratchは測定から除き、GPU process全体のVRAMは未測定。

```bash
python -m pytest -q tests/test_fourier_tiles_cuda.py tests/test_fourier_density_cuda.py
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/fourier-density-chebyshev-1024-a52428-b32-m32.json \
  --plans fourier-shared fourier-direct dense \
  --output output/fourier-density/cohort.json
```

実機結果、source/result hashes、job IDs、採否を検証後に追記する。
