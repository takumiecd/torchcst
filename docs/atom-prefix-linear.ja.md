# アトムから出力を直接求める Triweight 試作

対象は既存の Strip＋Torus（行方向の station、入力軸全体を含む tile）、
DirectAmpWidth、非正規化 Triweight。公開 backend には追加せず、
`prototypes/atom_prefix_linear.py` に PyTorch/autograd の参照実装を置く。
入力 `X[M,K]`、出力 `Y[M,N]`。重み `[N,K]`、重みタイル、
アトム別の `[A,N,K]` 寄与は作らない。

## 距離の変形

サイトの座標を `(rho[k] * u[n], z[k])`、アトム中心を
`(r[a] * v[a], z_a)` とする。`u,v` は2次元の単位方向。
既存の `geometry_factors` と中心 decode がこの座標を与える。

```text
delta[a,k] = (rho[k] - r[a])² + ||z[k] - z_a||²
s[a,n] = r[a] / 2 * ||u[n] - v[a]||²
distance²[a,n,k] = delta[a,k] + 2 * rho[k] * s[a,n]

t[a,k] = (sigma[a]² - delta[a,k]) / (2 * rho[k])
e[a,k] = 2 * rho[k] / sigma[a]²
phi[a,n,k] = e[a,k]³ * max(t[a,k] - s[a,n], 0)³
```

半径は正なので、支持条件は `t[a,k] > s[a,n]` となる。
これは radial kernel 全般の外積分解ではない。Triweight の三次多項式と
Torus の座標構造を使った縮約方法である。

## 三つの処理

1. アトムごとに `t` の降順で入力列を並べる。各出力行について、
   `t > s` を満たす列数 `q[a,n]` を二分探索で求める。
2. 並べた入力から4種類の累積和を求める。
   `P_j[a,m,q] = sum(first q columns, X[m,k] * e[a,k]³ * t[a,k]^j)`、`j=0..3`。
3. 各行で累積和を参照し、アトムの寄与を出力へ足す。

```text
Y[m,n] = sum_a amplitude[a] * (
    P_3[a,m,q[a,n]] - 3*s[a,n]*P_2[a,m,q[a,n]]
    + 3*s[a,n]²*P_1[a,m,q[a,n]] - s[a,n]³*P_0[a,m,q[a,n]]
)
```

実数演算では支持範囲の打切りも含めて等価。FP32の演算順は変わる。
`s` を大きな半径同士の内積の差として計算せず、方向の差の二乗で計算する。
最後の多項式は Horner 法で計算するが、境界付近の桁落ちを完全には防げない。

## 計算量とメモリ

準備はおおむね `O(A*K*log K + A*N*log K)`、入力の縮約は
`O(A*M*(K+N))`。距離の全 `A*N*K` 評価を必要としない。
一方、現在の試作は全アトムの入力列を走査し、全出力行の参照も行う。
支持される列だけの走査や出力行の絞り込みは未実装。

`prepare_prefix` は入力 `X` に依存しないが、アトム更新のたびに再計算する。
`contract_prefix` はその準備結果から出力を計算する。
固定 geometry の既存キャッシュだけを利用し、アトム由来の計算グラフは保持しない。
atom の所属順への repack はこの方式の正しさには不要。

アトムを chunk に分け、各 chunk の中間配列を `[chunk,M,K]` と
`[chunk,M,N]` に制限する。推論時の一時メモリには効くが、autograd は
backward に必要な中間値を各 chunk から保存するため、学習時の総メモリを
chunk 分だけに制限できるわけではない。

## 勾配と適用範囲

sort の値・gather・累積和を通して amplitude と中心への勾配を求める。
支持範囲の添字は微分しない。Triweight は境界で値と一次微分がゼロなので、
この扱いは一次勾配と整合する。帯域幅は既存の `tile_parameters` の
stop-gradient を維持する。activity 状態への勾配も既存同様ゼロ。

この試作は標準の一次 autograd 照合が目的。カスタム atom 勾配経路や
quartic optimizer への統合、他の profile、高階微分の検証は含まない。
PyTorch の sort、scan、gather、chunk ごとの起動コストがあるため、
演算量の削減だけで dense より速いとは判断しない。

```bash
pytest -q tests/test_atom_prefix_linear.py
python -m prototypes.benchmark_atom_prefix --output atom-prefix.json --source-commit COMMIT
```

A100 benchmark は同じ FP32 の保持済み dense を主基準とし、
既存の融合 CST、準備のみ、縮約のみ、準備を含む試作全体を併記する。
dense と CST のパラメーター数は合わせない。出力形状と重みの値を合わせる。

## A100 測定結果（2026-09-26）

NVIDIA A100 80GB PCIe MIG 3g.40gb、Torch 2.6.0+cu126。
FP32、TF32 無効、atom chunk=64。3ラウンドで計測順を反転させ、
CUDA Graph 反復時間の中央値を示す。以下は forward のみで、学習速度は未測定。

| N×K | M | atoms | 保持済み dense | 既存融合 CST | prefix 準備 | prefix 縮約 | prefix 全体 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64×128 | 16 | 64 | 0.0073 | 0.2038 | 0.1856 | 0.2373 | 0.4042 |
| 64×128 | 128 | 64 | 0.0135 | 0.2037 | 0.2320 | 0.7590 | 1.0364 |
| 64×128 | 16 | 256 | 0.0072 | 0.6188 | 0.2438 | 0.8584 | 1.1075 |
| 64×128 | 128 | 256 | 0.0131 | 0.6062 | 0.2447 | 3.0237 | 3.2561 |
| 256×512 | 32 | 512 | 0.0171 | 1.0349 | 0.5197 | 6.1669 | 6.7119 |
| 256×512 | 128 | 512 | 0.0249 | 2.1812 | 0.4639 | 24.4611 | 24.7854 |

単位は ms。準備・縮約・全体は独立測定なので厳密な加算内訳ではない。
**今回の PyTorch 試作は全ケースで既存の融合版より遅い。**
最大ケースの追加割当ピークは約117.7 MiB。これは保持済み dense 重みや
事前作成した計画等を含まない、各経路を warmed 実行した際の増分。
重みを生成しなくても、累積和の中間配列を大量に読み書きするコストがある。

また、A=512、N=256、K=512 では `A*(K+N)/(N*K)=3`。
4種類のモーメントの定数倍を考える前でも、入力に依存する処理の
オーダー上の項数は dense より少なくない。atom パラメーター数の削減を、
積和回数や実行時間の削減と同一視できない。現在は支持のない行も参照し、
各atomについて全入力列を走査している。専用カーネルへの移植だけで
dense に並ぶとは判断せず、対象形状・atom数・支持範囲の条件も検討する。

### 精度の結果と残る制限

CPU の18テストが通過した。FP32/FP64、intrinsic/ambient、可変帯域幅、
円環の継ぎ目、非連続入力、空バッチ、パラメーター更新後の再計算を含む。
A100 上でも18テストをCPUで実行したほか、benchmark の6ケースはCUDAで
出力・入力勾配・atom勾配を計算して照合した。

CUDAの全6ケースで出力・入力勾配は `atol=rtol=3e-5` を満たした。
atom勾配の要素単位 `atol=rtol=1e-4` は大形状2ケースで満たさず、
M=32 は3要素、M=128 は1要素が外れた。これは結果JSONにも残している。
速度の診断は有限値と相対L2誤差 `3e-5` 未満を確認して継続した。
この継続条件を公開backendの採用基準とは扱わない。

別途、同じFP32入力・パラメーター・固定bufferをdoubleへ変換し、
FP64 canonical dense/autograd を基準に大形状2ケースを検証した。
FP64 prefix の最大差は出力 `2.09e-15`、atom勾配 `1.99e-12` 以下。
FP32のatom勾配の相対L2誤差は dense と prefix の両方で約 `5.5–6.3e-6`。
この検証では式の誤りを示す結果はなく、両方式にFP32の数値誤差がある。
全ての支持境界・配置・高階勾配についての保証ではない。

### 再現情報

- 速度測定ソース: `28558f42a2a7da64d32d60ccc7de97a4a4cc58a4`
- Snapshot: `srv11/cst-lab/torchcst-atom-prefix-v3-20260926/`
- archive SHA256: `78d775b9a023d77a090debd06ba8a2574267dc94c5a77242e60ecbb04ca12124`
- 速度結果: `output/triton-a100-20260926/atom-prefix.{json,log}`
- 精度結果: `output/triton-a100-20260926/atom-prefix-precision.{json,log}`

精度診断は `105d9f3` のv2 snapshotに `check_atom_prefix_precision.py` を
追加して実行した。追加ファイルのSHA256は
`2136c87510c79de674f172376b58435b22dda779fadba74290da16277d6a5358`。
同じファイルを最終ソースにも収録している。転送hash、GPU名、ソースコミット、
速度6ケースと精度2ケースの完了をローカルでも確認した。
初回のGraph capture失敗と旧精度ゲートで停止した途中結果はサーバー側の
v1/v2 snapshotに残してある。
