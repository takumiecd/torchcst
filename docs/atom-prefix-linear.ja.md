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
