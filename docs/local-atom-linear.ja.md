# I/B配置を使う局所的なアトム直接縮約

`prototypes/local_atom_linear.py` と `local_atom_kernels.py` の試作。
公開backendや既存の標準経路は変更しない。
単独用 `I[g]`・境界用 `B[g]` の配置をそのまま利用する。
支持範囲の細かい位置に基づく再配置、列のsort、累積和は追加しない。

## Forward

GPUの１処理単位が、１つの出力特徴と少数の入力行の部分和を保持する。
この担当範囲は全atomの処理が終わるまで変えない。

```text
出力行nが属するタイルgを求める
出力の部分和 y[m:m+BM,n] = 0
for bucket in [B[g-1], I[g], B[g]]:
    for atom in bucket:
        if atomの支持球が行nのサイト群の外接boxに届かない:
            continue
        for 入力列の小区間:
            そのatomのkernel値 phi[n,k] を求める
            phiが非ゼロの列だけXを読み込む
            y[m:m+BM,n] += amplitude * sum_k(X[m:m+BM,k] * phi[n,k])
出力を書き戻す
```

外接boxは固定されたsection座標の最小・最大値から求める。
除外判定には小さい余裕を持たせ、実際のprofile評価式は変えない。
１atom・１行の寄与ベクトルはレジスタ上で評価して直後に縮約する。
複数atomを足した重みタイル、全W、atomごとの部分出力配列は作らない。
出力担当が固定なのでYへのatomicは使わない。

### 局所性を利用した範囲

- I/Bで対象外のタイルのatomを読み込まない。
- 行の外接boxで支持しない行を省く。
- 残った行の入力Xは、kernel値が非ゼロの列だけ読む。
- kernel値をBM行の入力で再利用する。

現段階では、除外できなかった行について全Kの小区間を走査し、
距離・kernel値を評価する。支持列を列挙する索引や細かなremapは作らない。
したがって「支持内だけで全演算が完結する」実装ではない。
同じ入力の複数atomでの再利用はハードウェアキャッシュに依存し、
明示的な共有メモリへの先読みは入れていない。キャッシュhit率は未測定。

## Backward

同じcanonical kernelの値と距離に対する傾きから一次勾配を求める。

- dX: 入力要素の担当を固定し、タイル・atomを巡回して局所的なdYを縮約する。
- dP: １atomを１処理単位で担当し、そのatomの１〜２タイルを処理する。
  各行・列区間で `sum_m(X[m,k] * dY[m,n])` を求め、振幅・中心の勾配へ直接足す。
- どちらもglobal atomicは使わず、全dWやatom別dWタイルも保持しない。

帯域幅のstop-gradient、振幅clamp、中心decodeとpackの逆置換は既存と共通。
この試作の微分対応は一次のみ。高階微分・専用optimizerとの統合は含まない。
FP32の縮約順序はdenseや既存融合版と異なるため、bitwise一致は要求しない。

## トレードオフ

同じ(n,k)にatomが重なると入力との積や部分和への加算をatomごとに繰り返す。
アトム総数だけでなく重なり、支持されるサイト数、入力行数によって有利不利が変わる。
dXの現実装は全タイルの候補を巡回するため、特に学習時の速度は別途評価が必要。

永続データは元のatom表と既存のO(A+N+K+G)の準備表。
新たに追加する外接boxのデータはO(D)。forward/backwardの作業領域は
出力・必要な勾配とレジスタ内の局所配列で、`[A,M,K]`の大きな中間配列を持たない。
固定boxは試作では毎forwardに小さいreductionで求め、測定時間にも含める。

## 実行

```bash
pytest -q tests/test_local_atom_linear.py
python -m prototypes.benchmark_local_atoms --output local-atoms.json --source-commit COMMIT
```

ベンチマークは保持済みdense、既存I/B融合版BM16/BM64、直接版BM4/BM16を比較する。
全体時間はI/Bの分類・sort・packも含む。準備済み経路は内訳の診断用。
既存の6形状に加え、I/Bが混在する7番目のケースを用意した。
