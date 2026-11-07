# 処理するmoment階層だけを起動する

[English](multirow-levels-20261108.md) | **日本語**

コード上、packed_momentsは各階層にR×(N-2)×17スレッドを起動するが、その階層の
2 bank以外は書かない。非活性階層も全gridを起動してからdevice maskでreturnする。
N8192/16384ではlevel2〜log2(N)の12/13回起動がある。これはスケジュールの費用で、
Fourier/セル質量の演算自体の変更ではない。

既存の呼出し内geometry再利用と階層を絞ったgridを組み合わせる研究Algorithmを別に用意する。
`cuda_radius_large_multirow_levels`は応答保存、
`cuda_radius_large_multirow_levels_stream64`は64行単位の再計算である。
階層lではR×2(N/2^l)×17スレッドを起動し、従来のcompact Z位置へ写像して同じ
compact_moment_elementを呼ぶ。device上のrequired bank mask、子bankの包含、階層昇順の
依存関係、加算順、moment保存layoutを維持。全階層の名目スレッド合計はR×(N-2)×17となり、
従来の12/13倍の起動を減らす。時間がこの比率で速くなるとは仮定しない。旧Algorithmのgrid・
既定値と、保存/64行単位再計算の保持方式は維持する。

計測前にsource・protocol・入力hashを固定する。A100 MIG3g.40gb限定、FP32 IEEE/TF32無効、
N2048で検証後、N8192/16384・R256・A8N・Xavier/q0/minimum-001を使う。初期P/x/target/dYは
幅比較の凍結値。対象CUDA回帰全体と、両候補のN2048全VJP・独立求積・matched updateを確認。
基準は変えず、失敗・OOM・非有限・timeoutで証拠を残して止める。GPU実験は同時実行しない。

fresh processで、Aは幅8192→16384・保存→levels→levels_stream64→Dense、
Bは幅16384→8192・Dense→levels_stream64→levels→保存。既存workerのwarmup10＋同期した
完全stepの計時51、全61更新がallocated/reserved peak対象。初期・更新後全VJP、Denseとの初期
演算子一致、optimizer、状態hash、source/extension hash、全sampleを維持。両順序・完全step時間・
メモリを記録し、精度・独立hardware再現・Graph・公開採用を推論しない。従来geometry/窓単位再計算と
比較する場合はsourceの異なる別実験として扱う。

これは計測前protocolである。この比較後は追加の推測的な最適化を止め、実測した選択肢を保全して
reviewできる状態に整理する。

## 研究用の明示選択

対応する大規模正方の`layer`に、既存selector APIで保存版か64行単位再計算を選ぶ。
限定cohort用のprivate opt-inで、公開CSTLinearの新しいflagではない。
厳密なPlan recipeは保存版で`save_response=True`、stream64で`False`とし、
別Algorithmが実行・保持方式を識別する。optimizerやbatch lossは変えない。

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.large_multirow import (
    LARGE_LEVELS,
    LARGE_LEVELS_STREAM64,
    RadiusLargeLevelsAlgorithm,
    RadiusLargeLevelsStream64Algorithm,
)

registry = make_algorithm_registry()
registry.register(RadiusLargeLevelsAlgorithm())
registry.register(RadiusLargeLevelsStream64Algorithm())
layer.selector = FixedSelector(LARGE_LEVELS_STREAM64, registry=registry)
# Saved responses instead:
# layer.selector = FixedSelector(LARGE_LEVELS, registry=registry)
```

許可範囲：FP32正方N2048/8192/16384、R1〜256、A4〜8N、origin0/spacing1、
Polar M8/base1/merge2、radius[.5,50]。実測はA8N。他の範囲・Graph・dtypeは未検証/拒否。
selectorは学習状態ではないためcheckpoint復元後に再指定する。厳密な再現には記録した
コミットのsource・catalog・凍結Caseを使う。
