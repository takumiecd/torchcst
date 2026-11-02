# Kernel 三分割 API の引き継ぎと検証（2026-11-02）

## 対象と保全

Antigravity の作業ブランチ `refactor/unify-kernel-chart-api` を引き継いだ。
Step 1/2 は `07f0637a`。未コミットだった Step 3 の差分と `factories.py` は
ignored `output/antigravity-handoff-20261102-154127/` に元の状態を保存した。
元の会話タスクリストは `~/.gemini/antigravity/brain/8170020f-e0e3-4b5b-a6ad-84169b2cec5a/task.md`。
Step 4 の grid / geometry 非 Tensor 化はこの変更には含めない。

## 宣言の契約

- Profile: `ChebyshevSpec` / `GaussianSpec` / compact profiles。
- Factor: `point` / `l2` / `cell_mass`。セル質量 1 は積分評価の定義に含め、
  点評価の `NormalizationSpec` と区別する。
- 座標則: Tensor を持たない `CoordinateSpec`。`polar` / `direct` / `physical` が返す。
- 結合: `product(factor, coordinates=...)` または `radial(factor, coordinates=...)`。
  完成した KernelSpec に合成し、既存の state 所有・実行・更新の経路へ渡す。

Weight 軸、Factor、中心列はすべて `(output, input)` 順。
`product` は各 Chart 全体から作った Factor の積であり、Chart 内の座標軸ごとの積ではない。
`radial` は同じ Chart ペアから次の値を作る。

\[
q_{oi,a}=\frac{d_{out}(o,c_{out,a})^2}{\sigma_{out,a}^2}
         +\frac{d_{in}(i,c_{in,a})^2}{\sigma_{in,a}^2},\qquad
W_{oi}=\sum_a A_a\mathcal N_{oi}[\phi(q_{oi,a})].
\]

`l2` の正規化は全 `(o,i)` サイトに一度だけ適用する。Gaussian は log-softmax、
compact は宣言された正の norm floor を使う。Polar / Direct の幅は既存と同じく task
勾配から detach する。境界は Chart ごとに設定できる。
`BandwidthBounds` の sigma と `RadiusBounds` の物理支持半径は別の単位として維持する。
`radial(cell_mass(...))`、点評価と半径座標の混在、固定幅と動的幅の二重宣言を拒否する。

二つの Chart の radial は Torch materialized 参照実装。専用 CUDA / factorization や
速度改善の主張はない。Sphere / FlatTorus の中心更新、optimizer moments の輸送は既存の
座標更新実装を使う。旧 presets の互換 alias は作らない。

## ローカル検証

- `python -m tools.kernel_dev test --suite cpu`: **1,688 passed / 1,842 skipped**。
  この後に追加した CUDA パラメータを含む対象検証は **80 passed / 17 skipped**。
- 独立した非対称 2D-output / 1D-input の scalar 式で、固定幅・Polar・Direct、
  Gaussian・Triweight、raw・L2 の Y / dX / 全 atom 勾配を比較。
- Gaussian の product / radial で Adam の3 step、Parameter・moments・step を比較。
- Sphere / FlatTorus 中心更新、weights-only checkpoint 復元後の次 step、空 atom、
  custom compact norm floor の forward と解析微分を確認。
- runtime `ruff check src`、wheel / sdist build 通過。
- `tools.kernel_dev check` で現存する radius の N128 / N1024 Case を検査。
  Step 1 の削除で切れていた CI の local-contraction 参照も置き換えた。

ログは ignored `output/handoff-*.log`、配布物は `output/handoff-dist/` に保全。
CUDA 回帰の実機結果は、共有 L4 pool の実行・hash 検証付き回収後に追記する。
性能測定・CUDA Graph の新規資格付け・DB / dispatcher 採用は今回の対象外。
