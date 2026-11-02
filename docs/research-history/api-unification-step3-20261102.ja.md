# Kernel 三分割 API の引き継ぎと検証（2026-11-02）

## 対象と保全

Antigravity の作業ブランチ `refactor/unify-kernel-chart-api` を引き継いだ。
Step 1/2 は `07f0637a`。未コミットだった Step 3 の差分と `factories.py` は
ignored `output/antigravity-handoff-20261102-154127/` に元の状態を保存した。
元の会話タスクリストは `~/.gemini/antigravity/brain/8170020f-e0e3-4b5b-a6ad-84169b2cec5a/task.md`。
Step 4 の grid / geometry 非 Tensor 化は、この Step 3 checkpoint には含めない。

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


## 実機で検出した追加修正

- `9c91fe86` の L4 回帰は **261 passed / 2 failed / 1 skipped**。
  失敗は二つの native public 3-step ケースで、Chart 単一階層化後も
  prepared Linear が `ChartSpec.coordinate_shape` を探していたため。
- `4ec54b8c` で検査対象を `RegularGridChartSpec.coordinate_shape` へ移し、
  その定義を seal する。検査を削除せず、callback 一覧には実在する関数を要求する。
  CPU の検査一覧テストと、property 置換時に prepared を拒否するテストを追加した。
- `8bbfca65` では KernelSpec / KernelState に `center_order="charts"` を記録。
  shape だけでは区別できない旧中心列順の model / optimizer checkpoint を拒否する。
  wrapper と native CSTAdamW の双方を検査した。
- 最終計算ソース `4ec54b8c`: CPU **1,692 passed / 1,854 skipped**、
  runtime ruff、wheel / sdist build 通過。ローカル Python 3.11.15 / Torch 2.13.0。
  ログは `output/handoff-cpu-qualified.log` / `output/handoff-build-qualified.log`。
- 先行 GitHub CI (`9c91fe86`) は Ubuntu / Torch 2.11.0 で
  **1,687 passed / 1,855 skipped**、宣言検査・build も通過。最終 HEAD の CI は別途確認する。

GPU の準備失敗はコードの数値結果から分ける。初回は Colab Python 3.13 の ensurepip、
次は job-local ninja の PATH が原因。いずれも raw logs を pool job に保全した。
システムへの依存追加はせず、job-local target と PATH、事前コンパイルで修正した。
最終 L4 job は `l4job-7e17660aad874c53a371b68cca96ed4f`、回収結果を追記予定。


### prepared の拡張検証

`4ec54b8c` の再実行では、主要な数値・native public 経路に新規 API / callback 一覧検査を
加えた **267 passed / 1 skipped** の後、追加 prepared-entry 検査の3件が失敗し停止した。
共通 `original_attribute` は「クラス属性が存在しない」という認証も担っており、その挙動を
変えると instance 所有属性の資格付けを失って prepared が常に None になるためだった。
共通の不在認証は元に戻し、`_definitions` の静的取得だけに default=None を指定する。
存在しない callback は関数として認証せず、instance 属性の不在認証と区別する。
この境界を CPU テストでも固定した（対象 **24 passed / 12 skipped**）。

macOS の明示的 native-guard 検査は `get_factory()` が None のため実行不能（2 setup errors）。
これは通常 CPU suite では明示的に skip される経路であり、L4 の Linux 上で実機検証する。


## Step 3 最終 L4 結果

計算ソース `cfcd9524` の job `l4job-ad0aa6f7068c49f1a74e5fe00a5cab0e` は
**433 passed / 1 skipped**。native public 3-step、フック・変更検知を含む prepared Linear / optimizer
の追加検査も通過した。NVIDIA L4 / driver 580.82.07 / Python 3.13.16 / Torch 2.11.0+cu130 /
CUDA 13.0 / Triton 3.6.0。source / result archive をローカルで SHA256 再照合済み。

- source archive: `475b26376e2ddb2cdc4ed6f9f592f90364a712482f60c50977d9969cfb9df7c9`
- result archive: `766039e0d53a9a63af51892274334c9464007add8714915cb1c531197175ec89`
- raw artifacts: `~/.local/state/colab-l4-pool/jobs/l4job-ad0aa6f7068c49f1a74e5fe00a5cab0e/results/`

この結果を Step 3 の境界とし、ユーザーの追加指示により Step 4 を続行する。
Step 4 の変更にはこの実機資格をそのまま流用せず、別 snapshot で確認する。
