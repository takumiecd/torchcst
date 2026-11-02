# 固定グリッド・幾何の非 Tensor 化（Step 4）

## 境界と変更

Step 3 の計算ソースは `cfcd9524`。共有 L4 で **433 passed / 1 skipped** を確認した後、
ユーザーの指示で Step 4 を続行した。Step 3 の証拠は
[API 統一記録](api-unification-step3-20261102.ja.md)に保全する。

- RegularGrid の origin / spacing と Geometry の radius / chart_margin / periods を
  immutable な Python metadata に移した。これらの State は固定 Tensor buffer を持たない。
- `.to()` は Tensor を保持しない placement hints を更新する。空の一時 Tensor に変換関数を
  適用して device/dtype を確認するが、固定値や Tensor storage を保存・量子化しない。
- 実行時の Grid 距離は中心 Tensor の device / dtype で座標を生成する。
  初期化後の CUDA 更新の placement は実際の `atoms.p` が正本。
- CUDA Polar の周期 Tensor は AlgorithmState.buffers に置き、周期・atom placement・定数の
  version が変わると再構築する。compiled launch も定数 version を確認する。
- 固定状態の signature は Spec の値を含め、native guard は Spec の辞書 version を確認する。
  property getter の code 変更も prepared callback の認証から外れることを確認する。
- checkpoint は全固定 metadata を extra state に保存し、復元先の一致を要求する。
  旧 buffer 形式や異なる固定値を読み替えない。

数学的な距離・積／radial・正規化・Polar 更新の式は変更しない。固定値は宣言の精度で保持し、
従来の FP32 buffer を後から FP64 化した値としては扱わない。
小さい spacing は metadata のまま保持し、出力座標の低精度丸めと区別する。
非有限の出力座標になる dtype / 配置は生成時に拒否する。

## 検証と再現

- CPU: `python -m tools.kernel_dev test --suite cpu` — **1,698 passed / 1,853 skipped**。
  最後に追加した getter 認証を含む対象検証は **9 passed**。
- runtime ruff、wheel / sdist build、README と Geometry の独立した実行例を確認。
- 固定 Tensor ゼロ、dtype 往復の metadata 精度、異なる constructor dtype のまま FP64 atom
  中心を渡す距離と勾配、weights-only reload、宣言の不正変更後の cache 検査を確認。
- 2 種の非対称周期を使う CUDA 更新と、geometry placement hints を変えた後も atom placement
  に従う検査を追加した。実機は別 snapshot で検証し、完了後に結果と hash を追記する。

ローカルログ: ignored `output/handoff-step4-*.log`。新しい速度や完全 step のピークメモリの
測定は行わない。固定状態の Tensor 数と学習全体の memory は区別する。
