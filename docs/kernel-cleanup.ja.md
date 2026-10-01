# Kernel の共通宣言への整理（2026-10-01）

旧 API の互換を廃止し、CSTLinear / CSTConv2d は KernelSpec を受け取る。
共通 KernelState / ProfileState が固定 Tensor と checkpoint を管理し、計算は
backend が担当する。学習する atom 座標は Atoms が所有する。

削除したのは Kernel / Profile ABC、Gaussian / compact Profile Module、Separable /
Amplitude / AmpWidth / DirectAmpWidth / PolarAmpWidth Module、
AmplitudeBandwidthSeparable alias と `_declarations.py` adapter。
旧 checkpoint の読み替え、custom subclass の演算メソッドも残さない。
旧実装を再現する場合は基準 Git revision `c21ea97` を使用する。

profile は関数の形、parameterization は座標から振幅・幅・中心を解釈する宣言。
KernelSpec はその合成と正規化・初期化・更新方針を固定する。preset は Spec の組立。
実行時の chunk / checkpoint は別の KernelOptions とする。動的幅と profile の
固定幅の二重指定、separable での operator-site 正規化は拒否する。
backend は exact な Spec 型・版・初期化・更新方針を確認し、未対応の意味を拒否する。

新 API の例と配置は [Kernel のガイド](../src/torchcst/kernels/README.md)を参照。
Torch / CUDA、Operator、CSTOptimizer、継続する benchmark と実験プログラムを
同じ共通状態へ接続した。数式は既存の backend から移し、input / output の異なる
profile 宣言はそれぞれの形で評価する。固定 radial と radial amplitude の共通実行も
用意した。log-width の汎用 Torch 参照は provided-atoms 契約で、Layer 自動初期化は
拒否する。公開 NormalizedStripLinear の既存 full / window 実行は維持する。

## CPU の検証

- source suite: **480 passed / 196 skipped**。skip は CUDA 等の実行条件による。
- 独立した wheel 展開先で suite: **480 passed / 196 skipped**。
- `torch.load(weights_only=True)` による model / optimizer checkpoint 往復。
- hot execution で declaration snapshot を呼ばないこと、Torus の温まった準備で
  `aten::item` / `_local_scalar_dense` が出ないことを確認。
- baseline `c21ea97` と新実装を別プロセスで比較。separable、amplitude、
  amplitude width（interpolating / inverse）、direct、polar と、Euclidean /
  intrinsic Sphere の組合せ **12 cases × 8 tensors** は float64 で完全一致。
  比較対象は初期 atom p、全 atom weight、出力、dX、全 atom gradient、勾配射影、
  座標更新、optimizer vector state の輸送。許容誤差は rtol=0 / atol=0。

比較プログラム・旧 source・Tensor・wheel・生ログは ignored
`benchmarks/cuda/linear/evidence/kernel-cleanup/` に保存した。
旧 constructor / alias / checkpoint 移行専用のテストは削除・置換し、数学的な
検証は新しい宣言と backend の関数へ移した。profile と幅の独立性、未知の型・版・
更新方針の拒否、入出力別の shape と勾配、radial amplitude を追加確認した。

log-width の追加検証では、極端な入力で先に log の範囲を制限してから指数を取る。
独立した radial oracle との値・全 atom 勾配、有限性、境界外の幅勾配ゼロ、
provided-atoms 初期化の拒否、Euclidean 更新を確認した。

## GPU の確認

共有 L4 キューで installed-wheel suite と既存の normalized Strip の独立 oracle、
Graph、完全 step の時間・peak allocated memory を確認する。性能改善や新しい
既定 dispatch への採用を目的とする変更ではない。GPU 結果は取得後に追記する。
