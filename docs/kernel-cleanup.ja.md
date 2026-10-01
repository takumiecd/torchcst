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

- source suite: **481 passed / 197 skipped**。skip は CUDA 等の実行条件による。
- 独立した wheel 展開先で suite: **481 passed / 197 skipped**。
- `torch.load(weights_only=True)` による model / optimizer checkpoint 往復。
- hot execution で declaration snapshot を呼ばないこと、Torus の温まった準備で
  `aten::item` / `_local_scalar_dense` が出ないことを確認。
- baseline `c21ea97` と新実装を別プロセスで比較。separable、amplitude、
  amplitude width（interpolating / inverse）、direct、polar と、Euclidean /
  intrinsic Sphere の組合せ **12 cases × 8 tensors** は float64 で完全一致。
  比較対象は初期 atom p、全 atom weight、出力、dX、全 atom gradient、勾配射影、
  座標更新、optimizer vector state の輸送。許容誤差は rtol=0 / atol=0。

比較プログラム・旧 source・Tensor・wheel・生ログ（失敗した GPU 検証も含む）は ignored
`benchmarks/cuda/linear/evidence/kernel-cleanup/` に保存した。
旧 constructor / alias / checkpoint 移行専用のテストは削除・置換し、数学的な
検証は新しい宣言と backend の関数へ移した。profile と幅の独立性、未知の型・版・
更新方針の拒否、入出力別の shape と勾配、radial amplitude を追加確認した。

log-width の追加検証では、極端な入力で先に log の範囲を制限してから指数を取る。
独立した radial oracle との値・全 atom 勾配、有限性、境界外の幅勾配ゼロ、
provided-atoms 初期化の拒否、Euclidean 更新を確認した。

## GPU の確認

最初の全体 suite は `fb00fcf` で 739 passed / 1 failed / 2 skipped。失敗は
`test_captured_geometry.py` が廃止済みの `profile.sigma` を変更していたためで、
実際の幅 buffer `kernel.scalar("sigma_min")` を変更するテストへ修正した。

| source | L4 job | 検証 | 結果 |
| --- | --- | --- | --- |
| `fb00fcf` | `l4job-46fe4e458c5b461597c6a2aecd6ec56e` | installed-wheel 全体 suite | 739 passed / 1 failed / 2 skipped |
| `b608e7d` | `l4job-c83a6a08431740d891bed39ed47d5f07` | Kernel、log-width CPU/CUDA oracle、checkpoint、optimizer、single-chart Direct | 102 passed |
| `77df7c2` | `l4job-0bf5860b2b2e46f9a8e66da35a05f82f` | 修正した capture、Kernel、normalized Strip、dispatch、Triton | 170 passed |

全体検証と対象の再検証を合わせると、重複を除き **742 passed / 2 skipped / 未解決の
failure なし**。最終 commit で全体 suite を新たに一括実行した件数ではない。
最後の wheel 内・展開先の Python 139 module は、現在の source と全て SHA256 一致。
wheel SHA256: `43e6cc76710a94843d21df3773d82da30329cee08043a722d5d1d13b1031f1ab`。

実機は NVIDIA L4 23034 MiB、driver 580.82.07、Python 3.13.15、Torch 2.11.0+cu128、
CUDA 12.8、Triton 3.6.0。結果 archive は共有 pool が SHA256 検証して回収した。
所有した runtime は全て停止済み。

## 完全 step の時間とメモリ

N=1024、batch=128、atoms=52429、float32、同じ精度・入力初期値・optimizer 契約。
forward / backward / optimizer を含む。eager と CUDA Graph replay の中央値を分け、
peak allocated は Graph capture を含む。幅の通常ケースと sharp-only ケースを分ける。

| fixture | 実行 | eager ms | Graph ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| broad sigma=3 | window | 2.839 | 1.510 | 46.162 | 86 |
| broad sigma=3 | full | 1.664 | 0.628 | 51.848 | 106 |
| broad sigma=3 | dense baseline | 0.724 | 0.126 | 50.502 | 106 |
| sharp sigma=.199 | window | 2.832 | 0.306 | 46.162 | 86 |
| sharp sigma=.199 | full | 1.635 | 0.549 | 51.848 | 106 |
| sharp sigma=.199 | dense baseline | 0.694 | 0.127 | 50.502 | 106 |

整理前の `3294bac` と初期 p の hash・shape・optimizer 契約が一致し、測定した
allocated / reserved peak は一致した。optimizer は plain fused capturable AdamW で、
CSTOptimizer の性能測定ではない。全 GPU process 使用量は未測定。小さい混合 fixture
の独立 FP64 oracle と Graph 更新検証が通過し、full-shape の独立全 atom oracle は
実施していない。時間の違いを性能改善や新しい既定 dispatch の採用根拠にはしない。

数値・error・job / source / archive hash は
[機械可読な結果](../benchmarks/cuda/linear/results/kernel-cleanup-20261001.json)に保存。
生ログ、失敗した job、wheel、source archive は ignored evidence に保存している。

継続して全体を再検証する場合は、clean な研究 branch で以下を実行する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" \
  --script "$PWD/benchmarks/cuda/linear/validate_dispatch_registry.py" \
  --timeout 900 --label kernel-validation -- "$(git rev-parse HEAD)" --full-suite
```

共有 pool の既存 supervisor がいなければ `serve --workers 1 --idle-seconds 0` を起動する。
詳細は AGENTS.md と共有 pool skill を参照。
