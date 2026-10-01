# CSTOptimizer 移行の記録

変更前の checkpoint は `33504c6`。ユーザーは旧 optimizer 全体の削除と
後方互換の破棄を明示的に承認した。

## 変更

- 公開 optimizer は `CSTOptimizer` と `OptimizerStateAdapter` の二つにした。
- 旧 AdamR / ParameterAdam / Normalized / Quadratic family、config、functional
  dense optimizer、N/D moment system・solver、専用 AtomGrad 経路を削除した。
- Linear / Conv の backward は通常の backend autograd のみを使う。
- 旧 optimizer 専用のテストと利用ガイドを削除し、Kernel / Geometry / execution
  の既存テスト中にある学習 step は新しい wrapper に移行した。
- historical GPU / MNIST research notes は旧 revision の記録として残す。
  新しい wrapper にその結果を転用しない。

## 検証

ローカル環境は Python 3.11.15、Torch 2.13.0、CUDA なし。
CPU 全 suite は 482 passed / 196 skipped。旧 API の削除により旧 optimizer 専用
テストも削除しており、以前の suite 件数と単純比較しない。
新規 optimizer テストは 39 passed / CUDA 専用 3 skipped。
Ruff と wheel build が通った。

SGD、Adam、AdamW、RMSprop、Adagrad、Adadelta、Adamax、Rprop の Euclidean
complete step を同一初期値・入力・勾配の base optimizer と照合した。
Sphere center と vector state の接空間制約、Polar / Direct のゼロ勾配時の活動
減衰、欠損勾配・lr=0 の扱い、group ごとの lr、scheduler、closure、checkpoint
復元後の次の step、所有と parameter order・state shape の拒否も確認した。

最初の新規テストでは Kernel constructor の必須設定の不足で失敗した。
fixture に amplitude / width 設定を与えて修正した。計算式を変えて解決していない。

GPU に関する実測は後続の L4 記録で区別する。CUDA Graph capture を wrapper
自体がサポートするという主張はしない。有限性検査が同期し、snapshot は atom
table サイズのため、旧 optimizer に対する速度・peak memory の改善も主張しない。
