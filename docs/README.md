# Design documents

`README.md` is the public API contract. The documents retained here describe
the supported continuous-only optimizer surface. Experiment runners and reports live in the companion
[cst-experiments repository](https://github.com/takumiecd/cst-experiments):

- [normalized-optimizer.ja.md](normalized-optimizer.ja.md) — composable normalized optimizers: LinearJGHAtomGrad, independent numerator/denominator moments, replaceable solvers, and elementwise box updates;
- [chart-geometry.ja.md](chart-geometry.ja.md) — Chart/Geometry/Profile/Kernel/optimizerの責務分割、球面retraction、storage幅とintrinsic自由度;
- [strip-torus-gemm-prototype.ja.md](strip-torus-gemm-prototype.ja.md) — Linear backendの分担、Strip + Torusの配置、Triton forward/backward、GPU検証;
- [optimizer-selection.ja.md](optimizer-selection.ja.md) — purpose-based optimizer choice, the Triweight + Polar MNIST comparison, fixed-LR best practices, and retained optimizer API;
- [parameter-adam.ja.md](parameter-adam.ja.md) — parameter-coordinate AdamW path and its bounded state;
- [direct-amplitude-bandwidth.ja.md](direct-amplitude-bandwidth.ja.md) — `(w, q)`座標、physical-displacement `q` 更新、Adam state、Polar checkpointとの非互換性、paired A100 evidence.
- [cst-1024-algorithm-redesign-20260929.ja.md](cst-1024-algorithm-redesign-20260929.ja.md) — L4の1024²小バッチでのCSTアンカー補間の初期誤差と、別モデルである直接低ランク因子の対照測定。アンカー学習の後続測定は下記.
- [flash-cst-l4-20260929.ja.md](flash-cst-l4-20260929.ja.md) — L4の1024²でCSTのW/dWタイルをon-chip融合した前向きとatom勾配の実測。現行分離経路より遅く、完全ステップへの採用を見送った.
- [l4-current-bottleneck-20260929.ja.md](l4-current-bottleneck-20260929.ja.md) — L4の現行1024²小バッチCST完全ステップを工程・kernel・Nsight指標まで分解したボトルネック分析.
- [l4-anchor-atom-training-20260929.ja.md](l4-anchor-atom-training-20260929.ja.md) — 評価siteを減らしatomを学習し続けるアンカーCSTのL4完全ステップ、勾配、メモリ、更新後誤差.

Removed optimizer experiments and their derivations remain available in Git
history but are not part of the package or current architecture.
