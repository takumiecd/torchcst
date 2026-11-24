# CSTConv2dからのFourierディスパッチ

[English](conv-fourier-dispatch-20261124.md) | **日本語**

`fourier` と `fourier_active` は、登録したPlanを共通Dispatcherで明示的に選ぶ。
チャネルタップ構成では空間タップごとにAtom IDのまとまりを割り当て、実行時にID順で
読む。チェックポイントと物理行の並べ替えがあっても対応は保たれる。Torchの重み合成も
同じ割り当てを使う。[先行のカーネル計測](radius-channel-fourier-wideresnet-20261124.ja.md)は
内部の重み生成経路を使った結果で、そのソースに紐付く。

RTX 3090（compute capability 8.6）、PyTorch 2.6.0+cu126、CUDA 12.6で
87テストを通過した。公開層の全9形状をFP64 Torchと照合し、3×3・stride 2の非対称
パディング、両Fourier Plan、半径割合と中心を変えた静的Graph replayを確認した。
CPUスイートは2,222件通過、2,650件スキップ。

別のスモークテストでは、cst-researchの `6529bc0c8e8c8cb89816016a4579cfe958eecfa4`
のWideResNet-22-2で対象となる畳み込み21層をすべて置き換えた。各タップ4 Atom、
合成入力バッチ2、Denseのstemと分類器で、CSTOptimizerを通してeagerのモーメンタム付き
SGDを1ステップ実行した。21層でAtom勾配が有限で、すべてのAtomテーブルが更新された。
更新後の評価logitは両Fourier PlanとTorchの重み合成で一致した。
モデル全体の速度と精度は未計測である。

検証したtorchcstソースは `fafae9033a3cfff62a8a1f2faadb64feb69be3b1`。
ソース、スクリプト、ログは作業ツリーの除外パス
`output/conv-fourier-dispatch-20261124/` と、Kyutechのホームからの相対パス
`cst-conv-channel-20261124/fafae903/` に保存した。同じランタイムの新しい実験ディレクトリで
アーカイブを `source/` へ展開し、`bash run-model.sh test` と
`bash run-model.sh wideresnet` を実行すると再現できる。

| ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `232498b24ffb98427aebdafc8bb07ef8bdf078e7eb14f16e22183a297f2b22ec` |
| `test.log` | `1c1249b4f602eaa4f9bbf54237b87db2606cdc3e861e79c04a70bf2d7d39c79c` |
| `run-model.sh` | `d78d60a4ad352e1b5215a0b3fecf4c219124f6ffb5a78c502478ca7aed2e1c05` |
| `verify_wideresnet.py` | `ea9a723aba6b46a09537e98a813364d6114e1c9acdffaf7395de8c8193ba6129` |
| `rigl_wide_resnet.py` | `a64ac48628c1a94f1e7f391cfed3b6dc1c9829fc61ecc13c0b90f4b336edaca6` |
| `wideresnet.json` | `dba9527baefd63d2622939efd8de69c87ab0dfb7f4167e4b61327986e8a35a31` |
