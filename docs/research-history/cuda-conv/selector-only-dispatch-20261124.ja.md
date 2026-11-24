# Selectorに統一した層のディスパッチ

[English](selector-only-dispatch-20261124.md) | **日本語**

CSTLinearとCSTConv2dの実行設定を `selector=` に統一した。Conv2dのパッチ構成は
従来のサイズ判定でfactoredとmaterializedを選ぶ。groups付きのパッチ畳み込みは
materializedを選ぶ。チャネルタップの既定はTorch重み合成。パッチの重み合成と
2つのFourier PlanはFixedSelectorで指定できる。すべてのConv2d Planが共通Dispatcherを通る。

CPUスイートは2,226件通過、2,650件スキップ。wheelとsdistもビルドした。
RTX 3090、PyTorch 2.6.0+cu126、CUDA 12.6で対象テスト126件を通過した。
WideResNetの9種類の畳み込み形状に対する両Fourier Plan、CUDA Graph replay、
Linearの既定選択と明示指定した有理半径の大幅Linearを確認した。

WideResNet-22-2のスモークテストでは、合成入力バッチ2でCSTOptimizerのモーメンタム付きSGDを
1ステップ実行し、畳み込み21層すべてのAtomを更新した。更新後の評価logitは
固定した両Fourier Planと既定のSelectorで一致した。stemと分類器には通常のTorch層を使った。
これは1ステップの正しさの検証である。

検証したソースは `985ec3dcff677839c7dc172c940b74b89ffc83e5`。証拠は作業ツリーの除外パス
`output/selector-only-dispatch-20261124/` とKyutechのホームからの相対パス
`cst-conv-channel-20261124/985ec3dc/` に保存した。同じランタイムで新しい実験ディレクトリの
`source/` にアーカイブを展開し、`bash run.sh test` と `bash run.sh wideresnet` を実行する。

| ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `6123d120bbf935e1f9d31a5aacad8225af47ea85daafe7f381dbbdec9b040c77` |
| `run.sh` | `dac21a3ceaf67a10498cd0e894da53093741457e0bce9db5b9a363ddff6f3cae` |
| `test.log` | `8869de207a8afdbfcc7caf673f0a5a5d3f021dfbd756ca911ee8b578eb37fe27` |
| `wideresnet.json` | `dba9527baefd63d2622939efd8de69c87ab0dfb7f4167e4b61327986e8a35a31` |
| `verify_wideresnet.py` | `d7cd94501515cb77234fedfed29c4acb728bc6fb37a2b724c48c51ceca262ba9` |
| `rigl_wide_resnet.py` | `a64ac48628c1a94f1e7f391cfed3b6dc1c9829fc61ecc13c0b90f4b336edaca6` |
