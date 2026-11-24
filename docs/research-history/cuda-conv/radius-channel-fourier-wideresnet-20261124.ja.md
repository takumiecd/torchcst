# WideResNet-22-2の層形状でのFourier Conv2d

[English](radius-channel-fourier-wideresnet-20261124.md) | **日本語**

RigLの公開CIFAR-10学習スクリプトでマスク対象になる畳み込み21層は、9種類の形状に
分かれる。この設定ではRGB stemはDenseである。Evciらの§4.3と[固定したRigLモデル](https://github.com/google-research/rigl/blob/d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9/rigl/cifar_resnet/resnet_model.py)
に従ってチャネル数、空間サイズ、strideを揃えた。3×3・stride 2では、偶数サイズの入力の
下端と右端だけに1ピクセルを追加し、両方式でTensorFlowのSAMEに合わせた。

## Fourier計算

空間タップごとにAtomを持つ。入力と出力のチャネルは、それぞれの周期を持つspacing 1の
トーラスとする。両軸とも1セルを基底タイルとして2倍ずつ広げ、両軸の周期を割り切り、
短い軸の半周期以内に収まる2のべき乗を半径上限とする。今回の上限は8、16、32、64。
短い軸の半周期まで広がる局所窓は、重複するトーラス上の像を足し合わせる。

Linearで使っているM8の各調和成分のセル積分と半径微分を再利用した。両軸の半径と
選択バンクが共通なので、順伝播では密度の係数、逆伝播ではセル積分後の係数と半径微分を
共有する。中心の位相と所有タイルの折り返しは軸ごとに計算する。逆伝播では畳み込みの
重み勾配を折り返し込みのFourier基底に射影し、Atom係数の微分と縮約する。
既存の安定ソートは実行順に適用し、保存されるパラメータと勾配のAtom順は保つ。

CUDAデコーダは半径の有理パラメータ化と振幅・半径割合への解析微分をデバイス上で
計算する。半径割合の両端はTorchデコーダと同じ値に符号化する。静的なFourier経路は
このデコーダと組み合わせて1層の学習ステップ全体をGraphにキャプチャできる。
eagerの占有バンク経路は、そのステップでAtomが入ったバンクを読み出して処理する。

## 学習ステップの比較

RTX 3090（compute capability 8.6）、PyTorch 2.6.0+cu126、CUDA 12.6、
cuDNN 9.5.1、driver 580.95.05で独立した2プロセスを実行した。seed 17と23で測定順を
反転した。バッチ128の合成入力、channels-last FP32、TF32無効とし、層ごとのDense重み数に
対してAtomの学習可能スカラーを約10%にした。初期半径割合は狭域が[0, 0.25]、広域が
[0.65, 0.98]の一様分布。半径デコード、重み生成とレイアウト変換、パディング、
畳み込み、dXとパラメータの勾配、fused SGD（モーメンタム0.9、学習率0.1、weight decay
5e-4）、Atomの射影を計測した。Denseの初期値はCSTが表現する重みと揃えた。
各条件でウォームアップ10回、計測30回。Graphは計測前に1回replayした。
速度の計測はプロセス5919と5934で実行した。値はプロセスごとのCUDAイベント中央値の範囲で、単位はms。

| 形状 | 入力 | stride | Dense eager | CST eager | Dense Graph | CST Graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1×1, 16→32 | 32×32 | 1 | 0.326–0.344 | 0.673–0.836 | 0.294–0.297 | 0.467–0.473 |
| 3×3, 16→32 | 32×32 | 1 | 0.526–0.534 | 0.937–0.962 | 0.492–0.493 | 0.699–0.702 |
| 3×3, 32→32 | 32×32 | 1 | 0.674–0.676 | 1.082–1.102 | 0.607–0.636 | 0.813–0.862 |
| 1×1, 32→64 | 32×32 | 2 | 0.274–0.278 | 0.624–0.681 | 0.233–0.234 | 0.459–0.461 |
| 3×3, 32→64 | 32×32 | 2 | 0.828–0.833 | 1.112–1.233 | 0.782–0.790 | 0.994–0.995 |
| 3×3, 64→64 | 16×16 | 1 | 0.651–0.688 | 0.985–0.991 | 0.632–0.634 | 0.888–0.893 |
| 1×1, 64→128 | 16×16 | 2 | 0.217–0.227 | 0.704–1.334 | 0.181–0.181 | 0.425–0.431 |
| 3×3, 64→128 | 16×16 | 2 | 0.957–0.961 | 1.066–1.239 | 0.866–0.878 | 1.220–1.231 |
| 3×3, 128→128 | 8×8 | 1 | 0.805–0.809 | 1.034–1.058 | 0.750–0.752 | 1.550–1.553 |

表は広域半径の結果。CST eagerはCUDAデコードと占有バンク、CST GraphはCUDAデコードと
静的バンクを使う。同じソースのeager占有バンク同士を比べると、CUDAデコードで
Torchデコードの1.18〜2.75倍の速度だった。両実行モードとも、全形状・両半径域でDenseが
速かった。Graphで最も差が小さかったのは3×3 32→64・stride 2の1.26倍。
広域の3×3 128→128ではeagerの占有バンク選択が速かった。静的Graphでは到達可能な全バンクを
射影・展開するので、この形状ではDenseの約2倍のreplay時間だった。JSONには両半径域、
従来のデコードとソートを使った条件、同期込みの壁時計時間も保存した。

複数条件を同じプロセスで回した計測では、Graphキャプチャの後にアロケータの使用量が
持ち越された。そこで広域の3×3 128→128をseed 17、各方式を別プロセスにしてメモリを
再計測した。表はPyTorchアロケータの総ピーク（MiB）で、Graphはキャプチャとreplayを含む。
GPUプロセス全体のメモリは計測対象外。

| 方式 | ピーク割当 | ピーク予約 |
| --- | ---: | ---: |
| Dense eager | 63.99 | 86 |
| CST eager、占有バンク | 63.41 | 90 |
| Dense Graph | 63.99 | 120 |
| CST Graph、静的バンク | 102.48 | 184 |

## 検証と再現

最終計測の両プロセスで全条件のW、Y、dX、4つの有理Atom座標の勾配をCPU FP64参照と
照合した。GPUテストでは独立したChebyshevのスカラー数値積分、タイルの切り替え境界、
トーラス上の窓の重なり、半径バンクと中心が変わるGraph replay、既存Linearの
有理パラメータ化の経路も確認した。

最大相対L2誤差は5.17e-6。GPUテストは65件通過した。CPUテストは2,219件通過、
2,640件スキップで、wheelとsdistもビルドできた。計測は合成入力の層単体ステップであり、
モデル統合、公開dispatcher、CIFAR-10での学習は別途検証する。

最終計測のソースは `74194a5d7ed45227a6c9c31af9f22fd69049835a`。
ソースアーカイブ、ドライバ、ログ、計測結果は作業ツリーの除外パス
`output/conv-wideresnet-fourier-native-20261124/` と、Kyutechのホームからの相対パス
`cst-conv-channel-20261124/74194a5d/` に保存した。初回のeager結果は
`output/conv-wideresnet-fourier-20261124/` と
`cst-conv-channel-20261124/33236517/` にあり、ソースは
`33236517aa1e7e92c883dc9a7745fc7d07d8e95c`。最終版の再現はアーカイブを
`source/` に展開し、`run.sh` と同じCUDA 12.6ヘッダ・Ninjaを設定して実行する。

```sh
bash run.sh test
bash matched-eager/run.sh
bash isolated-memory/run.sh
```

| ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `5feb5ad6564e16cf0e507339fb98bb80a9cc8b45be0ebf9991dcbe504ac7ea93` |
| `test.log` | `a4ba85addaac83e0ffc5b485f7c1155e962267877eddd499d5bc24bf7b58fda9` |
| `matched-eager/measure.py` | `3a26bd0e2ebe68b39bdee28e304da14e393edd9a3ad515c519d32eb624fae0a0` |
| `matched-eager/run.sh` | `97d4f0e592f3c6263da62aa3c5329ac78ec52abe0b6712e45211b6684b6ae377` |
| `matched-eager/run-17.json` | `fdfd67f8334904d223de07c579eb7ee36887028bb16a740e3e58e86243508b86` |
| `matched-eager/run-23.json` | `acaac2e2a7919ca73787c7eb5f18f1dd6b2b2e3c51c7d1e0eb38f47c98939f7c` |
| `matched-eager/final.log` | `ccfc5455dee2cb1fa2b82c1d5196a7ce58698654f9626d8efb6bcc4f17ec3eff` |
| `isolated-memory/measure.py` | `5c707afedfdf57e0a2e5d8d13745e181a00da8ca096aff170fd11f7354a3ee59` |
| `isolated-memory/run.sh` | `144ac7b6f300918ca226fa985695626634bab8ff188e3517d82fb4a3ced33cab` |
| `isolated-memory/run-dense.json` | `99b76244cd3433389ae74e74545f0ff14c7aa70eb8f854153f844d3911f2512d` |
| `isolated-memory/run-cuda-active.json` | `e2001d21eb7dd5f1d01cb143832b8eae70a7cd5d09d03908bd819bd080ab2518` |
| `isolated-memory/run-graph-dense.json` | `fb5a7331005bfabab2026b337864d7403cab6929ea4ed806eeadcf8a1d23fd3a` |
| `isolated-memory/run-graph-cuda-static.json` | `2cb7f30421be2fc1c7cd0362e5861de91c9b96498181f48cbc02fea4fd2bc670` |
