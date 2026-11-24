# Conv2d の全階層 Fourier チャネル重み

[English](radius-channel-fourier-all-banks-20261124.md) | **日本語**

[前回の実装](radius-channel-native-rtx3090-20261124.ja.md)は、どの経路でも Fourier
係数を使ってセルを積分する。直接加算は両軸の局所セル質量ベクトルを評価し、Atom ごとの
外積を重みに加える。混合経路は広い窓を Fourier 係数のまま集約し、空間タップごとに実行する。

新しい `channel_fourier_weights` は、周期2サイトの最小の窓を含め、すべての到達可能な
階層で Fourier 係数を集約する。チャネル軸ごとに周期境界を持つ Chebyshev M8、
spacing 1、base tile 1、merge factor 2、半径上限32の座標契約を使う。
空間タップはそれぞれ Atom 行を持ち、合成した重みを通常の Conv2d に渡す。

1. 元の各周波数をセル区間で積分し、owner タイルの位相へ移す。整数サイトでは局所周期だけ
   離れた周波数のサンプルが一致するので、sine の符号と偶数周期の Nyquist 項を含めて
   係数をまとめる。実係数の数は最大 `min(17, period_sites)` となる。
2. タップ・半径 bank・周期境界上の入出力 owner ごとに、Atom の係数の外積を集約する。
   ソート経路は実行インデックスを安定ソートし、同じ owner を縮約してから書き込む。
   ソートなしの経路は係数バッファに atomic 加算する。
3. 2または3タイルの窓を両側のチャネルトーラスに折り返し、bank ごとに全タップ分を
   行列積で展開する。backward は Conv2d の重み余接を同じ基底に射影し、Atom 係数の
   微分と縮約する。半径と中心の導関数にも同じ周波数の集約を適用する。

固定範囲の版は宣言された全 bank を処理し、準備済みの物理座標で Graph replay できる。
eager 版は実際に Atom が入った bank のビットマスクを読み、その forward で占有された
bank だけのバッファを確保して実行する。backward に使うマスク、幾何情報、ソート順は
各 forward が保存する。計測での rational 復号と optimizer は eager で実行した。

## 計測条件

RTX 3090で、FP32 channels-last の `[256, 96, 8, 8]`、出力128チャネル、stride 2、
1,773 Atom を使った。初期の物理半径は狭い条件で約1.44–6.71、広い条件で11.32–30.68。
いずれもチャネル方向の半径である。1×1、3×3それぞれで入力と重みのレイアウトを揃えた。
各ステップに rational の半径復号、重みの合成・レイアウト変換、畳み込み、backward、
射影付き fused AdamW（学習率0.001、weight decay 0）を含めた。Dense は対応する実効重みで
初期化し、通常の fused AdamW を使った。各条件で8回の warmup と24回の同期計測を行い、
二つの seed で比較順を反転した。TF32 は無効にした。

最終比較は compute capability 8.6、PyTorch 2.6.0+cu126、CUDA 12.6、cuDNN 9.5.1、
driver 580.95.05で、PID 4999、5014の独立プロセスで行った。表は各プロセスの
CUDA event 中央値の範囲（ms）である。

| 条件 | Dense | 直接加算 | 全 Fourier・固定範囲 | 全 Fourier・占有 bank | 占有 bank・ソートあり |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1×1 狭い半径 | 0.289–0.370 | 1.297–1.312 | 1.734 | 1.550–1.852 | 1.589–1.950 |
| 1×1 広い半径 | 0.268–0.311 | 2.425–2.596 | 1.740–2.003 | 1.510–1.788 | 1.538–1.801 |
| 3×3 狭い半径 | 1.009–1.015 | 1.591–1.631 | 2.332–2.464 | 1.909–1.925 | 1.955–2.165 |
| 3×3 広い半径 | 0.976–1.079 | 2.791–2.823 | 2.293–2.311 | 1.821–1.828 | 1.854–1.871 |

3×3の広い半径では、占有 bank を選ぶことで固定範囲の Fourier のピーク allocated が
60.69–61.49から42.53–42.57 MiBに下がり、reserved は80から66 MiBになった。
Dense は allocated 43.01–43.05 MiB、reserved 64 MiB。3×3の狭い半径では、占有 bank 版の
allocated は54.19–54.99 MiB、reserved は64 MiBだった。これらは PyTorch allocator の
総ピークであり、GPU プロセス全体は未計測である。

3×3では狭い半径で直接加算、広い半径で占有 bank 版の Fourier が最速だった。
後者はタップ単位の Fourier level 4の3.141–3.144 msを下回った。1×1の広い半径では
タップ単位の level 5が1.487–1.757 msで、占有 bank 版に近かった。占有 bank 版では、
全4条件と両 seed でソートすると時間が増えた。合成した中心・半径での比較であり、
実モデルでの採用には学習済み Atom の分布とモデル単位の計測を要する。層ステップは
全4条件で Dense が最短だった。

## 数値検証と再現

最終 extension は41件の個別テストを通過した。階層境界、両入力チャネル幅、独立した
セル積分、トーラスの周期シフト、全一次勾配、forward ごとの異なる eager マスク、
固定範囲の CUDA Graph replay を確認した。性能計測の条件では、全 Fourier 経路の
W、Y、dX、振幅、半径割合、両中心勾配が CPU FP64 参照と最大相対 L2 誤差1.85e-6で一致した。
最初の固定範囲 Fourier の試験では既存の幅2048の Linear CUDA テスト2件も通った。
最終 CPU スイートは2,219件成功、2,627件スキップ。wheel と sdist のビルドも通った。

最終計測ソースは `17c8a60248200b60018fb07172c74a5ad76ab8b7`。アーカイブ、driver、
ログ、結果は worktree の Git 管理対象外の `output/conv-active-fourier-20261124/` と
Kyutech ホーム相対の `cst-conv-channel-20261124/17c8a602/` に保存した。
最初の固定範囲の試験は `52d3607adc9e5b2a0430cff9bd2a1279d0245b78` を測定し、
`output/conv-all-fourier-20261124/` と `cst-conv-channel-20261124/52d3607a/` に保存した。
対応するソースを `source/` に展開して driver を展開し、
[前回のCUDA試験](radius-channel-native-rtx3090-20261124.ja.md)と同じく Ninja、
CUDA ヘッダーと隔離した extension cache を設定して実行する。

```sh
export PYTHONPATH=source/src:source:source/tests TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=2
python -m pytest -q test_radius_channel_weight_cuda.py
python measure.py --seed 17 --source-commit 17c8a60248200b60018fb07172c74a5ad76ab8b7 --output run-17.json
python measure.py --seed 23 --source-commit 17c8a60248200b60018fb07172c74a5ad76ab8b7 --output run-23.json
```

| 最終計測ファイル | SHA-256 |
| --- | --- |
| `source.tar.gz` | `ab8ca171e26540023df0c72a8b2759952ed1406952dbba826622783da34eb1e9` |
| `driver.tar.gz` | `42fa54623467f650d33f6a06ae7f160bb16cca14cc4c1e1c5ba72440b4a4704c` |
| `measure.py` | `2f6f2c4ad2cd0df9797981038b08272bb1fdea02bd97ccea79e2c8669dcd050d` |
| `run-17.json` | `68ef999a55d1a328185a7b1e7fbcb8486577ee73358fcac90940f7546b985dbb` |
| `run-23.json` | `3aadd857e3c140332afbb3b9653f7ee8e4dc1e854930729f2a7b1b1aaaf9e345` |
