# RTX 4090での1×1周期Gaussianショートカット

[English](pointwise-gaussian-20261118.md) | **日本語**

CIFAR-10の二箇所CSTヘッド版の最後のショートカットを対象とした。seed 17の
チェックポイントを読み込み、前回の単体検証と同じバッチと出力勾配を再生成した。
入力はchannels-lastの `[256, 96, 8, 8]`、stride 2、padding 0、groups 1。
出力は `[256, 128, 4, 4]` で、Linearとしては4,096行。CSTの全経路で
同じ1,774 Atom、ゲイン、両軸の固定σ=0.75のL2正規化周期Gaussian点評価を使った。
独立参照との照合には、同じ入力に対して、保存済みseed 17の学習後Atomとゲインも用いた。

## データフロー

CSTLinearとCSTConv2dで使うTorchの因子準備から `F_in[C_in,A]` と
`F_out[C_out,A]` を受け取る。振幅は出力因子に含まれる。対象画素を行に並べ、
Tritonの各ブロックで最大32行×32 Atomずつ入力応答と出力集約を続けて計算する。
Atom応答はブロック内で消費する。最終設定はAtomを4分割し、試験形状で一時的な
部分出力に8 MiBを使う。二段畳み込みは27.72 MiBの
`[256, 1774, 4, 4]` をbackward用に保存する。autograd hookで、新経路が
抽出済みの入力行と因子を保存することを確認した。部分出力の分割数はAtom数に
かかわらず最大4。

Backwardはforwardの因子を再利用する。dXは因子を交換して計算し、strideで飛ばした
画素にゼロを入れる。因子のVJPは出力チャネルごとに `dY_stripe.T @ X` を作り、
同じ区画を両方の因子に縮約する。今回の最大区画は32×96スカラー。振幅と座標の
勾配には既存の因子autogradを使う。計測は画素抽出、出力レイアウト、因子準備、
strideに沿ったdXの書き戻しを含む。

先行Linearのpacked-levels比較はM8 cell-massを使い、CIFAR-10の中間層は
2,048×2,048・256行だった。矩形recipeは特定の幅と最大256行に限定され、
owner-window foldの検証対象はN1024・最大4行だった。foldがまとめるのは、
セル積分済み有限Fourier基底上で同じ標本を持つ周波数である。今回は元の
周期距離と正規化でGaussianを点評価した。

## 測定

同じRTX 4090を使用し、各起動前に空きを確認した。CC 8.9、driver 570.86.15、
CUDA 12.8、cuDNN 9.10.2、Torch 2.9.1+cu128、Triton 3.5.1。経路の実行順を変えた
独立2プロセスで、各経路10回のウォームアップ後に同期付きCUDAイベント60試行。
表は2回の中央値の範囲。Tensorと累積はFP32。PyTorch matmulとcuDNNはTF32を
無効にし、新経路の内積は3回のTF32積でFP32相当の精度を得る。
層の完全stepはdXとCSTAdamW（lr=0.001）を含む。DenseはAdamW（lr=0.001）。
外部から渡すdYは固定した。

| 経路 | Forward ms | Backward ms | F+B ms | 層のstep ms | Forward追加allocated MiB | F+B追加ピークallocated MiB | 合計ピークreserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CST二段畳み込み | 0.994–1.005 | 1.125–1.130 | 2.120–2.136 | 2.238–2.245 | 37.42 | 97.42 | 182 |
| CST重み合成＋一段畳み込み | 0.575–0.582 | 0.299–0.305 | 0.874–0.888 | 1.607–1.628 | 7.46 | 23.46 | 62 |
| CST streamed | 0.833–0.839 | 0.404–0.460 | 1.232–1.304 | 1.868–1.896 | 17.29 | 22.71 | 62 |
| Dense FP32 | 0.020 | 0.084–0.089 | 0.103–0.109 | 0.147–0.207 | 2.00 | 18.00 | 54 |

追加allocatedは計測直前の24.35 MiBを引いた値。完全stepの追加ピークも
表の精度ではF+Bと同じ。reservedはPyTorch allocator全体のピーク。
層単体の計測にはPyTorch allocator統計を使った。CST全体の測定はeagerで行った。
既存RegularGridの因子準備にcapture中のホスト側境界検査があるためである。
準備済み因子からのCUDA演算は、入力と因子を変化させたGraph replayを通過した。
DenseのF+B capture/replayの値はJSONに収録した。

独立FP64参照は周期Gaussianを全格子で評価し、通常畳み込みをautogradで微分する。
全4,096行と全Atomを照合した。新経路の2 run・初期および学習後での最大相対L2差は、
出力2.50e-7、dX 2.22e-7、振幅2.23e-7、出力座標2.40e-7、入力座標3.28e-7。
最大絶対差は3.36e-6。CUDA回帰テストは、奇数の空間サイズ、非対称stride、
レイアウト、入力またはAtomの凍結、空バッチ、準備済み因子のGraph replayを含め
22件すべて通過した。

12のタイル設定を調べ、Atomを4分割にした。7分割のforward部分出力は14 MiB。
4分割は8 MiBで、設定比較でのF+B中央値が最短だった。最初のパイプライン設定はAdaの
共有メモリ上限を超えたため、実測ではパイプラインを1段にした。

## 3×3への判断

二段畳み込み比で、新経路はF+Bを39〜42%短縮し、追加ピークallocatedを76.7%削減した。
重み合成経路は両runで速く、F+Bの追加ピークは新経路より0.76 MiB多かった。
新経路は明示的に選ぶ研究用バックエンドとする。入力96チャネルの3×3は
864特徴になる。次の検証候補は、画素をチャネルとオフセットのタイルに直接読み込み、
重なる画素のdXを集約する経路である。重み合成Conv2dと時間、ピーク、独立参照を比較する。
3×3の入力Chartは別の配置を宣言できるため、因子はその並びを使う。

## 再現と証拠

測定コミットは `14b9a7792cd9cffcb5bf1ab2b7b4d7ddee8f24e8`。
Kyutechの隔離実行PIDは63739と63780。固定fixtureと元checkpointは実験ホストに保全した。
ドライバ、ログ、ソースと結果はignoredの
`output/cifar10-cnn-cst-conv-streamed-20261118-01/` に保存した。
ドライバは前回と同じラボルート内のリモートディレクトリ
`cifar10-cnn-cst-conv-streamed-20261118-01` を使う。そこに `reproduce.sh`、
`freeze_fixture.py`、`measure.py` とソースアーカイブを置く。アーカイブを
`source-v5/` に展開し、`frozen.pt` がなければ `bash reproduce.sh fixture2` を
実行する。起動コマンドはPIDを表示して戻る。最初の計測は `frozen.pt`、
2回目は `final_1.json` ができてから起動する。取得したJSONはローカルの
`verify.py` で照合する。

```sh
bash reproduce.sh final_1 factored,composed,streamed,dense_fp32
bash reproduce.sh final_2 dense_fp32,streamed,composed,factored
python3 verify.py final_1.json final_2.json
```

| ファイル | SHA-256 |
| --- | --- |
| `frozen.pt` | `d725da274f105c22aece28bfa89856953a41922d9bce93093f2f6ea342f7dd85` |
| `source-v5.tar.gz` | `1e094d11a7803a7650f2610528968381bda9abdc71bfe77f7435d1f98b84c595` |
| `measure.py` | `ae81dcf760fdf87f53df88e6a64c857fa00921bd7655f6b27ce63b87c5a63e46` |
| `final_1.json` | `57e8ee812b6347fb258206a857489d9c002bfe6964e7e1ebda73003d14d0d69c` |
| `final_2.json` | `bb4c1ef8d7f8113d5e95bc9aa7e4ac787e149a6f791c4f06e7b74a853831d930` |
