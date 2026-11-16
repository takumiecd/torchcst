# M8 FFN down の一時 Fourier 重み

[English](lm-ffn-fourier-weight-20261117.md) | **日本語**

明示選択の CUDA プランは 4096→1024 の M8 格子と 196,608 個の Atom を
対象とする。16 サイト以上のバンクでは Atom の所属タイルの組ごとに
17×17 の Fourier 係数行列を集約する。位相回転で両軸の 3 つの展開先に
係数を移し、周期境界で戻る寄与もすべて加える。2 回の行列積で係数を
展開する。小さいバンクではセル質量を直接加える。生成する一時重みは
4096×1024 の FP32 で 16 MiB。学習する状態は Atom の行である。

forward と dX には GEMM を使う。Atom の VJP は密な重み勾配 dYᵀX を
同じバンクの基底に射影し、Atom のセル質量および半径と中心の微分を
掛け合わせる。backward 向けにトークンごとの Atom 応答を保持しない。
プロファイル微分と演算の主部分は FP32。アーカイブの rational radius
デコードと VJP の内部では FP64 を使う。

## RTX 4090 での射影の再生

TinyStories の seed 17、step 4882 のチェックポイントから FFN down の
4 層を使った。アーカイブのモデルは全条件に同じ 256 行の入力と出力勾配を
渡す。各層と条件で、5 回のウォームアップ後に測定 24 回のブロックを
順序を反転して 2 回実行した。PyTorch 2.9.1+cu128 と CUDA 12.8 を使い、
TF32 は無効化した。forward は半径デコードと重みの準備を含む。
合計は同期を含む経過時間。各層で中央値を計算して、4 層の中央値を
取った。メモリは全測定ブロックと全層の最大値を載せた。

| 条件 | forward ms | backward ms | 合計 ms | 常駐分からの allocated 増分 MiB | ピーク allocated MiB | ピーク reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| アーカイブ版 | 39.90 | 46.13 | 85.95 | 529.22 | 997.23 | 1020 |
| 広域タイル共有 | 21.26 | 18.00 | 39.04 | 539.56 | 1002.57 | 1020 |
| Atom を整列して広域モーメントを集約 | 4.94 | 3.68 | 8.64 | 553.65 | 1019.67 | 1046 |
| 一時 Fourier 重み | 1.81 | 1.05 | 2.88 | 215.05 | 685.56 | 714 |

アーカイブ版との最大絶対差は Y が 9.16e-5、dX が 6.40e-10、Atom の
4 座標の勾配が最大 7.62e-8。最大相対 L2 誤差は 1.15e-5 だった。最初の
層は行数 1、7、9、65、137 でも検証を通過した。独立した CPU FP64 の
因子計算では全 196,608 個の Atom と有効 3 行を使い、GPU では 72 行に
ゼロパディングした。最大絶対誤差は半径分率の勾配で 2.47e-4、
最大相対 L2 誤差は 6.00e-6 だった。

## 学習更新全体

各測定は step 4882 のモデル・オプティマイザ・乱数状態を復元し、
次の 256 トークンのマイクロバッチ 4 個を使う。loss、backward、勾配
クリップ、CSTAdamW の更新を計時に含む。状態の復元は計時外。
各条件について 1 回のウォームアップの後、4 更新の測定ブロックを
順序を反転して 2 回実行した。

更新内共有では層ごとに一度だけ重みを組み立てる。重みを detach した
leaf に各マイクロバッチの密な重み勾配を加算し、個別の backward で
その活性値を解放する。勾配クリップと最適化の前に Atom の VJP を
層ごとに一度実行する。[学習再生の補助コード](../../../benchmarks/cuda/linear/lm_ffn_down_shared_step.py)
は benchmarks に置いた。呼び出しごとの経路には
RADIUS_ENVELOPE_DENSE_FOURIER_WEIGHT_LM を選ぶ。

| 条件 | 更新時間の中央値 ms | ピーク allocated MiB | ピーク reserved MiB |
| --- | ---: | ---: | ---: |
| アーカイブ版 | 1436.27 | 3126.36 | 3496 |
| 広域タイル共有 | 658.06 | 3123.64 | 3502 |
| Atom を整列して広域モーメントを集約 | 169.58 | 3148.98 | 3630 |
| 呼び出しごとの一時 Fourier 重み | 79.43 | 2057.80 | 2588 |
| 更新内で重みを共有 | 46.78 | 2272.43 | 2522 |

全条件で loss、勾配ノルム、Atom の更新、AdamW モーメントの照合を通過
した。更新内共有のクリップ前ノルムは 1.72426033、loss の絶対差は
5.96e-8 だった。Atom の更新とモーメントの最大相対 L2 誤差は 2.89e-4。
Dense パラメータの更新量全体に対する相対誤差は 1.12e-5 だった。
RTX 4090 の幅別回帰テストは片側勾配と周期タイル境界を含む 29 件が
通過した。計測で検証したのは指定の FFN down 形状の eager 実行である。

### 連続更新

アーカイブ版と更新内共有版を同じチェックポイント・データ順序・
オプティマイザのスケジュールで step 4883〜5010 まで進めた。loss の
最大差は 3.12e-4、クリップ前勾配ノルムの最大差は 1.50e-3 だった。
step 5010 における個々の半径分率 q の最大差は 0.0988、q 全体の
層単位での最大相対 L2 誤差は 1.03e-3 だった。初回を除いた中央値は
それぞれ 1441.17 ms と 46.40 ms。連続更新の結果の SHA-256 は
cf61121f9d2b6d30fe274a761635fdeff272bdf1863da25906babdf1a56c3a42。

## 根拠と再現方法

チェックポイントの SHA-256 は
8409ddbbfb6e7b0a2a9706952657156bec52bf3074e136cb49c68467a1e4dce3。
ソースのコミットは 63e445901a2167a1ea8333d16d310efd9f2fb465。
独立した実行ディレクトリ lm-fourier-weight-final-20261117-01 にドライバ、
ログ、ソースのマニフェストがある。単体と学習更新の結果の SHA-256 は
それぞれ 67b0ec2bb91029d80aef3e72ec6c000b5993ad633ce6b60092e88ff789e0be58 と
40eb4975205ecab9c4cd745cef91730dd260622309ab72477e02dc2e307ad47c。

各条件で固定したソースと対応するネイティブビルドを使う。
TORCH_CUDA_ARCH_LIST=8.9 を指定し、TORCH_EXTENSIONS_DIR には対応する
ビルドのディレクトリを指定する。元の[広域タイルの再生](lm-shared-wide-tiles-20261117.ja.md)
にソースとチェックポイントの準備がある。

~~~sh
export PYTHONPATH="$SOURCE/torchcst/src:$SOURCE/torchcst:$SOURCE/cst-research/src"
python "$EVIDENCE/compare_direct.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$EVIDENCE/result.json" --profile
python "$EVIDENCE/compare_step.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$EVIDENCE/step.json"
~~~

一時重みの初回の再生は Y の基準を満たさなかった。単一 Atom をバンクごとに
確認し、Fourier 集約の転置で隣接タイルを逆向きに参照する誤りを特定した。
最終版の周期境界テストで修正後の向きを検証した。失敗した再生と確認結果は
それぞれ lm-shared-step-fourier-weight-20261117-01 と
lm-dense-weight-debug-20261117-01 に保存した。
