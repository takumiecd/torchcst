# M8 FFN down の一時 Fourier 重み

[English](lm-ffn-fourier-weight-20261117.md) | **日本語**

明示選択の CUDA プランは 4096→1024 の M8 格子と 196,608 個の Atom を
対象とする。16 サイト以上のバンクでは Atom の所属タイルの組ごとに
17×17 の Fourier 係数行列を集約する。位相回転で両軸の 3 つの展開先に
係数を移し、周期境界で戻る寄与もすべて加える。2 回の行列積で係数を
展開し、最後の GEMM は重みに直接加算する。小さいバンクではセル質量を
直接加える。生成する一時重みは
4096×1024 の FP32 で 16 MiB。学習する状態は Atom の行である。

forward と dX には GEMM を使う。Atom の VJP は密な重み勾配 dYᵀX を
同じバンクの基底に射影し、Atom のセル質量および半径と中心の微分を
掛け合わせる。backward では dX に一時重みを、Atom の VJP に幾何を
再利用する。プロファイル微分と演算の主部分は FP32。アーカイブの rational radius
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
| アーカイブ版 | 39.84 | 46.11 | 85.90 | 529.22 | 996.23 | 1022 |
| 広域タイル共有 | 20.95 | 18.03 | 38.80 | 538.56 | 1002.57 | 1022 |
| Atom を整列して広域モーメントを集約 | 4.93 | 3.68 | 8.63 | 553.65 | 1020.45 | 1034 |
| 一時 Fourier 重み | 1.76 | 1.05 | 2.82 | 199.05 | 668.56 | 702 |

アーカイブ版との最大絶対差は Y が 9.16e-5、dX が 6.40e-10、Atom の
4 座標の勾配が最大 7.61e-8。最大相対 L2 誤差は 1.15e-5 だった。最初の
層は行数 1、7、9、65、137 でも検証を通過した。独立した CPU FP64 の
因子計算では全 196,608 個の Atom と有効 3 行を使い、GPU では 72 行に
ゼロパディングした。最大絶対誤差は半径分率の勾配で 2.47e-4、
最大相対 L2 誤差は 6.00e-6 だった。

GEMM の直接加算を入れる前に、step 512 と 2441 でも射影の照合を通過した。step 512 の合計は
アーカイブ版 80.54 ms、一時 Fourier 重み 3.69 ms。step 2441 は
それぞれ 85.41 ms、3.03 ms だった。各ブロックで 12 回測定し、
lm-fourier-weight-history-20261117-01 に保存した。

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
| アーカイブ版 | 1435.97 | 3125.33 | 3680 |
| 広域タイル共有 | 658.80 | 3123.74 | 3490 |
| Atom を整列して広域モーメントを集約 | 169.89 | 3144.12 | 3512 |
| 呼び出しごとの一時 Fourier 重み | 78.63 | 2057.63 | 2572 |
| 更新内で重みを共有 | 46.59 | 2271.51 | 2576 |

全条件で loss、勾配ノルム、Atom の更新、AdamW モーメントの照合を通過
した。更新内共有のクリップ前ノルムは 1.72426033、loss の絶対差は
5.96e-8 だった。Atom の更新とモーメントの最大相対 L2 誤差は 2.89e-4。
Dense パラメータの更新量全体に対する相対誤差は 1.07e-5 だった。
RTX 4090 の幅別回帰テストは片側勾配と周期タイル境界を含む 29 件が
通過した。計測で検証したのは指定の FFN down 形状の eager 実行である。

全 Dense の対照は、専用の step 4882 チェックポイントから同じ
次の 1,024 トークンと手順で測定した。中央値は 36.81 ms、
ピーク allocated は 2215.19 MiB。結果を
lm-archived-dense-step-20261117-01 に保存した。SHA-256 は
c3eb75983d85906d04d307c3f0892c46cb34a12f821ad45a27cb4edbb80dc67d。

### 連続更新

同じチェックポイント・データ順序・オプティマイザ設定を用い、
候補を 1 回、アーカイブ版を 2 回、step 4883〜5010 まで進めた。
アーカイブ版の 2 回は、全更新の loss と勾配ノルム、および記録した
7 時点の q が一致した。候補の loss の最大差は 3.00e-4、クリップ前
勾配ノルムの最大差は 1.22e-3。step 5010 では 786,432 個の Atom 中
654 個で q の差が 0.01 以上、13 個で 0.05 以上となり、806 個の
半径の階層レベルが異なった。個々の q の最大差は 0.0991、層ごとの
q 全体の最大相対 L2 誤差は 9.88e-4。初回を除いた中央値は、
アーカイブ版 1442.99 ms、更新内共有 46.24 ms。結果の SHA-256 は
4c18cc3ab8785fdbe50835328f08c6da7fb42edb4a64cc6a6a38e74b2348e2f7。

### rational radius の FP32 実験

アーカイブの rational 式に合わせて、半径デコードと VJP を FP32
で行う Triton 試作も計測した。チェックポイントの 196,608 個の Atom
で、FP64 版に対してデコードの中央値が 0.251 ms から 0.035 ms、
VJP は 0.216 ms から 0.048 ms になった。この単体計測の VJP の
相対 L2 誤差は 3.38e-7。射影全体とチェックポイントからの更新の
照合も通過した。更新内共有での同一条件の学習時間は FP64 デコードで
46.71 ms、FP32 試作で 45.01 ms。ドラフトの eager プランは
アーカイブの FP64 デコードを使う。根拠は
lm-rational-decoder-fp32-20261117-01 と
lm-rational-decoder-fp32-step-20261117-01 に保存した。結果の SHA-256 は
それぞれ f57ddc95cf052cd1126587550b3edff7aa63ab204c16574d5e3cfb96dca3540d と
2dee1dd70e6fa5c07e9dc605b5d7d2d37add3d9c0b2581b90d2735561d8a3174。

## 根拠と再現方法

チェックポイントの SHA-256 は
8409ddbbfb6e7b0a2a9706952657156bec52bf3074e136cb49c68467a1e4dce3。
ソースのコミットは 7daf1ddea5ccbcb3a17e61f46cdd7f8485579944。
射影の再生は lm-fourier-weight-release-20261117-01、学習更新と q の監査は
lm-ffn-q-drift-audit-20261117-01 にドライバと測定結果がある。
単体と学習更新の結果の SHA-256 は
それぞれ b819197e8dec9be2a40164778245f21413e51b06d6866eec27423cf658361612 と
e154bc7e154e88ae42181a17d30750e4a383ca4d73428ccadd2a6c19674f35ed。

各条件で固定したソースと対応するネイティブビルドを使う。
TORCH_CUDA_ARCH_LIST=8.9 を指定し、TORCH_EXTENSIONS_DIR には対応する
ビルドのディレクトリを指定する。元の[広域タイルの再生](lm-shared-wide-tiles-20261117.ja.md)
にソースとチェックポイントの準備がある。PROJECTION_EVIDENCE と
UPDATE_EVIDENCE にそれぞれの実行ディレクトリを指定する。

~~~sh
export PYTHONPATH="$SOURCE/torchcst/src:$SOURCE/torchcst:$SOURCE/cst-research/src"
python "$PROJECTION_EVIDENCE/compare_direct.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$PROJECTION_EVIDENCE/result.json" --profile
python "$UPDATE_EVIDENCE/compare_step.py" --training-run "$TRAINING_RUN" \
  --source "$SOURCE" --capture-driver "$PROFILE_RUN/profile_down.py" \
  --output "$UPDATE_EVIDENCE/step.json"
~~~

一時重みの初回の再生は Y の基準を満たさなかった。単一 Atom をバンクごとに
確認し、Fourier 集約の転置で隣接タイルを逆向きに参照する誤りを特定した。
最終版の周期境界テストで修正後の向きを検証した。失敗した再生と確認結果は
それぞれ lm-shared-step-fourier-weight-20261117-01 と
lm-dense-weight-debug-20261117-01 に保存した。

## 振幅上限の CPU 丸め

Atom 更新は、検証済みの CPU 側の宣言から現在の上限を取得する。
CPU でゼロ方向に丸め、上限値とパラメータの dtype ごとに結果を
キャッシュする。GPU には丸め済みのスカラーを渡して clamp する。
上限を変更した後やチェックポイントを復元した後は、固定状態の
メタデータを再取得する。

RTX 4090 の両条件で、FP32 の rational radius 試作、step 4882 の
チェックポイント、step 4883 の同じ 1,024 トークンを使った。eager
計測ではサンプルごとにモデル、AdamW、乱数状態を復元した。順序を
反転したブロックから、各条件で全更新 36 回、optimizer のみ 40 回の
中央値を求めた。optimizer のみの計測では同じ勾配を再生した。

| 上限処理 | 学習更新全体 ms | optimizer ms | 学習更新のピーク allocated MiB |
| --- | ---: | ---: | ---: |
| GPU で FP64 比較 | 44.953 | 7.524 | 2234.34 |
| CPU で先に丸める | 44.857 | 7.454 | 2235.09 |

同じ勾配を用いた更新では、52 個すべてのパラメータテンソルと
AdamW の状態がビット単位で一致した。学習更新全体では loss の差が
5.96e-8、Atom の更新量またはモーメントの最大相対 L2 差が 3.59e-5
だった。optimizer の演算トレースでは GPU の FP64 比較が 4 回から
0 回になった。宣言を読み取る FP64 スカラー操作は両条件で 24 回だった。

ソースはコミット 421f90073631e39bd7a6704418ef5703ebde053f。
lm-radius-cpu-cap-20261117-01 にソース差分、compare_cap.py、ビルドと
実行のログを保存した。結果 JSON の SHA-256 は
f13add958ae086231542bdb49a3760f739161bc8a6204516a45677d32a988a8f。
CPU テストは 2312 件が通過した。
