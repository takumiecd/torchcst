# 小tileから親へ積み上げるFourier集約のCPU検証

2026-10-21。ユーザーの指示は「小tileのatomを処理し、親用の値を集約し、
親が揃ったらその尺度のatomへ適用する」というbottom-upのデータフロー。
親から子へ入力集約を作る方式とは区別する。GPU実装・公開dispatcherではない。

## 共通kernel

多項式との混合を使わず、全sigmaで有限Fourierを使う。
母多項式(1-(r/(sqrt(11)*sigma))²)^4_+への制約付き連続L2射影で係数を定義する。
M4は端の0/2階微分0、M8は0/2/4階微分0、中心値1。
[混合kernelの係数定義](tile-hybrid-profile-design-20261021.ja.md)と同じ固定射影行列。
今回そのPはFourier係数を決める母形状であり、実行時にPへ切り替えない。
負の小さい裾は許容した数学prototype。厳密な非負性を主張しない。

spacing=.1、L_j=4*spacing*2^j (j>=0)、支持[-L_j,L_j]、周期2L_j。
R=sqrt(11)*sigmaが.85L_jの±12%にある間は、quintic smoothstepでF_LとF_2Lを接続する。
区間外は単一F。各成分は支持外0、kernel全体は最大active Lの外で0。
各atom・軸の中心/sigmaは連続で自由。尺度割当だけを離散化する。

sampled L2を各軸で正規化する。2Dは直積格子のprofile productで、L2²=Ex*Ey。
全ての方式は同じ係数・支持・正規化・mu/sigma微分を使う。
小Lでは格子上の周波数折り返しが重要。
正規化のgeometry和でも整数周波数を周期内へ折り返し、aliasしたDCを正確なnsite和として扱う。
以前のsinc(integer)/sinc(integer)式を小Lにそのまま使うと精度が不足したため、
新prototype内だけでalias-awareな幾何和を実装した。旧測定source/helperは変更していない。

## 計算方式

入力は1D線分または2D正方格子。最小leafは4/8/16点（2Dは各軸の点数）。
これは論理集約単位であり、GPU launch tileが4点等であるという意味ではない。
支持区間/長方形を、完全に含むdyadic nodeと部分leafへ厳密に分解する。
部分leafは短い1D/2D prefixで処理し、境界サイトのkernel直接評価や裾の省略を使わない。

- direct: 同じFourier関数をatomごとの支持内で直接評価し、Hと各軸mu/sigma pullbackを求める。
- independent: 同一nodeを使うatomは共有するが、別node/尺度のmomentは入力から独立に構築する。
  完了したnodeはqueryへ適用後に破棄する、streaming baseline。
- bottom_up: 小leafで必要な周波数を一度集約し、そのleafのatomへ適用する。
  親へ必要なmomentを足し上げ、親のatomへ適用したら子の作業配列を破棄する。

atomの位置やsigmaからquery metadataを先に決める。親が要求する周波数集合を子へ伝えるのは
metadata処理だけであり、入力依存の値の計算はleafから親への順序である。
親用の奇数周波数もleafで計算する。親の偶数成分だけを再利用して残りを無料とは扱わない。
同一周波数、格子alias、実入力の共役対称性をまとめて重複計算を減らす。
2Dは片軸だけの共役折り返しではなく、(qx,qy)と(-qx,-qy)の対をまとめる。

各leafの入力を連続コピーし、周波数chunk32で積和する。
複数atomの部分leaf queryにも同じ短いprefixを使う。
親用だけの周波数にはprefixを作らない。Xが変わるstepでは入力依存momentを再構築する。
同じ相対leaf位置のsin/cos表は固定geometryとして共有する。

## 比較範囲とカウンタ

M4/8、1D N256/B4/A16または128、2D 32²/B2/A8または64。
leaf4/8/16、small/mixed sigma分布。各atom・軸のsigmaは異なり、stepごとにX/mu/sigmaを更新。
smallは約.1〜.15、mixedはlog(.1, side*spacing/8)。同じfixtureを全M/leaf/方式で使う。
3動的step、各状態3回のCPU時間中央値。係数・L2・group/metadata・準備・query込み。
固定geometryは先にwarmし、相対leaf basisのsetup時間と配列payloadを別記する。

独立Torch quadrature/autogradで、合成出力・振幅/中心/sigma勾配を照合する。
identity入力で実装の全入力Jacobianを取り出し、Torchの全入力勾配も確認する。
性能測定は正規化profile-bank Hとmu/sigma pullbackまで。
高速なbackward executor、CST全体のY/dX、optimizer、GPU完全stepではない。

カウンタは下記を別々に記録する。

- input_values: 各leafでXを連続ロードする論理要素数（batch込み）。
- weighted_products: 親用の追加成分も含む、入力*Fourier basisの複素積和項数。
- prefix_values: 部分leafだけのscan要素数。
- merge_adds: 子から親へ足す複素要素数。
- traffic_model_bytes: 上記の主要配列・moment・係数・query出力更新の論理read/writeモデル。
  hardware transaction、cache miss、GPU帯域を実測した値ではない。
- scratch/carry payload: 実装が数えた配列の容量モデル。プロセスpeakではない。
- cached_bank_payload: query nodeの全bankを保存した場合の仮想payload。実行時にはstreamして破棄する。

代表条件のみ、call内の追加tracemalloc peakも測る。既存fixture/geometryの保存は別。
入力読み取り削減だけで、全メモリアクセスや速度が同率に改善したとは主張しない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_bottom_up_fourier_cpu \
  --output output/bottom-up-fourier-cpu-20261021
```

測定結果は検証済みcheckpointで追記する。
