# 過去のkernelからHの生成・再利用・寿命へ移せるもの

2026-10-11。過去の実装・測定報告を読み、現在のRegularGrid profile-productへ
移せる処理順序を整理した。新しいGPU測定やruntime変更は行っていない。
以下の数値は各報告内の比較であり、GPU・数学・測定境界が異なる行を順位付けしない。
設計の正本は[Hの寿命を軸にした設計](../../h-lifecycle-kernel-design.ja.md)。

## 実装と実測が残る候補

| 過去の方式 | 当時の同条件比較 | メモリと寿命 | 現行へ移す部分 |
| --- | --- | --- | --- |
| Strip `block_direct_reuse` | A100 MIG、4096²/B128/A8192、準備込みforward 47.102→14.842ms | 入力・モデル・preparedを除く追加allocatedは両方2.252MiB。X断片をatomループの外で読んで保持、出力担当内で累積 | 入力支持が重なるatomへ同じX断片を共有。旧演算はraw radialでforward限定、profile-product Hそのものではない |
| local-productのlayout/launch移植 `reuse` | L4 N128/rho3、完全step 152.99→77.90µs。逆順152.51→77.51µs | ordered/reuseともallocated316416B、reserved6MiB | H生成と出力担当、atom/batch分割、支持順序を一緒に整える。個々の効果は未分離 |
| owner-local backward融合16 | L4 N64/rho1.25、matched repair4の完全step約6.7–7.2%短縮。N128/広支持は悪化 | GをdXとsource VJPでCTA内共有、global H/Gなし。matched allocated+5120B、reserved同じ | dXを担当するownerの一つだけがatom勾配を出す。Gのglobal往復を省く代わりにowner間で再計算する費用も測る |
| parameter VJPのbatch16 loop | L4 N64/rho3、split2の50.63µsに対しloop52.64µs、loop/owner2は56.86µs | owner2 allocated99328B対owner4 115712B。CTA内でbatch部分の勾配を累積しpartial buffer/reductionを省く | 小batchでH/Gを作って捨て、atom勾配だけを残す。並列性を減らすため速度の勝者とは限らない |
| Torusの選択保存 | L4 N1024/sigma3、完全step292.653→269.534ms、独立逆順でも約7.9%短縮 | recompute allocated/reserved12.194/24MiB、VJP保存34.737/116MiB。Hだけの保存は0.28–0.61%改善に留まる | 値Hだけでなく、Xと入力profile微分の縮約を残して同じ入力走査を省く。現在のdense reserved条件では当時のVJP保存版は超過 |
| grouped profile-product matrix G8/P8 | L4 Product1024/rho3、prepared因数分解0.561313→native matrix0.309897ms、逆順でも改善 | allocated25,604,096→16,688,640B、reserved102,760,448→56,623,104B。全Hではなく集約Wを保持 | 多数atomの寄与をbatchに依存しないWへ先に集約する交換。大batch等の対照として保持。全W方式でありlocal Wの実測証明ではない |

出典・source anchors:

- [Strip入力再利用](notes/block-direct-reuse.ja.md)、source
  `e356ed95c0b0cd9356346d9cefaacc2ee2ca0f85` の
  `prototypes/block_strip_kernels.py::block_direct_reuse`。
  `git show <source>:prototypes/block_strip_kernels.py`で参照できる。
  同じ値のWを持つ旧directとの比較。正規化・GPU backwardを移植済みとは扱わない。
- [profile-productへの移植](../local-product/20261007-profile-product-reuse.md)、
  `634cb32b2a37122f16b8a6ca6d06d6bc9e665482`。
- [backward融合](../local-product/20261007-backward-fusion.md)、
  現存`local_product/kernels.py::_fused_general_block`。
  swap時のh/dhはG/dGで、ParamOwnerだけがH/dHを計算しsource VJPを書く。
- [batch loop](../local-product/20261007-parameter-batch-loop.md)、
  `889d9ef1`、現存`local_product/kernels.py::param_vjp`。
  最終統合の[完全step・peak](../local-product/20261007-pr-integration.md)も参照する。
- [Torus選択保存](../local-product/20261008-torus-hybrid-reuse.md)、
  `8dd36f8e2aef7690cacb02678998ff77d98a3d5e`、
  現存`torus_profile_product/onchip/kernels.py`のHCache。
  intrinsic S1×S2で現在のFlatTorusとは数学が異なる。N2048比較は未完了。
- [grouped matrix](../local-product/20261008-profile-product-grouped-matrix.md)、
  implementation `1bf450644f807147fd5bf513a10ee55af2a25f64`、
  現存`profile_product_matrix/grouped_kernels.py`。Euclideanの当時の比較。

## 不採用結果から残す制約

[small onchip H](../local-product/20261005-onchip-h.md)では、32/64-siteの広い
出力ownerでHを使い回す候補は16-site対照より遅かった。Hの保持を減らしても、
CTA数・ライブ値・再計算を変えるため高速とは限らない。
[初期Torus onchip](../local-product/20261008-torus-onchip-hg.md)はspill0を確認したが、
全atomのY/dP精度gateを超え、速度を測らず不採用。その後の選択保存版とは分ける。

[Torus compact IDs](../local-product/20261009-torus-compact-site-ids.md)は、
site IDをint32からint16へ無損失保存してallocatedを約18–21%削減したが、
3%以上の速度改善はなかった。現行の支持範囲・routingへ大きいatom×site表を
新設する根拠にはしない。保存するindexを先に減らし、必要なら型を縮める順序にする。

## 今の実装との違い

基準`input-site-reciprocal-bm16-g16-h32`は、すでに次を実装している。

- output ownerによるY集約とinput ownerによるdX集約。単純scatterへ戻す必要はない。
- backwardの`produce_g_parameters`はGを一度作り、その場でH/dH・dGを使って
  parameter partialを作る。同じGをglobal bufferへ書き、別kernelのdX ownerが読む。
- forward Hはbackwardへ保存しない。backwardではforward snapshotとXから再生成する。
- Gとparameter partialはbatch方向に制限して作業領域を再利用する。

ただしH32はatom方向に制限されていない。B32/A209715では
FP32 Hが4×A×32 bytes≈25.6MiB、G16は≈12.8MiBの値容量になる。
これは完全step peakではなく、bufferの容量である。
細分化したbatchが全atomのbufferをなくすわけではない。

[詳細診断](regular-grid-fine-bottlenecks.ja.md)でinput順group8のH側支持は、
独立要求数/支持unionが約7.9。G側は約1.01–1.03。
同じXを参照できる機会は多いが、異なる中心のVは異なるためHの値を共有できるという意味ではない。
packedを並べ替えるだけの実測と、Xのunionを一度読んで共有する実装は別である。
実L2/DRAM転送量は未測定。

## 次の比較へ落とす二つの候補

候補Aは旧Stripの入力断片共有をH producerへ移すこと。
近い入力支持のatom群について、連続するX小片を一度読み、各atomのVと縮約する。
狭い支持の直接縮約と、union小片から複数Hを生成する方式を同じ条件で比較する。
正規化と微分用縮約を同じ入力読み取りへ載せる。G側へ同じ共有率を仮定しない。
最初はH/Gの外部buffer・Y/dX ownerを維持して、producer変更の効果を分離する。

候補Bはsmall backward融合の担当規則をFlatTorusへ移すこと。
GをdXとparameter VJPで同じCTA内に使い、最後の利用後に捨てる。
現在のG共有はkernel間なので、重複G生成とglobal保存/再読込の交換が変わる。
周期支持は二つの端区間へまたがり得るため、canonicalなparameter担当を一意にし、
空支持・singleton・norm-floor・各勾配要求・retained backwardも検査する。
owner間のG再計算が増えるなら、広支持では既存buffer方式を残す。

その後でatom方向にHを制限する。支持の出力側と入力側の順序は独立なので、
atom chunkだけ導入しても全出力ownerの再走査、partial Y、追加launchが発生し得る。
Hの最後の利用を近づけるrouting/担当を決め、その費用も比較に含める。
batch-loopによるparameter partial削減とforward→backward選択保存は独立の後続比較にする。

新候補は現在の数学・更新・厳格なFP64 Y/dX/全dP gateを維持し、準備・復元込みの
未計装完全stepとcapture/replay allocated/reservedで判断する。
同条件denseの両peakを上回らないという目標は変えない。
歴史的な改善率を現行Adaの予測値には使わない。

## 追加の方針: Stripは最後の選択肢に回す

ユーザーは、Stripで一つの次元へ複数の意味を持たせ、短い軸と長い軸へ配置を
組み直すことを避けたいと指定した。現在のRegularGrid＋FlatTorusのgrid_shape、
weight shape、spacing、座標とprofile-productの意味を保ち、Strip chartへの変更は
通常の最適化候補から外す。Strip自体を不可能と結論したものではない。

次の優先対象は候補Bのbackward融合とする。現在のproducerで作ったGはparameter
partialと別kernelのdX ownerに共有されるため、単にG共有を追加する実験ではない。
dX owner内でGを生成・消費し、atomごとに一意なownerだけがH/dHとdGを使って
parameter partialを出す方式と、現行のG保存方式を比較する。
周期支持の重複、owner間のG再計算、勾配用partial、準備とroutingの費用を含める。
数学・更新を変えず、Gの寿命とkernel境界だけを変更するのが実験の目的である。

候補Aの入力断片共有もRegularGridの同じ座標上で実装でき、Stripへの変更を必要と
しない。ただし今回の順序では後続候補とし、Stripの幾何・tile配置を移植しない。
