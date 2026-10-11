# 支持幅の生成粒度とdH/dG寿命を別々に比較

2026-10-11。実装・計測準備中。公開defaultの変更はない。

対照は[支持タイルBM16](regular-grid-tile-dot.ja.md)と現行BM16/G16/H32。
2候補を組み合わせず、同じ32-site所属routing、8 owner partial、IEEE FP32/TF32 off、
全operator-sitesの正規化・床・singleton・source VJPを維持する。
全支持の縮約後にparameter勾配を作り、タイル間の交差項を落とさない。

`tile-window16-bm16`は生成粒度だけを変える。同じ支持開始タイルのatom束について
liveな支持区間の和集合を求め、16-site panelでH/Gを縮約する。空支持は0 writerを持つ。
H/dH/G/dGの全atom×小batch slab、生成回数と後段のowner集約は対照のまま。
rhoによる固定分岐ではなく現在のpacked supportsを使用する。広支持は既存の正確なfallback。

`tile-derivative-bm16`は32-site panelを保ち、dH/dGをproducer内で消費する。
1. Hのvalueだけを全支持で縮約し、小batchのH slabへ書く。
2. G/dGを全支持で縮約し、Hと掛けてdAと出力中心勾配を作る。G valueだけをslabへ書く。
3. H/dHを再計算し、Gと掛けて入力中心勾配を作る。H/dHはここではglobalへ書かない。
4. Gを既存dX ownerで集約し、physical→source VJPを行う。

数式は `dA=sum_b H G`、`dc_in=A sum_b dH G`、`dc_out=A sum_b H dG`。
batch chunkをまたぐphysical勾配は元と同じ順序で加算する。H/Gの2 slabは残り、dH/dG slabは
作らない。H valueの追加再計算と融合勾配の費用も完全stepに含める。
forwardとneed_x-onlyは元の固定panelを使う。retained forward snapshot/routingを維持し、
次forwardで更新後の支持に基づき再構築する。gradientsはatomicなしで唯一writerが書く。

比較はRTX 6000 Ada N2048/B32/A209715/rho_initial3/8/seed41、同じMSEとAdamW/Polar。
現行、固定tile BM16、各独立候補、matrix/factor/onchip/dense対照をfresh workerで測る。
rhoごとに独立2cohort（2回目reverse）、24実更新、21 primary samples。
GPU資格検証を先行し、初期/post24全shape・全atomのFP64 Y/dX/全4 dP gate（4e-4）を維持する。
準備・所属sort・集約込みの未計装完全stepとcapture/replay allocated/reserved peakを判断に使う。
別Graph診断は加算せず、実cache hit/DRAM/stall/達成occupancyを未測定のまま断定しない。
