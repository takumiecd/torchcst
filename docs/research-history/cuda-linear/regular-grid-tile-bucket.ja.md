# RegularGrid: 支持幅bucketと可変タイル数

## 実装と仮説

各軸の支持タイル数Kを1/2/3–4/5–8…のdyadic区分へ分類し、
区分と開始タイルをkeyとしてatomを一度だけsortする。
保存するのは各軸A個のint32 IDと、小さい区分×タイルのint32 bounds。
支持所属の全展開、入力×出力タイル組合せ、全tile×atomのH/G partialは作らない。
次forwardで現在の幅と中心から準備し、同じforwardのbackwardへindexを保存・再利用する。

各タイルownerは区分上限K_upperに基づく開始タイル範囲を読む。
周期端では最大2区間へ分け、全周期区分は全タイルを一度だけ読む。
実際のKで所属を絞る。各側1次元では全タイル合計のlive候補参照数は
正確な所属数の2倍未満。これは候補数の上界であり速度比の保証ではない。
末尾の短いタイル、全周期・seam・空支持でも同じsite/tileを二重に数えない。

H/G producerは幅区分と開始タイルが近いatom束を担当する。
16-site共有panelを読み、支持区間全体をregister内で縮約して唯一writerがH/Gを保存する。
区分境界を跨ぐ束は区分ごとに処理し、広いatomで狭いatomのpanel loopを長くしない。
入力・出力側は独立してsortし、H/Gは元atom順の小batch slabへ書く。
全量packedコピーやH/Gの並び替えpassは追加しない。

H/G寿命は既存 `tile-combined-bm16` に揃える。
H value → G/dGをHと即時消費 → dHをGと即時消費 → dX集約。
dH/dG slabはなくH/Gの2 slabを保持する。入力consume2ではH valueのdotを生成しない。
H/Gを全支持で完成してからパラメータ勾配を作り、交差項を維持する。
全operator-sitesの正規化・floor・singleton・source VJPは既存の準備とVJPを使う。
H/Gとphysical gradientsはatom唯一writer、Y/dXはtile×8partsと別reductionでatomicを使わない。
広支持を2タイルの制限によるall-atom fallbackへ切り替えず、そのまま処理する。

研究plan `tile-bucket16-bm16` / `tile-bucket32-bm16` は同じAlgorithmのtile幅だけを変える。
今回の対応範囲はD2/full-period RegularGrid/FlatTorus/Triweight、各側1次元である。
一般Chart/APIや公開defaultは変えない。

## 固定比較

RTX 6000 Ada、N2048/B32/A209715、rho_initial3/8、seed41、MSE、
IEEE FP32/TF32 off、実AdamW/Polar更新24回、primary21 samples。
現行、固定tile、支持幅単独、組み合わせ、bucket16/32とmatrix/factor/onchip/denseを
fresh workersで比較する。独立2cohort/rho、2回目はplan順を反転。
準備・sort・IDのint32変換・生成・集約・backward・optimizerの費用を全て含める。
capture/replay peak allocated/reservedを同じdense対照と比較する。
10 workers × 4 cohorts =40 workers/840 samples、72 full FP64 gates。
初期/post24全shape・全atom Y/dX/全4 dPを既存4e-4基準で検証し、失敗した計測を再選択しない。
GPU qualification251件（bucket/schedule/tile-dot）を全PASS後にprimaryへ進む。
支持が3個以上のタイルを跨ぐ場合、非整除N、周期端、空・全周期、保存状態と可変幅更新を検証する。
固定post24の10診断は別Graphで行い、primaryへ足し合わせない。
