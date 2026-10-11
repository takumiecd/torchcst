# packed と parameter partial の入力site順を揃える比較

2026-10-11。候補 `input-canonical-bm16-g16-h32` を、既存
`input-site-reciprocal-bm16-g16-h32` と比較する。計測前にこの条件を固定する。
GPU未検証の研究候補であり、公開dispatcherには接続しない。

## 配置と寿命

Parameter `[A,4]` と optimizer state のatom順序は変えない。forwardごとの
raw Source snapshotから入力中心siteのint32 keyを作り、sortのpermutation
`raw_order[pos] = raw_atom_id` を得る。唯一のpacked `[13,A]` を、このpos順に
直接生成する。raw packedを作ってからgatherする方式は使わない。

input-site境界と最大支持距離はforwardのsnapshotと一緒に保存する。
backwardのGは従来通り `[G16/BM8,A,BM8]`、勾配partialとphysicalはpos順とする。
producer、dX ownerの有限scale経路・fallbackとも、input routingのID gatherを省く。
最後のsource VJPだけがraw_orderを参照し、raw Sourceを読んでraw dPへ直接書く。
元の順序のphysical bufferは作らない。saved packed/sourceの書換えも行わない。

forwardのoutput routingはpos空間のpackedに対して再構築する。このため同じoutput
site内の加算順序は従来と変わり得る。数学的契約、正規化floor・singleton、幅の
FP64補正、raw polar VJPは維持し、独立FP64の同じ閾値で検査する。

saved stateにはraw-order permutation int64[A]、入力境界 int64[NI+1]、距離scalarが
追加される。全量packedコピーが無いことと、実測peakが下がることは別である。
入力sortをforwardへ移すことで、forward H32 scratchとindexが同時に生存する費用も
capture/replay込みのallocated/reserved peakに含める。

## 固定比較

- RTX 6000 Adaのみ。N=2048、B=32、A=209715、rho=3/8、seed=41、FP32、TF32 off。
- 対象2方式にmatrix-torch、saved factor、onchip-h、denseを加えた6方式。
- rho3/rho8 × 正順/逆順の4 cohort。それぞれfresh workerで24実更新、21測定replay。
- 正規化product、AdamW、Polar/periodic更新、sort・snapshot・準備・VJP復元を含む
  未計装の完全stepをprimaryとする。準備済みpackedだけの速度では選ばない。
- qualificationのfloor、seam、stride、tail、gradient masks、retained snapshot、
  Graph/public更新、非有限scale fallbackを通す。初期とpost24の全shape・全atomの
  Y/dX/all_dPについて、独立FP64 maxabsとrelative L2の両方を4e-4以下とする。
- capture/replay allocated/reservedを別々に比較し、denseを両方以下にできるか確認。
  更新回数・同一入力・source hashも監査する。失敗runの保存、閾値据え置き、再選択なし。
- 計装stageは別copyの固定post24でのみ行う。候補の実経路のsort/prepare/bounds、
  forward H/owner、backward G/H/dX/physical、raw-ID source VJPを区切る。
  stageを足してprimary complete stepと見なさない。

API接続は当初HTTP 403を返した。ユーザーによるtoken更新後にreadyとなり、
RTX 6000 Ada、0%利用、39MiBを確認した。GPU検証281件を先に通し、その後に
4 cohort / 24 worker / 504 primaryサンプル / 40 full FP64 gateを実行する。
CPU suite: 1741 PASS、3832 GPU/DB等skip。catalog 34件のdecodeとTriton遅延import確認もPASS。
GPUの正しさ・時間・peakは、このsource checkpointではまだ未計測。
