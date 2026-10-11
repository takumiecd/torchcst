# packed と parameter partial の入力site順を揃える比較

2026-10-11。候補 `input-canonical-bm16-g16-h32` を、既存
`input-site-reciprocal-bm16-g16-h32` と比較する。計測前にこの条件を固定する。
benchmark-localの研究候補であり、公開dispatcherには接続しない。

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

## 実測結果と扱い

測定source `2a0daa31ff8806ecee67de51e003038a026f8259`。
RTX 6000 Ada Generation 48GB、Torch 2.9.1+cu128 / CUDA 12.8 / Triton 3.5.1。
GPU qualification 281 PASS、skip/error/failure 0。初期/post24の全40 FP64 gateが
同じ4e-4条件でPASS。24 worker、504 primary sampleであり、504独立runではない。

| rho / cohort | 既存完全step ms | canonical完全step ms | 短縮 | dense ms |
| --- | ---: | ---: | ---: | ---: |
| 3 / 正順 | 1.428415 | 1.392744 | 2.50% | 0.329180 |
| 3 / 逆順 | 1.436255 | 1.398125 | 2.65% | 0.328626 |
| 8 / 正順 | 1.828160 | 1.767291 | 3.33% | 0.329367 |
| 8 / 逆順 | 1.837863 | 1.766626 | 3.88% | 0.329176 |

完全stepは入力sort、direct packed準備、H/G生成、Y/dX owner、partial/physical、
raw順へのVJP、loss、AdamWとPolar/periodic更新を含む。primaryは未計装。
両研究方式それぞれの独立再実行ではpost24 P/Y/dX/all_dPがbitwise一致した。
方式間のpost24 Parameter同一性は要求していない。

| capture/replay peak MiB（全rho/cohortで同値） | allocated | reserved |
| --- | ---: | ---: |
| 既存 BM16/G16 | 57.430664 | 104 |
| canonical BM16/G16 | 59.846191 | 120 |
| dense | 81.752441 | 106 |

packed全量コピーは追加していないが、allocatedは2.415527 MiB、reservedは16 MiB
増えた。saved raw-order indexはint64[A]約1.6 MiBで、forward/backwardをまたいで残る。
入力境界/距離もsnapshotと同時に保存する。reservedはindexのtensor payloadではなく
Graphを含むallocator予約領域であり、16 MiB全体を特定のtensorの大きさと解釈しない。
allocation履歴による個々の予約blockの帰属は今回取得していない。

**全4 cohortでreservedがdenseを14 MiB超えるため、canonicalを基準へ採用しない。**
既存BM16/G16を維持し、候補はbenchmark-localな研究controlとして保存する。
canonicalもdenseに対してrho3で約4.23–4.25倍、rho8で約5.37倍の完全step時間が必要。
速度の小改善とメモリの悪化を一緒に記録する。

別copyの更新付きphase計装でも、backwardが短縮する一方でforwardが長くなった。
このphaseは初期clock24から30までの別診断で、primaryには混ぜていない。

| phase ms（各5replay中央値・2 cohort範囲） | 既存 rho3 | canonical rho3 | 既存 rho8 | canonical rho8 |
| --- | ---: | ---: | ---: | ---: |
| forward + loss | 0.517856–0.524032 | 0.554688–0.557088 | 0.661664–0.666784 | 0.693312–0.694592 |
| backward | 0.548320–0.552544 | 0.472288–0.478112 | 0.803424–0.808960 | 0.707392–0.708288 |
| optimizer | 0.374016–0.374688 | 0.373600–0.374464 | 0.374912–0.375904 | 0.374560–0.374656 |

backwardはrho3で約13.5–13.9%、rho8で約11.8–12.6%短縮するが、forwardで
約0.027–0.037 ms増える。準備をforwardへ移す費用を含めたprimaryの改善は
2.5–3.9%に留まる。これらのphase時間も足してprimary値を作らない。

## 準備・復元の別計装

実際のcanonical executorを別copyの固定post24で計装し、非計装のY/dX/dPと
bitwise一致を検査した。primary24を変更せず、optimizer更新も行わない。
以下は各5replayの中央値、2 cohortの範囲。primaryの内訳を足した推定ではない。

| canonicalの区間 ms | rho3 | rho8 |
| --- | ---: | ---: |
| 入力sort + direct prepare + bounds/distance | 0.147680–0.147776 | 0.164288–0.164992 |
| forward output index/hot + H生成 + Y owner | 0.395648–0.398528 | 0.512544–0.515552 |
| backward input hot + G/H/partial + dX + physical | 0.455808–0.460736 | 0.686368–0.693408 |
| raw-ID source VJP・直接書戻し | 0.012960–0.012992 | 0.013152–0.013248 |

先頭の区間はraw Source/amplitude snapshot cloneの後に始まる。clone費用は
primary完全stepとforward phaseには含まれる。既存のsnapshot/prepare区間は
rho3約0.100、rho8約0.116 ms、backward側のinput routing/hotは約0.061–0.064 ms。
canonicalは入力sortをforwardへ移したため、この二つの区間を単独で同義とは扱わない。
既存source VJPは約0.0097–0.0098 msで、raw IDへの直接復元にも約0.003–0.004 msの
追加費用がある。準備・復元を省いたproducer局所速度だけで選ぶと結論を誤る。

次に検討するなら、保存indexとsort scratchの寿命・再利用を先に調べる。
元Parameter/optimizer順の変更、packedの追加コピー、全H/G保存はこの結果から正当化しない。

## 証拠と再現

- durable job: `/home/jovyan/runs/cst-canonical-g16-ada-2a0daa31-20261011/`、controller PID 151201。
  `nohup python controller.py 2a0daa31ff8806ecee67de51e003038a026f8259`、`controller.log`、
  `campaign-status.json` と `result-manifest.json` を保存。各cohort前にGPU compute processなしを確認。
- local: `output/regular-grid-h/canonical-results/`、source archive、operating notes、
  controller/qualification/audit/download検証script、CPU/build logを保存。
  raw oracle `.pt` はclusterに残し、88 compact entryをdownload後に全件SHA検証した。
- source archive SHA256: `ffe38f660db7e5ec1c3db0b27adf1c0a57688d257e9cd41d9705e0b6ab4cfa42`。
- compact archive SHA256: `8c722fcf2eef602532897aae5be6c3d119a4a4c8cabfe026ee459bbe495ef493`。
- audit PASS_COMPLETE SHA256: `2373ac448a78c76123e1dc357c19e99a7cfb0dc26aab550eda75adc690e4a9bb`。
  全source map、入力hash、24更新、40 snapshotのSHA/shape/全勾配、peak/時間、独立repeatを監査。
- source CPU/build CI PASS: <https://github.com/takumiecd/torchcst/actions/runs/38118541969>。
  ローカルwheel/sdist buildもPASS。

各cohortの例（rho8はcase名を変更、`--reverse`で第2順序）：

```bash
PYTHONPATH=src:. CST_FROZEN_SOURCE_COMMIT=2a0daa31ff8806ecee67de51e003038a026f8259 \
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --plans matrix-torch factor onchip-h input-site-reciprocal-bm16-g16-h32 \
    input-canonical-bm16-g16-h32 dense \
  --isolated-oracle --phases --output output/new-canonical-cohort/comparison.json
```
