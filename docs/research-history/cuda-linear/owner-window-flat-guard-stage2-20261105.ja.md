# G1 Stage2: 一回のprospective lifecycle memory検証

Stage1追加試行 `l4job-b48fffb7e8894db7805dce763fc161fc` の正誤・費用screenを通過したため、
次の一回を追加GPU確保前に固定する。これはStage1の第三試行でも、正式な256更新のtrialでもない。

## 一回の固定範囲

- shared poolのL4同時1台、1 job、hard timeout900秒、再試行0。
- fresh processを最大4つ、順序はCST qualified→CST flat→dense flat→dense qualified。
- 同じ7凍結入力、N1024/A52428/真B4、FP32 FLAT_C＋既存geometry/Polar、TF32off/math threads1。
  `src/`242ファイルは#132と同一、G1の四つのguard/coreファイルはStage1と同一。
- `full_step_graph_learning --mode graph --flat-c --guard ... --epochs 8 --steps-per-epoch 16`。
  初期8 ordinary更新を含む128実更新、120 replay。仮capture更新は復元し、実更新数へ再加算しない。
- 最初のCST processはcompile込み最大500秒、他processは各150秒。ただし全体の残り900秒を超えない。
- 正式performanceセル、追加eligible費用probe、45,000更新は0。K/Oは併用しない。

## 変更しない実行と、時計外の来歴

既存のeval/save/train、資格失格時のreprepare/recapture、Graph継続・退役条件、数値bodyと計時区間は保つ。
各captureのPlan/recipe/entry・実extension SHA・backend/flat有効性をscalarで記録し、
`captures.json`を逐次保存する。source・入力・actual moduleの同一性をdriverで照合する。
新しいprovenance hash/file出力は時計外の診断であり、アルゴリズムのprepare/capture費用は従来どおり計測する。
capturesへmodel/Tensor/Graphの追加参照を持たせない。

CSTのcapture位置は`[8,16,32,48,64,80,96,112]`、評価境界の退役7回と最終128更新での退役を確認する。
denseは評価後も自然に適格なら同じGraphを継続してよく、CSTに合わせるための強制recaptureは足さない。
dense自身のqualified/flatでは、capture/退役eventと実更新・replay数を一致させる。
全captureはStage1で観測したnative backend、予定guard・entryを使い、途中でbackendを強制変更しない。

## CSTの公式memory-v2.1

`evaluate_lifecycle(..., evidence_scope="prospective")`と既定の`ENDPOINT_PHASE`を変更せず使う。
8 cycle固定、cycles3–8のendpoint allocated/reservedが一定、candidateの累積peak増分は
allocated≤2MiB/reserved≤8MiB。単一resetはoptimizer作成前、capture・評価・再準備・最終退役まで累積し、
途中のpeak resetや`empty_cache`はしない。dataset初期配置は既存runnerと同じscopeである。

まずqualified baselineの自己照合でschema/phase/128実更新/120 replay/plateauを検証する。
自己照合の差分0を採用結果とは呼ばない。baseline自体が成立しなければcandidateを起動せず、比較は未判定で停止する。
両CST processの各8 epoch PTについて、42状態・5勾配・totals・orderを直接byte照合し、
JSONのhash/metricsとも対応させる。全clock、capture復元、評価の数値非変更、通常fallback0を確認する。
baselineが有効でcandidateの予算/plateauが不成立ならmemory gate不合格。CST不通過時はdense診断を起動しない。

## denseの対称診断

CST gate通過時だけ、dense二方式にも同じ8 cycleを与える。既定v2.1判定器は42状態固定なので、
denseの20状態を42へ加工して合格扱いにはしない。denseは20状態・5勾配・totals/order/metricsの直接byte比較、
128実更新/120 replay、自然なevent一致、同じendpoint phase、cycles3–8 plateau、peak増分2/8MiBを手動診断する。
この診断をCST公式v2.1合格とは表示しないが、**Stage3へ進むための継続条件**とする。
qualified denseの基準自体が不成立ならcandidateのmemory判定は未確定、G1 denseだけの不成立なら副作用として停止する。

## 停止と次段階

source/input/Plan/entry/actual module/backend不一致、byte/finite/clock/復元不正、予期しないfallback、
必要なepoch/capture/退役証拠の欠落、memory条件不成立、runtime/環境/通信の失敗で停止する。
都合のよいsuffix選択、再測定、phaseを変えた再判定はしない。失敗したrunと原本、完成/未完成範囲を保全する。

両CSTの公式gateとdenseの対称継続条件が通った場合だけ、既定の正式first pairを開始する。
別のtrial256は追加しない。両状態各6 pair・最大48セル、最初の有効CST負けで停止、両状態5%以上かつ6/6勝利、
byte一致/fallback0とcapture込みmemory予算を維持し、全て通過時だけ二順序45kへ進む。採用/mergeはこのStage2の結果で決めない。
