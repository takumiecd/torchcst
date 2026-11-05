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

## 観測: Stage2通過、正式性能は未実施

job `l4job-2bcc5ab92d02417ea71963385e73ecb4`、source
`a45a2f13d52fc5ef8c5dd45c830bd5d9046c3498`で4セルを完了した。
CSTのprospective memory-v2.1、denseの20状態の手動継続条件がともに通過。
peak allocated/reservedのcandidate増分は両モデルとも0 bytes、cycles3–8の終端値も一定だった。
CSTは予定8capture/退役、denseは自然なGraph継続を行い、qualified/flatのevent列は一致した。

ローカルでは1,039 rawファイルと32 PTの元bytesを照合し、各epochのCST47項目×8＋dense25項目×8、
計576 Tensorを直接byte比較した。metrics/order/clock/capture条件も一致し、missing/unverified/errorsは0。
初版のCPU verifierは保存workerのTorch2.11を読取環境2.13と比較して停止したため、元の失敗資料を保存し、
比較先を保存済worker情報へ適合したv2で再検証した。workerが2.11/cu13/L4である条件は維持し、
ローカルTorchのversionは偽装せず、GPU実験や判定閾値も変更していない。

source archive SHA256は`f30512c0e216eac1e48f3033985367c39a47acd32ddd359d55ccb477c7bac7e4`、
result archiveは`69d5145e983a5ebf565ea2136dad211dadcaa3d572be4d7985abc5f8cc0bd75e`。
driverは`548899bf3715d130d5a2343e92895b569c313a4cb4372a090e257cb722029060`。
raw/復元PT/初版エラーとv2 verifierをignored outputに保持し、pool所有VM停止を確認済み。
詳細は同名JSON。正式256更新・45k・採用はまだ未実施である。
