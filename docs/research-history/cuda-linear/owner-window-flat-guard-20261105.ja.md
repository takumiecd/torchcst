# 署名3列のflat化: 一つのCPU prototypeの事前仕様

## 探索上の位置

#136のguard費用pilotは事前screenを通った。Proレビュー後、K/Oは待機のまま、G1を一つだけ
CPUで試作する。pilotの約155µsを回収可能時間とせず、enqueue/drainの相殺からCPU/GPUの律速を
断定しない。ここに書く工程は性能採用やGPU Gateの実施報告ではない。

基準runtimeは#132 `fba2e72d733325fbd7be5aac334a0a7463d96b27`、N1024/A52428/真B4、
FP32 FLAT_C＋既存FP64 geometry/Polar。数式、現在値による分類、支持、正規化、全勾配/dX、
optimizer則、hooks/live settingsを維持する。runtime `src/`、CUDA、optimizer数値処理を変えない。

## 正しさの共通対照と性能差分

R0/#132と#136のG0は過去のreference。次の対照は**資格補強済みG0**と呼ぶ。
有限stock資格、既知code/default穴、capture前後の検証と限定grad refreshを共通のcommitで補強し、
その上にG1の性能変更だけを別commitで載せる。安全補強費用の回収を元#132からの純高速化とは表示しない。

G1は`_signature()`中の次の三列だけを変更する。

1. `named_parameters()`の名前・metadata列。
2. `named_buffers()`の名前・metadata・従来対象version列。
3. `modules()`のidentity/type/training/forward binding列。

coldで全raw登録edgeを順序付きで認証し、weakなModule/Tensorと旧first-path/dedupのleaf表を作る。
liveではowner→現在dict/登録先、全Tensor metadata/pointer、必要versionを既存native helperで照合する。
比較用の巨大tuple、名前生成、再帰的な構造再発見を省く。全edgeとdedupした署名leafは区別する。
古い登録dict・grad・削除subtree・Tensorを追加所有しない。外部ownerのmodel rebindも照合する。

optimizer state/groupのlive列挙・順序・repr、grad、入力/period、hooksのModule走査、TLS/precision、
CST native roots/guard、Polar preflight/launch hook検査はそのまま残す。body、index供給、replay後の
version通知、guard頻度は性能差分へ含めない。新C++ helperや汎用Python namespace解析器は作らない。

## 有限のstock資格とcaptureの復元

固定wheel/sourceのclean worker起動を信頼の起点とし、model/利用者callbackより前にlibrary定義を
anchorへ記録する。現在の任意first-use関数をstock認定しない。stock optimizer初回生成のprofiling
wrapperは、信頼した宣言関数そのものを包む既知shellとして確認する。関数だけでtemplateを作り、
dummy Tensor/optimizerを資格のために保持しない。未知cold overrideはordinary eagerへ戻す。

Module call/traversal、Sequential、Linear/ReLU/functional、CE直接依存、optimizerの有限catalogを
対象にする。既存native helperを使う場合も、stock出自の証明とlive consistencyを区別する。
source固定のnative/autogradや明示したDynamo内部の実装境界を信頼し、任意の先行monkeypatchや
cohortへの競合mutationまで認証する汎用機能とはしない。

`zero_grad`の既知lazy Dynamo cacheは、trusted bootstrapで通常の関数専用factoryによってだけ
初期化できる。既存の未知cacheを上書き/認定しない。build/bootstrap費用を別記し、各cohortの
証明構築費用はprepare時間へ残す。unsupported native APIは従来Python predicateかeagerへ戻し、
検査を省いたacceptへ劣化させない。

captureは最初の仮bodyより前に全eligibleを検査する。P/Adam/clockはuniqueな実TensorとCPU複製で
transactionとして退避し、metricsも保存する。serialization callbackは呼ばず、外部が変更した定数
Bufferやoptionを数値復元で取り消さない。仮3更新＋capture1更新後、数値をbyte確認して復元し、
非grad署名・全hooks/TLS/native資格が元のままか検査する。成功時だけ現在のgrad署名列を更新する。
失敗時はGraphとcaptured temporaryを退役し、復元不能ならcohortを停止する。学習値/versionの通常更新と
固定する実行資格を混同しない。grad値の仮更新前への復元までは主張しない。

## 結果を見る前の工程と停止条件

### Stage 0: CPU

固定stockの正常値更新を対照/G1が受理し、`G1 accept => 資格補強済みG0 accept`をshadowで確認する。
旧19反例とcallable8例、alias/None/rebind、params/group順序、nested mutable、storage/grad変更、
定数Buffer、hooks/TLS、cold callback、capture後の非grad再anchor禁止、weakGCを検証する。
CE kwdefaults例はmetadata拒否であり、意味差の再現数へ加えない。

追加資格検査込みの比較費用を一緒に記録する。部品の速度をCUDA eligible/全モデルへ換算しない。
既存helperだけでは拒否保存・weak寿命が閉じない、または検査全体の費用を減らせなければ、
この固定G1を保全終了する。新C++/一般namespace frameworkへ拡張して救済しない。

### Stage 1: CPU通過時だけ1 L4の短いGate

initial8 / 自身45kの二状態、CST/dense、対照/G1で同じsourceを使う。
selected Plan/entry/実extension同一、同じX/dYの全atom Y/dX/dP byte、16実Graph更新の第1/終端の
状態＋勾配/metrics/order byte、clock8→24 / 45000→45016、capture復元、正常fallback0を確認する。
意図的変更のno-replayと、現在eagerに一致するfallbackは別cohortで確認する。
CUDA算術が同一のhost変更なので、新しいquadrature/cold Adam探索を追加しない。

initial capture cohortで、**共通追加資格込みeligibleが全5 blockで対照より短い**ことを継続条件とする。
1024呼出し×5block、対照/G1順序は均衡固定、shadow/証拠採取は時計外、状態/version不変を確認。
denseも同じprobeを記録する。未達は停止し、追加反復で通さない。

### Stage 2: prospective memory-v2.1を一回

8cycle×16更新＝128実更新/120 replay、eval/save/train/reprepare/recapture、cycles3–8 plateau、
capture前から最終退役まで累積peak、allocated増分≤2MiB/reserved増分≤8MiBを既存判定器/endpointで評価。
全cycleの状態/勾配/metrics/orderを保存PTでも直接照合し、fallback0。Gpilotのpeak一致は流用しない。

### Stage 3: 正式な第1pairへ直接進む

試しの256を別に追加しない。両状態各256更新×6 balanced fresh-process pairsで、各5%以上の
幾何平均短縮と6/6勝利、47 CST state/gradient＋metrics/order byte、fallback0、capture込みmemoryを要求。
denseにも対照/G1の二条件を与えるため、**最大48セル**。有効なCST負けが一つ出たら固定候補を停止する。
全勝でも5%未達は不採用。結果に合わせたhelper構成変更/再試行/反復追加をしない。
全gateを通った場合だけ二順序45,000更新へ進む。

長期/lifecycleの初期8実更新は自身の更新数へ含め、8更新後checkpointからの256比較へ再加算しない。
bootstrap/build、証明構築、capture/復元、更新、再prepare、退役/GCの時間を重複加算しない。

Pro回答原本・source/CPU診断はignored `output/design-search-20261104/`へ保全する。

## Stage 0の観測（GPU未実施）

共通安全補強は`e695af9a` / `21a96b2c`、三列だけの性能変更は`53ac91f6`へ分離した。
CPUのfocused検証は153 passed / 2 GPU skipped、CPU suiteは1990 passed / 2299 skipped。
skipをGPU検証とは数えない。Torch2.13.0 / Python3.11.15 / macOS arm64であり、
対象L4のTorch2.11 / Python3.13の結果ではない。

有限資格には、Tensor/Parameterのmetadata providerも加えた。instanceの`data_ptr`や`stride`を
shadowして値を変更するcallbackは、結果のmetadata tupleだけを比較すると見逃せたため、
providerを実行する前に拒否する。登録Tensor以外のgrad/state/input/periodにも同じ拒否を適用する。
capture rollbackではTensor宣言が変わっていれば復元済みと認定せず、以後の更新を止める。

追加の共通資格とroot edge確認を含む**全signature部分**を1024回×5 blockで測った。
CSTの対照は89.194–91.161µs（中央値90.314）、G1は61.840–62.894µs（62.305）。
denseは42.227–43.333µs（42.803）から34.938–36.330µs（35.447）。
両モデルとも全5 blockでG1が短く、CPU試作の研究費用screenは通過した。
ただし、同じまま残したTLS/hooks/CST nativeのeligible検査はこのCPU時計に含まず、
全CUDA eligible、GPUの律速、完全step短縮や採用を示さない。CPUのAdam状態はclock8の合成値で、
学習更新はしていない。対照とG1のstate byte・version不変を確認した。

最終JSONは`g1-stage0-whole-signature-cost-provider-final.json`、固定測定scriptのSHA256は
`eaa1f93dcd47834946630048aca89e6d16727a9a5cfeb9e44fa046dc7f5b50f6`。
四つの実装fileのSHAをJSONに含め、保存済scriptと照合した。
初期の安全補強前の予備測定JSONは、改訂source測定時の上書きによりblock別配列を失った。
stdoutの中央値だけが残り、その原配列は復元できない。`g1-stage0-preliminary-evidence-limit.json`に
この限界を記録し、Stage 0判定には上の最終sourceの完全な5 block記録だけを使う。
GPU/checkpointの消失ではなく、旧配列を後から再測定して原本扱いもしない。

Stage 1は新しいfresh processの同一sourceで実施予定。未知callbackのcold拒否と
ordinary eager fallback、live変更後のno replay、正常capture復元を両guardで確認し、
状態byteと実eligible費用を通過するまでlifecycle/正式throughput/45,000更新へ進まない。

## Stage 1の最初の実行: 検証runnerの仮定により未完了

job `l4job-6ee82518b27947cb94f8d5b7058875e6`、source `852403bf`。L4上の各guardのsuiteは
28件ずつ、合計56 passed / skip 0。この内40実行がGPU依存、16実行がCPUのtopology検証である。
正常capture、全atom/境界、変更検知、cold/live未知forwardのno-replayとordinary fallbackの
状態・全勾配・totals byte比較を両guardで実行した。

その後の最初の`numerical-initial-cst-qualified`は`sample_vjp`内の検証assertで停止した。
公開出力のgrad_fnは`torch::autograd::CppNode<(anonymous namespace)::ReadLease>`だったが、
runnerが`len(next_functions)==1`と仮定していた。実装の`ReadLease::apply(input,marker)`には
Tensor入力が二つあり、後者はrequires_gradなしのCPU lifetime markerである。
forwardはinputを返し、backwardは第一入力へgradient、第二入力へundefinedを返す。
従って、出力保護をunary nodeとしたrunner側の仮定がsourceと合わなかった。

完成した数値セル0、eligible費用セル0、正式性能セル0。G1の性能や数値比較の合否、
memory-v2.1、長期更新の成立を示さない。共通資格のnative backendも数値セルの完成記録で
確認する前に停止したので、56件のsuite通過だけからその経路を認定しない。
事前規則どおりこのjobを終了し、自動再投入はしていない。別のboundedなrunner修正実験を
許すかどうかは、測定ゼロであることを明示してレビューし、改訂protocolを先に記録する。

rawの21ファイル・tar size・SHA、source/spec/driver、7入力、JUnit、保存された3つのbuild済
共有libraryをローカルで再照合した。数値セルによる実loaded libraryの報告は未完成。
source archive SHA256は`9caadf168f3a17039ed8cae5485b971036f3abecb18bf14d8a6ccd21d9f0b056`、
result archiveは`f52290c899ddd806ec87a6f0c3614994c2fc9420c635a75f4be28f360205b1d6`。
原本と`g1-stage1-retrieval-verification.json`をignored outputへ保全し、pool所有VMの停止を確認した。
簡潔な機械可読記録は同名JSONへ保存した。

## 未実施のrunner修正案

`native_read_lease_path`は固定native経路に限定し、出力ReadLeaseの2 edge、marker側の
`(None,0)`、第一入力のnative本体を確認する。そのnativeの第二入力もP側ReadLeaseであり、
第一入力の`AccumulateGrad.variable is site.atoms.p`まで確認する。native本体名は、同じextensionの
直接entry出力の`grad_fn.name()`と完全一致させる。未知wrapper/Python fallbackは受理しない。
汎用的なautograd graph探索や、検査を消して実験を通す変更にはしない。

実C++ ReadLeaseをCPU上で使う検証は4 passed（Torch2.13/macOS）。正しい二入力の接続、
微分、余分なwrapper/leaseなし/異なるParameterの拒否を確認した。これは対象L4での新runner
実行を示さない。数値body、guard候補、runtime src、workload、採用/停止閾値は変更していない。
改訂された限定実験に研究費用を払うか、G1を未判定で保全してKへ移るかをProへ相談する。

Proの完了レビューは、一回だけの追加Stage1を支持した。初回失敗を消さず、合計最大2試行・
第三試行なしとする例外、backend条件、6 tests/8数値セル/2費用processの上限を、追加GPU確保前の
[amendment仕様](owner-window-flat-guard-stage1-amendment-20261105.ja.md)へ分離した。
この判断自体を追加試行の完了や採用とは扱わない。
