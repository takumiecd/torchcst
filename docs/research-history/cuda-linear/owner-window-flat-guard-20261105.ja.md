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
