# owner-window: 次設計の事前screening（2026-11-04）

## 目的と証拠の境界

#132を基準に、次に実装費用を払う設計を一つへ絞る。第一候補は **G: Graph stepのguard費用診断** とする。ただし、flat certificateはまだ実装しない。親Sequentialのordered edge検査を先に補強し、未計装の費用診断で実装価値を判定する。CPU律速、flat化で削れる時間、全モデルの高速化はいずれも未確定である。

- 基準R0: [PR #132](https://github.com/takumiecd/torchcst/pull/132)、`fba2e72d733325fbd7be5aac334a0a7463d96b27`。
- 対象: N1024、52,428 atoms、真batch 4、FP32 bodyの中央CSTを含むMNIST全モデル、Graph学習。数式、支持、正規化、dX、全atom勾配、optimizer則、hooks、live settingsの契約を保つ。
- #135の8係数保存Aは負結果として保全する。初期の正式pairは約0.578078→0.617372 ms/updateとなり停止した。同案の再実装・有利な再測定で救済しない。
- 独立したsource/CPU調査とPro回答を参照した。本記録は次のpilotの**事前仕様**であり、pilot完了、GPU上のalias拒否、flat性能、正式採用を報告するものではない。

Pro回答原本はignored `output/design-search-20261104/pro-screening-answer.txt`、SHA256 `18b347ed5d2c78deb8ed76f43819f95bfff8b0b32b2b0c512e37b338263220c1`。関連する同ディレクトリの `execution.md`、`kernel.md`、`guard-redteam.md`、`guard-callable-config-redteam.md` とJSON/CPUプログラムを保全する。ローカルCPUはTorch 2.13/Python 3.11であり、対象L4のTorch 2.11/Python 3.13上の実証と区別する。公式v2.11 sourceの照合は行ったが、Pro自身はprivate trace/checkpointや実wheelを取得せず、GPUを再実行していない。

## G/K/Oを残す理由と順序

| 候補 | 減らし得る費用 | 増える費用・未確認事項 | 今回の判断 |
| --- | --- | --- | --- |
| **G: guardのflat certificate** | 毎stepの再帰探索、名前生成、first-path/dedup、metadata容器の再構築 | live検査は残る。構造、callable、alias、寿命の証明が必要。未計装の対象費用とCPU/GPUの重なりが不明 | root edgeの正しさを直した後、費用診断一回だけを先行 |
| **K: bank period除算の厳密な整数丸め** | 限定periodに対するgeneric FP32 RN除算 | bit抽出、整数演算、分岐、register、fallback。生成コードと動的時間は未確認 | 数学的根拠を保全し待機。CUDA helperはまだ作らない |
| **O: 既存AdamW functorのCTA再配置** | 大きいTensorをviewへ分割し、CTA当たりの仕事を短縮し得る | CTAごとのbias correction/metadata/scheduling、alignment、alias/version、実wheel依存 | G/Kより後順位で保全。並列性不足が律速とは未確認 |

Gで消せるのはguard全部ではない。Tensor identity/data pointer/shape/stride/dtype/device/requires_grad、元の固定buffer version対象、optimizer state/groupの順序・値、hooks/TLS、live settings、CST native guard、Polar preflightは維持する。旧profileの累積時間や再帰call数を加算して未計装の回収時間としない。

Kの限定前提は当該N1024/M8/base tile 1/merge 2/spacing 1経路の `P=2·2^n` または `3·2^n`、`n=0..10`。normal FP32を `|x|=m·2^(E-127-23)` とすると、Pが3倍の2冪の場合、`s=1 (m≥3·2^22)、それ以外s=2`、`m_out=floor((2^s m+1)/3)`、`E_out=E-n-s` で正しい最近傍丸めを構成できる。fastpathは**計算した正規化出力exponentが1..254**の場合に限定し、その他は元の除算へ戻す。input `0x00ffffff` / 2は丸め後に最小normalとなるが、計算exponentが0なのでfallback対象。±0、subnormal、NaN/Inf、非対応periodを近似で扱わない。CPU全mantissa検算はGPU payload、codegen、速度の証明ではない。generic caller全体へ広げず、近似reciprocalや他の加算順変更を混ぜない。

Oのsource費用モデルは、**denseも二つのparam group**であることを反映する。各group最大Tensorを32 viewsに分ける同一policyの場合の予測は次のとおり。

| 条件 | ordinary CTA | middle CTA | fused AdamW launches |
| --- | ---: | ---: | ---: |
| CST基準 | 16 | 4 | 2 |
| CST候補 | 35 | 32 | 2 |
| dense基準 | 16 | 16 | 2 |
| dense候補 | 35 | 32 | 2 |

chunk 65,536/block 512/depth4上限36を前提としたsource計算であり、GPU観測値ではない。以前のdense「36 Tensor/48 CTA」はgroup統合または別分割policyが混ざるため今回の比較に使わない。wrapperが元clockを一回進め、functorがclockを読む設計には根拠があるが、P/g/m/vの非重複・隙間なし区間、options、alignment、version通知、一時view/旧grad/Graph poolの寿命、実wheelのschema/git versionを別途証明する。clock一回とversion counter増分は同じ問題ではない。同functorだけでは互換性を証明できず、総要素算術や主要trafficも減らない。

## G0: root ordered edge補強と残る課題

旧`_signature()`はdedupされた`named_modules()`等を用いる。一方Sequentialはraw `_modules.values()`の全edgeを実行する。CPU toyでは既存ReLUを末尾aliasとして追加するとeager logitsが変わるのに旧signatureは同じだった。CST/optimizer native rootに親Sequentialのedge監視は確認できなかった。これはsource/CPUの反例であり、旧L4 cohortが`eligible=True`で誤replayした実測とは呼ばない。構造不変だった過去の測定全体を無効とも扱わない。

現在のG0補強は、親Sequentialだけのordered `(name, child)` edge列をprepare時に記録する。`RootTopology`はroot/childをweakrefで持ち、毎`eligible()`の冒頭に個数、順序、名前、生存Module identityを照合する。idだけのABA検査や削除subtreeのstrong所有を使わず、変更時はreplay前に全stepを元eagerへ戻す。既存signature、hooks/TLS、metadata/native guardは残す。全nodeのflat certificateを入れた変更ではない。

- **R0**: 純#132。過去との参照用。
- **G0**: #132 + root ordered edge検査だけ。以後の共通の正しさ基準。
- **G1**: screenを通過した場合に初めて検討するflat候補。現時点では未実装。

G0追加費用の回収だけを、R0を上回る性能改善と表示しない。正しさ補強と性能変更は別に扱う。

G0は次のcallable穴を解決したとは主張しない。CPU source-equivalent検査では、capture前にroot/ReLUのinstance/class forwardを変えると、constructorの型条件を満たしたまま固定Graph bodyとeagerが異なる場合があった。また、同じ関数の`nn.Linear.forward.__code__`変更や`F.cross_entropy.__defaults__`のmean→sum変更は既存signatureに現れず、意味を変え得る。CEの`__kwdefaults__`変更はsignatureに出ないが、現CEにkeyword-only引数がないため同様の意味差は確認していない。これらは実GPU eligible/replay未検証である。

将来G1がtraversal呼出しを省くなら、instance/class/code/defaultのcold資格確認とlive変更検出が必要になる。辞書version/identityだけでは、params listのreverse、nested mutable option、同じ関数のcode変更をカバーしない。固定node/edgeはweakな参照とし、古いgradやGraph poolを延命しない。capture・復元後のgrad storageを基準に確定し、再prepare/recaptureで更新する。任意Python heap全体の万能検査ではなく、現guardの対象と明示的callable契約を維持する課題として扱う。

## 一回のpilot: 実装費用を払うかの診断

### 正しさと測定対象

変更はG0のみ。CUDA算術、Graph body、optimizer、guard頻度は維持し、flat/K/O/NCU探索を同時に行わない。CPU ordered edgeテストの後、同じL4の使い捨て変更cohortでcanonical eligible、alias追加後の拒否、拒否時のreplay非到達を確認する。fallbackを実行する別テストでは、現在のeagerの結果・更新と照合する。guard-onlyの状態不変と、正当なfallback updateは別テストにする。

pilotは凍結した**初期8更新後だけ**を使う。guardは学習数値やowner/rank分布を読まないため費用screenを限定する判断であり、学習後の性能を保証しない。意図的な拒否testを通常性能セルのfallbackへ混ぜず、通常セルはunexpected fallback 0とする。

### guard-only

G0のCST/dense正常capture・復元cohortについて、`eligible-only`と`_signature-only`をそれぞれ**1,024呼出し × 5 block**測る。`perf_counter_ns`をblock境界だけで用い、各blockの1 call時間と5 blockの中央値/最小/最大を残す。二probeの順序は事前固定して交替し、profiler/cProfileを入れず、1,024個のsignatureを蓄積しない。eligibleはsignatureを含むため両時間を足さない。

実更新は0、clockは8のまま。計時外にCSTの42状態+5勾配、metrics、clock、対象version、live settingsの前後不変を照合する。denseも自身のstate schemaで対称に確認する。

### real-stepと時間・メモリ

guard-onlyとは別のfresh processで、同じ凍結状態から **R0/G0 × CST/dense × 3反復 = 最大12セル**、各256実更新を測る。R0/G0順序は事前指定で入れ替える。正式な6 paired adoption試験とは呼ばない。

区間境界で同期し、未計装の全step時間、最後のenqueue終了時刻、最終同期完了時刻を残す。開始/終了state・勾配・metrics・order・fallbackも照合する。enqueue後のdrainはqueue残存の手掛かりであり、enqueue自体のbackpressureを含むので純CPU/GPU時間へ分解しない。guard-only時間を全step時間から引いてno-guard性能を予測しない。

重複しない区間を、state load/必要な証明の準備、capture・復元、256実更新、最終退役・GC・同期として記録する。clockは**8→264**。済んだ8更新やcapture用仮更新を実更新に二重計上せず、仮更新後のP/Adam/clock/計数済みmetrics復元を確認する。

capture込みのpeak allocated/reserved、guard-only前後current、退役後currentを両方式で記録する。途中のpeak resetや`empty_cache`で値を整えない。新たな数値cache/flat Tensor所有を導入しないこのpilotでは8 lifecycleを要求しないが、**memory-v2.1合格とは表示しない**。将来のG1は新所有設計に対してprospective 8 lifecycleを実施する。

### 結果を見る前に固定するscreen

G0/CSTの3セルの完全step中央値を `T`、5個のsignature-only blockの1 call時間を `S_i` とする。

| 結果 | 次の判断 |
| --- | --- |
| alias拒否、状態不変、通常セルbyte一致/fallbackに問題 | 性能解釈を止め、正しさ修正だけを行う |
| 全5 blockで `S_i/T ≥ 0.10` | 一つのCPU中心flat実現性prototypeを検討する価値あり |
| 全5 blockで `S_i/T < 0.05` | Gの性能研究をこの段階で見送る |
| 5–10%付近、またはblock間で結論が揺れる | 保全して実装へ進まない。有利な結果を探す追加反復をしない |

10%は実装・証明費用の予算screenであり、半分を削れる予測、厳密な性能上限、5%採用gateの代用ではない。eligibleだけが大きくsignatureが小さい場合は、flat案が残すnative guard等の費用かもしれずG1の根拠にしない。Gを見送っても研究全体を終了したとは扱わず、K/Oは待機案として残す。

## 将来の正式採用条件

screen通過後のG1には、初期8更新後/自身45,000更新後の両状態で各256更新×6 paired比較、両状態で5%以上の幾何平均短縮と6/6勝利、CST42状態+5勾配/metrics/orderのbyte一致、fallback 0を要求する。allocated増分≤2 MiB、reserved増分≤8 MiB、prospective memory-v2.1も維持する。denseにも同じ実行基盤変更を与え、共通基盤改善とCST固有改善を分ける。全gate通過時だけ、準備・再capture・退役/GC込みの二順序45,000更新へ進む。pilot結果だけで採用・mergeしない。

## worktreeとraw証拠の保全

12個の旧worktreeはfull snapshotとして `/Users/ware10sai/.codex/research-archives/torchcst/worktrees-20261104T130657Z` へ保全し、Gitの登録だけを解除した。旧絶対pathは互換symlinkとして残る。Git作業用checkoutではなく、復元手順を通すarchiveである。main、#132、#135、新design-searchの4 checkoutをactiveとする。

HEAD pin refs、verify済みbundle、元index/admin、非Git payloadのhash/inode/metadata/symlink照合、named/detached復元probeを記録した。payload削除は0であり、空き容量を回収したとの主張はしない。private driver間の絶対path/相対wrapper依存もsnapshotと互換pathで保持する。詳細と復元helperはmainのignored `output/worktree-reorganization-20261104/` にある。raw checkpoint/trace/outputはPR本文や短いJSONから再生成できるとは扱わず、archive原本を保持する。
