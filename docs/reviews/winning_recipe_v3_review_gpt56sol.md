総評は「役割分離の方向はよいが、現状は実装開始不可」です。特に realized profit の脱落、slot/ID の混同、cross-entity 意味論が blocking です。

### 1. realized-profit gate を型から消している

対象節: §3、§4.1、§7.2

判断: **Reject**

具体例: schedule が birth を許可した後、固定 tape 上で有限 polish した結果、損失改善が 0.01、構造価格が 0.02 なら棄却すべきです。しかし `Judge(view, ages, strikes)` には before/after objective がなく、受理済み birth を棄却・rollbackできません。

破綻する理由: [ROADMAP.md](/Users/ware10sai/Desktop/personal_research/cst/ROADMAP.md:35) は「最終受理は realized profit」、理論の[運用形](/Users/ware10sai/Desktop/personal_research/cst/theory/sections/03_support_dynamics.tex:734)も「実現損益ゲートと rent が決裁」と明記しています。損失による「発火」と、schedule 発火後の損益「監査」を混同しています。merge/prune も同じ問題を持ちます。

最小の代案: loss を `Schedule`/`Proposer`/rent からは遮断したまま、scheduled trial 専用の `RealizedProfitCourt(before, after, price, trial_id)` を追加する。trial apply → 有限 polish → accept/rollback の状態機械を契約に含める。

### 2. 「retired slot 再割当禁止」は packed storage と矛盾する

対象節: §2.3、§2.4、§3、§7.4

判断: **Reject**

具体例: capacity=100、K=100 の cSET が10本 deathし、同じeventで10本birthする場合、物理slotを再利用禁止にすると最初のrewireでcapacity exhaustedになります。cRigLも同様です。現実装の [`SlotPool.apply()`](/Users/ware10sai/Desktop/personal_research/torchcst/src/torchcst/storage/common.py:104) と `BalancedSlotPool` は、まさに死亡rowをbirthへ再利用します。

破綻する理由: 混同されているのは、物理rowであるslot、never-reuseのentity ID、再提案禁止対象となる候補/chart IDの三つです。物理slotまでretireすると、packed storageと固定K baselineが成立しません。

最小の代案: 不変条件を「entity IDはnever-reuse」「物理slotは全follower/optimizer stateを初期化して再利用可」に修正する。再birth禁止は別の `RetiredCandidateRegistry` または lineage key に置く。

### 3. kernel × domain だけでは凍結済みconv familyを一意に表せない

対象節: §2.1、§6、§7.8

判断: **Concern**

具体例: ResNet stage3の `C_out=C_in=64` で、unfold全体に DotKernel を適用すると原子は `u∈R^64, v∈R^576`、価格641です。一方、凍結済みPhase 3 A-atomは[per-offset rank-1](/Users/ware10sai/Desktop/personal_research/cst/docs/framework/framework_phase3_design.md:53)で、各原子価格は `64+64+1=129` です。§6の「per-offset site」を読めば後者にできますが、その分解は kernel/domain config には含まれていません。

破綻する理由: familyには、kernelと座標domain以外に、offset分解、parameter sharing、site topology、予算配分単位が必要です。また整数座標 `[K,1]` と球面座標 `[K,C]` ではstore tensor形状も変わり、既存instanceに対する単純config交換ではありません。

最小の代案: `RepresentationSpec = domain + kernel + factorization/layout + sharing + atom_cost` とする。Phase 3用には `DeltaOffset × DotChannel` のper-offset仕様を明記し、§7.8は「同じstore機構・Policy契約」に弱める。

### 4. 座標制約とoptimizerの契約がない

対象節: §2.1、§2.4、§3「輸送保護」

判断: **Reject**

具体例: momentum SGDで単位球面上の `u` を更新すると、通常のstep後は `‖u‖≠1` です。単純に再正規化してもmomentum bufferには半径方向成分が残ります。Adam + box clampでは境界に押し付けられた座標に外向きmomentが蓳積します。現実装は [`s/t/w`をすべてParameter化](/Users/ware10sai/Desktop/personal_research/torchcst/src/torchcst/storage/synapse.py:157)し、NeuronStoreの学習座標は未実装です。

破綻する理由: domainは静的な値検査ではなく、birth初期化、勾配射影、optimizer step後のretraction、moment state処理まで含む動的契約です。これがないと輸送凍結防止のgaugeが毎step壊れます。またGaussianでは同じ `|w|` でも中心・σによって `‖K_in[:,k]‖‖K_out[:,k]‖` が違うため、全familyのfingerprintを `|w|` に統一できません。

最小の代案: `CoordinateDomain` に `validate_birth / parameter_role / project_grad / retract / project_optimizer_state / functional_mass` を持たせる。entry座標はinteger buffer、sphereはRiemannian retraction、Gaussian fingerprintは少なくとも `|w|‖K_in‖‖K_out‖` とする。

### 5. CompositeCourtの「incident」の意味がentry以外で成立しない

対象節: §2.2、§4.1、§7.9

判断: **Reject**

具体例: entry familyでneuron 7をretireするなら、端点ID=7のsynapseをcascade deleteできます。しかし rank-1原子 `cuvᵀ` は通常すべてのneuron成分に作用します。Gaussianもcompact supportでなければ全neuronへ非零です。「incident」を非零作用と定義すると、neuron一個のretireで全原子がdeath対象になります。

破綻する理由: 座標一致によるgraph incidenceと、kernel supportによる解析的作用を同じcross-entity opにしています。さらに現Engineはsite batchを順次applyするため、neuron apply成功後にsynapse applyが失敗してもrollbackできず、「同一Plan」は原子的ではありません。`Dormant → Ungate` と永久的な `Retired` の状態も未分離です。

最小の代案: family別のretirement semanticsを定義する。entryはstable neuron IDを端点としてcascade delete、rank-1はgate-onlyまたは全因子の該当成分をzero→再正規化するprojection op、Gaussianは当面cascade対象外とする。Planには全storeのprepare/validate後にcommitする二相適用を入れる。

### 6. 宣言的Policyが重要な学習則を落としている

対象節: §4全体、特に§4.1・§4.3

判断: **Reject**

具体例: 次は現在の `EntityRule(proposer→BirthOps, judge→DeathOps)` では自然に書けません。

- merge候補の射影、有限polish、realized-profit受理
- 全層ERK予算内で、ある層のunused budgetを別層へ移すglobal allocator
- neuron ungateとincident block導入を同時決定する複合birth
- trial中だけ一時的に構造を追加し、後刻rollbackする規則

破綻する理由: 旧API v0.3はSC-MRG-1/SC-LIFE-2の病理を根拠に「全site・全entityを見るglobal `decide()`」を残していました。新設計はこの横断性を、neuron death掃除だけの `CompositeCourt` で解消済みとしていますが、根拠がありません。このままではmerge・profit gate・global budgetの時点でRawPolicyへ落ちます。

最小の代案: 宣言部品を `OperationProposer`（Birth/Death/Merge/Ungate/Trial）＋global `BudgetAllocator`＋`Court`＋状態機械へ広げる。受け入れカタログにmerge一則とcross-layer budget一則を追加し、それらがRawPolicy不要であることを設計試験にする。

### 7. Scheduleの3テンポはgradient accumulationを定義できていない

対象節: §4.1、§5.2、学習ループ

判断: **Reject**

具体例: accumulation=2でmicrobatch勾配が `g₁=+1, g₂=-1` の候補は、optimizer update単位では勾配0です。しかしmicrobatchごとに `|g|` をEMAすると正の高scoreになります。accumulation数を変えただけでcRigLのbirthが変わります。可変長runでは「総stepの75%」も `Clock.step` だけでは確定しません。

破綻する理由: 「観測窓」と「event」を追加しても、観測値の集約単位、重み、finalize時点が定義されていません。scheduleがoptimizer-step時計を所有しても、hookはその間のmicrobatch境界を知る必要があります。

最小の代案: Engine契約を `begin_update → observe_microbatch(weight) → finalize_backward → optimizer_step → structural_event` に分ける。`Clock`には `update_step, micro_step, samples_seen, total_updates, phase` を持たせ、Instrumentごとにsum/mean/EMA/absolute-after-sumを宣言する。

### 8. 分散のID・観測・op broadcast契約が互いに矛盾する

対象節: §2.4、§3「再現性」、§6 Engine、§7.12

判断: **Reject**

具体例: rank0がIDを含まない `SynapseBirth(s,t,w)` をbroadcastし、各rankのStoreがapplyすると、上位bitにrankを持つ `IdAllocator` は異なるIDを発行します。その後rank0 IDを含む `SynapseDeath` をbroadcastすると、他rankではunknown IDになります。

破綻する理由: 「rank別ID発行」と「rank0 decide→同一op適用」は同時には成立しません。また各rankのlocal `g_out,x` から作ったCertificateSubspaceも、all-reduce前なら異なります。現在実装が [`world_size>1` を拒否](/Users/ware10sai/Desktop/personal_research/torchcst/src/torchcst/engine/engine.py:78)しているのは、この契約が未決だからです。

最小の代案: rank0がIDまで割り当てた完全なserialized Planをbroadcastする。観測は「統計量all-reduce後にrank0で決定」か「候補scoreをall-reduce」のどちらかをInstrumentSpecに固定し、全rankでprepare成功後にcommitする。

### 9. BackwardContext/queueの削除は早すぎる

対象節: §5、§8「BackwardContext不要」

判断: **Concern**

具体例: DDPのTensor hookで即時にcertificateを更新すると、parameter gradientのcollective完了前のrank-local値を読みます。Certificate SVDまでhook内で行えばbackwardを直列に停止します。複数forward、reentrant backward、`retain_graph=True`では同じupdateへのObservation帰属も曖昧です。

破綻する理由: 「Observationを長期保持しない」ことと、「backward完了境界が不要」は別です。現実装の [`BackwardContext`](/Users/ware10sai/Desktop/personal_research/torchcst/src/torchcst/backward/capture.py:47) は、この境界を作っています。旧RFCでもgradient accumulationとreentrant forwardは未決事項でした。

最小の代案: hookではdetach済みの軽量factまたは可約統計だけをqueueし、`finalize_backward()`後にInstrumentへ渡す。`update_id/forward_id/microbatch_weight`をObservationに追加し、SVDなど高価な処理はevent直前に行う。

### 10. 受け入れ条件が「不可能・統合試験・実験結果」を混在させている

対象節: §7、§9

判断: **Concern**

具体例:

- §7.1: PythonのJudgeは過去に保存したIDを返せるので、「免疫中pruneが書けない」は型だけでは保証不能。
- §7.12: tape/RNG/op logだけでは初期checkpoint、optimizer、Instrument state、演算決定性がなく、任意runのbit replayは不能。
- §7.13: thrash 0・静止・compact再現はframework契約ではなく統計的実験受け入れ。
- Gaussian統一は一級受け入れに残す一方、理論の本体であるprofit trial/mergeが削られています。
- `calibrate`が実装順7なのに、手順2のLCはすでに較正済み免疫を必要とします。

破綻する理由: 合否原因がAPI欠陥、GPU非決定性、学習結果の分散のどれなのか区別できません。性能条件もなく、generic DeltaKernelが巨大な `K_in/K_out` を生成しても形式上PASSします。

最小の代案: 受け入れを三層に分ける。

1. 決定的contract test: immunity runtime rejection、constraint、cross-store atomicity、checkpoint/resume。
2. 数値・性能test: dense等価、有限差分勾配、entry specialized pathのメモリ/FLOPs上限、accum/DDP parity。
3. experiment reproduction: pinned config/seeds/toleranceでPhase 3/5cを別管理。

v1からGaussianを外し、entryと凍結済みper-offset rank-1に集中する。

## この設計で実装を始める前に変えるべき点

1. schedule発火とloss監査を分離し、realized-profit trial、rollback、mergeを宣言的Policyへ戻す。
2. slot/ID/candidate retirementを分離し、RepresentationSpec・optimizer制約・family別neuron retirement・cross-store原子性を定義する。
3. optimizer-update単位のObservation lifecycleと、gradient accumulation/DDP/checkpoint契約を先に固定し、受け入れ条件をcontract・性能・実験へ分割する。

ファイルは変更していません。


---

# Round 2: Claude応答への回答 (収束)

結論は「全体方針に合意、ただし型に残る修正が3点」です。ProfitCourtのopt-in化には同意します。常設を要求する証拠はありません。

## 反論1への回答

提案された整理に合意します。

- トリガ: 常にloss遮断
- LC/5c・Phase 3 A-entry/A-atom: RentCourtのみ、ProfitCourtはOFF
- Phase 1 C1、canonical merge、profit-gated実験: ProfitCourtをopt-in
- realized profitの事後計測: auditで可能だが、構造受理には影響させない

5cとPhase 3 §8Bはrent-onlyで凍結されており、常設ProfitCourtを支持しません。理論の realized-profit gate は「利用可能であるべき決裁則」であって、凍結済みchampionの必須段と読むべきではありません。

また、選択済みの1操作をtrial評価するProfitCourtと、多数候補をprofitで比較するfull screeningは別物です。後者だけをoracle auditへ限定すればROADMAP運用ルール7と両立します。

一点だけ修正があります。最終型の「RentCourt必須」はPolicy全体では成立しません。cSET/cRigLはMagnitudeDropでありrentを使わないためです。

- LC系Policy: RentCourt必須
- SET/RigL baseline: MagnitudeCourt
- ProfitCourt: opt-in
- canonical merge: 原則ProfitCourt必須

つまり「何らかのretention/adjudication courtが必要」であって、「全PolicyでRentCourt必須」ではありません。

## 反論2への回答

提示された上限で4例中3例は書けます。1例だけ、複合birth用の最小抽象が不足します。

| 学習則 | 判定 | 必要な契約 |
|---|---|---|
| merge＋profit受理 | 書ける | `MergeCandidate`がop・価格・trial specを持ち、ProfitCourtへ渡す |
| cross-layer budget | 書ける | Allocatorが全siteの候補価格・proposal score・quotaを見る |
| ungate＋incident block | 現状では書けない | 複数opをall-or-nothingに束ねる `ProposalBundle` が必要 |
| trial/rollback | 書ける | ProfitCourt専用の `TrialTransaction` が必要 |

`UngateProposer`と`SynapseBirthProposer`を独立部品のままにすると、allocatorがungateだけ採択しincident blockを落とす半端状態を作れます。これは旧APIが避けようとしたcross-entity半端均衡の再発です。

最小追加は一般Controllerではなく、次のデータ型だけで足ります。

```python
@dataclass(frozen=True)
class ProposalBundle:
    bundle_id: str
    ops: tuple[CandidateOp, ...]
    cost: CostVector
    atomic: bool = True
```

各proposerはop種別ごとのまま、宣言的な `BundleComposer` が参照キーで結合し、Allocatorはbundle単位で採否を決めます。profitによる大域ランキングは不要です。

## 修正案への確認

functional massの再解釈には条件付きで合意します。

\[
m_k=|w_k|\,\|K_{\rm in}[:,k]\|\,\|K_{\rm out}[:,k]\|
\]

これは孤立原子が作る行列のFrobenius normなので、representation-space fingerprintとして綺麗です。両kernel列を単位normに固定すれば `m_k=|w_k|` へ退化し、entry/rank-1の凍結済みrentとも整合します。

ただし次を明記すべきです。

- これは「実現効用」ではなく原子のfunctional massである。原子間coherenceやデータ分布は含まない。
- gateがある場合は、gate適用後のeffective kernel列を使うか否かを固定する。
- Gaussianではσ・境界位置でnormが変わるため、entry/rank-1と同じrent定数を自動継承しない。
- massはkernel/store versionに対してcache可能にする。

その他の受け入れ修正には合意します。

## 残る見落とし

### 1. rollbackの範囲

有限polish後のrollbackは構造rowだけ戻しても不十分です。polish中には既存weight、optimizer moment、BN running stats、RNG、Instrument stateも変わります。

ProfitCourt専用の `TrialTransaction` は少なくとも以下を対象にする必要があります。

- model parameters/buffers
- optimizer state
- Store・follower・version
- RNG streams
- Instrument accumulators
- Clock/event state

またはclone/shadow上でtrialし、accept時だけcommitします。これは重いですがopt-inなので許容できます。

### 2. observesはPolicy全体でなく部品別capabilityにする

平坦な `Policy.observes` だと、RentCourtにもloss readingが届くなど、せっかくの遮断が崩れます。

```python
proposer.requires
allocator.requires
court.requires
```

のように部品ごとに宣言し、Engineが必要最小限のReadingだけ配線すべきです。ProfitCourtだけがobjective delta capabilityを持つ形にします。

### 3. retirementはPolicy値ではなくRepresentationSpec側

最終型にある `retirement: RepresentationSpec依存` は、Policyフィールドとして重複させず、`RepresentationSpec.retirement_semantics` とExecutorの契約に置くのがよいです。Policyは `NeuronRetire` を提案し、具体的なcascade/projectionはRepresentationSpecがPlanへ展開します。

## 最終設計に必ず入れるべき契約

1. **Identityとtransaction**  
   ID never-reuse、物理slot再利用、candidate retirementを分離し、cross-store Planをprepare/commitで原子的に適用する。

2. **RepresentationSpecと連続更新**  
   domain・kernel・site分解・sharing・原子価格・retirement semantics・functional mass・optimizer retraction/state処理を一体の契約にする。

3. **Engine時間境界**  
   `begin_update → observe_microbatch → finalize_backward → optimizer_step → structural_event` を固定し、集約規則・DDP reduction・Clock意味論を定義する。

4. **宣言的Policy合成**  
   op別Proposer、ProposalBundle、global Allocator、部品別observes、RetentionCourtを持つ。LCではRent必須、ProfitCourtはopt-in、canonical mergeではProfitCourt必須とする。

5. **Profit trialの隔離**  
   trial/rollback状態機械はProfitCourt専用に閉じ、完全なTrialTransactionを保証する。通常のLC/SET/RigLループにはtrial能力もobjective readingも配線しない。

この3点の残修正――`RentCourt必須`の限定、`ProposalBundle`、完全な`TrialTransaction`――を反映するなら、最終設計として合意です。
