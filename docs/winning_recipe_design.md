# CST-native設計: 勝ちレシピからの再構成 (v4 — 収束版)

> 状態: **GPT-5.6-sol(codex)との2ラウンド敵対レビューで収束した版** (2026-07-20)。
> 審査履歴の全文は
> [`reviews/winning_recipe_v3_review_gpt56sol.md`](reviews/winning_recipe_v3_review_gpt56sol.md)。
> 双方が合意した5契約(§10)を反映済み。
>
> v4の変更(v3レビューの反映):
> 1. **Identity三層分離**: entity IDのnever-reuse / 物理slotの再利用(state初期化つき) /
>    候補retirement(Policy側registry)を分離。v3の「slot再利用禁止」は誤りだった。
> 2. **RepresentationSpec**: family = kernel × 座標domainでは足りず、site分解・
>    sharing・原子価格・retirement意味論・optimizer契約まで含む一体の仕様にする。
> 3. **fingerprint → functional mass** m_k = |w_k|·‖K_in[:,k]‖·‖K_out[:,k]‖。
>    gauge制約はこれを|w|へ退化させる条件、と再定義。
> 4. **RealizedProfitCourtをopt-in決裁部品として追加**(トリガは常にloss遮断のまま)。
>    trial/rollbackは完全な`TrialTransaction`でProfitCourt専用に隔離。
> 5. **ProposalBundle**: cross-entity複合提案(ungate+incident block)のall-or-nothing化。
>    半端均衡(SC-MRG-1/SC-LIFE-2病理)の再発防止。
> 6. **Engine時間境界**: begin_update → observe_microbatch → finalize_backward →
>    optimizer_step → structural_event。gradient accumulation・DDPの契約を固定。
>    BackwardContext(現torchcst)は復活。
> 7. **observesは部品別`requires`宣言**へ。分散はrank0がID込み完全Planをbroadcast。
>    受け入れ条件は3層(contract / 数値・性能 / 実験再現)に分割。
>
> v3以前の変更履歴: v3 = SynapseStore/NeuronStore統一(torchcst形採用)・Schedule
> 3テンポ化。v2 = capture一級化・entity座標系・Policy復活。v1 = 勝ちレシピ棚卸し。
>
> 証拠の出典: `../cst/ROADMAP.md`、`theory/sections/03_support_dynamics.tex`
> (有限予算七法則・運用形)、`docs/framework/framework_conn_5c_report.md`、
> `docs/framework/framework_phase3_design.md`、`docs/cstf_api_draft.py` v0.3。

## 1. 何が勝ち、何が死んだか(棚卸し)

| 確定事項 | 出典 |
|---|---|
| **運用形の4役割: 地図=監査・行き先=流れ・決裁=rent・ペース=スケジュール** | `sec:operational-lifecycle`、CONN-5cで機構完成(thrash 0・静止・compact) |
| 損失トリガ全廃。構造変更の**駆動**は時計スケジュールのみ | CONN-5b X4-C、5c Y4 PASS |
| birth供給は有界窓。窓閉鎖後はrent掃除のみ→静止 | oversupply補題、5b「birth停止でK 15→7・静止」 |
| rentの接続定数: 新生免疫 τ_imm>τ_rise・live母集団median×0.3・2連続ヒステリシス | 接続定数原理、5b X4-A/B、Phase 3 §8B(τ_rise較正) |
| 配置は被覆指向。greedy top-\|g\|(頂点買い)は買わない | N2A・G4・CONN-0/1/3/6の横断確定線 |
| ただし被覆指向の最良形は**反勾配 = certificateを読んで避ける**(B4) | CONN-2: B4 91.5% > random 90.2% > SVD 88.4% |
| score場は監査地図としては厳密、順位表としては系統的に偏る | 近視眼原理・遅延選択原理 |
| **realized profitは理論の決裁則として維持**(S-5/U-1)。ただし凍結済みchampion(5c/Phase 3 A-entry/A-atom)はrent-onlyで、profit gateを常設しない | S-5・運用形「決裁」、5c Y4、Phase 3 §8B |
| 輸送は破壊可能な資源。因子norm正規化(gauge固定)は不変条件 | 輸送凍結補題、CONN-3 C3の崩壊機構 |
| lifecycleの価値は非定常応答に集中 | 5c確定結論、Phase 3-B |
| packed storage機構: id never-reuse・slot pool・follower・version・op log | FW-NEURON live birth契約、cstf v0.3 P1–P5 |
| NeuronGate論理幅 + scheduled ungate | CONN-5系、5c H2応答(+2.85pt) |
| prune済**候補**の再birth禁止・lineage・age追跡 | 5c契約(reuse/rebirth 0)。※物理slotではなく候補identityの禁止(§2.3) |
| 全arm同一apply経路・同一batch tape・RNG分離・provenance hash | 運用ルール、N2A監査訂正、5c integrity |
| 粒度スケール則 γ=(m+n+1)/mn: 表現コスト会計は一級市民 | 粒度補題、Phase 3 P2 |

死んだ・降格した:

| 事項 | 裁定 |
|---|---|
| conv channel添字の連続幾何化 | 打ち切り(「偽幾何」、G0–G4)。連続座標はPDE/作用素向け特殊例へ降格 |
| gradient-greedy birth(top-\|g\|・SVD頂点買い) | 敗北系列で確定。**対照armとして生存** |
| **多数候補をprofitで比較するfull screening** | 「oracle上限としてのみ測り、既定実装にしない」(運用ルール7)。※選択済み1操作のtrial監査(ProfitCourt)とは別物 — こちらは理論の決裁則として型を残す |
| 損失停滞トリガ・fingerprint絶対閾値 | CONN-5死因分析で機構ごと排除 |

**captureの位置づけ**: 死んだのは「勾配の頂点を買う」選択則であって、勾配観測
そのものではない。B4(反勾配)はcertificateへの直交化のためにcaptureを必要と
し、S-1N(dormant gate score)はcapture駆動neuron提案器の理論的根拠になる。
captureは一級サブシステムとし、制約は用途に置く: **観測は提案と監査の情報に
なる。トリガ(いつ)には決してならない。決裁は原則rent、opt-inでprofit trial。**

**帰結**: `cst_policy_embedding.md`はsuperseded(oracle screening枝を本線と誤認)。

## 2. Entity座標系とRepresentationSpec

storeは2つで足りる(現torchcstの統一形を正式採用):

- **SynapseStore**: 原子 = (s, t, w)。synapse = atom(ν = Σ_k w_k δ_(s_k,t_k))。
- **NeuronStore**: 標本点 = (mu, gate)。

### 2.1 family = RepresentationSpec (kernel × domainだけでは足りない)

CSTの1本の式 W_ij = Σ_k w_k κ_in(μ_i, s_k) κ_out(μ_j, t_k) が全familyを
含むが、**凍結済みPhase 3 armを一意に指定するには構成情報がもっと要る**
(例: A-atomはper-offset rank-1で原子価格C_out+C_in+1。unfold全体にDotKernelを
当てると価格が変わってしまう)。familyの単位は次の一体仕様とする:

```python
@dataclass(frozen=True)
class RepresentationSpec:
    domain_in: CoordinateDomain       # 整数格子 / 球面 / box
    domain_out: CoordinateDomain
    kernel_in: KernelSpec             # δ / 内積 / Gaussian
    kernel_out: KernelSpec
    sites: SiteFactorization          # per-offset分解・sharing・site topology
    atom_cost: int                    # 原子1個の価格 (γ会計の分子)
    retirement: RetirementSemantics   # neuron退役時のincident処理 (§4.4)
```

| family | domain | kernel | 座標学習 | atom_cost | retirement |
|---|---|---|---|---|---|
| entry | 整数格子 | δ | しない | 1 | 端点ID cascade |
| per-offset rank-1 | 球面 S^{C-1} | 内積 | する | C_out+C_in+1 | 成分projection op |
| 連続座標 | box [0,1]^d | Gaussian等 | する | 2d+1 | 当面cascade外 |

「偽幾何」裁定とは矛盾しない: 死んだのはchannel添字に距離を導入すること。
δ kernel + 整数座標は距離を持たない。metricを持ち込むのはkernelであって
storeではない。

### 2.2 CoordinateDomainは動的契約である

domainは静的な値検査ではない。momentum SGDは球面上のuを更新すると
‖u‖≠1になり、単純再正規化ではmoment bufferに半径方向成分が残る。
domainが所有するのは:

```python
class CoordinateDomain(Protocol):
    def parameter_role(self) -> Role       # Parameter(学習) / Buffer(entry整数)
    def validate_birth(self, coords) -> None
    def project_grad(self, coords, grad) -> Tensor    # 接空間射影
    def retract(self, coords) -> Tensor               # step後のretraction
    def project_state(self, coords, opt_state) -> None  # moment等の接空間処理
```

輸送凍結補題のgauge固定は、球面domainのretract + moment射影として実装
される。**Policyやユーザーの規律ではなく、optimizer stepに接続された
storeの不変条件**である。

### 2.3 Identityは三層に分離する(v3の誤りの訂正)

| 層 | 規則 | 根拠 |
|---|---|---|
| entity ID | int64・**never-reuse**・単調増加 | v0.3 P2。永続参照・op log・replayの基盤 |
| 物理slot | **再利用可**。ただし再利用時に全follower(optimizer moment・計器state・age)を初期化 | packed storageの成立条件。cSETのcapacity=Kのrewireは再利用が必須 |
| 候補identity | 5cの「再birth禁止」の対象。同一runで一度retireした**候補**(chart上の位置・lineage key)は再提案禁止 | 5c契約。Policy側の`RetiredCandidateRegistry`が所有 |

物理容量(SlotPool倍々拡張 + packed K→K+M growth + optimizer state follower)と
論理生死(live集合 / gate)の二層区別は従来どおり。

### 2.4 Neuron = (mu, gate)。状態は Dormant / Live / Retired の3値

- 物理幅 N_max 固定(chart)。Computeのtensor形状は安定。
- **Dormant**(ungate可能な休眠) と **Retired**(永久退役・再ungate禁止) を
  状態として分離する。birth = Dormant→Live(ungate)、death = Live→Retired。
- muはentry/rank-1では固定(index/標準基底)、連続familyのみ学習可能。
- neuron deathのincident synapse処理は`RepresentationSpec.retirement`が
  Planへ展開する(§4.4)。「incident」の意味はfamily依存であり、
  座標一致(graph incidence)とkernel support(解析的作用)を混同しない。

### 2.5 Storeの共通契約とfunctional mass

```python
class EntityStore:
    site: str
    def view(self) -> View            # packed読み取り。mass列を必ず含む
    def apply(self, op) -> None       # トランザクション、version++、op log
    def prepare(self, ops) -> Ticket  # 二相適用: 検証のみ (§4.4)
    def commit(self, ticket) -> None
    def followers(self) -> FollowerHub
    # 標準follower(framework供給): age列・lineage列・optimizer state影列
```

**fingerprintの正式定義はfunctional mass**:

> m_k = |w_k| · ‖K_in[:,k]‖ · ‖K_out[:,k]‖ (孤立原子のFrobenius norm)

- entry(δ, one-hot列)とrank-1(球面gauge)では m_k = |w_k| に退化する。
  **gauge制約とはfunctional massを|w|へ退化させる条件**である。
- これは原子の表現空間上の質量であり実現効用ではない(coherence・データ
  分布を含まない)。gateがある場合はgate適用後のeffective kernel列を使う
  (規約として固定)。
- Gaussianではσ・境界でnormが変わるため、entry/rank-1のrent定数を
  自動継承しない(較正し直す)。
- massはkernel/store versionでcacheする。

## 3. 設計原則: 法則を型と不変条件に翻訳する

| 法則・契約 | 実装での保証 |
|---|---|
| 損失トリガ全廃 (Y4) | Schedule/Proposer/Allocator/RentCourtの入力型にloss/objectiveが存在しない。objective delta capabilityを持つのはProfitCourtだけ(§4.3) |
| 免疫中prune = 0 | rentの読み値からage < τ_immの行を除外(runtime必須拒否 + 型でIDを渡しにくくするergonomics。「型だけで不可能」とは主張しない) |
| 供給有界 (oversupply) | 提案器を呼べるのはscheduleが窓内で払い出した予算だけ |
| 候補の再birth禁止 | Policy側`RetiredCandidateRegistry`(lineage key)。物理slotは対象外(§2.3) |
| 輸送保護 | 球面domainのretract + moment射影がoptimizer stepに接続されたstore不変条件 |
| 地図≠順位表 | 監査計器は集計値のみ公開。per-candidate順位Readingは部品別`requires`宣言を通してのみ配線(greedy対照armの依存が宣言から見える) |
| 半端均衡の禁止 (SC-MRG-1/SC-LIFE-2) | cross-entity複合提案は`ProposalBundle(atomic=True)`。Allocatorはbundle単位で採否 |
| 同一apply経路 | baselinesも同じStore/Op/Follower経路。mask直接書き換えなし |
| 接続定数は較正対象 | τ_rise較正ルーチン(`lab/calibrate`)。新規設定では較正必須、凍結済み設定(5c/Phase 3 §8B)は凍結定数を使用 |
| 再現性 (N2A教訓) | 名前付きRNG stream・tape hash・op log・provenance SHAはEngineの標準出力。**構造replay(op log決定性)は必須契約、full bit-replayはpinned determinism flags下のbest-effort** |

## 4. Policy: 差し替え可能な学習則

### 4.1 型 (収束版)

```python
@dataclass(frozen=True)
class Policy:
    """1つの構造学習則。差し替え = この値を替えるだけ。"""

    schedule: Schedule                       # ペース: 観測窓・event・応答窓の3テンポ
    proposers: tuple[OpProposer, ...]        # 行き先: op種別ごと(Birth/Merge/Ungate/…)
    composer: BundleComposer | None          # 複合提案の宣言的結合 (§4.2)
    allocator: BudgetAllocator               # 予算・quota・cross-layer配分。bundle単位で採否
    retention: RetentionCourt                # 決裁(必須): RentCourt / MagnitudeCourt
    profit: ProfitCourt | None               # 決裁(opt-in): scheduled trial監査 (§4.3)
    # 各部品は自分の requires: tuple[InstrumentSpec, ...] を宣言する。
    # Engineは部品ごとに必要最小限のReadingだけを配線する(平坦なobservesは廃止 —
    # RentCourtにloss readingが届くような漏れを構造的に防ぐ)。
```

- `Schedule.phase(clock)` / `observing(clock)` / `event(clock)` — 入力は
  Clockのみ。3テンポ(毎stepの流れ / 観測窓 / event)をすべて所有する。
  観測の「いつ」もschedule依存である。
- retirement意味論はPolicyのフィールドではない。Policyは`NeuronRetire`を
  提案するだけで、incident展開は`RepresentationSpec.retirement`が行う。
- 決裁courtは「何らかのRetentionCourtが必須」であり「RentCourt必須」では
  ない(cSET/cRigLはMagnitudeCourt)。canonical mergeは原則ProfitCourt必須。

### 4.2 ProposalBundle: 複合提案のall-or-nothing

`UngateProposer`と`SynapseBirthProposer`を独立のままAllocatorに渡すと、
ungateだけ採択されincident blockが落ちる半端状態を作れる(旧APIが
SC-MRG-1/SC-LIFE-2で避けた病理の再発)。最小の追加はデータ型1つ:

```python
@dataclass(frozen=True)
class ProposalBundle:
    bundle_id: str
    ops: tuple[CandidateOp, ...]   # 例: (NeuronUngate, SynapseBirth×G)
    cost: CostVector               # γ会計の単位
    atomic: bool = True            # Allocatorはbundle単位でしか採否できない
```

proposerはop種別ごとの独立部品のまま、宣言的な`BundleComposer`が参照キーで
結合する。一般のController抽象は導入しない。

### 4.3 ProfitCourt: opt-inの実現損益決裁

理論の決裁則(S-5: 正のrealized profitのみ受理すれば価格込み目的は減少)を
型として提供する。ただし:

- **トリガではない**: 発火は常にschedule。ProfitCourtは発火済みtrialの
  受理/rollbackだけを行う。「lossによる発火」と「schedule発火後の損益監査」
  は別物であり、後者はY4と両立する。
- **championではOFF**: 5c/Phase 3 A-entry/A-atomはrent-only凍結。catalogの
  LC系はprofit=None。ProfitCourtを使うのはPhase 1 C1系実験・canonical
  merge・profit-gated則。
- **full screeningとは別物**: 多数候補のprofit比較はoracle計器(audit)に
  留める(運用ルール7)。ProfitCourtは選択済み少数操作のtrial監査。
- **rollbackは完全な`TrialTransaction`**: 構造rowだけでなく、model
  parameters/buffers、optimizer state、store/follower/version、RNG stream、
  Instrument accumulator、Clock/event stateを対象にする。または
  clone/shadow上でtrialしaccept時のみcommit。重いがopt-inなので許容。
  **trial能力とobjective readingは通常のLC/SET/RigLループには配線されない。**

### 4.4 cross-storeの二相適用

neuron death + incident synapse death(および全bundle)はprepare/commitの
二相で適用する: 全対象storeが`prepare(ops)`で検証し、全成功後にのみ
`commit`。現行の逐次apply(部分適用の危険)は廃止。workbench §8の未決は
「二相が必要」で確定(ProposalBundleとProfitCourtのrollbackが要求するため)。

### 4.5 学習則カタログ

| Policy | schedule | proposers | retention | profit | requires |
|---|---|---|---|---|---|
| **cSET** | Periodic(ΔT, cosine frac) | RandomBirth | MagnitudeCourt | — | — |
| **cRigL** (対照) | Periodic | GradFieldTopKBirth | MagnitudeCourt | — | GradField |
| **LC** (5cチャンピオン) | BirthWindow+FrozenSweep | CoverageRandomBirth | RentCourt(免疫/median×0.3/2連続) | — | — |
| **LC-anti** (B4系・次の本命) | BirthWindow | OrthogonalBirth | RentCourt | — | CertificateSubspace |
| **LC-response** (Phase 3-B) | +ResponseWindow | CoverageRandom + ScheduledUngate(bundle) | RentCourt | — | — |
| **LC-merge** (canonical merge) | BirthWindow | + MergeProposer | RentCourt | **ProfitCourt** | — |
| **C1系** (Phase 1実験) | 任意 | 任意 | RentCourt | **ProfitCourt** | 宣言 |

「今はrandomが買っているが今後はcaptureが買っていく」は、**proposers列だけが
進化する**形で吸収される。schedule=時計、retention=rentの勝ちレシピは
据え置いたまま提案器を差し替える。設計試験: LC-merge(merge+profit受理)と
cross-layer budget則(Allocator)が**RawPolicyなしで**書けること。

### 4.6 Escape hatch

`RawPolicy`(自由なdecide)は残すが、構成的保証(免疫・供給有界・loss遮断・
bundle原子性)が失われることを型名が示す。本線はカタログで運用する。

## 5. Observation / Instrument / Engine時間境界

### 5.1 Engineの1 update lifecycle

gradient accumulationとDDPを最初から契約に入れる:

```text
begin_update(update_id)
  └ forward/backward × microbatch回
      └ hook: detach済み軽量factをqueue (BackwardContext — 現torchcstを踏襲)
      └ observe_microbatch(weight)      # 集約重み。DDP reductionの前後関係を固定
finalize_backward()                      # queueをInstrumentへ配信。集約を確定
optimizer.step()                         # domainのretract/moment射影もここに接続
structural_event()?                      # scheduleのevent時のみ: 提案→採否→決裁→二相apply
```

- `UpdateClock(update_step, micro_step, samples_seen, total_updates, phase)`。
  scheduleの「総stepの75%」等はupdate_step基準で定義する。
- InstrumentSpecは集約規則を宣言する: sum / mean / EMA / **abs-after-sum**
  (microbatchごとの|g|EMAはaccumulation数で結果が変わる — g₁=+1, g₂=−1は
  update単位では0。既定はupdate単位で集約してから絶対値)。
- 高価な計算(certificate SVD等)はhook内で行わず、event直前の観測窓で行う。

### 5.2 配線

1. **Observationは1種類**(x, g_out, version, update_id, micro_weight)。
   family別の解釈はInstrument側が持つ。
2. **活性化は「部品別requires宣言(何を) × scheduleの観測窓(いつ)」の積**。
   宣言がなければhookは張られず、窓外では休止する。RigLのΔT窓蓄積、
   B4のevent前certificate構築、LC/推論時のゼロオーバーヘッドが同一機構。
3. **同じInstrumentを提案器と監査が共用**(B4のcertificate = Phase 3媒介計器
   P1c)。二重実装を作らない。
4. per-atom状態はFollowerとしてstoreに追従。slot再利用時は初期化(§2.3)。

### 5.3 分散契約

- **rank0がIDまで割り当てた完全なserialized Planをbroadcastし、全rankが
  同一適用する。** IdAllocatorのrank上位bitは撤回(rank別発行とrank0 decide
  は両立しない。分散decideを将来やるならその時に別契約)。
- 観測は「統計量をall-reduceしてからrank0で決定」をInstrumentSpecに固定。
- 全rankでprepare成功 → commit(二相はここでも効く)。

## 6. 全景

```text
Storage ──View──→ Compute ──Observation──→ [Instruments] ──Readings──→ Policy ──Bundles/Ops──→ prepare/commit ──→ Storage
   ▲                                                          ▲
   └── age/lineage/optimizer follower                UpdateClock┘ (schedule)
```

```text
cstf/
├── storage/
│   ├── mechanics.py      # IdAllocator / SlotPool / FollowerHub / age・lineage
│   ├── synapse.py        # SynapseStore (s,t,w) + functional mass + 二相apply
│   └── neuron.py         # NeuronStore (mu,gate) + Dormant/Live/Retired
├── representation/
│   ├── spec.py           # RepresentationSpec / CoordinateDomain / RetirementSemantics
│   ├── kernels.py        # DeltaKernel / DotKernel / GaussianKernel
│   └── domains.py        # IntegerGrid / Sphere(retraction) / Box
├── compute/
│   ├── cst_linear.py     # kernel合成 (delta/dotはgather/scatter専用path)
│   ├── cst_conv.py       # unfold視点・per-offset site
│   └── capture.py        # hook + BackwardContext (finalize_backward境界)
├── instruments/
│   ├── certificate.py    # CertificateSubspace (提案器・監査共用)
│   ├── gradfield.py      # GradField / CandidateField
│   └── coverage.py       # CoverageMap
├── policy/
│   ├── contract.py       # Policy / OpProposer / ProposalBundle / Allocator / Courts
│   ├── schedules.py      # Periodic / BirthWindow / ResponseWindow
│   ├── proposers.py      # Random / CoverageRandom / Orthogonal / GradFieldTopK / Merge
│   ├── courts.py         # RentCourt / MagnitudeCourt / ProfitCourt+TrialTransaction
│   ├── registry.py       # RetiredCandidateRegistry
│   └── catalog.py        # cSET / cRigL / LC / LC-anti / LC-response / LC-merge
├── audit/                # economy(K(t)/thrash/realized profit) / accounting(γ会計)
├── baselines/            # SET / RigL / static (同一apply経路)
├── lab/                  # tape / streams / ledger / calibrate(τ_rise) / arm
└── engine.py             # update lifecycle・配線・routing・分散broadcast
```

## 7. 受け入れ条件 (3層)

### 7a. 決定的contract test

1. 免疫中IDへのDeathOpをruntimeが必ず拒否する。rentの読み値に免疫行が
   含まれない。
2. Schedule/Proposer/Allocator/RentCourtの型にloss/objectiveが現れない。
   objective deltaに触れるのはProfitCourtのみ。
3. birth供給はscheduleの払い出し以外に経路がない。
4. entity ID never-reuse。物理slot再利用時に全follower stateが初期化される。
   RetiredCandidateRegistryが同一候補の再提案を拒否する。
5. 球面domainのretract + moment射影がoptimizer step後に不変条件を回復する。
6. ProposalBundle(atomic)の部分採択が不可能。cross-store Planはprepare全成功
   後のみcommitされ、prepare失敗時に全store無変更。
7. requires宣言のないPolicyでhookが1本も張られず、観測窓外で休止する。
8. ProfitCourtのTrialTransactionがrollback後に(param/opt state/store/RNG/
   instrument/clock)を完全復元する。LC/SET/RigLにはtrial能力が配線されない。
9. `catalog`の全Policyが`Engine(model, opt, policy=X)`の1引数交換で切り替わる。
10. 構造replay: tape + RNG stream + op log + 初期checkpointから、構造イベント
    列がbit一致で再生される。

### 7b. 数値・性能test

11. dense等価性: 各RepresentationSpecでW実体化との数値一致(有限差分勾配含む)。
12. entry専用path(gather/scatter)のメモリ/FLOPsが一般kernel pathの上限以下。
13. gradient accumulation数を変えてもupdate単位集約の観測値が不変。
    DDP world_size=1と=2でPlanが一致(統計all-reduce後決定)。
14. functional massがentry/rank-1で|w|に退化し、Gaussianで‖K列‖を反映する。

### 7c. 実験再現 (pinned config/seeds/tolerance — frameworkの合否と分離)

15. 5c H1相当構成でthrash 0・構造静止・compact化を再現(登録tolerance内)。
16. Phase 3-A2凍結spec(§8B: 免疫定数・rent定義・birth窓・予算parity)が
    configとして直接表現でき、較正ルーチンがτ_rise変換則を再現する。

## 8. 既存資産との関係

- **torchcst `storage/`**: SynapseStore/NeuronStore統一形を正式採用。追加:
  age/lineage標準follower、slot再利用時state初期化、mass列、二相apply、
  Dormant/Retired、CoordinateDomain接続。
- **torchcst `backward/`**: **BackwardContextは復活**(hookは軽量factのqueue、
  finalize_backwardで配信 — v2の「不要」判定はDDP/accumulation/reentrantを
  見落としていた)。`compute/capture.py` + `instruments/`へ再編。
- **torchcst `policy/`**: cSET/cRigLは`catalog`の2行に。instrumentsの
  GradEMA/CandidateProbeは`instruments/`へ。
- **workbench / MVP RFC**: 三角形・entity縦割り・Engine中立性は吸収済み。
  未決だったmulti-store atomicityは「二相prepare/commit必須」で確定。
  gradient accumulation(RFC未決7)は「update単位集約 + 集約規則宣言」で確定。
  両文書はクローズ可能。
- **`cst_policy_embedding.md`**: superseded。
- **`../cst`**: 数値oracle(5b/5c/Phase 3 probeのSHA付きJSON)としてread-only参照。

## 9. 実装順

1. `storage/mechanics` + SynapseStore + IntegerGrid×DeltaKernel(entry) +
   ID三層・mass・age/lineage契約。二相applyの骨格。
2. `policy/contract` + schedules/courts(Rent/Magnitude) + `catalog.LC`。
   単層toyでthrash 0・静止を再現(**5cの凍結定数を使う — 較正不要**)。
3. `lab/` tape・streams・ledger。構造replay一致テスト(7a-10)。
4. Engine update lifecycle(begin_update〜finalize_backward) + `capture` +
   `instruments/` + `catalog.cRigL`。accumulation不変テスト(7b-13)。
5. Sphere domain(retraction) + DotKernel + per-offset SiteFactorization +
   `cst_conv` + `catalog.LC-anti`(B4提案器 = capture駆動提案の本命)。
6. NeuronStore gate意味論 + ProposalBundle + retirement展開 +
   `catalog.LC-response`(Phase 3-B形)。
7. `audit/` + `lab/calibrate`(τ_rise変換則) + Phase 3-A2 arm config。
   **新規設定の免疫定数はここから較正で生成**。
8. ProfitCourt + TrialTransaction + `catalog.LC-merge`(opt-in経路の検証)。
9. GaussianKernel(連続family)が同じ枠で動くことの確認は最後。

## 10. 審査履歴と合意契約

GPT-5.6-sol(codex, reasoning=high)による敵対レビュー2ラウンドで収束。
全文: [`reviews/winning_recipe_v3_review_gpt56sol.md`](reviews/winning_recipe_v3_review_gpt56sol.md)。
指摘10件中、v3の誤り4件(slot/ID混同・座標optimizer契約欠如・incident意味論・
accumulation未定義)を全面受け入れ、2件(profit常設・宣言的Policyの範囲)は
反論の上「profit=opt-in」「Bundle+Allocatorまでで拡張上限」で双方合意。

双方が最終合意した5契約:

1. **Identityとtransaction**: ID never-reuse / slot再利用 / candidate
   retirementの三層分離と、cross-store Planのprepare/commit原子適用。
2. **RepresentationSpecと連続更新**: domain・kernel・site分解・sharing・
   原子価格・retirement意味論・functional mass・optimizer retraction/state
   処理を一体契約に。
3. **Engine時間境界**: begin_update → observe_microbatch → finalize_backward
   → optimizer_step → structural_event。集約規則・DDP reduction・Clock意味論。
4. **宣言的Policy合成**: op別Proposer・ProposalBundle・global Allocator・
   部品別requires・RetentionCourt。LC系はRent必須、ProfitCourtはopt-in、
   canonical mergeはProfitCourt必須。
5. **Profit trialの隔離**: trial/rollbackは完全なTrialTransactionとして
   ProfitCourt専用に閉じ、通常ループにはtrial能力もobjective readingも
   配線しない。
