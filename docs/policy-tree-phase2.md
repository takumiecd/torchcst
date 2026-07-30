# Policy Tree — Phase 2 設計ノート: 一直線の protocol stack

起草: 2026-07-30（ユーザー裁定を受けて）。ステータス: **設計・レビュー待ち**。
前提ノート: `policy-tree-design.md`（Phase 1 と語彙・family 裁定はそのまま生きる）。

## ユーザー裁定（2026-07-30・本ノートの公理）

1. **後方互換は全部切る。** 今の使いやすさではなく今後の使いやすさを最適化する。
2. **4層分割（engine / 親 policy / 子 policy / storage）を framework の軸にする。**
   今後のアルゴリズムはこの軸の上にそのまま乗ることを狙う。
3. **依存は四角形ではなく一直線**: engine → 親 → 子 → storage。
   engine は storage への矢印を持たない。親も storage を持たない。
   子だけが storage と密にやり取りし、構造変化を起こす。

## なぜ一直線が成立するか（四角形の斜め辺の由来と消滅）

旧・依存の四角形の斜め辺（engine→storage の書き込み専権）は3不変量の防御だった:
跨 store 原子性・監査の一方向性・単一書き手。これらは「family は閉じた密結合単位」
（policy-tree-design.md「契約の細さと family 密結合」）の裁定によって、
**場所（engine）ではなく位相（protocol の commit phase）で守れる**ようになった。
親＋子はフレームワークが一体で出荷する閉族であり、ユーザーコードに書き込み権限が
渡ることはない。よって斜め辺は不要になり、依存は線に畳める。

## 層構造と protocol

```
engine   clock・update 生活環（begin_update/観測/finalize）・capture サービス・
   │     トランザクション進行役。知っているのは root だけ。storage を知らない。
   ↓↑    上がるもの: requires 宣言（起動時1回）・Plan（毎イベント）
   │     下りるもの: EventView 照会・prepare・commit の各 verb
root     裁定層。cadence の唯一の所有者（「今はイベントか？」に答える）。
   │     family 裁定（RentEconomy: λ 価格フィルタ / QuotaRegime: budget top-k）。
   │     跨 site の計画組立て = カスケード展開（neuron 死→接続 synapse 死）・
   │     dedup・immunity 床。知っているのは自分の子だけ。
   ↓↑    上がるもの: 子の priced proposals・prepared ack
   │     下りるもの: 承認済み per-child plan・prepare/commit
children 局所知性。構築時に自分の store への参照を所有（site 文字列は存在しない）。
   │     自 store の view / 計器を読み、PricedProposal(op, gain, cost) を親へ返す。
   │     commit 位相で、承認された op を自 store の二相 API で実行する。
   ↓↑
storage  真実。版つき二相 mutation・slot 機構・lineage。
         follower 通知と監査イベントの発火点（commit した層が記録を発する）。
```

### 不変量の書き換え

| 旧（四角形） | 新（一直線） |
|---|---|
| engine だけが storage に書く | 書き込みは protocol の commit 位相でのみ・store を所有する子のみ・承認済み Plan どおりのみ |
| engine が原子的 unit を準備/commit | storage の二相 mutation ＋ root の跨子 coordination（prepare 全員完了→commit 一斉） |
| engine が監査を発行 | commit の当事者（storage 層）が発火、監査は従来どおり一方向 sink |
| op はデータ（engine が検査して適用） | **op はデータ（不動の軸）**。Plan は全層境界をデータとして通過し、監査・replay・follower が commit 前に全容を見る |

### イベント意味論の変更（bit 互換の放棄）

イベントは**観測状態の純関数**になる: `Plan = root.propose(EventView)` →
（監査記録）→ prepare → commit。旧 engine の「absorb を commit してから
retention を裁く」という段階的 interleave は実装事故であって理論の要請ではなく、
snapshot 一括裁定の方が J_λ = L + λK の裁定と同型である。
**op_log の bit 互換は放棄**し、検証は統計的等価（同 seed 分布・到達損失・
K 軌跡がベンチ解像度で一致）に切り替える。旧挙動との bit 比較はしない。

## 消すもの（後方互換の切除リスト）

- composed `Policy` の engine 直渡し経路・`StructuralPolicy` 経路・
  `policy/arbiter.py`（ComposedArbiter / WholePolicyArbiter）ごと削除。
  **木が唯一の実行経路**になる。
- `Independent` 根。「調整しないという調整機構」は線の思想に反する。
  対照実験は `QuotaRegime(budget=∞)` を明示で書く。
- 実行時の site 文字列（`_SiteProposer` の fnmatch 照合）。overrides は
  構築時に子→store の束縛として解決し尽くす。
- `LC` / `LC_anti` / `LC_merge` / `LC_response` / `GrowthByProfit` の
  dataclass preset 形。**内容は recipe（木の名前つき構成）として再表現して救う**
  （recipe は既定1＋対照2の上限規律を維持。5c 系 preset は歴史的価値のぶんだけ
  移す。捨てる場合は commit メッセージに記録）。
- engine から: `_expand_retirements`（→root）・`_check_immunity` 系（→root）・
  `_deduplicate_synapse_deaths`（→root）・`_prepare_atomic_unit` /
  `_commit_atomic_unit` / `_apply_atomic_unit` / `apply_proposals`（→storage 二相
  ＋root 進行）・監査組立ての大半（→storage 発火）。

## 残るもの（動かさない）

- storage の slot/lineage/二相 mutation・optimizer follower の仕組み
  （通知の発火点が storage に一本化されるだけ）。
- capture / 計器の実体（engine のサービスとして残る。根が合算した requires
  宣言**だけ**から構築する。engine が policy の内部構造を覗く経路は削除）。
- op データ型・監査レコード型・replay の思想。
- Phase 1 の語彙: `SynapseLifecycle` / 方法コンストラクタ / family gate /
  cadence 根所有 / thinned()。子ノードはこの語彙の実行時対応物になる。

## neuron の座席

NeuronLifecycle は「隣接2層の界面」の子として木に住む（policy-tree-design.md
どおり）。retire→synapse 死カスケードは**跨子調整なので根の計画組立てに属する**。
旧 engine に残っていた `_expand_retirements` の座りの悪さはこの再配置で解消する。

## 段階計画（各段の合格ゲートつき）

1. **S0 統計的等価ハーネス**: 旧 main（3e6d40f）で基準統計（固定 seed 群の
   到達損失・K 軌跡・op 種別頻度）を採取し、以後の各段の合格判定器にする。
   bit 比較はここで退役。
2. **S1 storage 発火**: 変異の帰結の発火点を storage に一本化。実体は
   optimizer-state 整合の二重化解消（store.commit の FollowerHub 通知が唯一の
   経路になる。engine 側の post-commit `reconcile_optimizer_state` 再実行を削除）。
   **監査はここでは動かさない**: イベント級 AuditRecord は裁定文脈（rent 閾値・
   prune 時点の age）を要するため、S3/S4 で「root の Plan＋子の commit ack」から
   組み立てる形に置換する。store 級 commit 通知は FollowerHub が既に担っており、
   それが「storage が発火する」の実体である。
3. **S2 子の実行時化**: SynapseLifecycle → store 束縛済み子ノード
   （view 読み・PricedProposal・二相実行）。
4. **S3 根の実行時裁定**: RentEconomy / QuotaRegime が毎イベント
   propose(EventView)→Plan を実装（λ フィルタ / budget top-k・カスケード・
   dedup・immunity）。cadence 応答も根へ。
5. **S4 engine 削減**（進行中・sub-step 分割）:
   - S4a 済: RuntimeTree を QuotaPolicy 駆動に一般化（windowed 供給・prune cap は
     root の計画行為）。recipes.LC が runtime 経路で指紋 PASS。
   - S4b 済: profit trial = root の execute_trial サブプロトコル。checkpoint
     (TrialTransaction) は全 family 共通の機構として engine が factory で貸すだけ
     （trial の存在を知らない）。ordinary 半（absorb/retention）commit 後に
     checkpoint → priced 半（birth/merge）→ polish → 裁定、が旧境界と同一。
     recipes.GrowthByProfit 指紋 PASS（rollback 込み）。
   - S4c 未: NeuronLifecycle（界面子）。実装指針: compose_response は
     `neuron_id=` 明示指定を既に受けるので、snapshot の dormant_ids から root が
     ターゲット列を計画し、bundle 間の依存は view_after を「計画済み birth の行
     追加（合成負 id・lineage は op が持つ）」に拡張して解く。retire カスケードは
     root の計画行為: SiteBinding.endpoints で当該 neuron store に接する synapse
     子を特定し `spec.incident_synapse_ids` を子経由で照会して死を計画に加える。
     LC_response recipe が lc_response_cascade 指紋を PASS したら完了。
   - S4d 未: StructuralPolicy の後継 = 「root を自作する」研究者向け seam。
     structural_policy_direct fixture を自作 root で再表現して指紋 PASS。
   - S4e 未: 切除実行 — arbiter.py・composed/StructuralPolicy 経路・Independent・
     _SiteProposer/_SiteRetentionCourt・compile() 経路・catalog preset
     （LC_anti / LC_merge は歴史的 preset として recipe 化せず削除を許容、
     コミットメッセージに記録。移すのは指紋が pin する LC / LC_response /
     GrowthByProfit のみ）。engine の instruments reset・capture は現状維持
     （A束は Phase 2 スコープ外の継続課題）。
6. **S5 テスト再憲法化**: 旧テストのうち意味論に依存しない層（storage/compute/
   audit/replay）は維持、イベント編成系は新意味論で書き直し。S0 ハーネスで
   旧 main との統計的等価を最終確認。
7. 全段完了までベンチ箱への rsync 禁止は継続（rebaseline アーク完走が先）。

## 裁定済み（2026-07-30 ユーザーレビュー）

判断原理（ユーザー）: **まとめられるものはまとめる、無理にまとめない。**
全 family に共通して同一の処理なら共通機構に置いてよいが、演算ごと・family ごとに
処理が異なるものを共通層（engine）へ持ち上げてはならない（family 閉族裁定と同根）。

1. **prepare 失敗 = イベント丸ごと破棄**（案A確定）。部分 commit は「誰も裁定して
   いない中間状態」を生むので選ばない。失敗は記録し、次イベントが新状態から再提案。
2. **profit trial は root の裁定内サブプロトコル**（案A確定）。trial は
   「実測による値付け」＝値付け方法の一種であり、必要とする family（profit 系）と
   不要な family（解析的値札の RENT 等）があるため、共通層に持ち上げる資格がない。
   engine は trial の存在を知らない。
3. **observe_window は根の cadence 所有に含める**（案A確定）。同上の原理。
