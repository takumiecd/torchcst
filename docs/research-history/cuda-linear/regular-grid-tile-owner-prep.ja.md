# タイル16のY/dX集約: 準備へ移す処理の分離

## 比較の目的と固定条件

`tile-bucket16-bm16` のownerで、幅区分の走査、周期境界を含む候補範囲の
構築・8分割、atomの所属情報の再構築を別々に準備へ移す。
H/Gのproducer、16行のbatch slab、32 atomの局所積、ownerの8分割と最終reduce、
正規化・floor・source VJPは維持する。新しい研究Algorithm
`research_cuda_regular_grid_tile_owner_prep_h` は公開dispatcherへ接続しない。

| Plan | 準備側へ移す処理 | 保存する追加情報 |
| --- | --- | --- |
| `tile-owner-bands-bm16` | 非空の幅区分の抽出 | 各軸 `LEVELS+1` 個のint32: 個数と昇順の区分 |
| `tile-owner-ranges-bm16` | 候補区間の構築とowner partへの分割 | 各軸 `[tiles,8,LEVELS,2,2]` int32: 開始・終了位置 |
| `tile-owner-membership-bm16` | 周期先頭タイルと所属タイル数の構築 | 各軸 `[2,atoms]` int32 |
| `tile-owner-all-bm16` | 上記3つ | 上記すべて |

幅区分は実際の支持幅から毎forwardで作る。GPU→CPUのcount同期を挟まない。
範囲は既存のsorted order/boundsに対する位置であり、tile×atomのIDを展開しない。
所属情報を保存してもownerでの `distance < count` 判定は残す。任意の幅でこの判定を
省くと、幅区分の上限が実際のタイル数を上回る場合に候補を誤って採用するため。

準備した情報はforwardのpacked snapshotと共にbackwardまで保持し、次のforwardで
再構築する。古いParameter・幅のindexを再利用しない。H/Gを保存する寿命は変えず、
dH/dGは局所積の直後に物理勾配へ消費する。H/G全支持の縮約後に勾配を積算するので、
タイル間の交差項も維持する。

N=2048、A=209715では、両軸の追加tensorの合計はbands 72 B、ranges 262144 B、
membership 3355440 B、all 3617656 B。これはtensorの論理量であり、capture/replayを
含むallocated/reservedピークの実測とは区別する。sortのtemporaryの上界は未知のため
`workspace_bound=None`を維持する。

## 検証・計測

CPUの宣言・catalog往復を確認する。GPUでは任意区間・周期wrap・非divisible形状・
全周期支持のmetadataを独立に列挙し、Y/dX/全4成分のatom勾配を独立FP64 oracleと
比較する。元のbucket16とのbitwise一致、floor、retained snapshots、20回のGraph
更新、勾配mask、dH/dGの保存量も検査する。

RTX 6000 Ada、N=2048/B=32/A=209715、rho_initial=3/8、IEEE FP32・TF32無効、
seed=41、AdamW/Polarの24回更新、21サンプル。現行の候補削減版、bucket16、4つの
準備版、denseを同じcohortのfresh workersで比較し、各rhoで独立2cohort・2回目は
逆順とする。各CST workerは初期とpost24の完全形状FP64 gateを測定前に通す。
失敗した場合は性能比較に使わず、同じ条件をretryせず証拠を残す。

固定post24の別Graphでは、準備全体とbands/ranges/membershipの内訳、Y/dX owner、
最終reduceをCUDA eventsで計測する。内訳は準備全体と重複し、診断時間を完全stepへ
足さない。単独版との差は実装変更の効果であり、特定命令の排他的な実行時間とは
呼ばない。完全stepのallocated/reservedピークと合わせて判断する。

## 性能測定前の共通精度修正

最初のsource `d62a7fd6060114319c0b2da23b1c3991e00bb7c9` の405回帰で、
sigma0.7、shape(257,65)、513 atom/17 batchの全dP誤差が0.000472860604026で
4e-4を超えた。性能測定には進まず、404 PASS/1 FAILとsource/logを保存した。
固定したCUDA RNG streamで入力を再現し、元bucket16と4候補のY/dX/all_dPが
bitwise一致することを確認した。失敗は所属range処理ではなく共通のprofile計算にある。

複数周期ずれた代表座標を一周期内へ戻してからnorm/profileを計算する診断では、
最大all_dP誤差が0.000012331545051へ下がった。source/optimizer/VJPに使う元座標は
変えない。bucketとowner準備版は、既存のprepare kernel内で必要なatom groupだけ
FP64周期reductionを行う。中心のpacked保存はFP32、norm/floor/gradの定義は維持し、
全量source copyや新しいglobal scratchは増やさない。他のPeriodic経路の既定は維持する。

失敗入力を固定した5経路の回帰を追加し、未変更の4e-4で再検証する。最初のcampaignは
性能結果なしのFAILとして残し、修正後は別source・別remote directoryへfreezeする。
完全stepは修正後のbucket16を対照とし、過去のbucket campaignと時間をpoolしない。

実測結果は検証終了後に追記する。
