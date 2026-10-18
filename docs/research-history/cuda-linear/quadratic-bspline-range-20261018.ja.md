# B-splineのW区間集約と累積和

B64のmixedで7.386→4.310 ms（41.65%短縮）を独立2 jobで再現した。
B32 mixedは改善せず、smallは遅い。メモリはB64で54.194/114→86.840/188 MiB
（allocated/reserved）へ増え、Dense以下のメモリ目標を満たさないため既定採用しない。
形・学習規約を保った大batch用の研究候補として、数式・負の結果とともに保留する。

profile・Polar更新則・task幅stop-gradient・全観測siteの離散L2は変更しない。
本番dispatcherへの採用とは分ける。Hをbatchごとに生成する現行版に対し、batch非依存の
W組立と、その線形随伴による全atom VJPを比較する。

## 数式

軸dのgrid座標を `z_d(n)=2n/(N_d-1)-1` とする。二次B-splineの三つの区間の
多項式係数を `c_{d,s,p}` (`s,p=0,1,2`) とし、観測domainへclampした四つの
切替siteを `t_{d,r}` (`r=0,1,2,3`) とする。

\[
 f_{d,a}(n)=\sum_{r=0}^3\mathbf1[n\ge t_{d,a,r}]
              \sum_{p=0}^2 J_{d,a,r,p}z_d(n)^p,
\]
\[
 (J_0,J_1,J_2,J_3)=(c_0,c_1-c_0,c_2-c_1,-c_2).
\]

このindicatorは計算方式の区間表現である。profile自体は従来と同じC1関数であり、
追加窓・幅の量子化・支持境界の近似は使わない。domain外の寄与は数えない。
空支持の軸は全係数の相殺を計算せず明示的にゼロとする。

`w_a=amp_a/D_a`、`D_a=max(sqrt(Q_o Q_i),floor)`として、9枚の係数面へ
各atomの4×4の切替siteで加算する。

\[
 \Delta_{pq}[t_{o,a,r},t_{i,a,s}]
 \mathrel{+}=w_a J_{o,a,r,p}J_{i,a,s,q},\qquad
 C_{pq}=P_oP_i\Delta_{pq},
\]
\[
 W_{ji}=\sum_{p,q=0}^2 z_o(j)^p z_i(i)^q C_{pq}[j,i].
\]

`P`はinclusive prefix sum。添字がNの切替は保存不要。係数面はFP64で、Wだけ
最後にFP32へ丸める。小さいsigma、または片軸のclipped supportが2点以下の寄与は従来の支持別grouped W構築を併用する。
係数面は1/3/9枚ずつ処理して、一時メモリとlaunch/reductionの交換を調べる。

`E=dY`、`G=dW=E^T X` と置く。逆向きのinclusive累積和 `S=P^T` を使い、

\[
 B_{pq}=S_iS_o\bigl(G_{ji}z_o(j)^p z_i(i)^q\bigr),
\]
\[
 T_a=\sum_{r,s,p,q}J_{o,a,r,p}J_{i,a,s,q}
       B_{pq}[t_{o,a,r},t_{i,a,s}].
\]

中心勾配の生の収縮 `T'_o,T'_i` は片側のJをその中心微分へ置換して求める。
既存のprepareから得るamp微分・`D'/D`を使うpullbackは変更しない。
切替siteが動くときもC1接続によりprofileの勾配は連続である。各stepの現在の
係数・normを再計算し、forwardのMetaとplacement snapshotをbackwardに保存する。
retained backwardは後続forwardの配置・Parameter変更から独立である。

計算はW構築 `144 A_broad` 個までのFP64 atomic加算、係数面のscan `O(9 No Ni)`、
atom VJP `O(144 A_broad)`、三つのGEMMで構成する。batch依存のatom処理を除くが、
全行列のscanとその随伴の費用・一時面の容量が加わる。

## CPUで棄却した表現

1枚の差分行列に各軸の3階差分を注入し、各軸3回の累積和でWを復元する案も確認した。
数学的には同値だが、長い積分区間で丸め誤差が増幅する。NumPy FP64、seed9、
N1024/A52428/B32、sigma .1/.4/1.6/6.4/12 の直接計算との比較では、Y最大絶対誤差
.0161、生の中心収縮誤差 .542。固定の精度基準を満たす候補としては扱わない。

9係数面も小さいsigmaをすべて通すと桁落ちする。同じCPU条件でY最大絶対誤差.0133。
大きいsigma (1.6/6.4/12)のみではY最大絶対誤差3.35e-8、中心収縮1.49e-7だった。
このCPU probeはGPU精度・性能の証拠ではなく、新候補の振り分け設計に使う。
生の収縮はL2補正前であり、全Parameter勾配そのものとは区別する。

raw CPU probeとGPU driver/source/resultは ignored `output/bspline-range-20261018/`、
共有poolのjob directoryにも保全する。

## 実機検証

独立FP64 Y/dX/全atom勾配、可変sigma・Graph/optimizer・retained backwardを通過。
最終runtimeと独立repeatの完全step・capture込みallocated/reserved peakは末尾に記録した。

## 係数合成とscanの融合候補

v7は両軸のprefixを9係数面すべてに適用する。v8ではtensor積の線形性から、
各出力次数pについて、先に入力側の3面をprefixして入力多項式へ合成する。

\[
 T_p[j,i]=\sum_q z_i(i)^q(P_i\Delta_{pq})[j,i],\qquad
 W_{ji}=\sum_p z_o(j)^p(P_oT_p)[j,i].
\]

入力側のscanは計9面、出力側は計3面となり、中間合成・読み書きも融合する。
随伴では出力側を先に計3面処理し、入力側を計9面へ展開してreverse scanする。
同じ9係数面の演算順序を変える候補であり、kernelの近似・切り捨てではない。
配列の一時容量は3係数面と1中間面。FP64の合成用WとFP32のWも別に存在する。
Triton scanの性能、ピークメモリと丸め誤差はGPUで別途比較する。

## v7単純版の実測（修正前の探索結果）

NVIDIA L4、1024×1024、A52428、B32、spacing .1、seed41、warmup5/21 samples。
AdamWとfused Polar更新、live sigma・構築・厳密L2を含む未計装Graph完全step。
各候補の前に全A・3状態の独立FP64 oracleを通している。

| sigma条件 | 従来最速 ms | 1面ずつ ms | 3面ずつ ms | 9面同時 ms | Dense ms |
|---|---:|---:|---:|---:|---:|
| mixed | 4.318150 | 8.855427 | 5.661841 | 7.387453 | .082610 |
| small | 1.205669 | 7.208419 | 4.511889 | 5.004908 | .082248 |
| switch | 4.634804 | 10.589937 | 7.690621 | 8.713041 | .082533 |

新候補のcapture/replay込みpeak allocated / reserved MiBは、1面70.590/138、
3面102.590/202、9面198.590/490（3条件とも同じ）。mixed従来51.688/116、
small従来50.231/114、Dense49.002/106。process全体のGPU memoryは未測定。

別計装Graphのmixed・3面版ではrange組立2.308 ms、range VJP2.142 ms。
smallではbroad atomが0でも、それぞれ1.698/1.452 msの固定費を払う。
このstageにはscatter/query・scan・係数合成を含む。未計装完全stepの代用として
足し合わせたり差し引いたりしない。DRAM/cache/atomicの物理counterは未測定。

v7単純版は速度・メモリの両方で不採用。3面版の別job再測定もmixed5.660750、
small4.512089、switch7.688989 msで、同じ結論だった。

## 分離した研究ブランチ

今回のv7/v8試作は `kernel/quadratic-bspline-range` に隔離する。
基準の `kernel/quadratic-bspline-prefix` とPR109は `e4e266e3` のまま維持し、
既存の最速版や公開dispatcherへ新方式を置換しない。
分離前に現在worktreeで作成した `689d57fa` / `a4940ea0` の両commitは新研究branchに保持。
source/result archiveとrawログは削除しない。worktreeのpathは従来と同じである。

## 検証・取得時の注意

v7実装の初期2 jobはTritonのconstexpr再代入とscalar/vector条件式のコンパイルで失敗。
数学・性能の結果には数えず、修正後のsmokeで25件通過を確認した。
`l4job-52a0fe25c3304a25b5159a1bed1e0876`はdriver正常終了・全result archiveの取得と
SHA256照合まで完了したが、取得後のremote cleanupがWebSocket切断で失敗した。
GPUはserver照合で停止済み。データは保全するが、採用判断用の独立run数には数えず、
以後のclean jobで融合版の精度・性能を再確認した。

CPU回帰1968 passed（CUDAのskipはGPU確認ではない）。wheelとsdistをbuildし、
全338 runtime fileがcheckoutとbyte一致することを確認した。
最初のno-isolation buildはローカル環境にhatchlingがなく失敗したため、標準の
隔離build環境で再実行し成功。ローカル環境の依存を変更していない。

B32 mixed/switchの閾値比較とB64 mixedを新しいGPUで確認した。
full-shape oracleは各候補・各case・3つのlive状態でY/dX/全source Parameter勾配の
maxabs/relative L2をともに4e-4未満とする。基準の緩和や幅の固定化はしない。

最終監査で、Tritonの`where`だけで作ったknot fraction tensorが暗黙のFP32に
なり得る点を修正した。従来の値・勾配gateは通っていたが、ceil直近の区間選択も
元のFP64 geometryと揃えるため、±1/3をFP64 tensorに直接入れる。
微小offsetを持つ33 atomでcut index自体をFP64式と完全一致させる回帰テストを追加。
最終の速度・精度・メモリ判断には、この修正後の同一runtimeを使う独立jobを使う。
修正前の表は探索経過として保持し、最終版の数字と混ぜない。

追加のCPU auditでは、大きいsigmaでも片軸の支持が領域端で1点へ縮むと、
L2 floorの倍率で全体係数の相殺誤差が増幅することも確認した。
N1024・sigma .3・片軸中心 `-sqrt(6)*sigma+1e-7` の1 atomではY最大絶対誤差
7.74e-5だが、真値が小さいためrelative L2 gateを満たさない。
この既知のケースを放置せず、v7/v8は **sigma <= cutoff または、どちらかの軸の
clipped supportが2 site以下** のatomを既存のnative W/VJPへ戻す。
全atomの小支持キーをdeviceで毎forward検査し、中心だけの移動でもrouteを更新する。
旧streamの分類は変更しない。profileやfloorを変える救済ではない。
1 atomだけを使う小真値のY/dX/全atom勾配gateと、中心だけを動かすGraph replayの
route反転テストを追加した。通常のbenchmarkにも、このguardの構築費を含める。

## 最終版の比較と判断

最終runtimeは `ee5343502fd91df4482c7f7f5144184ced985ac6`。
FP64 knotと2-site guardを含み、以下の2 job・checkout・wheel/sdistの全338 runtime file、
runner/manifest/oracle/test/catalogの8 fileがbyte一致している。
同じL4上の独立job（各planは新規process）であり、別GPUでの再現とは数えない。

NVIDIA L4 / driver580.82.07 / Python3.13.15 / Torch2.11.0+cu130 /
CUDA13.0 / Triton3.6.0。N1024×1024、A52428、spacing .1、seed41、warmup5、
21 samples、FP32・TF32なし、AdamW lr1e-4 + fused Polar更新。
Aは行列site数の約5%であり、Parameter数の5%ではない。
Denseは通常の学習可能FP32行列。CSTと同じParameterizationではなく、同じ入力・target・
batch・precision・optimizer条件の性能対照である。

初期sigmaはsmall=.1/.14/.18、mixed=.1/.4/1.6/6.4/12、
switch=.1/.4998/.5002/.501/1.6を各等比率でshuffle。boundsは.1..12.8。
旧最速はsmallでmatrix-support4a16、mixedでstream-reuse8（cutoff.5）、
switchでstream-reuse8c02。新方式はrange-fused-c02（cutoff.2）。

| 条件 | 旧最速 ms | 新方式 ms | Dense ms | 旧比 |
|---|---:|---:|---:|---|
| B32 mixed | 4.316159 | 4.319431 | .082487 | 改善なし |
| B32 small | 1.205816 | 3.230222 | .082837 | 遅い |
| B32 switch | 4.631252 | 4.329603 | .082995 | 6.51%短縮 |
| B64 mixed | 7.386281 | 4.309852 | .104415 | 41.65%短縮 |

独立repeatはB32 mixed=4.314593→4.326242（Dense .083255）、
B32 switch=4.631833→4.323370（.082710）、B64 mixed=7.382172→4.307182（.104663）。
B32 mixedは両方とも改善せず、switchは6.51–6.66%、B64は両方41.65%短縮。
smallは最終runtimeの独立job1つ、他の3条件は2つ。21 samplesを21独立runとはしない。

| 条件 | 旧最速 allocated / reserved MiB | 新方式 | Dense |
|---|---:|---:|---:|
| B32 mixed/switch | 51.688 / 116 | 86.590 / 186 | 49.002 / 106 |
| B32 small | 50.231 / 114 | 86.590 / 186 | 49.002 / 106 |
| B64 mixed | 54.194 / 114 | 86.840 / 188 | 49.502 / 106 |

capture/replay込みの実測peakであり、process全体のGPU memoryではない。
repeatでも該当条件のpeakは一致した。新方式はDenseを上回らないというメモリ目標を
満たさず、既定採用しない。batch64の速度候補として研究branchに保留し、
batch32 mixedおよびsmallを新方式へ置換しない。

### 何が変わったか

新方式の別計装Graphでは、B32 mixedのrange組立2.039 ms、range VJP1.725 ms。
B64も2.044/1.724 msで、batchを倍にしてもこの部分の時間はほぼ一定だった。
旧H版のstream forward/backwardはB32で1.041/1.972 ms、B64で2.054/3.956 ms。
W構築をbatch間で共有する効果は確認できたが、9係数面由来の全行列scanと随伴の
固定費がB32の改善を相殺する。smallではrange atomが0でも1.044/.812 msを払う。

新方式のprepare（L2込み）は.162 ms、guardを含む分類/配置は.032 ms、Meta pack .042 ms。
これらのstageは別計装Graphであり、足して完全step時間の代用にしない。
compiler reportでは選択したscan/組立/queryにspill/local load/storeを検出しなかったが、
物理DRAM/cache/atomic counterによる原因の特定はしていない。

選択した4条件とも52 call中1 rebuild、51 reuse（98.08%）。warmup/eager/replayを含む
主測定全体のcounterであり、timed replayだけのreuse率ではない。
W・sigma係数・normを使い回した割合ではなく、GPU上で検証した配置の再利用率である。
全52428 atomのsigmaが測定前後で変化したことを両jobで確認し、every forwardのlive prepareと
独立3状態oracle、forced Graph移動の単体テストで動的な数値・支持・配置を検証した。
各atomのsigmaがtimed replayの一回ごとに必ず異なったとまでは主張しない。

### 正しさとpackage

312 tests passed（metadata1、actual GPU311）。その前にcritical 8件のpreflightも通過。
2-site guard、中心だけの移動でのGraph route反転、ceil直近のFP64 cut index完全一致、
空/単一/小/広い支持、floor1e-6/1000、矩形/異なるorigin/spacing、retained backward、
個別gradient要求、live sigma、公開CSTOptimizerとのParameter/moments/step比較を含む。
CPUは1968 passed / 3087 skipped / 18既存warnings。CUDA skipはGPU検証として数えない。

全A・全shapeの独立FP64 oracleを候補/対照の各case・3つのlive状態で実行。
同じ初期P/X/target hashを候補間・repeat間で確認した。最終新候補の両job全状態の最大誤差は：

| Tensor | maxabs | relative L2 |
|---|---:|---:|
| Y | 1.788291e-5 | 1.196098e-6 |
| dX | 1.903907e-5 | 1.254305e-6 |
| 全source atom勾配 | 1.407709e-4 | 4.037001e-7 |

全て同じ4e-4のmaxabs/relative L2 gateを維持した。数学の変更や近似の導入ではないが、
加算順序は変わり、旧H版とbit一致するわけではない。全ての単体テストの誤差上限が
この表の最大値以下であるとは主張しない。上表は性能caseの全shape oracleの範囲である。

最終wheel SHA256 `19eebda7d5c48edbcc0a38213d970b2b1a779b055386afea5de3e1098f91e27f`、
sdist `4dabcb15a9f2cdee0773a5f5cf6ac622b908a389f06630f9c238f1b3942c58ec`。
両方とも全338 runtime fileが最終runtimeとbyte一致する。検証後の変更は研究記録のみ。

### 最終証拠と再現

| job | source archive SHA256 | result archive SHA256 |
|---|---|---|
| l4job-645610c8538d4fcfb57d50e219b5e8b5 | 41172936e748967d86739f5a8e5f52eb58be82fee95d28ffbdc14e8e839d7daf | 4cb683e1ec9901a587be14f781bef9567257bf7a517c7ed04c37c822845c8da5 |
| l4job-82b8f225cd64406b8659f8a1a4b2860d | 67a2594d2bed7865f75148457309bbe1d56e70cd826ee509722c309fb574d93d | a70d7f17f8ced08de80faac58444fe82e5a0f30b42badd92e9a16a47f39729da |

両archiveと全manifest fileを再hashし、returncode0・完成runner JSONの提出前検査を確認。
通信後片付けが失敗した52a0 jobやqueued cancelを、この独立run数に含めない。
raw evidence保全先はこのworktreeの `output/bspline-range-20261018/verified-jobs/`、
元のpool job directoryも保持する。最終結果の初期条件hash/全精度/メモリ集約は
`output/bspline-range-20261018/audit-final.json`、source byte proofとpackage proofも同所にある。

最終runtimeをcheckoutしたGPU環境で、既存runnerを使う例：

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-b64-mixed-range-selected.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --source-commit ee5343502fd91df4482c7f7f5144184ced985ac6 \
  --output output/bspline-range-reproduction.json
python -m benchmarks.submissions check output/bspline-range-reproduction.json
```

所有者L4では共有poolに `output/bspline-range-20261018/selected-guard.py` を提出する。
commit引数だけならcritical/full testsと4条件、続けて`repeat`を渡すとmixed/switch/B64の
3条件を新規processで再測定する。構築済み結果のhash検証は `verify.py`、
同一入力・全勾配gateの集約は `audit_final.py`、runtime照合は `verify_sources.py` を使った。

実装・数式・負の結果はDraft [PR110](https://github.com/takumiecd/torchcst/pull/110) に保存。
PR109をbaseとする独立研究PRで、現行最速版は変更しない。最終2 jobのremote cleanupと
owned GPUの停止を確認し、queued/running jobが残っていないことも確認済み。
