# 2-tile・低M profile の形と実装検証

2026-10-14。前の3-tile revision4とは別の数学的演算revision5。
形の探索を先に行い、ユーザーは狭いsigmaでの幅下限を含めて候補を比較する方針を選んだ。
既存profileへの近似誤差を目的にせず、有限profileそのもの、裾、L2、計算量を評価する。

## 形の探索

sigma_max=10、tile幅T=20、局所周期P=40、spacing0.1。
1 tile200 sites、選ぶ2 tiles400 sitesが一周期。owner中心の左側または中心ちょうどなら
左隣+owner、右側ならowner+右隣を選ぶ。常に2枚を使う。
global Torusのseamではtile IDをmoduloで扱うが、中心位相はunwrapped pair原点から計算する。
2 tiles以外は厳密0、2 tiles内の半径10外は有限Fourierの小さい裾が残る。

M16/24/32、22 profile設定、84 sigma、17 centre位置のCPU FP64比較。
Chebyshev、Kaiser、DPSS、Fejer、Jackson、raised cosine、heatとその窓掛け、
幅テンプレートの直接定義を含む。負値を持つ比較profileは正値候補と区別する。
5544 profile rows、94248 site二乗和/Parseval比較、330 branchwise centre微分checks。
tail領域は1e-6単位の整数距離で閉区間を正確に判定し、端点の丸め誤差で漏れを増やさない。
16384点で測った連続形状・tail値は格子による推定で、解析的全域保証とは区別する。

## 選んだ新profile

`KaiserTemplateSpec`、eta=1、knots=(0.1,0.3,1,3,6,10)。
各knotsでnative Kaiser Hを用意し、隣接2つのK=H²をlog(sigma)で凸結合する。
非負性と周波数Mを保つ新しい数学的定義であり、単一Hの平方と同じものではない。

```text
r0 = eta * P/(M+1), R=P/4
r(sigma) = [r0^4 + (1-(r0/R)^4)*sigma^4]^(1/4)
B = pi*(M+1)*r/P
h_k proportional to I0(B*sqrt(1-(2k/M)^2)), k=-M/2..M/2
sum h_k=1; K_j=H_j^2; c_j=h_j*h_j
t=log(sigma/sigma_j)/log(sigma_{j+1}/sigma_j)
K_sigma=(1-t)K_j+t K_{j+1}; c_sigma=(1-t)c_j+t c_{j+1}
```

sigmaは形を制御するwidth knobで、literal support radiusや標準偏差ではない。
最小sigmaでも、実際の半値半径には以下の下限がある。

| M | 半値半径 sigma0.1 → 10 | 半径10外のsite L2 energy比（検証点最大） | midpoint切替のnormalized vector L2 jump（最大） |
| --- | ---: | ---: | ---: |
| 16 | 1.350 → 2.502 | 1.254e-7 | 5.007e-4 |
| 24 | 0.908 → 2.002 | 6.349e-8 | 3.562e-4 |
| 32 | 0.684 → 1.716 | 3.405e-8 | 2.608e-4 |

sigma0.1そのものの半径の外にあるsite L2 energy比は、M16で85.3〜90.2%、
M32で71.3〜80.8%。これは幅下限に対応し、半径10外の低漏れと区別して記録する。

支持pairの切替は微小な裾を入れ替えるので有限の不連続を持つ。
中心微分は選んだbranch内の微分で、整数routingを微分しない。
sigma task VJPは既存Polarと同じstop-gradient。幅はoptimizer更新後にliveに読み直す。

## L2と共有集計

N=400>2Mなら、完全な一様周期の観測なので`||K||²=N*sum|c|²`は中心非依存。
テンプレートのnormは各隣接pairの3つのimmutable Gram値から計算できる。

```text
g00=<c_j,c_j>, g01=<c_j,c_next>, g11=<c_next,c_next>
||K||²=N*((1-t)^2*g00+2t(1-t)*g01+t^2*g11)
```

input/output軸ノルムの積にfloorを一度適用する。
一つのatomのtask widthに対して、per-stepのBessel・FFT・Chebyshev recurrenceは不要。
immutable6幅tablesを一度準備し、毎callはwidth interval/fraction、中心位相、routingを更新する。
tile momentは一度共有し、隣tileへのhalf-period rotationは`(-1)^k`になる。
出力側は同じrotationの随伴で集約結果を配る。

M16は実数33 features、M24は49、M32は65。2 tiles400 sitesに対する圧縮率は
約12.1/8.16/6.15倍。ただしatomごとの縮約は残る。
前のN1024/A52428/B32に対する算術量の参考値は、中心微分も含む現在の分解では
約`6BAF`、dense3 GEMMは`3BN²`、比は約3.30/4.90/6.50。
tile transforms・係数準備・optimizerを除く概算であり、GPU速度の予測や勝利証明ではない。
一般にA=rho*N²ならこの比は`2*rho*(2M+1)`。M16/32でMAC量だけがdenseを下回る
rhoは約1.52%/0.769%未満だが、これは性能dispatcherの境界ではない。
共有集計によってsite走査をatomごとに繰り返す負担は消えても、全atomの縮約と勾配は残る。
そのN1024はT20/spacing0.1の等幅格子にはならないので、GPU fixtureには流用しない。

## GPU検証の事前条件

初回のGPU比較はN1000、A50000=int(0.05*1000²)、B32、5 equal tiles、
global period100、T20/P40/spacing0.1、M16/32。
旧N1024/T12.8/P38.4とはshape・数学的profileが異なり、同条件の高速化比とは扱わない。

研究Algorithm `research_cuda_two_tile_template_triton/v1`をbenchmark Registryだけへ登録。
共有routeとnative H²-template direct route、denseを既存periodic_comparisonで比較する。
独立CPU FP64 oracleは全sitesのnative H²を補間し、実際のsiteノルムを微分する。
CUDA係数や共有moments、Parseval shortcutはoracleに使わない。
Y、dX、全atom dPのmax_abs/relative L2 gate4e-4、初期/24更新後、
retained invocation、live幅、Graph/public eagerとoptimizer moments/clockを確認する。
未計装complete step、capture/replay込みallocated/reservedを隔離workersで測る。
CPU passは1810 tests、2758 GPU/DB skips、18既存warnings。
M24はCPU形状比較のみ。GPUの数値・完全step qualificationはM16/32で行う。
GPU結果は同じ数学的profileの研究実装の検証で、公開CUDA dispatcher採用とは別判断。

```bash
python -m pytest -q tests/test_fourier_two_tile_template.py tests/test_two_tile_template_protocol.py
python -m pytest -q tests/test_two_tile_template_cuda.py
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/two-tile-template-1000-a50000-b32-m16.json \
  --plans two-tile-shared two-tile-direct dense --output output/two-tile/cohort.json
```

形の生データ・図・独立auditはprimary checkoutのignored
`output/fourier-two-tiles-20261014/`に保全する。GPU jobのsource/result hashesも同じ名前の
research-worktree outputに記録し、終了時にprimaryへ照合してコピーする。

## 初回GPUと追加候補

source450d5438の実NVIDIA L4検証は54GPU tests、全50k atomsの初期/24更新後gateがPASS。
M16 shared complete step8.756 ms、M32 10.592 ms。allocated46.23/58.51 MiB、
dense約0.083〜0.084 ms、allocated32.67 MiB。各2独立cohorts。
native directはFP64 libdevice trigで全site factorsを生成する実装で、最適compactbaselineではない。
別GraphのM16診断ではprepare5.529 ms、atom縮約約0.983 msが各方向に残った。
tile momentは各方向約0.0092 ms。診断値をprimary時間から差し引かない。

同じ数学的profileの追加候補は、FP64で中心・位相の範囲縮約を行ってからFP32 sin/cosを使い、
owner順に並べたatom blockで同じmomentをIEEE dotに再利用する。
異なるownerが混ざるblockではtileごとのmasked dotを使う。
width fractionはknotsの端でclampし、FP32上限丸めによる非凸weightを避ける。
この変更は独立FP64 gateを維持して再検証・完全step再測定する。


## 端点検査で停止した二つのjob

- `l4job-f6ba9e12972f42b4b1eb33519888c989`（sourcee8d694b8）は既存54 GPU testsがPASS、
  追加2 testsは宣言時にFAIL。9.6というimmutable knotに対し、FP32 scalar上限が9.600000381へ
  丸まり、厳密coverage判定に落ちた。sourcea6266c34ではmetadataの表現誤差だけにrel1e-6を
  許し、真の未coverage範囲は拒否するCPU testsも追加した。Y/dX/all-dPの4e-4 gateは変更しない。
- `l4job-8ddaf4261cab4bea98fd4bcddf10be42`（sourcea6266c34）も既存54 testsはPASS、
  追加2 testsはfixture assertionでFAIL。大振幅のPolar入力がwidth envelopeを抑えるため、
  テストが意図したsigma上限に届かなかった。source7f9e172eで振幅0/radial alpha1に変更。
  振幅微分は残り、clamped fractionの上限とnative all-VJPを検査する。

どちらもpytest段階でdriverを停止し、完全step性能値を作っていない。
失敗ログ・receipt・source/result archivesはignored evidenceへ保持する。


## 最終GPU qualificationと採否

`l4job-03dd37c788554691833db0c8318b5e07`、実NVIDIA L4、Torch2.11.0+cu130、
CUDA13.0、Triton3.6.0。frozen source commitは
`7f9e172eeddb26f7a4a126d5487512b6fe28a405`。
source archive SHA256は`b802fc8f0adf9de91d6252271345d0c4b74309632ed39e8d3fd0959ea94e4515`、
result archive SHA256は`90d973d4fdd9e4cc7541cb3053efa423820aaff342912955bcb767759018f82e`。

56 GPU tests、skip0、FAIL0。12 isolated workers、337 Python source hashesの照合、
同cohortのinput/target/dyとCST初期Parameter照合がPASS。
CST2方式×M16/32×2順序反転cohortsについて、初期/24更新後の16回のfull native oracleがPASS。
全検査でのmax_absはY3.3351e-5、dX3.4605e-5、all-dP1.5486e-5、
relative L2最大1.8421e-6で、固定4e-4 gate以内。

| M / route | 完全step中央値ms（独立cohort1 / 2） | capture/replay peak allocated MiB | reserved MiB |
| --- | ---: | ---: | ---: |
| M16 shared | 6.040730 / 6.035935 | 46.23 | 80 |
| M16 native direct | 857.860890 / 856.971778 | 830.77 | 906 |
| M16 dense | 0.084244 / 0.083930 | 32.67 | 86 |
| M32 shared | 8.564684 / 8.639916 | 58.51 | 94 |
| M32 native direct | 1631.651893 / 1631.644581 | 830.82 | 906 |
| M32 dense | 0.083759 / 0.083182 | 32.67 | 86 |

完全stepはlive幅/係数/L2準備、forward/MSE/backward、fused AdamW、Polar projection、
periodic centre更新を含む。各workerはwarmup2、initial step1、timed replay21で24更新。
21 replay samplesは21独立runsではない。primaryは未計装。memoryはwarmup/capture/replayを
それぞれ記録し、上表はcapture/replayのピーク。独立oracleのscratchとGPUプロセス全体のVRAMは
測定対象外で、allocated/reservedを混同しない。

低Mの非負profile、低漏れ、静的3-GramによるL2、複数atomでのtile moments共有は成立した。
ただしこのN1000/A50000/B32構成ではsharedでもdenseよりM16約72倍、M32約102〜104倍遅く、
allocatedも多い。M16 reservedだけはdenseより小さいが、時間/allocated改善にはならない。
形の新profileと研究Algorithmを保存し、dense置換や公開CUDA dispatcherへの性能採用は見送る。
Native directは全site factorsを保持しFP64のphase/accumulationを使う実装であり、
compact supportを最大限使ったbaselineの代表速度ではない。
`assessment.qualifies=true`はnative directとの比較だけを表し、dense勝利を意味しない。

この比較から全てのtile/Fourier方式が原理的に不可能とは言えない。
現在の分解では共有site走査を省けても、5万atom×33/65 featuresの縮約、係数更新、
全atom勾配と保存量が残る。5% atom条件でその負担がdenseの3 GEMMより軽いことは示せなかった。
別density・別分解の採用は、その条件での独立完全step測定を要する。


## 方針停止

2026-10-14、ユーザーがFourierを止めて`(1+|x-p|/s²)exp(-|x-p|/s²)`の検討へ移る方針を指定。
PR105はmergeせずclose。正しい新profile・Torch/Triton実装・全実験の証拠は研究branchと
ignored evidenceに保持し、dense dispatch採用も追加Fourier tuningも行わない。
停止指示時に実行中だった別Graphの最終M16診断job
`l4job-e65de87ebced4cef946ab9deca73adb9`は完了結果を回収・保存した。
新しいGPU測定は投入しない。この診断をprimary完全stepへ加算/減算しない。
