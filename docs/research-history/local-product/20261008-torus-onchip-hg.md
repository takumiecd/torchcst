# Torus profile product: H/GをCTA内で再利用する候補

この候補は不採用。H/G用global bufferなし・16構成spill0は確認できたが、
全atomのN1024 sigma3ではdP、N2048 sigma3ではYの数値gateを超えた。
性能測定へ進まず、sigma8と逆順確認は未実施。本番へprototypeを統合しない。
三角関数・section中心の精度を次の対象とし、誤差許容値と予算を変更しない。

## 対象と計算

所有者の追加指定は「中間値を保存というよりcacheに置き、DRAMへ置かない」。
前段PR #75のH/saveはB×Aのglobal Tensorを作る対照であり、L2への常駐を保証しない。
この段階では、forwardのHとbackwardで再計算するH/Gを同じCTA内で消費する。
forward/backward間のregister/shared保持は行わない。X/Y、dX/dP、古いParameterと
live geometry/scalar/queryのsnapshotは必要な入出力・微分状態としてglobalにある。
H/G、A×site factor、W/dWのglobal scratchは渡さない。

主対象は単一Strip + intrinsic S1×S2 Torus、circle出力、Triweight/Triweight。
Euclideanを主対象にしない。普通のSphereのprofile productはまだ宣言していない。
CUDA FP32、2D batch1..64、入出力axis<=2048、現在の幅を毎step再decodeする。
本番Registry/dispatcherへの追加はしない。

1 atom = 1 CTA、site tile64、4warps。最初は完全axis走査し、近似しない。
raw circle/section profileをu/vとすると

\[
 n_u^2=\sum_i u_i^2,\quad n_v^2=\sum_j v_j^2,\quad
 D=\max(\sqrt{n_u^2}\sqrt{n_v^2},\epsilon),\quad
 H_b=\sum_j X_{bj}v_j,\quad G_b=\sum_i dY_{bi}u_i.
\]

forwardは同じCTAでHを保持し、出力tileへamp H u/Dをatomic加算する。
backwardはH/Gとprofile微分の縮約を再計算し、GをdXとatom VJPの両方へ使う。
単一chart全体のnorm、一回のfloor、norm微分、section中心による
circle半径R+r q0の微分、S² expmap Jacobian、Polar amplitudeのVJPを保持する。
widthのtask VJPは元の契約どおりdetached。旧forward用のsource/radii/scalars/queriesは
snapshotし、次のforwardやGraph replayではlive状態から再構築する。

compilerのregister/shared報告とspill slots、PTXのlocal load/storeを記録する。
H/G用Tensorを作らないことだけでオンチップ実装の成立や速さを主張しない。
unsupported layout/dtype/autocast/deterministic要求は明示的に拒否する。

## 事前固定する検証と予算

これはPR #75の失敗した確認runをやり直すものではなく、新CUDA Algorithmの別段階。
過去の消費予算、無効order、失われたN2048 primaryを変更しない。

最初のcorrectness campaignは合計900 driver秒、最大3つのsource版。
各job上限300秒、pytestは220秒、compiler auditは60秒、setupもjob時間に含める。
失敗したsource版は記録し、結果に合わせてgate/許容誤差を緩めない。
失敗がある間は性能測定・本番採用を行わない。

- independent embedded-fibre FP64 oracleでY/dX/全source gradientを検査。
  max absolute/relative L2の両方<=4e-4。
- partial site tiles、batch1/3/32/64、floor1e-6/.5/100、section origin、
  empty atoms、各gradient要求、旧forward後のp/geometry/pitch/amp変更。
- 同じAdamWと既存Torus updateで20 Graph replay。Y/dX/dP/Parameterは上記gate、
  moments/stepは既存2e-5 atol/rtol。geometry/pitchとwidthのlive変化を確認。
- 全site1024/2048、sigma3/8、23 atomsの非自明な全中心勾配gate。
  これは全atom性能fixtureのoracle gateを代替しない。
- N1024/N2048 B32/64のforwardとbackward x+p/x-only/p-onlyについて、
  n_spills==0かつPTXのld.local/st.localがないことを要求。報告とPTXを保存。
- 次の性能campaignはこのgateの後で別途事前固定する。未計装完全step、
  capture/replay allocated/reserved、全atomoracle、21 samples、独立逆順を使う。
  支持範囲を絞る次の候補は、完全走査の費用を測ってから判断する。

## 初回CPU checkpoint（GPU未検証時点）

research branch `kernel/torus-profile-product-onchip`。実GPUの正しさ/placement/性能は
まだ未確認。基準Torch比較はPR #75、merge `18077420569c09b207063c1e2cb34f28e4a3173a`。
復旧bundle `output/research-recovery/20261008-torus-h-reuse/reference-final.bundle` を保存・検証し、
raw evidenceと旧branchは残している。

CPU checkpoint: full suite1327 passed /2216 skipped /18 warnings、38.23s。
GPU skipsは実機検証を示さない。Ruff/diff check、contributor prepare/checkもPASS。
raw CPU logはresearch worktreeの`output/onchip-cpu.log`。

## 第1 GPU gate: 数学はPASS、placementはFAIL

source `899f479dce16ee5c6f7adcf0679a21d387dbb59f`、
job `l4job-3da2e2de56034021ac6f68fa90f68e3f`、L4/Torch2.11.0+cu130/Triton3.6。
26 tests passed/0skip、32.95s。独立physical oracle、20step Graph更新を通過。
しかしN1024 B32 forwardのcompiler auditでregister48/shared8bytes/spill8、
PTXにld.local/st.localがあり、採用gateはFAIL。性能測定なし。
source archive `0809c5f30198ba2a3f4e315d3fe46f4d9ed87f7848cf16abc333a2ffc903164d`、
result archive `3a0db3cee663b72fc4d8eda400973aa2ea9698f627f9714f816396a0a2ca7b6b`。
driver43.585s消費、900s campaign内。runtime停止確認済み。

PTXのlocal配列は`__internal_trig_reduction_slowpath`の大角度処理にある。
これをH/Gのregister不足だと断定しない。circle angleはFP64で±piへwrap済み、
valid intrinsic sectionのangleもsection chart契約でpi以下。
第2版はこのbounded argumentにtl.sin/tl.cosを使い、不要な大角度slow pathを避ける。
数学的な演算・oracle・誤差gate・launch shapeは変更しない。
同じ26 testsと16compiler variantsを再検証し、残存spillがあれば不採用とする。

## 前段の保全とmerge後の整理

`reference-final.bundle` SHA256
`dbc0ebca22fe7e7a328bb770620210dd7268929b101b67c8211c9b87f0182f03`。
standalone restore HEAD35acc32eを照合し、git fsck成功。
`reference-evidence.tar.gz` SHA256
`129cdc61d4bd77d8ac0a66469a44e8d22ab317b04b6b15ddf02a1600dc03db51`。
raw evidenceと前段10jobsの381filesを抽出し、全size/SHA256を照合した。
manifestは同じrecovery directoryの`reference-final-manifest.json`。
旧branch/worktree/raw copiesは維持し、merge済み#74/#75のremote branchのみ整理する。

## 第2 GPU gate: frontend変更だけではlocal memoryは消えなかった

source `1e3615f57273cd4644fe7bbde173c2a6d04f1e60`、
job `l4job-db3add834dca43c2bad8d18f12cf1291`。
26 GPU tests passed/0skip、32.56sだが、同じforwardにregister48/shared8bytes/spill8。
Triton3.6ではこのtl.sin/tl.cosもPTXのlibdevice slow pathへ下りていた。
metadataから8spillを推測するだけでなくPTXのlocal load/storeを確認し、gate FAIL。
source archive `26807f08f37d735f97f3820fa1a6c9d5d50731174be3f134eda2219e7f26c079`、
result archiveはverified receipt/raw evidenceへ保存。driver42.916s消費、全slot停止。

第3版はbounded sine/cosineを明示的なPTX sin.approx.f32/cos.approx.f32にする。
range reduction、profile、正規化、VJP、launch shape、既存error gateは変えない。
これはFP32三角関数の数値近似を変えるため、同じoracle/20step gateで再検証する。
[NVIDIA PTX sin/cos仕様](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin)
の±2piの絶対誤差保証範囲内に引数を置く。離散site/支持/演算を近似して間引かない。
このcampaignの最後のsource版。失敗時は予算・gateを変更せず記録する。

merge済み#74/#75のremote branch削除は保全後に完了。
元の3stashesは変更せず、reflog上の旧stashまでarchive refsと
`stash-history.bundle`へ保全し、復旧cloneで全3commit/tree存在とfsckを確認した。
追加bundle SHA256 `c69418faca4463dbd991fbae2dfe1ad55c01cc2f08ded7969c6ce91678f90763`。

## 第3 GPU gate: correctnessとplacementがPASS

source `b63046d789acd3571431e5170dd104eb6add0a2b`、
job `l4job-7069188df24343a7a22adf71505080c2`。
26tests/0skipとN1024/2048 B32/64の16compiler variantsを通過。
全variant spill0、PTXにld.local/st.localなし。B32 forward52register/shared8bytes、
backward x+p96register/shared16bytes。B64 forward60、backward x+p128register。
H/Gのglobal Tensorも渡していない。全step速度/peakの確認はまだ別。
source archive `e137a68b02b03080653ea1fb81617d2f695600131aebdda73e2b284f68281d2b`、
result archive `218fb5366d8f5fc352892af266d182262b5e9b89a894c031bf3de50c3aa64eb5`。
23manifest files+source/result archive hash照合済み。driver43.175s、
3sourceの実消費合計129.676s。全slot停止確認済み。
第2版result archive `0330547902e79b3fac0b1217265d4405fd1eff1ec5e4fe8a8271f557cb16c3b3`も
全manifest/source/result hash照合し、raw失敗結果を保全した。

## 完全step比較: 実行前に固定する条件

runtime sourceは第3版のb63046d、準備・図示・文書変更はruntimeを変更しない。
N1024/N2048、sigma3/8、B32、FP32（TF32なし）、約5%atoms、seed41、
同じ全chart L2/一回floor1e-6/live widths/Torus update/AdamW。
4routesの実行順はh-saved,h-recompute,w-gemm,onchip、denseは最後。
各workerのfresh process、全atomのphysical FP64 Y/dX/全dPのstrict二重gate4e-4、
exact update comparison2e-6、20step widthの変化、21 timing samplesを使う。
主結果は未計装完全step、capture/replay allocated/reserved。
phase診断は別Graph。既存fixture/適合条件/数値gateを変えない。

primary2jobsは各2100 driver秒、各sigma runner850秒、pytest220秒。
新Algorithmが追加された比較であり、前段の無効/消失した確認runの修復ではない。
速度/メモリ改善はこの新cohort内だけで比較し、旧stageとの独立確認を主張しない。
有望とする条件は、onchipが速いTorch H経路よりGraph中央値3%以上速い、
またはallocated5%以上減かつGraph回帰3%以内。正しさ/spill/幅変化が必要条件。
その場合のみ各Nにつき独立inverse1job（各2100秒、sigma850秒）を使う。
inverseはCase plansもonchip,w-gemm,h-recompute,h-savedへ逆順にし、
事前にcase/recipe物理条件が同じこと、結果の実測order/initial hashesが一致することを検査する。
failed runのretry、予算増加、gate緩和は行わない。改善条件のないNは追加確認しない。
small gain/negativeを全routes含め保存する。公開fast path採用は別判断。

比較用standalone cloneはprimary checkoutのignored recovery directory内
`onchip-comparison-source`、clean HEAD `9b67781`（b63046dからruntime変更なし）。
primary N1024 job `l4job-c4a08498f1704ea69fad94d790f9d76e`、
primary N2048 job `l4job-724473c62bd6477a837183377666400a`を提出。
同じdriver/catalog/固定code、順番はCase plansで検査済み、共有L4を1台ずつ使う。
現在のdraft PRは#76。結果待ちであり、速度改善・本番採用はまだ主張しない。

## 全atom gateで不採用: 時間/peak測定前に停止

N1024 sigma3、52428atomsのonchip dPでmax absolute
`0.00045135537893159494` /relative L2 `1.9286866400503026e-5`。
固定したmax<=4e-4に違反。Y/dXは先に通過した。
job `l4job-c4a08498f1704ea69fad94d790f9d76e`、
source archive `dc71e6f927b05f7e2222280c8d3de42b7787728d05f9c9f232f0fd2bc8f920b8`、
result archive `67a2ab67773581e9f10c046c20b0e7541157309c72484082fd4d3d9fb6ff820d`。

N2048 sigma3、209715atomsはonchip Yでelementwise gate違反、
41/65536要素、報告された最大絶対差`0.0007270520718771767`、
位置(batch3, output1649)。この値はassert_closeが報告した差で、
全tensorのrelative L2は検査完了前に停止しているため未測定。dX/dPも未評価。
job `l4job-724473c62bd6477a837183377666400a`、
source archive `911f9221b236d03b34ccf6832bd58750b0f7baa04295aa6df376e29cd9f66d1f`、
result archive `f9d80073529f96d443dd2f614c8b808872a91455b111733b88f0c3b97bd3c338`。

両jobは26 GPU tests/0skipを再通過し、全atomのTorch3対照もPASSしたが、
onchip correctnessでrunnerが停止した。各13manifest filesとsource/result hashを照合。
recordsは4 correctness workersのみ。未計装完全step/phase/dense/peakはこのcohortで
一つも測っていない。sigma8には進まず、改善条件を評価できないためinverseなし。
failed primaryのretryやbudget変更はしない。全owned L4停止確認、supervisor正常終了。

元PR #75のH lifetime/方式比較はresearch referenceとして統合済み。
本候補のPR #76はclosedとしてsource/branch/全失敗結果を保全し、
この採否noteだけを別PRでmainへ統合する。
H/G register/shared reuse自体の不成立を示す結果ではなく、現実装の数値適合性不足。
次はbounded trigの数値精度とsection expmapのFP32計算順序を、
失敗atom/出力の診断で切り分ける。寄与の小さい要素や失敗caseを捨てて採用しない。
