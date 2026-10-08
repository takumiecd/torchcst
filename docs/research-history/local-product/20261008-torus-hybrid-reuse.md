# Torus profile product: selective saved forward contractions

ユーザーは全中間値をon-chipに置く要件を撤回し、backwardで再利用する価値がある
値をDRAMに保存してよいと指定した（2026-10-08）。精度を通過したPR #78の
bounded-poly版を基準に、保存の費用と再計算の費用を同じcohortで比較する。
単一Strip + intrinsic S1×S2 Torus、全chart L2、一回floor、coupled centre-radius
微分、幅task VJP detachとlive幅更新、現在のAdamW/Torus更新は変更しない。

## 保存内容

Algorithm revision v4、same launch: 1atom/CTA、site64、4warps。
`save=recompute` は元の数学とFP32縮約順を保持し、Hをbackwardで再計算する。
`save=h` はraw H[B]とq[3]/sinc/nu²/nv²（6scalars/atom）を保存する。
入力側微分用の縮約とnorm微分はbackwardで再計算する。
`save=vjp` はHに加えてH0/H1/H2（入力profileのq各成分に対する微分との縮約）、
5つのnorm微分縮約nua/nu0/nv0/nv1/nv2dをforwardで計算・保存する。
forwardで読み込んだXとprofileを再利用し、backwardの入力微分準備走査を省く。
G/ga/g0はbackwardで計算してdXと全atom勾配に同じCTA内で再利用する。
全A×axis profile/W/dWは保存しない。forwardごとのsource snapshot/cacheは
autograd ctxで所有し、旧forwardやretain backwardで別forwardの値を使わない。

intentional FP32保存容量はh: 4A(B+6)、vjp: 4A(4B+11) bytes。
B32、A52428/209715ではh7.969/31.877MB、vjp29.150/116.602MB（十進）。
これはbufferの容量であり、完全stepの実測peakではない。
compiler register/shared/spill/PTX local load/storeを各方式で保存する。
意図したglobal保存とcompiler spillを区別する。spillがあればその方式のGを
on-chipだと主張せず、数値/完全step/実測memoryで採否を判断する。

## 実行前に固定するcohortと予算

L4一台、B32、N1024/2048、sigma3/8、全atoms約5%、seed41、FP32/TF32なし。
初期p/X/dY、chart、live widths、AdamW/Torus update、全chart norm/floorは同じ。
許容値はmax absolute/relative L2とも4e-4かつelementwise4e-4のまま。
更新比較は既存2e-6、20step momentsは既存2e-5。

correctness: 最大2source版、各900driver秒。3方式の既存78tests（GPU skips不可）、
20Graph steps、partial sites/floor/各gradient要求/empty/旧forward snapshot、
全atomN1024/2048 sigma3/8の3方式Y/dX/全dP/更新を検証する。
48compiler variantsは記録し、値が不明な場合はon-chipを主張しない。
失敗版を保存し、gateを緩めず、同じ失敗jobをretryしない。

性能: correctnessを通過した同じsourceでNごとprimary1job（各1500driver秒）。
各sigmaは既存runnerで最大650秒、isolated process順は
onchip(recompute),cuda-h-saved,cuda-vjp-saved,w-gemm、dense最後。
21samples、未計装完全stepを主結果、capture/replay込みallocated/reserved peak。
phase診断は別Graph。runnerの全atomoracleも改めて要求する。
全route/遅い結果/負の結果を残す。source/result archive hashと実測orderを保存。

保存候補が同じNのsigma3/8両方でrecomputeより3%以上速い、またはallocated5%以上
減かつ時間回帰3%以内なら、そのNだけinverse1job（1500driver秒）を実行する。
逆順はw-gemm,cuda-vjp-saved,cuda-h-saved,onchip（dense最後）、Caseのplansも逆順。
結果に合わせた再計測・追加予算・gate変更はしない。同じ初期fixture/hashと実際の
worker順を確認する。独立確認を通過しなければverified winnerを主張しない。
本stage最大総driver予算7800秒。元の完了済み精度stageの予算は変更しない。
公開dispatcherの既定は変えない。PR #78をこの最終scopeの記述へ更新し、採否を残す。
