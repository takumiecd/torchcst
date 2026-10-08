# Torus profile product: H/GをCTA内で再利用する候補

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

## 現時点

research branch `kernel/torus-profile-product-onchip`。実GPUの正しさ/placement/性能は
まだ未確認。基準Torch比較はPR #75、merge `18077420569c09b207063c1e2cb34f28e4a3173a`。
復旧bundle `output/research-recovery/20261008-torus-h-reuse/reference-final.bundle` を保存・検証し、
raw evidenceと旧branchは残している。

CPU checkpoint: full suite1327 passed /2216 skipped /18 warnings、38.23s。
GPU skipsは実機検証を示さない。Ruff/diff check、contributor prepare/checkもPASS。
raw CPU logはresearch worktreeの`output/onchip-cpu.log`。
