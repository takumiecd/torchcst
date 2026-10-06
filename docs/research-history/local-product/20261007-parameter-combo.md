# 2026-10-07: combine parameter atom16 and batch split2

Branch `kernel/parameter-atom-batch-split`, based on PR #38 `45e8cdd8`.

L4/G4ではatom16とbatch split2がそれぞれcopy8より速かった。
この2 routesはparameter VJPを16 atoms/CTAにし、batchを16 rowsずつ2 CTAへ分担する。
N64/A204のparameter gridは13*2=26、N128/A819は52*2=104 CTA。
consumer output16/input16次元owner、atom chunk32、owner split4とcopy8 preparationは維持。
標準warps (N64 4 /N128 8)とforced4を比較する。

runtime kernel/reductionは既存のものを使う。canonical DQ partialsは2*C*4 FP32で
元param2と同じ。振幅/位置/幅のPolar pullbackは固定sourceでtask cotangentsに線形なので
部分和のpullbackを足す意味は同じ。FP32の丸め差・20captured optimizer updates/
moments/stepsを独立FP64/referenceで確認する。singletonにもgeneral Hを保存しない。
位置/dX/全atom勾配・正規化・support・independent snapshotsは変更しない。
新しいglobal H/stateは追加せず、partial/reduction込みの完全step時間/ピークを測る。

Host:863 passed /656 skipped (22.60s)、declaration1 passed/18 GPU skipped、
changed-file ruff、8 prepare/check、isolated wheel/sdist build成功。
GPU正しさ・速度は未検証。18 GPU testsは2routesのslices/B1/32/64、N64/N128
20captured live updates/moments/steps、old backward、empty-neighbor FP64 checks。

N64/N128 B32 A204/A819 seed41、FP32 IEEE/TF32off、初期rho>1、production
fused AdamW/Polar。最初はrho3/8の2casesずつでcopy8・param2・atom16/warps4を
controlsに同時比較し、有望案だけrho1.25/3/8/mixedの独立2回へ進める。
全8caseをprepare済み、21 samples/execution。raw scripts/evidenceはignored
parameter-combo-20261007へ保存する。L4 supervisor97800のgather/repair batchに続ける。

source `8e17b2c2960cd45dd48f8838c335f367ec848036`、draft PR #43。check `l4job-23fac7b94c774159a7a4cc44ec5078a6`、N64 rho3/8 `l4job-16460a0c8a1e4fde9f31dd1d09487f36`、N128 rho3/8 `l4job-70d2d8416cba40c589cf82c7349b2124` queued。L4 supervisor97800は先のgather/repair jobsとこの3 jobsをsource分離して実行する。
