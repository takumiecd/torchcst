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
GPU検証19 passed (18 GPU + declaration)、80.31s。submitted/worker/commitの196 runtime files、source/results archivesを照合済み。18 GPU testsは2routesのslices/B1/32/64、N64/N128
20captured live updates/moments/steps、old backward、empty-neighbor FP64 checks。

N64/N128 B32 A204/A819 seed41、FP32 IEEE/TF32off、初期rho>1、production
fused AdamW/Polar。最初はrho3/8の2casesずつでcopy8・param2・atom16/warps4を
controlsに同時比較し、有望案だけrho1.25/3/8/mixedの独立2回へ進める。
全8caseをprepare済み、21 samples/execution。raw scripts/evidenceはignored
parameter-combo-20261007へ保存する。L4 supervisor97800のgather/repair batchに続ける。

source `8e17b2c2960cd45dd48f8838c335f367ec848036`、draft PR #43。check `l4job-23fac7b94c774159a7a4cc44ec5078a6`、N64 rho3/8 `l4job-16460a0c8a1e4fde9f31dd1d09487f36`、N128 rho3/8 `l4job-70d2d8416cba40c589cf82c7349b2124` completed。L4 supervisor97800は先のgather/repair jobsとこの3 jobsをsource分離して実行する。

## L4 rho3/8 screening (1 independent execution)

完全step median us。全20 full-shape FP64 Y/dX/all-atom-gradient comparisons passed、初期rho>1・位置勾配・live width更新を確認。隣接JSONにraw hash proof/21 samples/DB proofを保存。4 measurement artifactsのDB idempotent/export checks passed。

|N/rho|copy8|param2 atom32|atom16 single/4warps|atom16 split2/default|atom16 split2/4warps|dense|
|---|---:|---:|---:|---:|---:|---:|
|64/3|59.14|52.65|53.33|51.36|51.25|39.14|
|64/8|56.82|53.10|51.82|51.15|51.29|39.56|
|128/3|70.40|68.25|69.14|70.88|68.19|46.06|
|128/8|73.91|69.60|70.13|74.66|69.98|45.66|

N64 default/forced4は同じconfigであり別方式ではない。双方91 registers/0 spills/shared12288 (param2 atom32は128/0/20480)。N128 default8は255 registers/4 spills/shared24576で遅い。forced4は80/0/24576、param2 atom32の80/2/40960に近い時間。N128に改善が一般化したとの主張はしない。

ピークallocated N64 115712/N128 303104 bytes、reserved6291456、全routes同じ。compiler spillなしでもDRAMアクセスゼロやcache residencyは保証しない。L4 supervisor97800 terminal0、slot stopped、Session terminated/server No active sessionsを確認。次はforced4をcache修復・低メモリowner分割と組み合わせて独立2回へ進める。
