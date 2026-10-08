# Torus on-chip numerical diagnosis

前段b63046dの全atom数値gate失敗を保存したまま、新しい原因診断を行う。
単一Strip + intrinsic S1×S2 Torus、Triweight/Triweight、全chart L2/単一floor、
widthのtask VJP detachとstepごとの再decode、現在のPolar/Torus updateを維持する。
H/Gを同じCTA内で再利用する設計を保つ。結果に合わせて4e-4 gateを緩めない。

## 実行前に固定する診断

研究branch `kernel/torus-onchip-numerics`。基点main 3e99d4f0。
不採用prototypeを研究branchへ復旧し、Algorithm revision v2に三角関数のみの
比較recipe（hardware / fp64）を追加する。FP64版も三角関数の入力と出力はFP32。
profile・norm・H/G縮約・手書きVJP・launchは同一。FP64三角関数の内部spillは
この原因診断でのみ許す。オンチップ採用には別途spill0 gateが必須。

L4一台、1診断job上限900 driver秒。N1024/2048、sigma3、B32、seed41、
約5%全atoms（52428/209715）、TF32なし。各Nのphysical FP64 oracleは一度計算。
hardware→fp64、各2回でY/dX/dPをすべて保存し、atomic出力の変動も区別する。
各tensorにmax absolute、relative L2、elementwise check（全てtol4e-4）、
最悪位置・actual/truthを報告。正しさの不合格は診断データとして保存し、計測しない。
Torch H対照も同じoracleで確認する。baseline dPの最悪16atomsを選び、中心とraw
profileを保存する。diagnostic出力tensorはruntimeのH/G storageではない。

その後は診断根拠に基づく修正を最大2source版、各600 driver秒以内で検証する。
最大総driver予算2100秒。既存26tests/20 Graph stepsと16compiler variants、
全atomN1024/2048 sigma3/8のY/dX/全dP strict gateを要求する。
いずれかが失敗した版を保存し、自動retry/予算延長/許容値変更はしない。
今回は精度と配置の検証までとし、完全step性能探索は別の事前固定段階にする。
未検証・不合格runtimeはmainへ統合せず、採否と証拠をPRから残す。

## 三角関数だけの比較で改善（性能未測定）

source 7dfefe44e4bbad391ca8453b776cd05cc7d355f1、
job l4job-ca7d9aaee6604f51b893e9e9a9b5b308、driver28.326秒。
L4/Torch2.11.0+cu130/Triton3.6。source archive
950c926c1567d97db07fcfcaeae0612ff2e0c9b59e639467343201264f489ef6、
result archive17346d0962cb80e8318fb7ace21e6e3f82be1b2c0f71f540b4b41d6a34f83e2f。

N1024 hardware dP最大差0.000451355（atom23779, arc成分2）を再現。
三角関数だけFP64評価/FP32出力にすると最大差0.0000115437へ低下。
N2048 hardwareではY/dX/dP最大差0.000849486/0.000730504/0.000726073、
FP64三角関数では各2回の最悪0.0000289756/0.0000690053/0.0000297362。
同一全atom physical oracleの3tensorすべてstrict4e-4 gateを通過した。
Torch H対照も通過。atomic出力の反復差は小さく、dPは両反復で同一。
このcohortでは近似三角関数がgate失敗へ大きく寄与した。残る誤差の内訳は未確定。

## 第1修正版: bounded polynomial（実GPU検証前）

三角関数以外を固定し、valid区間[-pi,pi]のsin23次/cos24次Taylor多項式を
FP64 Hornerで評価してFP32へ一度丸める。候補recipeはbounded-poly、revision v3。
高精度libdeviceの大角度slowpathを呼ばず、H/GのCTA配置を維持する狙い。
CPUの200001点ではFP64 sin/cos最大差1.71e-13/2.08e-14（GPU保証ではない）。
完全step時間・高速化はまだ測っていない。前記600秒の第1修正版gateへ提出する。

## 第1修正版: 精度・配置gate PASS（速度未測定）

source 0f71858、job l4job-8a52ac83d0f44704baddd1a382401462。
26GPU tests/0skip/31.78秒、20step Graph replayとmoments/Parameter比較PASS。
N1024/2048 B32/64の16compiler variantsは全てspill0、PTX local load/storeなし。
B32 forward48register/shared8B、backward x+p122register/shared16B。
B64 forward59register、backward x+p128register。H/G global Tensorなし。

全atom physical FP64 oracle（B32/seed41/TF32なし）、strict二重gate4e-4とelementwise検査：

| N | sigma | Y max | dX max | dP max | 更新 max |
| --- | --- | --- | --- | --- | --- |
| 1024 | 3 | 1.1770205e-05 | 4.0190149e-05 | 1.1543707e-05 | 0.0 |
| 1024 | 8 | 1.8090612e-05 | 8.4947537e-05 | 2.5279361e-06 | 0.0 |
| 2048 | 3 | 2.8975575e-05 | 7.2819956e-05 | 2.9736204e-05 | 0.0 |
| 2048 | 8 | 4.2766455e-05 | 0.00021012335 | 5.3932007e-06 | 0.0 |

全relative L2も4e-4以下、Torch H対照もPASS。今回の最悪絶対差は
N2048 sigma8 dXの0.000210123（gate0.0004）。floor・支持・可変width・chart定義を変更していない。
最大2修正版のうち第1版で通過し、第2版は未使用。driver123.122秒、診断と合計151.448秒。

source archive2ace709c54437f5b98e9ac235d25ec946d6a7c3e0d155fa8146e2ed0d59a54f5、
result archived36dde5beb5e25ce2736bece31b808cb60fa7cf2903c0b9993d6f732f434e5cc。
全manifest/source/result SHA256を照合してraw logs/tensors/PTXを保存した。

CPU checkpoint1327pass/2216skip、targeted metadata2pass/24GPUskip、Ruff、
plan/case宣言、wheel/sdist buildはPASS。GPU skipsを実機検証と数えない。

保全先: research worktreeの `output/torus-onchip-numerics/`。
`trig-diagnosis/` と `bounded-poly-gate/` が各pool job全体のコピー。
各driver、CPU/build logs、source bundleも同じdirectoryにある。
本段階では精度・on-chip配置の成立まで。完全step時間、allocated/reserved peak、
独立逆順は未測定であり、速度改善/本番採用を主張しない。draft PR #78に研究sourceを
保持し、mainのruntimeは変更しない。次は同じvalidated版でtime/peakを比較する。
