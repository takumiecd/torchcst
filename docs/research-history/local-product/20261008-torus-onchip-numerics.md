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
