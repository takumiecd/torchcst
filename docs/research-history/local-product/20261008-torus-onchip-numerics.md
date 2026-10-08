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
