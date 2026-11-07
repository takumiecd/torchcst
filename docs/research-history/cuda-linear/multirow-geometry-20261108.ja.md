# トークン行をまたぐAtomごとのgeometry再利用

[English](multirow-geometry-20261108.md) | **日本語**

N16384の行数対照実験の最初の順序では、大きいRの時間はoptimizerより
forward/backwardが支配した。R256の別計測中央値はforward+loss/backward/
optimizerで41.59/59.91/1.42ms（足したり引いたりして主計測を構成しない）。
コード上、projectionとexpansionは同じ半径・wrapした中心・Fourier係数9個を
8行ごとにdecodeし、backwardでも再実行している。

候補`cuda_radius_large_multirow_geometry`は既存のFP32・25値/Atomのgeometryを
forwardごとに一度だけ計算し、projection・expansion・勾配の各groupでbackward
まで再利用する。そのautograd呼出しが入力・Parameterと通常のversion検査付きで
保持する。Algorithm上のcache、CPUへのAtom読出し、数式・Polar・LRの変更、
dense W/dWはない。追加forward保持は100A byte（N8192/16384で6.25/12.5MiB）。
従来もbackwardに同じ一時tableを作るため、保持が無料と仮定せずピークを測る。

既存の保存・再計算経路も選択可能に保つ。既定値・Tensor変換fallback・一次微分
制限・旧scopeは維持し、新templateも既定は旧経路。半径変化・周期画像・R1/9/256
端数・大きいowner key・非連続入力・片側だけの勾配・saved hook・retain backward・
古いParameterの拒否を確認する。性能の前に既存CPU FP64/独立求積/matched-state
基準で全N2048/A8NのY/dX/dPとoptimizerを検証する。

R256/N8192/16384/A8Nで幅比較の凍結初期値・データを使う。既存workerのwarmup10＋
計測51の同期した完全step、allocatorピークは全61更新。fresh processで両順序：
Aは8192→16384、保存→geometry→Dense、Bは16384→8192、Dense→geometry→保存。
演算・勾配・Adam・Parameter基準と初期・更新後全VJPを維持。失敗を残し、基準違反・
OOM・非有限・timeoutで停止。独立source/buildを使い、GPU実験は同時実行しない。
同条件の両順序の最終hashを確認し、geometryと保存のbyte一致は、事前計算による
コンパイラの丸め差もあり得るため仮定せず記録する。公開・既定採用は別の判断とする。

これは計測前の仮説とprotocolであり、高速化の完了結果ではない。
