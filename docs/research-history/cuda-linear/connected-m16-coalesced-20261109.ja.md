# Connected M16の行連続配置実験（2026-11-09）

このベンチマーク限定アルゴリズムはrevision 9のconnected physical kernel、
FP32の正係数生成、完全な交差項、一次VJPを維持する。本番dispatcherには登録しない。

入力と出力の余接ベクトルを`[site, row]`に転置し、共有モーメントを
`[node, feature, row]`で保存する。1 Atomを256スレッドのCTAに割り当て、
隣接laneが隣接行を処理する。Atom応答は`[atom, row]`となる。
パラメータVJPはwarp shuffleで行を集約し、dX用に計算済みの応答を再利用する。
owner集約と階層構造は直前の方式を維持する。転置とモーメント再計算を
計測するeager stepに含め、forwardの応答とモーメントはbackward用に保存しない。

既存Linearベンチマークrunner、別Planカタログ、B=256、A=8N、
N=2048/8192/16384のCaseで検証する。対照は直前のowner方式とDenseで、
既存の識別子の意味を変更しない。forward、loss、dXを含むbackward、
fused AdamW、物理領域への復元を含む完全stepとallocated/reservedメモリピークを測る。
初回コンパイルは除外する。

速度を評価する前に、独立FP64 oracleによるY/dX/全物理勾配、fixtureの
optimizer状態ゲート、複数forward・retain・version guardテストを通し、
その後に直前の検証済み方式と実規模A/Bで比較する。
[結果は別記録](connected-m16-coalesced-results-20261109.ja.md)に保存した。
学習品質、CUDA Graph、本番既定への採用は主張しない。
