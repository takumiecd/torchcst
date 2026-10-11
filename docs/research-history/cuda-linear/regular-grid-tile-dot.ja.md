# RegularGridの支持タイルと小行列積

## 数学と寿命

`W = U diag(A) V` の全operator-sites L2正規化と床、source VJPを維持する。
入力I、出力J、atom束S、小batchTで
`H[T,S] = sum_I X[T,I] V[S,I]^T`、
`Y[T,J] += H[T,S] (U[J,S] diag(A[S]))^T`。
backwardは `G[T,S] = sum_J dY[T,J] U[J,S]`、
`dX[T,I] += (G[T,S] diag(A[S])) V[S,I]`。
中心のセル所属と違い、各軸の支持区間が重なるタイルを一覧化する。
1つのatomが複数タイルへ所属しても、各siteは所有タイルから1度だけ出力する。
H/Gの全支持の集約後にdA/dcenterを作り、入力・出力タイル間の交差項を残す。
タイル単位で再正規化しない。

## 実行

`TileDotHAlgorithm` はD2 full-period Torusの既存数学契約に限定する研究候補。
32-siteタイル、32-atom束、8/16-row batch、8 owner partialを使用する。
支持長<=32かつ各軸が32の倍数なら各atom/軸の所属は最大2個。
packed支持区間から入力・出力それぞれ2A個のkeyをsortし、boundsを作る。
forwardのroutingをbackwardで再利用し、次forwardでは現在の値から再構築する。
空支持もH/G writerを持ち、全atomの結果を初期化する。
幅が広い／短い端タイルではGPU上で全支持の正確なfallbackを選ぶ。
容量超過による切り捨てはない。

H/G producerは同じ支持開始タイルのatomでX/dYの連続panelを共有し、
最大2枚のpanelに対して `tl.dot(input_precision='ieee')` を行う。
タイル一覧の第1entryが唯一のH/G writerとなる。
ownerは支持が重なるatomのH/Gを行列積で消費し、8 partialを別kernelで集約する。
Y/dX/dPへのscatter atomicは使用しない。

Hはforwardから保存せず、backwardで再計算する。
H/dH/G/dGは小batch分だけの `[A,BM]` slabを繰り返し利用する。
したがって寿命はbatch chunk単位であり、全atomを跨ぐslabは存在する。
完全にregister内でatom束ごとに使い切る実装ではない。
入力×出力タイル積のscratchや全量Wは作らない。

## 検証・採否

CPU宣言44 plans PASS。GPU数値検証・完全step計測は未完了。
比較は現行BM16/G16/H32、tile-dot BM8/BM16、matrix/factor/onchip/dense controls。
RTX 6000 Ada、N2048/B32/A209715、rho3/8、seed41、IEEE FP32/TF32 off、
24 actual updates、21 samples、2独立cohort/rho（2回目reverse）。
初期・post24の全site/all-atom FP64 Y/dX/dP gateを変更しない。
