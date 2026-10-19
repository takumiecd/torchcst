# H方式の係数前計算・入力問い合わせ・出力集約

同じ二次B-spline、全観測siteの離散product L2、Polar更新、task幅stop-gradientを保ち、
H方式の入力累積bank・係数・H生成・Y集約とbackwardを分けて比較する。
Tensor Core向けの密なprofile因子展開は導入しない。研究用v12で、本番登録は変更しない。

## 数学と共有単位

正規化index `z_d(n)=2n/(N_d-1)-1`。軸profileを4切替点のjoined係数で表す：

\[
f_{d,a}(n)=\sum_{r=0}^{3}\mathbf1[n\ge t_{d,a,r}]\sum_{k=0}^{2}J_{d,a,r,k}z_d(n)^k.
\]

入力bank `P_k(t)=sum_{i<t} X_i z_i^k` を使うと、

\[
H_{ba}=-\sum_{r,k}J_{i,a,r,k}P_{b,k}(t_{i,a,r}).
\]

Yへの寄与は `c_a=amp_a/max(sqrt(Q_o Q_i),floor)` として、

\[
\Delta_{k,b,t_{o,a,r}}\mathrel{+}=c_a H_{ba}J_{o,a,r,k},\qquad
Y_{bj}\mathrel{+}=\sum_k z_o(j)^k\sum_{t\le j}\Delta_{k,b,t}.
\]

入力側も出力側も現在のsigma/中心を使う。前処理は各forwardで更新し、forward時点の
Meta/cuts/Jをctxへ保存する。同じsnapshotをbackwardとretained backwardで使い、
後続forwardやoptimizer更新で上書きしない。norm/floorは従来prepareで一度だけ適用。
H/Gとその中心微分は従来のFP32保存相当への丸めを維持し、係数と収縮はFP64。

## 候補

- uncached split: 従来と同じ幾何計算をH問い合わせとY注入に残して、HをFP32で保存し別々のkernelで測る。
- cache fused: 各atom・各軸の4cutと3係数/2中心微分を一度準備し、register内Hから直接注入する。
- cache split: 前計算後のH queryとY injectionを実際に分離し、保存/読み戻し込みの完全stepを測る。
- compact cache: 3係数だけ保存し、固定sigmaに対する平行移動の恒等式
  `J'_0=-J_1/half, J'_1=-2J_2/half`で中心微分を得る。同じ数式だが丸め順の差は独立gateで判定。
- edge reduction: 終端Nへの未使用差分書込を省き、atom群内でcut0の寄与を足してからatomicする。
- segmented injection: 4cut eventを出力site別にinteger histogram/scan/placeで並べ、
  固定長のevent block内で同じsiteの寄与をsegmented sumしてからatomicする。
  入力側の同じ配置はbackwardのdX注入に使う。site×atomの全接続表は作らない。

segmented版のH/Gはforward/backwardの局所workspaceで、backwardへH自体を保存しない。
同じ形のworkspaceをallocatorが再利用でき、retained backwardで保存Hを破壊しない。
出力event orderはforwardで使い終われば解放し、入力event orderだけbackwardまで保存する。
4cut/axisは現在の値で毎forward構築するので、integer histogram/scan/place費用も主測定に含む。
clipped supportが2点以下の軸はnativeへ回し、norm-floorで増幅する係数相殺を避ける。

## 計測規約

既存Linear runner、L4、1024²/A52428、B64 mixedから開始。
基準は既存H `stream-reuse8`、最速W `range-reuse-a16w1`、Dense。
atom別sigmaがoptimizer stepで変わる同一fixture・初期P/X/target・AdamW/fused Polarを使う。
独立FP64の全shape・全atom・3 live状態でY/dX/全P勾配のmaxabsとrelative L2を両方4e-4未満に保つ。
可変sigmaのcutoff跨ぎ、切り出し/floor、保存snapshot、Graph/optimizerを別途検証。
主指標は未計装の構築・L2・optimizer込み完全stepとcapture/replayのpeak allocated/reserved。
別計装Graphのcoef_prepare/H_query/Y_inject、bank/scan/restore、back_query/dX_injectを記録する。
fused版のh_query/y_injectイベントは空で、stream_forward/backward spanに両処理が入る。
この空イベントをHが無料になった証拠として扱わない。部品時間は主時間から足し引きしない。

raw source/results/logsとhash検証はignored `output/bspline-h-precompute-20261019/`に保全。
性能と採否は実機結果の回収・検証後に記録する。

初回 `l4job-15b6cb5104204de3927a664e0f5e1536`（7325d2f2）は、GPUの係数準備kernelで
unrolled loop内のconstexpr再定義がコンパイルエラーになった。数値・性能結果は採用しない。
修正ad0954d0で再提出。初回のsource/result/log/manifestは保全している。
