# 局所座標のB-spline moment：CPUでの数値screen

Denseの2倍以内・allocated/reservedともDense以下という目標に対し、v18/v19はメモリ条件だけを満たす設定があるが速度は悪化した。この文書は、同じB-splineを数値的に扱いやすい局所座標へ書き直す次候補の数式を保存する。GPU実装・GPU速度・GPUメモリ・採用の証拠ではない。

## 同じprofileをdyadic blockで表す

R=√6σ、spacing h、N=1024。S=2^ceil(log₂(R/h))をNでcapする。今回のNは2冪。σやshapeを丸めず、Sだけを離散選択する。tile tの中で

\[
 \xi_i={i-tS-S/2\over S/2},\quad \psi_i=(1,\xi_i,\xi_i^2),\quad
 B^{S,t}_p(j)=\sum_{i=tS}^{tS+j-1}\psi_{ip}X_i.
\]

profileの各quadratic pieceをq(i)=c₀+c₁ξᵢ+c₂ξᵢ²として表す。cは実際のμ/Rとtileの中心から求める。tile幅がRと同程度なので、全domainの座標で展開した大きい係数同士の打ち消しを避ける。

4つのpiece境界はκₖ=clip(ceil((μ+ηₖR−o)/h),0,N)、η=(-1,-1/3,1/3,1)。各κは

\[
 t(\kappa)=\operatorname{clip}\left(\left\lfloor{\kappa-1\over S}\right\rfloor,0,N/S-1\right),
 \quad j=\kappa-tS
\]

へ置く。ちょうどblock境界のκは前blockのj=Sへ置き、κ=0だけ最初のj=0とする。κ=Nも最後のblockに属する。この選択でdomain端とpiece境界を同じ離散和で扱う。

全支持長は2Rなので、blockをまたぐ非ゼロ部分に必要な補正は最大2つ。最初のκのblockに続く境界b=(t₀+q+1)S（q=0,1、b<Nかつb<κ₃）を列挙する。境界直後のpieceは整数κとの比較で選ぶ。古いblockの座標での係数cを、新blockへ移すと

\[
 c_{new}=(c_0+2c_1+4c_2,\ c_1+4c_2,\ c_2).
\]

HからYへは、4つのκで係数jump Δcをlocal deltaへ加え、block境界では新blockのj=0にc_newを加える。scanはblock内だけで行う。支持siteの直接評価や追加窓は不要。

この操作の離散adjointがH/G queryである：

\[
 H=-\sum_{k=0}^3\Delta c_k\cdot B^{S,t(\kappa_k)}(j_k)
   +\sum_{\text{active boundary }b}c_{after,old}(b)\cdot B^{S,t_{old}}(S).
\]

後半はblockを変えてprefixを0へ戻す補正。4境界だけのglobal式をそのままlocal prefixへ適用しない。これは区間内のquadraticを厳密に集計する同じprofileの恒等式であり、窓や別のkernel形状を導入しない。

各piece・jumpの中心微分はRをtask VJPで固定し、half=hS/2として、d=(−c₁/half, −2c₂/half, 0)。同じqueryにdを使ってHμ/Gμを得る。L2の中心補正とPolar source pullbackは従来通り含める。

## L2を正の項の和へ書き直す

piece区間内のn個のsiteを、その区間の中心に対して正規化した座標zでq=a₀+a₁z+a₂z²と書く。従来と同じ離散moment M₂=Σz²、M₄=Σz⁴から

\[
 m=a_0+a_2M_2/n,\quad V_4=M_4-M_2^2/n,\quad
 Q=n m^2+a_1^2M_2+a_2^2V_4.
\]

これは同じ有限site二乗和で、積分近似ではない。半幅H=h n/2とすると

\[
 \sum q\partial_\mu q=-{a_1\over H}\left(nm+2a_2M_2\right).
\]

n=0は寄与0。正規化は全piece・全siteを合計し、D=max(√(QᵤQᵥ), floor)を一度だけ適用する。FP32 screenでは局所座標の区間判定をFP64で行い、tiny clipped supportとfloor判定の近傍にはFP64 normを残した。これはfloor条件や精度gateを緩める操作ではない。

## CPU screenの範囲と結果

N1024/A52428、spacing .1、batch32/64、floor 1e-6/1000、5状態の計20組。最初の3状態はPolar sourceと中心とX/dYを変更し、最後の2状態はdyadic幅の境界の両側へσを振った。追加のdomain外/端付近の中心も含めた。

候補はFP32 local moments、FP32の4 knot+2 boundary query、FP32 broad delta加算/scan。native側の最終投影はこのscreenではFP64直接評価であり、GPU native実装の代わりにはならない。task source勾配の縮約はFP64。独立した直接FP64のprofile/全site normからY/dX/全4 atom source勾配を比較した。入力seedは41..45。

| normの算術 | 最大Y誤差 | 最大dX誤差 | 最大全atom勾配誤差 | 最大relative L2 |
|---|---:|---:|---:|---:|
| 従来の直接FP64 norm | 5.900e-5 | 9.225e-5 | 7.672e-5 | 7.010e-6 |
| 正の離散moment式・FP32 coreとFP64 guard | 6.464e-5 | 8.962e-5 | 7.650e-5 | 7.010e-6 |

全20組が従来のmaxabs / relative L2 <4e-4を満たした。丸め誤差はFP64 global式と同一ではなく、GPUでatomic順序・全backward・retained state・Graph replayを検証するまでは精度保証や採用判断をしない。CPU時間はGPU速度の根拠にしない。

二軸のCPU bank payloadは14,940,672 bytes（B64、9尺度）。これは最終GPUのallocated/reserved peakではない。固定のdyadic座標weights/moment tablesはstep間共有できるが、X/dYのbankは毎step構築、μ/σの係数は毎forward更新する。入力や幅が変わっても古い数値bankを使う方式ではない。

screen source・全JSON・logs・hashはignored `output/bspline-native-h-20261020/` の `block-event-dynamic-cpu.py` / `block-event-norm32-cpu.py` と対応する `*-dynamic.json` に保持する。実行例：`PYTHONPATH=src:. python output/bspline-native-h-20261020/block-event-norm32-cpu.py 3`。次はこの数式をruntimeの明示的な研究Algorithmとして実装し、同じGPU gateを最初に確認する。
