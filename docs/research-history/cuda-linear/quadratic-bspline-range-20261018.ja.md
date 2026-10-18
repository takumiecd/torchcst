# B-splineのW区間集約と累積和

研究候補。profile・Polar更新則・task幅stop-gradient・全観測siteの離散L2は変更しない。
本番dispatcherへの採用とは分ける。Hをbatchごとに生成する現行版に対し、batch非依存の
W組立と、その線形随伴による全atom VJPを比較する。

## 数式

軸dのgrid座標を `z_d(n)=2n/(N_d-1)-1` とする。二次B-splineの三つの区間の
多項式係数を `c_{d,s,p}` (`s,p=0,1,2`) とし、観測domainへclampした四つの
切替siteを `t_{d,r}` (`r=0,1,2,3`) とする。

\[
 f_{d,a}(n)=\sum_{r=0}^3\mathbf1[n\ge t_{d,a,r}]
              \sum_{p=0}^2 J_{d,a,r,p}z_d(n)^p,
\]
\[
 (J_0,J_1,J_2,J_3)=(c_0,c_1-c_0,c_2-c_1,-c_2).
\]

このindicatorは計算方式の区間表現である。profile自体は従来と同じC1関数であり、
追加窓・幅の量子化・支持境界の近似は使わない。domain外の寄与は数えない。
空支持の軸は全係数の相殺を計算せず明示的にゼロとする。

`w_a=amp_a/D_a`、`D_a=max(sqrt(Q_o Q_i),floor)`として、9枚の係数面へ
各atomの4×4の切替siteで加算する。

\[
 \Delta_{pq}[t_{o,a,r},t_{i,a,s}]
 \mathrel{+}=w_a J_{o,a,r,p}J_{i,a,s,q},\qquad
 C_{pq}=P_oP_i\Delta_{pq},
\]
\[
 W_{ji}=\sum_{p,q=0}^2 z_o(j)^p z_i(i)^q C_{pq}[j,i].
\]

`P`はinclusive prefix sum。添字がNの切替は保存不要。係数面はFP64で、Wだけ
最後にFP32へ丸める。小さいsigmaの寄与は従来の支持別grouped W構築を併用する。
係数面は1/3/9枚ずつ処理して、一時メモリとlaunch/reductionの交換を調べる。

`E=dY`、`G=dW=E^T X` と置く。逆向きのinclusive累積和 `S=P^T` を使い、

\[
 B_{pq}=S_iS_o\bigl(G_{ji}z_o(j)^p z_i(i)^q\bigr),
\]
\[
 T_a=\sum_{r,s,p,q}J_{o,a,r,p}J_{i,a,s,q}
       B_{pq}[t_{o,a,r},t_{i,a,s}].
\]

中心勾配の生の収縮 `T'_o,T'_i` は片側のJをその中心微分へ置換して求める。
既存のprepareから得るamp微分・`D'/D`を使うpullbackは変更しない。
切替siteが動くときもC1接続によりprofileの勾配は連続である。各stepの現在の
係数・normを再計算し、forwardのMetaとplacement snapshotをbackwardに保存する。
retained backwardは後続forwardの配置・Parameter変更から独立である。

計算はW構築 `144 A_broad` 個までのFP64 atomic加算、係数面のscan `O(9 No Ni)`、
atom VJP `O(144 A_broad)`、三つのGEMMで構成する。batch依存のatom処理を除くが、
全行列のscanとその随伴の費用・一時面の容量が加わる。

## CPUで棄却した表現

1枚の差分行列に各軸の3階差分を注入し、各軸3回の累積和でWを復元する案も確認した。
数学的には同値だが、長い積分区間で丸め誤差が増幅する。NumPy FP64、seed9、
N1024/A52428/B32、sigma .1/.4/1.6/6.4/12 の直接計算との比較では、Y最大絶対誤差
.0161、生の中心収縮誤差 .542。固定の精度基準を満たす候補としては扱わない。

9係数面も小さいsigmaをすべて通すと桁落ちする。同じCPU条件でY最大絶対誤差.0133。
大きいsigma (1.6/6.4/12)のみではY最大絶対誤差3.35e-8、中心収縮1.49e-7だった。
このCPU probeはGPU精度・性能の証拠ではなく、新候補の振り分け設計に使う。
生の収縮はL2補正前であり、全Parameter勾配そのものとは区別する。

raw CPU probeとGPU driver/source/resultは ignored `output/bspline-range-20261018/`、
共有poolのjob directoryにも保全する。

## 実機検証

未完了。独立FP64 Y/dX/全atom勾配、可変sigma・Graph/optimizer・retained backwardを
確認してから、同条件の未計装完全stepとcapture込みallocated/reserved peakを比較する。

## 係数合成とscanの融合候補

v7は両軸のprefixを9係数面すべてに適用する。v8ではtensor積の線形性から、
各出力次数pについて、先に入力側の3面をprefixして入力多項式へ合成する。

\[
 T_p[j,i]=\sum_q z_i(i)^q(P_i\Delta_{pq})[j,i],\qquad
 W_{ji}=\sum_p z_o(j)^p(P_oT_p)[j,i].
\]

入力側のscanは計9面、出力側は計3面となり、中間合成・読み書きも融合する。
随伴では出力側を先に計3面処理し、入力側を計9面へ展開してreverse scanする。
同じ9係数面の演算順序を変える候補であり、kernelの近似・切り捨てではない。
配列の一時容量は3係数面と1中間面。FP64の合成用WとFP32のWも別に存在する。
Triton scanの性能、ピークメモリと丸め誤差はGPUで別途比較する。
