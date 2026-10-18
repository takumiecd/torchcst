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

## v7単純版の実測

NVIDIA L4、1024×1024、A52428、B32、spacing .1、seed41、warmup5/21 samples。
AdamWとfused Polar更新、live sigma・構築・厳密L2を含む未計装Graph完全step。
各候補の前に全A・3状態の独立FP64 oracleを通している。

| sigma条件 | 従来最速 ms | 1面ずつ ms | 3面ずつ ms | 9面同時 ms | Dense ms |
|---|---:|---:|---:|---:|---:|
| mixed | 4.318150 | 8.855427 | 5.661841 | 7.387453 | .082610 |
| small | 1.205669 | 7.208419 | 4.511889 | 5.004908 | .082248 |
| switch | 4.634804 | 10.589937 | 7.690621 | 8.713041 | .082533 |

新候補のcapture/replay込みpeak allocated / reserved MiBは、1面70.590/138、
3面102.590/202、9面198.590/490（3条件とも同じ）。mixed従来51.688/116、
small従来50.231/114、Dense49.002/106。process全体のGPU memoryは未測定。

別計装Graphのmixed・3面版ではrange組立2.308 ms、range VJP2.142 ms。
smallではbroad atomが0でも、それぞれ1.698/1.452 msの固定費を払う。
このstageにはscatter/query・scan・係数合成を含む。未計装完全stepの代用として
足し合わせたり差し引いたりしない。DRAM/cache/atomicの物理counterは未測定。

v7単純版は速度・メモリの両方で不採用。3面版の別job再測定もmixed5.660750、
small4.512089、switch7.688989 msで、同じ結論だった。

## 分離した研究ブランチ

今回のv7/v8試作は `kernel/quadratic-bspline-range` に隔離する。
基準の `kernel/quadratic-bspline-prefix` とPR109は `e4e266e3` のまま維持し、
既存の最速版や公開dispatcherへ新方式を置換しない。
分離前に現在worktreeで作成した `689d57fa` / `a4940ea0` の両commitは新研究branchに保持。
source/result archiveとrawログは削除しない。worktreeのpathは従来と同じである。

## 検証・取得時の注意

v7実装の初期2 jobはTritonのconstexpr再代入とscalar/vector条件式のコンパイルで失敗。
数学・性能の結果には数えず、修正後のsmokeで25件通過を確認した。
`l4job-52a0fe25c3304a25b5159a1bed1e0876`はdriver正常終了・全result archiveの取得と
SHA256照合まで完了したが、取得後のremote cleanupがWebSocket切断で失敗した。
GPUはserver照合で停止済み。データは保全するが、採用判断用の独立run数には数えず、
以後のclean jobで融合版の精度・性能を再確認する。

CPU回帰1968 passed（CUDAのskipはGPU確認ではない）。wheelとsdistをbuildし、
全338 runtime fileがcheckoutとbyte一致することを確認した。
最初のno-isolation buildはローカル環境にhatchlingがなく失敗したため、標準の
隔離build環境で再実行し成功。ローカル環境の依存を変更していない。

B32 mixed/switchの閾値比較とB64 mixedを新しいGPUで確認する。
full-shape oracleは各候補・各case・3つのlive状態でY/dX/全source Parameter勾配の
maxabs/relative L2をともに4e-4未満とする。基準の緩和や幅の固定化はしない。

最終監査で、Tritonの`where`だけで作ったknot fraction tensorが暗黙のFP32に
なり得る点を修正した。従来の値・勾配gateは通っていたが、ceil直近の区間選択も
元のFP64 geometryと揃えるため、±1/3をFP64 tensorに直接入れる。
微小offsetを持つ33 atomでcut index自体をFP64式と完全一致させる回帰テストを追加。
最終の速度・精度・メモリ判断には、この修正後の同一runtimeを使う独立jobを使う。
修正前の表は探索経過として保持し、最終版の数字と混ぜない。

追加のCPU auditでは、大きいsigmaでも片軸の支持が領域端で1点へ縮むと、
L2 floorの倍率で全体係数の相殺誤差が増幅することも確認した。
N1024・sigma .3・片軸中心 `-sqrt(6)*sigma+1e-7` の1 atomではY最大絶対誤差
7.74e-5だが、真値が小さいためrelative L2 gateを満たさない。
この既知のケースを放置せず、v7/v8は **sigma <= cutoff または、どちらかの軸の
clipped supportが2 site以下** のatomを既存のnative W/VJPへ戻す。
全atomの小支持キーをdeviceで毎forward検査し、中心だけの移動でもrouteを更新する。
旧streamの分類は変更しない。profileやfloorを変える救済ではない。
1 atomだけを使う小真値のY/dX/全atom勾配gateと、中心だけを動かすGraph replayの
route反転テストを追加した。通常のbenchmarkにも、このguardの構築費を含める。
