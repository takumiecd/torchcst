# Quadratic B-spline: FP64演算と係数・prefix読み取りの切り分け

2026-10-20。研究用revision v16。[H/Hμ保存](quadratic-bspline-h-save-20261019.ja.md)を対照に、[端点共有・atomic削減の比較](quadratic-bspline-adjoint-20261019.ja.md)に続いて、queryの計算式自体を比較する。公開dispatcherの変更とは区別する。

## 維持する数学

profile、支持、全siteの正確なsampled product L2とfloor、中心微分、振幅0での勾配、Polar update、task幅stop-gradientは変更しない。係数、prefix moment、縮約、差分集約はFP64、H/Hμ/G/Gμの保存・使用境界は既存FP32丸めを維持する。自動FMA融合は無効のまま、独立FP64 oracleのY/dX/全atom4成分のmaxabsとrelative L2をともに4e-4未満で検査する。

入力側と出力側に同じ1D演算を使う。以下はG側で説明する。格子の中心o+h、半幅h、正規化座標u=(x-o-h)/h、B-splineの支持半径R=√6σとする。z=(μ-o-h)/h、r=R/h、q=1.5(h/R)²。切替点はμ+R·(-1,-1/3,1/3,1)、その最初のsite indexをκ₀..κ₃とする。格子外のindexは[0,N]にclipする。

\[
P_{p,k,b}=\sum_{j<k}u_j^p dY_{bj},\qquad
\mathcal R(P,z)=P_2-z(P_1+(P_1-zP_0)),\qquad
\mathcal T(P,z)=P_1-zP_0.
\]

### 展開済み係数

既存のqueryは各切替点につき(c₀,c₁,c₂,d₀,d₁)を読み、

\[
G=-\sum_k(c_{k,0}P_{0,κ_k}+c_{k,1}P_{1,κ_k}+c_{k,2}P_{2,κ_k}),\quad
G_μ=-\sum_k(d_{k,0}P_{0,κ_k}+d_{k,1}P_{1,κ_k})
\]

とする。query内部にはprefix scanもatomicもなく、4点のgatherとFP64演算がある。scanはPの事前構築とY/dX復元の別段階にある。v16のexpandedは同じ既存関数を呼ぶ対照。

### 共通因子を括る式と左端の差し引き

zₖ=z+r·(-1,-1/3,1/3,1)ₖ、w=(1,-3,3,-1)とすれば、

\[
G=-q\sum_kw_k\mathcal R(P_{κ_k},z_k),\quad
G_μ=(2q/h)\sum_kw_k\mathcal T(P_{κ_k},z_k).
\]

Σwₖ=Σwₖzₖ=Σwₖzₖ²=0なので、各PをDₖ=P_{κ_k}−P_{κ₀}に置換しても実数上は同じである。factored実装は先にこの差を取る。D₀=0、残り3項を(0−R₃)−3(R₁−R₂)として合算する。大きいprefix値をそのまま二次式へ入れる初案は狭いσで誤差を増やしたため、採用しない。

### 区間差分

Eₛ=P_{κ_{s+1}}−P_{κ_s}、s=0,1,2とすれば、3つの多項式区間を直接積分和として評価できる。

\[
G=E_{1,0}+q\{\mathcal R(E_0,z-r)-2\mathcal R(E_1,z)+\mathcal R(E_2,z+r)\},
\]
\[
G_μ=-(2q/h)\{\mathcal T(E_0,z-r)-2\mathcal T(E_1,z)+\mathcal T(E_2,z+r)\}.
\]

これは連続積分によるL2近似ではなく、離散prefixから求める同じsite和である。各区間の境界siteを個別評価しない。両候補ともGだけでなくforwardのH/Hμにも使い、既存のY/dX差分書き込み・復元は維持する。

## 保存と実行の範囲

追加のfma候補だけは、展開済み係数の同じ積和を明示的なFP64 `tl.fma` で計算する。dtypeを下げず、丸め順序だけが変わる。自動融合flagは全候補でFalseを保つ。fmaも独立oracleの同じ閾値で検査し、失敗時は採用しない。

factored/piecesはquery中に展開済み係数を読まず、Metaの中心・半径・逆数から共通因子を再生成する。Y/dXの差分注入には既存係数が引き続き必要で、係数snapshot全体を廃止したわけではない。

compact候補は既存の3-plane snapshotを利用する。係数は40A doublesから24A doublesになり、A=52,428では6.400MiB減る。queryのために全atom×batchのG/GP配列を追加しない融合経路を完全stepで比較する。staged経路は検証と診断用であり、融合経路の時間分割とは扱わない。

σはatomごと・stepごとにdecodeし、配置の検証・再配置、係数構築、L2、forward/backward、AdamW+fused Polarを完全stepに含める。保存値は同じforwardのsnapshotで、XやParameter更新後の値に置き換えない。

## 診断の設計

`benchmarks/cuda/linear/bspline_query_probe.py`は実fixtureのMeta/Cuts/CoeffとdY prefix bankを使い、G/Gμのみを別Graphで測る。8atom tile、batch contiguous、FP64、2つのFP32出力。21 replay、1 replayにつき20 query。

- full: 元の読み取りとFP64演算。
- checksum: 同じ係数・prefixを読み、FP64演算の代わりにbitwise checksumを出力。機能的に不正な診断であり、精度を変えない候補の速度として扱わない。
- uniform_coeff: 係数のみ先頭の広いatomの値を共有し、prefixの参照先は元の各atomのまま。
- uniform_bank: prefixの参照先だけを同じ切替点にし、係数は各atomの元の値。
- uniform_both: 両方を共通化。
- factored/pieces: 正しい式の候補。fullとのG/Gμ差も確認する。

参照先共通化はアドレスの多様性の診断であり、物理DRAM転送やcache hit率の直接測定ではない。checksumは命令構成・register量も変えるので、差をそのままFP64の専有時間と断定しない。PTX・register/spill/shared量も保存し、hardware counter未測定の原因は仮説として区別する。scan-onlyは既に重み付けしたTensorのcumsumを出力確保済みで測り、bank全体は変換・重み付け・scan・転置・確保を含めて別に測る。

## 途中で除外した結果

初回ede4ee55はunroll内のconstexpr再代入によるcompile失敗。87d46c95のc9ee5e2cでは35件合格・6件失敗。4件はdXだけを要求する経路のkernel引数、2件は左端差し引き前のfactoredによるGraph精度失敗（最大差5.920e-4）だった。引数と数値式を修正し、閾値は変更していない。9e683916は診断モードのTriton文字列比較でcompile失敗した。これらの失敗を速度比較や最終合格の根拠には使わない。

## 初回B32探索

job `l4job-17e9ea4b68954884ab9f6523b0aec0a9`、runtime source `6e5fa174`。archive・manifest・source・cleanupを照合し、全recordがPASS。新規41件（40 CUDA + 1 metadata）も34.65sで合格。1024²、A=52,428、B32、spacing .1、混合σ=.1/.4/1.6/6.4/12を各20%、上限12.8、TF32 off、AdamW+fused Polar、warmup5/21 replay median。

| 方式 | 完全step ms | allocated / reserved MiB | forward / backward register |
|---|---:|---:|---:|
| v14 H/Hμ保存 | 1.705895 | 81.288 / 172 | 48 / 48 |
| v16 expanded・同条件 | 1.711550 | 81.288 / 172 | 48 / 48 |
| expanded + compact | 1.778371 | 75.688 / 158 | 56 / 64 |
| factored | 1.823597 | 81.288 / 172 | 72 / 64 |
| factored + compact | 1.804168 | 75.688 / 158 | 72 / 64 |
| pieces | 1.709196 | 81.288 / 172 | 64 / 64 |
| pieces + compact | 1.687377 | 75.688 / 158 | 64 / 64 |
| W版 | 3.230818 | 78.590 / 154 | — |
| Dense | .082738 | 49.002 / 106 | — |

全H系候補のcompiler spillは0。新revisionのexpandedは対照と同じregister量・同じ時間帯であり、wrapper追加による改善とは扱わない。単に係数を縮小すると中心微分係数の再生成が必要で遅くなった。piecesは中心微分を式から求めるので、この追加演算を避けてcompact化できる。ただし初回の約1%差だけで安定した高速化とはしない。

独立FP64 gateの全H系最大誤差はY 1.537410e-6、dX 1.511848e-6、dP 1.601655e-5、relative L2は1.281e-7以下。全52,428 atomのσが各測定で変化し、初期PとDenseを含むX/targetのhashは一致した。peakはcapture/replay込みのallocatedとreservedで、総GPUプロセス使用量は未測定。

同じ測定の別Graph診断では対照のbank_adjoint .031744ms、scan_output .023552ms、scan_input .024576ms、係数構築 .061440ms、融合broad backward .592896msだった。norm_replay .214016msは診断だけの追加処理で、完全stepには存在しない。空markerは処理時間と扱わない。これらを足して主測定の時間内訳とはしない。


診断job `l4job-f8d19e5a422a43d1bd22c198355cb37e` はWebSocket connection lostにより結果archiveを回収できず、poolの外側timeoutで終了した。poolのrecover後に全slot stoppedと所有sessionのserver assignment消失を確認した。runtime性能・精度の根拠には含めない。source/spec/transportと復旧ログをignored outputへ保持し、B32/B64を分けて明示再提出した。未実行の旧比較jobはcancelし、FMA候補を含む固定snapshotに揃えた。queuedの待ち時間をkernel時間として扱わない。

## FP64積和の候補

expandedの1切替点はGに3mul+3add/sub、Gμに2mul+2add/subを使う（自動FMA無効）。4切替点で40個のFP64算術操作となり、広いatom 41,942・B32では約5,369万操作、B64では約1億737万操作。geometryとpullbackの演算はこの数に含めない。

fmaは各端点のGの3項を2個の明示的FP64積和と1mul、Gμを1積和+1mulで求め、その結果を累積値から引く。各端点の算術命令は10から7へ減る。fma_chainは各積を直接累積値へFP64積和として加える候補で、各端点5個の積和とする。演算の実数上の意味・dtypeは同じだが丸め順序は異なるため、bitwise同一とは主張しない。正確なsampled L2/floorや中心微分は従来どおり。

初回fma job `l4job-d9cc391a705c47eda0c67d77782ccb44`、source `3683bec7`は61件（60 CUDA + 1 metadata）を97.66sで合格し、全候補の3状態の独立FP64 gateも通過した。archive/manifest/source/cleanup照合済み。

| B32混合σ | 完全step ms | allocated / reserved MiB | forward / backward ms (別Graph診断) |
|---|---:|---:|---:|
| v14対照 | 1.709664 | 81.288 / 172 | .455680 / .592896 |
| FP64 fma | 1.618376 | 81.288 / 172 | .451584 / .506880 |
| fma + compact | 1.651668 | 75.688 / 158 | .458752 / .561152 |
| pieces + compact | 1.691437 | 75.688 / 158 | .457728 / .596992 |
| Dense | .083630 | 49.002 / 106 | — |

fmaと対照は係数とprefixの参照先、forward/backwardのregister/thread=48、spill=0、shared=1024byteが一致する。B32の初回完全stepは約5.3%短縮し、backward診断は約14.5%短縮した。FP64算術命令の減少が実時間の改善につながることを示すが、FMA pipelineやcacheのhardware counterの測定とは区別する。forwardの差は小さく、G/Gμだけを速くしてforwardの集約まで同率に速くなるとはいえない。

縮小snapshotは6.400MiB少ないが、測定peakの削減は5.601MiBだった。容量とlive lifetimeの異なるpeakを同一の数値として扱わない。

診断B32/B64の初回分割job `630288f9` / `1cbaeb55` はchecksum内のTriton tupleループでcompile失敗した。失敗したjob内の部分timingを性能の根拠にせず、ループを明示的な5個のXORへ変更して再提出した。

## G/Gμ単独の切り分け: B32

job `l4job-45843cd49a2340b78024d332866ddd6e`、source `bc7186f5`。全mode完了・return0・archive/manifest/source/cleanup確認済み。実混合σのgeometryとprefixを固定し、繰り返しqueryする診断であり、cold-cache時のDRAM帯域や可変σ完全stepの代用ではない。

| query mode | ms/query | register/thread | static PTX FP64算術 | static global load |
|---|---:|---:|---:|---:|
| full | .286822 | 42 | 80 | 37 |
| checksum（診断のみ） | .062618 | 56 | 0 | 37 |
| 係数の参照先だけ共通 | .287181 | 42 | 80 | 38 |
| prefixの参照先だけ共通 | .286106 | 40 | 80 | 40 |
| 両方共通 | .286054 | 40 | 80 | 40 |
| factored | .357118 | 70 | 101 | 23 |
| pieces | .293120 | 58 | 82 | 23 |
| FP64 fma | .195789 | 40 | 56 | 37 |

すべてspill=0、PTX barrier=0。このtableのPTX数はコンパイル結果の静的な命令数で、実行回数・FLOPS・DRAM transactionとは異なる。B32の1threadが複数のatom×batch要素を処理するため、上記の1要素40操作と80命令も直接比較しない。

係数の参照先とprefixの参照先を理想的に共通化しても、この診断のquery時間は約.287msのままだった。FP64をchecksumに置換すると短縮するが、registerや命令構成も変わるため、その差.224msをFP64の排他的な占有時間とはしない。正しいFP64 fmaで、global loadの静的数を保ちながら算術80→56、query .287→.196msとなり、完全stepでも改善したことを合わせると、この条件のqueryではFP64積和の命令実行が支配的な費用だと判断できる。

factoredは読み取り命令37→23でも、FP64算術80→101、register42→70となり遅くなった。piecesも読み取りが減っても算術は82で、このB32 query単独では速くならなかった。式の項数や係数plane数だけからGPU演算量を推定しない。

同じprobe processのscan-only .019405ms、変換・重み付け・scan・転置・確保を含むbank全体 .028621ms、dX復元側scan-only .021504ms。G/Gμのkernel内にはscan・局所共有gather・atomic・block barrierが存在しない。scanを直列処理として想定して、この.287msを説明するのは適切でない。

正しいquery候補のfullとの差は、fmaでG max4.768372e-7 / relative L2 1.310e-10、Gμ max2.980233e-8 / relative L2 1.187e-11だった。これは単独query同士の照合であり、独立oracleの全atom勾配gateとは別に記録する。

## G/Gμ単独の切り分け: B64

job `l4job-bc91f439c2444f1b8be2c8122f94c383`、source `bc7186f5`。演算・全mode・driverはreturn0で完了し、結果archiveと全manifestを回収・hash照合した。その後のremote cleanup応答だけを失ったため、poolが所有VMを停止し、serverのactive assignment消失と全slot stoppedを確認した。`cleanup-proof.json` と `resource-probe64-stopped.json` / `lifecycle-probe64-stopped.log`を保持する。未回収のf8d19e5aとは区別し、この回収済み完全結果は停止確認を根拠に利用する。cleanup ackを偽造・書換えしていない。

| query mode | ms/query | register/thread | static PTX FP64算術 | static global load |
|---|---:|---:|---:|---:|
| full | .571290 | 64 | 160 | 49 |
| checksum（診断のみ） | .223590 | 102 | 0 | 49 |
| 係数の参照先だけ共通 | .571187 | 64 | 160 | 50 |
| prefixの参照先だけ共通 | .569549 | 72 | 160 | 52 |
| 両方共通 | .569856 | 64 | 160 | 52 |
| factored | .657203 | 95 | 185 | 35 |
| pieces | .530022 | 96 | 148 | 35 |
| FP64 fma | .412518 | 72 | 112 | 49 |

全spill=0、bar.sync=0。scan-only .035072ms、bank全体 .047869ms、dX側scan-only .021760ms。B32と同じく参照先共通化だけでは大きく変わらず、FP64積和の命令削減がqueryの約27.8%短縮につながった。checksumの差をFP64だけの排他的時間と扱わない。B64 fmaのG差はmax4.768372e-7 / relative L2 1.248e-10、Gμはmax2.384186e-7 / relative L2 4.994e-11。

以上は同じbankと初期geometryを繰り返すprobe。更新・構築・L2込みの評価は完全stepの比較を使う。

## 累積まで積和にする候補

job `l4job-b3100ed66e514fd983ac6748c7f75f02`、runtime `bc7186f5`。新規対象81件（80 CUDA + 1 metadata）は121.95sで合格し、全候補の独立FP64 full-shape gateもPASS。archive/manifest/source/cleanup照合済み。

| B32混合σ | 完全step ms | allocated / reserved MiB | 融合backward診断 ms |
|---|---:|---:|---:|
| v14対照 | 1.710214 | 81.288 / 172 | .592896 |
| fma_chain | 1.645537 | 81.288 / 172 | .531456 |
| fma_chain + compact | 1.676759 | 75.688 / 158 | .586752 |
| fma（端点内の積和） | **1.618422** | 81.288 / 172 | .507904 |
| Dense | .083039 | 49.002 / 106 | — |

3つの非compact kernelはforward/backwardともregister48、spill0、shared1024byte。命令数を減らしたchainが最速にはならなかった。chainは累積値への依存が長い式であり、独立に計算できる端点内積和の方がこの実測では速い。ただし依存待ちやissue率のhardware counterは測っていない。

最終runtimeに対するCPU全体は1,976 passed / 3,495 skipped、41.68s、既存warning18件。wheel/sdist内の348 runtimeファイルをcheckoutとbyte照合した。GPU回帰は変更対象のquery・saved-H・adjointを対象にし、非対象の旧range方式のGPU全件再実行とは区別する。
