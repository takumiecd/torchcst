# Quadratic B-spline: FP64演算と係数・prefix読み取りの切り分け

2026-10-20。研究用revision v16。[H/Hμ保存](quadratic-bspline-h-save-20261019.ja.md)を対照に、[端点共有・atomic削減の比較](quadratic-bspline-adjoint-20261019.ja.md)に続いて、queryの計算式自体を比較する。公開dispatcherの変更とは区別する。

## 維持する数学

profile、支持、全siteの正確なsampled product L2とfloor、中心微分、振幅0での勾配、Polar update、task幅stop-gradientは変更しない。係数、prefix moment、縮約、差分集約はFP64、H/Hμ/G/Gμの保存・使用境界は既存FP32丸めを維持する。FMAを無効にしたまま、独立FP64 oracleのY/dX/全atom4成分のmaxabsとrelative L2をともに4e-4未満で検査する。

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
