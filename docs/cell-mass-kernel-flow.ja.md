# 区間確率方式：prepareからbackwardまでのkernelの流れ

## 設計の正本と、現在の実装範囲

**実装の区切り（2026-10-28）：** 既存σ版と検証済み最適化はPR #122から
`main`へmerge済み（`0047e29c`）。次期半径APIと以下の設計は、そこから切った
`kernel/support-radius-pipeline`で進める。
現在は半径APIのCPU参照とN<=128の明示CUDA Plan、対称な融合forward/backward、
FP32半径PolarのGPU更新まで実装している。以下では初版・後続候補・測定結果を区別する。
この半径版でN1024以上、Graph、自動dispatch、dense相当の速度を達成したとは扱わない。

**半径APIの初版：** revision7の半径宣言とCPU/Torch参照を追加した。
`RadiusBounds`・`ChebyshevRadiusSpec`・物理／Polar用preset・`CSTModule.support_radii()`を公開する。
旧σ版のAPI・checkpointは維持し、半径版との相互読み替えを拒否する。
新CUDAの初版は以下の別経路へ実装する。融合・保存量削減・速度／メモリの目標達成は別の段階とする。

CPU検証は全体2,324 passed／5,596 skipped／既存警告18件。
新規69ケースには、独立Chebyshev数値積分とのセル確率照合、FP32/64のY・dX・全物理atom列の勾配、
中心／半径のgradcheck、奇数M、周期端・繰り返し像、retained backward、空batchを含む。
全物理列の独立勾配gateはN16・B3・2atomの小さい検査であり、新しいN1024全atomのGPU gateではない。
同じ外部勾配を与えた旧／新Polarの5更新では、ParameterとSGD／Adam／AdamWのmoment・stepがbit一致した。
旧σ checkpointと半径checkpointの取り違え、Gaussian・未対応device／dtypeの拒否も検査した。
READMEのCPU例を実行し、wheel/sdist双方の352 packageファイルをsourceへbyte照合した。
ログ・配布物・hashはignored `output/support-radius-implementation-20261028/`に保持する。
再現は`PYTHONPATH=src:. python -m tools.kernel_dev test --suite cpu`。

### 半径CUDAの初版：処理と保存の対応

`_backends/cuda/algorithms/linear/radius_cell_mass/`に専用Algorithmと
明示Plan `REFERENCE`／`DIRECT`を置く。旧σ版のPlan・数式・checkpointは変更しない。
最初の対応範囲は1次元周期格子、N<=128、A<=4096、FP32/64、M=1..16（奇数も可）。
CPUで初期化したCSTLinearをCUDAへ移し、FixedSelectorで選ぶ。
この段階はLinearのforward／一次backwardの数値基準であり、半径PolarのGPU optimizer、
Graph、autocast、高階微分、dense以下の完全stepメモリ／速度を達成した実装ではない。

| 呼び出し | 1 threadが担当するもの | 保存先・寿命 |
| --- | --- | --- |
| `prepare_layout` | 1 atomの入出力bank・最初のtileを分類 | 一時keyをsortし、元atom行番号とrange offsetsだけ残す。atom値のコピーは作らない |
| `build_moments` | batch×tile×1実基底の和 | `input_moments[B,node,2M+1]`。backwardまで保存 |
| `compute_response` | batch×atom。必要なK枚すべてを足し終える | 完成`atom_response[B,A]`のみ公開し、backwardまで保存 |
| `compute_coefficients` | batch×出力tileの全基底 | sorted range内の完成responseを集約して`output_coeffs`をglobal scratchへ保存 |
| `expand_coefficients` | 最終出力の1 site | 各階層の寄与をregisterへ加算し、Yへ1回だけ書く。coeffsはforward後に解放 |
| backwardの`build_moments`／`compute_response` | dYを同じ仕組みで処理 | `grad_output_moments`、完成`grad_response`をbackward中だけ保持 |
| backwardの`compute_coefficients`／`expand_coefficients` | 入力tile／最終入力site | 逆向きsort rangeから`grad_input_coeffs`を作り、dXへ1回だけ書く |
| `parameter_grad` | 1 atomの全batch | dX側で作ったmoments・完成Gを再利用し、保存H・X側の情報と合わせてatom勾配を集約。Polarは半径のtask勾配を切る |

momentsはセル積分済みの基底を使う。`merge`ではDCとmerge_factorで割り切れる周波数を
子tileから正確に継承し、それ以外の周波数は入力から直接計算する。
有限個の子周波数だけでは親の全周波数を復元できないため、近似補間で埋めない。
子level→親levelには依存があるが、同一levelのbatch・tile・基底は並列である。
`direct`はlevel依存なしに同じ和を生成する。

各levelにK=2とK=3のbankを持ち、現在のRから使う方を選ぶ。
P=(ceil(2R/T)+1)Tを全周Qで打ち切らず、K枚の窓がチャートを一周以上する場合も、
同じtileの繰り返し像を異なるunwrapped座標で加算する。
H/Gの途中の和からcoeffsや勾配を作る経路はない。

この初版ではshared memoryは使わず、thread内の係数・累積はprivateな配列／scalar、
kernel間の受け渡しはglobalで行う。private配列がregisterに収まるかspillするかはcompiler依存である。
すべてのbankのmoments／coeffsをFP64で確保するので、これを最終メモリ設計とはしない。
例えばN128・base1・merge2・M8・B4では1軸510 nodesで、1つのmoments／coeffs配列は277,440 bytes。
H/Gはそれぞれ8BA bytes、元Parameterは4A要素で、batchに比例するscratchが主な削減対象になる。
将来のactive bank省略・block内融合・選択保存と比較できるよう、この初版の明示境界を残す。

launch順序で依存を保証し、blockを越えたspin waitは使わない。
sort前の異常値検査は1 error wordをCPUへ読み戻すため、この版では同期ゼロ／Graph対応を主張しない。
positiveなR/Tが数値的に0へunderflowする尺度は、K=1の不正bankへ進まず拒否する。

**初版の検証（2026-10-28）：** CPU全体2,339 passed／5,625 skipped／既存警告18件。
共有L4で44ケース成功（PyTorch 2.11.0+cu130、CUDA13.0、Python3.13.16）。
N16/27/128、M1/3/8/16、merge2/3、非対称軸、K=2/3、P>Qの繰り返し像、
セル中心one-hot近似、空batch／atom、retained backward、公開FixedSelectorを確認した。
Y/dXは独立Chebyshev数値積分へ、全物理列・Polar列の勾配はCPU/Torch参照へ照合した。
N128・B4・A819（5%）のFP32/64でもY・dX・全atom列の勾配を照合した。
これは速度測定／完全optimizer step／memory peakの検査ではない。
初回はテストの参照行列の余計な転置で7失敗し、修正後に42件、5%条件追加後に44件が通過した。

最終GPU jobは`l4job-beac5aa0da4a406c881746807f620eb7`、snapshot SHA256は
`ed26030d0a401a394aa45fb2ca13f14c639d7020822bc4edf5a4ac477d70bd75`。
native moduleは`torchcst_radius_cell_mass_94e7b1de3936`、回収binary SHA256は
`6d7aa4a78cd2500fa053f6cb5621d423560747b1fe40bd0560b6b36d52566c13`。
再現は同snapshotで`python -m pytest -q tests/test_radius_cell_mass_cuda.py`。
source照合・pool receipt・結果・binaryはignored `output/radius-cuda-20261028/`へ保持する。

### direct／階層を独立に選び、依存のない計算を重ねる

2026-10-28。上の初版が全atomにmoments→response→coeffsを使うのは、数値照合のための
比較基準であって、全atomにその直列依存があるという意味ではない。追加する候補は次のとおり。

| 入力側の選択 | 出力側の選択 | atomの計算経路 |
| --- | --- | --- |
| direct | direct | Xのセル確率から完成responseを作り、そのままYのセルへ展開。moments／coeffsなし |
| direct | coeffs | momentsなしで完成responseを作り、出力側coeffsへ集約 |
| moments | direct | momentsから完成responseを作り、coeffsを介さずYへ展開 |
| moments | coeffs | moments→完成response→coeffs→Y |

Recipeの`input_direct_levels`／`output_direct_levels`より小さい階層をdirectとする。
0なら従来どおり全階層を使い、level数を指定すると全atomをdirectで処理する。
入力側と出力側の設定は独立で、profileのR・P・M・セル積分・正規化は変更しない。
省略した低階層のmoments／coeffs用global領域は確保しない。
最下位の残るmomentsは入力から直接作り、それより上で一致周波数を子から継承する。

`HYBRID_SERIAL`は順次実行の比較用、`HYBRID_OVERLAP`は次の依存を使う。

```text
prepare/sort ─┬─ direct response ─────────────────┐
              └─ moments構築 → hierarchy response ┤
                                  完成値をjoin ──┴─ direct展開／coeffs → Y
```

別streamは同じresponse配列の異なるatom要素だけに書き、使用前にeventでjoinする。
全GPUのCPU同期やblock間spin waitではない。streamを分けてもSMの専有率は指定できず、
資源競合やlaunchのタイミングにより実際には重ならない場合もある。
この候補の出力展開は完成response全体のjoin後に始める。
直接経路のYまで先行させる二重出力bufferは、この段階では追加しない。

`BLOCK_LOCAL`はsort後のforward computeを1 kernelへまとめる。
1 CTAがbatchの1行と出力site範囲を持ち、sharedにinput momentsと完成responseを作る。
block同期後、moments領域をoutput coeffsのscratchへ転用してYを一度だけ書く。
direct atomはmoments／coeffsを使わず、同じprofileのセル積分で処理する。
block内では明示した段階の同期を行い、direct responseとmomentsを別teamで同時実行する版ではない。
複数stream版と、block内の段階を融合する版を別の候補として比較する。

この版は出力CTAごとに入力moments／responseを計算するので、共有と重複演算が交換条件になる。
完成momentsとresponseのglobal保存は出力CTA0だけが担当し、各CTAは自分のshared値を使う。
forwardは`(Y,input_moments,atom_response)`を返し、後二者をbackwardへ保存する。
backwardは同じcompute kernelを逆向きに使い、`(dX,grad_output_moments,grad_response)`を返す。
その後のatom勾配kernelは保存済みX側moments/Hと、逆向きに作った同じmoments/Gを再利用する。
input momentsを再構築する段階はない。coeffsは両方向ともshared scratchで、global表を作らない。
prepareを除いてforward computeは1 kernel、通常backwardは逆compute 1 kernel＋atom勾配1 kernelになる。
sharedの上限は1 CTAあたり48KiBで、超える設定は拒否する。
`output_tile_sites`は出力CTAの担当範囲であり、profileの階層tile幅を変更する設定ではない。

**backwardの実行順：** 全候補で、出力側moments／直接投影→完成G→dX→grad_atomとする。
grad_atomはdXの値そのものではなく、dX側で生成した同じ出力moments・完成Gを使う。
X側momentsとHは、block_localも含めforwardの保存値を使う。
dX用coeffsのscratchは展開後に解放し、grad_atom段階へ残さない。
片方の勾配だけが必要なら、不要なdX／grad_atomの段階は実行しない。
これは同じstreamへのenqueue順であり、CPU同期を追加したり、前の層のbackwardまで同時に開始できると保証したりしない。
順序変更後もL4で106件成功（宣言等31件、CUDA実行75件）。
job=`l4job-895949990509471f8bf01b3daff41078`、snapshot SHA256=
`c9ac2bab18b5fb8f094a9a2e6d16b0b45e3369ed0f2b37c38347d6aee27af6cd`。
native sourceは不変で、検証済みbinaryをsource／runtime照合して再利用した。
新しいPython実行順のsource hash、結果、receiptはignored `output/radius-backward-order-20261028/`へ保持する。

#### 対称な融合computeと、保存値の契約

2026-10-28。前節の順序変更だけでは十分でないため、block_localのbackwardも融合する。
momentsは使用する階層分だけを保存し、全階層directなら0要素でもよい。
directなprofile微分を含むatom勾配には、raw X/dYも使用する。

```text
forward_compute(X)              -> Y,  input_moments,       atom_response   [1 kernel]
backward_compute(dY)            -> dX, grad_output_moments, grad_response   [1 kernel]
parameter_grad(保存値と逆向き値) -> grad_atom                                [1 kernel]
```

3つを返すのはnative computeの契約である。公開`CSTLinear.forward`の戻り値はYのまま維持し、
input momentsとatom responseはautogradが保存してbackwardへ渡す。

atom勾配だけ必要なら`backward_project(dY) -> (grad_output_moments,grad_response)`を
1 kernelで計算してからparameter_gradへ進む。dX展開・dX用coeffs・逆向きsortは行わない。
dXだけ必要なら逆computeだけを実行し、parameter_gradを起動しない。
moments/Gを作り直す段階や、block間で完成を待つspin waitはない。
kernel内部では依存順にblock同期し、momentsを保存してからshared scratchをcoeffs用に再利用する。
これらはcomputeの起動数であり、prepareの分類・sort等まで1 kernelに含む意味ではない。
旧REFERENCE／HYBRID_SERIAL／HYBRID_OVERLAPは分割実行の比較候補として残し、融合版とは区別する。

**対称融合後の検証：** CPU全体2,355 passed／5,697 skipped／既存警告18件。
L4で132件成功（宣言等31件、CUDA実行101件）。新規26件は、実際のnative呼び出しを記録して
forward triple、逆向きtriple、atom-onlyの2値、不要段階の省略を検査する。
forwardのmoments/Hと逆向きのmoments/Gが同じstorageのままparameter_gradへ渡ること、
input momentsの再構築がないことも検査した。FP32/64のN128・A819・B4、非対称軸、
奇数M、全direct、Polar別bounds、retained backwardを参照値へ照合した。

N128・A819・B4・M8の実GPU traceでは、通常経路はforward＋reverseのtransform kernelが2回、
atom勾配kernelが1回で、別moments／response／coeffs／展開kernelは0回だった。
atom-onlyはforward＋projectionの2回とatom勾配1回、dX-onlyはforward＋reverseの2回だけだった。
この観測は準備・sortを除くcomputeの起動数であり、未計装の完全step速度比較ではない。
同条件で1方向の保存momentsは`[4,126,17]` FP64＝68,544 bytes、responseは`[4,819]` FP64＝26,208 bytes。
これはcache本体の容量であり、完全stepのallocated/reserved peakではない。
勾配有無・directの組合せとA819を含む18ケースでmemcheckは0 errors、racecheckは0 hazards。

gate job=`l4job-f77c56e6f1c04ff69097c8c483292748`、snapshot SHA256=
`047b898cc3e964f8791de3945b56a1f8befed908aa03bf24719e0d2b44ed1d1a`。
起動数・sanitizer job=`l4job-41ca88725b1d458e82be7c5cb60ae924`、snapshot SHA256=
`e95cd93dceeadf8841a57f526c0aa83d4c5d2295e1ccacb2b5ca4af9c771a525`。
native binary SHA256=`0f103d5bfa3c86a8e351aacb1baf8b8cef9171588efae8f20b28ff8e3230eb63`。
source照合・receipt・binary・Chrome trace・検査logはignored `output/radius-fused-backward-20261028/`に保持する。
再現gateは前節の2ファイルに`tests/test_radius_cell_mass_fused_backward.py`を追加する。

初期候補のdirect_levels=2は比較用の明示設定で、最適閾値や全GPU共通の既定としては採用しない。
なお従来の`DIRECT` Planはmomentsの直接生成（`moment_method="direct"`）であり、atomのdirect経路とは区別する。

#### 半径版の最適化に使う完全stepの基準

2026-10-28。既存Linear runnerに`polar_radius_cell_mass` fixtureを追加し、公開CPU初期化から
FP32のCSTLinear、同じ初期密行列のdense、loss・dX・全atom勾配・実AdamW/Polar更新まで測る。
更新用の明示Planは`RADIUS_FLAT_TORUS_FUSED`。旧σ Planを拡張して流用せず、
revision7の半径Polarだけを受け付け、既存のPolar CUDA更新式をそのまま使う。
固定宣言の検証結果はbuffer/specの署名でcacheし、設定変更時に再検証する。
atom更新ではcacheを無効にせず、毎stepの固定値のCPU読戻しを省く。

比較条件はN128・B4・A819（約5%）・M8・seed41、base1/merge2、FP32 IEEE、
eager AdamW（lr=1e-4、weight_decay=.01、fused=True、capturable=False）。
CSTOptimizerの`check_finite=False`はPyTorch流儀の比較設定であり、live設定の検証は維持する。
異常値検査を有効にした場合の拒否・非更新も別テストで確認する。
公開初期化のRは両軸とも約3.98552〜3.9999999994に集中し、819atom全てがlevel2/K3である。
階層に均等配置したfixture、旧σのMNIST条件、N1024の結果として読み替えない。

最適化前の独立job（warmup3・21サンプル、独立run数1）：

| Plan | 完全step中央値 ms | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| REFERENCE | 34.741301 | 973312 | 2097152 |
| HYBRID_SERIAL | 34.335889 | 346624 | 2097152 |
| HYBRID_OVERLAP | 34.274785 | 346624 | 4194304 |
| BLOCK_LOCAL（担当32site） | 34.774483 | 291328 | 2097152 |
| dense | 0.702979 | 17313280 | 23068672 |

allocator peakはfresh workerのresident model・勾配・AdamW状態・eager一時領域を含む。
Graph capture/replayとGPU process総量は未測定。denseの通常ライブラリworkspaceも含み、
小行列での比率を大行列やMNISTへ外挿しない。21サンプルは21独立runではない。
BLOCK_LOCALの別計装stepではforward/loss約15.46ms、backward約19.34ms、optimizer約2.13ms。
これらは完全stepから差し引いたり足し合わせたりしない。forward/backwardの係数再生成が次の調査対象である。

正しさは全819atomのY/dX/dPをFP64公開Fourier参照へ、別の8atomを独立多項式数値積分へ照合する。
optimizerは3つの同一状態FP32更新でParameter・dX・AdamW moment/stepを照合する。
各更新直前にGPUのParameterとAdamW状態をCPU参照へコピーし、GPU軌道だけを進める。
これは自由に走らせたCPU/GPU学習軌道の一致を意味しない。
初回の独立FP64参照軌道はR=4の離散階層境界をFP32と異なるstepで跨ぎ、3更新gateに失敗した。
その負の結果と診断を保存し、profileや境界を変更せず、同一状態からの更新検査へ範囲を明示して変更した。
許容差はParameter最大絶対差2e-5、他5e-4。固定外部勾配を与えた5更新の別検査では、
SGD/AdamW・2 activity mode・有限値検査の有無を含むPolar更新式と状態を照合した。

基準job=`l4job-f4d2b6c0052c4f39943471f95b423a92`、source snapshot SHA256=
`52383a6c03d512a679e11019ef13b042928d578c8af82f4275cdc9937264ad0c`、
result archive SHA256=`d38ef0b45d8225290cc8ade2a9bb357ac0986f9f8b4a43978dfad7463355fe3e`。
native binaryは前節の`0f103d5b…`をsource/runtime照合して使用した。
失敗job=`l4job-8d571176e4ed48438a009467be3b36d4`、
診断job=`l4job-2a60d5dcc160414e83e0e67c48c8e8e6`も
ignored `output/radius-optimize-20261028/`へreceipt・結果・診断ごと保持する。
これらの測定sourceはparent commit `2f9a055`に未commit差分を含み、正確な内容はsnapshot/hashで固定する。

再現コマンド（このfixtureは研究用eager経路で、DB提出／dispatch adapterの対応は別段階）：

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-128-initial.json \
  --polar-update radius --phase-diagnostics \
  --output output/radius-initial.json
```

#### 最適化：M8の定数位相をコンパイル時に畳み込む

`common.cuh::profile_coefficients_m8`はM=8だけを固定サイズにし、17標本とDFTのloopを
展開可能にした。標本のcosとDFT位相は定数で、動的なcos評価とprivate配列のindex処理を削れる。
関数本体はdeviceの`noinline`として、呼出し元ごとの巨大な展開を避ける。
Chebyshev漸化式、scale、k優先の加算順、面積正規化と半径微分は元の処理を維持する。
他のMは元の汎用関数へ進む。Mを増減する、精度を下げる、globalなatom係数表を作る変更ではない。

初回L4 jobでは147テストが成功し、同条件の完全stepはREFERENCE17.666807ms、
HYBRID_SERIAL17.190587ms、HYBRID_OVERLAP17.378848ms、BLOCK_LOCAL16.515928ms、
dense0.719887msだった。全候補のallocated/reservedは前表と同じ。
BLOCK_LOCALは初回同士では約2.11倍高速化したが、まだdenseの約22.9倍の時間を要する。
これは速度目標の達成ではない。後述の同一GPU再比較でも改善を確認した。
host側の1632条件でも汎用版へ、独立高精度多項式oracleへは別の条件集合で係数と微分を照合した。
GPU正しさのgateは前節のY/dX/全atom勾配と、保存値・省略段階・公開optimizerの検査を含む。

job=`l4job-307b65f5ebce410fb92acc73684e4568`、source snapshot SHA256=
`9199cd678a450cd8e8f2953fcb2dd6ab19cd144306b974ba88864ab1883b0360`、
result archive SHA256=`caa47fed24b1f01936843fa6017ee91d35deba7457c1fc557c955e2de58c0a0d`。
native binary SHA256=`31a624b6dd4a9940d6dcc2744ecaedb486305dad631fc5a79aafd1776e34992e`。
このjobのsourceもparent `2f9a055`に未commit差分を含む。正確なsource・binary・結果はhashで区別する。
CPU全体は2,361 passed／5,708 skipped／既存警告18件、wheel/sdistもbuildした。
再現gateは前節の3ファイルに`test_radius_polar_cuda_update.py`と`test_radius_benchmark.py`を加える。
研究fixtureはrevision7の宣言を復元できるが、既存DB／submission adapterは専用対応が済むまで
明示的に拒否する。通常のGraph付きschemaへ読み替えて取り込まない。

**同じL4での再確認：** committed source `bde973ff`で既存runnerをfresh processから実行し、
旧native→新native→新native→旧nativeの順で各2回比較した。
旧nativeには元の`common.cuh`を持つ隔離sourceを使い、両binaryのhashとnative source digestを
照合してロードした。Python・optimizer・fixtureは共通で、測定入口は既存runnerのまま。
初期Parameter・X・target・両軸の半径分布はbit一致し、全819atomの更新後levelも両版とも2だった。

| native / 担当site数 | fresh process 1 ms | fresh process 2 ms | peak allocated / reserved bytes |
| --- | ---: | ---: | ---: |
| 旧 / 32 | 34.216068 | 34.172785 | 291328 / 2097152 |
| M8固定 / 32 | 16.470235 | 16.475394 | 291328 / 2097152 |
| 旧 / 64 | 34.326092 | 34.301078 | 291328 / 2097152 |
| M8固定 / 64 | 16.538569 | 16.488370 | 291328 / 2097152 |

denseは同じ4実行で0.701691〜0.737726ms、peak allocated/reservedは
17313280/23068672 bytesだった。採用候補はM8固定・担当32siteとし、担当64への変更は採用しない。
改善は完全stepで約2.08倍だが、denseに対しては依然約23倍の時間である。
4実行とも独立Y/dX/全atom勾配と同一状態optimizerのgateを通過した。

別計装の新32site版はforward/loss約7.8ms、backward約9.0ms、optimizer約1.8〜1.9ms。
次の候補は`compact_coefficient_tile`内のatom直列走査と、出力CTAごとのmoments/Hの重複生成である。
この段階では各要因の完全stepへの寄与を分離したとは主張しない。
`cuobjdump`ではblock_localのregister数が110→255、stackが1104→1376 bytesへ増えた。
atom勾配kernelもregister92→218。private配列の扱いと並列実行数は引き続き交換条件になる。
Torch allocator peakが同じことを、register・stack・全GPU processメモリが不変という意味にしない。

再確認job=`l4job-073147cd17c64b6bb044d5c9a1022d1b`、snapshot SHA256=
`9a980e08a97c3b1dce14c458de600062a725bf20011103a2dd6026c0d3959c3f`、
result archive SHA256=`f2f2354ddf81ea972735f3802979b40fe6fc3bc390a4362f23ea9c63dcba7a03`。
raw JSON・binary metadata・resource出力・receipt・source snapshot情報は同じignored保全先に置く。
最終wheel/sdistの361 packageファイルはsourceへbyte照合した。

#### 旧cell massから戻す最適化：係数の正の多項式評価

旧`cell_mass_hierarchical/common.cuh`には、標本からDFTを取らず正の多項式から係数を直接求める
実装がある。新半径版の次の候補は、この式をM8に移す。
旧σからbetaを求める式は移さず、現在の半径と周期から
`z=cos²(pi R/P)`、`u=1-z`を求める。旧raw係数へ共通の`z^8`を掛けると、

\[
Q_k(z)=\frac{\mathbf{1}_{k=0}}2 z^8+
\frac4{8+k}\sum_{j=1}^{8-k}A_{kj}z^{8-k-j}(1-z)^j\quad(0\leq k<8),
\qquad Q_8(z)=\frac14
\]

となる。`A_kj`は旧多項式と同じ正の整数定数で、atomに依存しない。
共通のscaleは面積正規化で消えるので、

\[
c_0=1/P,\quad c_k=Q_k/(P Q_0),\qquad
\frac{\partial c_k}{\partial R}=
\frac{Q'_k-Q_k Q'_0/Q_0}{P Q_0}\,
\left[-\frac{2\pi}{P}\cos(\pi R/P)\sin(\pi R/P)\right].
\]

Pは選択bank内で固定して微分する。負のべきやsecを作らず、係数と微分を同じ式から求める。
保存する数学定数は8×8のdouble（512 bytes）のconstant領域だけで、atom係数のglobal表ではない。
privateな`z^j/u^j`は各9要素。M8以外は従来の汎用計算へ進み、odd Mを切り捨てない。
元のDFT版を保存し、旧cell massのtail／σ／bankの制約は新経路へ持ち込まない。

hostの1,632条件で前段との係数・微分を照合し、別の102条件を独立70桁多項式oracleへ照合した。
後者の誤差は`|Δc|P<=2.64e-16`、`|Δdr|P²<=5.82e-15`。
R/Pが0または0.5へ近づく場合も含み、M8以外の差は0だった。
このhost検査と、以下のGPU検査・完全step測定を分けて記録する。

**GPU検証と採用判断：** L4で149テストが成功した。N128・A819のFP32/64について、
追加のmemcheckは0 errors、racecheckは0 hazards（各2ケース）。
同じL4で前段の固定DFT版→多項式版→多項式版→固定DFT版の順に、
既存runnerをfresh processから実行した。各2回、warmup3・21サンプルである。

| 係数の求め方 | fresh process 1 ms | fresh process 2 ms | peak allocated / reserved bytes |
| --- | ---: | ---: | ---: |
| M8固定標本・DFT（前段） | 16.516910 | 16.466935 | 291328 / 2097152 |
| M8正の多項式 | 10.282051 | 10.274260 | 291328 / 2097152 |

denseは同じ4実行で0.725859〜0.752762ms、allocated/reservedは17313280/23068672 bytes。
多項式版をこの半径版のM8候補として採用する。完全stepは前段から約1.60倍速く、
最初の汎用DFT版の約34.2msからは約3.3倍だが、denseの約14倍の時間であり速度目標は未達である。
これはN128/B4の研究fixtureの結果で、MNISTや半径版N1024へ外挿しない。
Graph・全GPU process総量は未測定。数学定数以外のglobal保存表を増やしていない。

`cuobjdump`のblock_local register数は255→130、atom勾配は218→93へ減った。
stackはそれぞれ1376/1112 bytesで前段と同じ。これはcompilerのresource情報で、
Torch allocator peakやGPU process全体のメモリ量とは異なる。
別計装のstepはforward/loss約5.19ms、backward約5.42ms、optimizer約1.95ms。
計装値を完全stepへ足したり差し引いたりしない。

旧版を次に参考にする箇所も明確である。
`cell_mass_hierarchical/contraction.cu`の32atom単位のshared係数再利用と、
`common.cuh::HarmonicStream`の1回のsincosからの位相漸化式は移植候補になる。
各threadが別atomの係数を作り、共有したchunkからmode別にCを集約すれば、
現在の`radius_cell_mass/hybrid.cuh::compact_coefficient_tile`の1 threadによるatom直列走査を減らせる。
これらは未実装。完成responseを使う順序、48KiB予算、保存値のsingle writer、
radius版固有のP・tile窓を保った別の変更として検証する。
旧版のσからのbeta変換、固定tail／bank、globalなwindow-moments追加表は自動的に持ち込まない。

job=`l4job-21b44424ff0e4eaaa751dac050951d80`、committed source=`494c1243`、
snapshot SHA256=`000f4fe707298a6db8ae3962ce69963d47a0832fa79ac179a465cb955dbaa6e2`、
result archive SHA256=`88a02b8038fc5e7bf52a804b0f5fed2d1e5942feb718e4ab599c30458b012d42`。
native module=`torchcst_radius_cell_mass_1ca51a7c912d`、binary SHA256=
`df616996d2e10c1d37f9e40c540421aa839d3812444558afac989163c2f3c64c`。
raw結果・両版のnative source照合・resource情報・sanitizer・oracle・receiptは
ignored `output/radius-optimize-20261028/`に保持する。元のDFT候補もcommit `bde973ff`から復元できる。

#### 次の旧cell mass移植：shared C、位相再利用、分類キーの寿命

2026-10-28。次の候補を実装し、上の10.27ms版をbaselineとして別々に測る。
この段落は実装内容であり、採用性能の結果は後続測定で判断する。

- `HarmonicPhase`は1回の`sincos`から昇順のharmonicを漸化式で作る。
  H/G投影、C、directと階層のsite展開へ使い、セル積分のsincにも位相を再利用する。
  unwrapped座標、P、R、窓、正規化、和を取る順番は維持する。
- Cの各出力nodeを1 warpが担当し、各laneが別atomの係数を作る。
  32atomのchunkを`[warp,basis,33]`のshared列へ置き、各laneは担当modeを元のsorted-slot順で足す。
  M16の33番目のmodeはlane0が追加で担当する。完成Hを使い、Cのglobal表やatomicを追加しない。
  対象出力区間と重なるnodeだけを連続した仕事としてwarpへ配り、粗い階層が1 warpに偏るのを避ける。
- C用sharedは残り48KiB予算から0〜4 warp分を選ぶ。0なら元の1 thread/nodeの処理へ戻す。
  全directやprojection-onlyではC領域を追加せず、以前受け付けた設定を新たに拒否しない。
- BLOCK_LOCALのforwardは分類済みの整数keyをdXが必要な場合だけ保存する。
  backwardはそこから逆向きsort/rangeを作り、atomの再分類とerror wordのCPU読戻しを繰り返さない。
  保存はint64をA個、つまり8A bytes（A819なら6552 bytes）。atom値やFourier係数のコピーではない。
  PyTorchのsaved-tensor version検査で、forward後にatomを変更したbackwardは使用前に拒否する。
  別forwardでは現在のatomを必ず分類し直す。atom-onlyのbackwardはkeyを保持しない。

hostでは位相の9,856条件、出力nodeの割当992,448条件、shared予算147,456条件を確認した。
予算検査では以前有効だった119,414設定を維持し、うち出力あり1042設定はCの元処理へfallbackした。
CPU全体は2361 passed／5768 skipped／既存警告18件。GPU向けには偏ったowner、32/64atom境界、
M1/3/8/16、FP32/64、P>Q、M16のshared fallback、分類の再利用と更新検査を追加した。

**先行候補（対称融合前）の検証結果：** CPU全体2,355 passed／5,671 skipped／既存警告18件。
L4で旧参照と新候補を含む106件成功（宣言等31件、CUDA実行75件）。
入力directのみ／出力directのみ／両方direct／両方階層、全階層direct、非対称軸、
空shape、retained backward、Polar別bounds、非既定stream上の一時tensor寿命を確認した。
N128・A819・B4・M8のFP32/64でも、全3候補のY・dX・全atom列をCPU参照へ照合した。
全direct時にmoments／coeffsが実際に0要素であることも検査した。

同じN128・A819・B4・M8をinstrumented profilerで1回ずつ観測すると、
overlap版にはdirect response（stream7）とcompact moments（stream13）の実行区間の重なりがあった。
serial版の同じ2処理には重なりがなかった。これは同時実行の観測であり、時間の改善率や
完全stepの採用判定ではない。このtraceの先行block_local版はforwardのみ融合し、backwardは分割していた。
上記の対称な融合版とはnative sourceが異なるため、その検証結果として読み替えない。
N128・A819の3候補×FP32/64の6ケースで、memcheckは0 errors、racecheckは0 hazardsだった。

gate job=`l4job-ee6e51862006463780e2b1ac74072cf9`、snapshot SHA256=
`fcb5e52f93a3e764d04b7fc2ab0a2ef7a06dc8246af5ce46f480de747b179f71`。
trace／sanitizer job=`l4job-7bd1ce0eb46b4647862507fcf1500c5b`、snapshot SHA256=
`373fb409d7e4341d512d2b82e1c1ff08a9a08c5fd0300be98700c090c8c74a87`。
同一gate済みnative binary（SHA256 `9d69a3e28a106b2b0c19db430c7e82b2a2bac3424b182879acc24a9db89c26a6`）を
runtime／source照合して再利用した。raw結果・Chrome trace・検査log・source照合は
ignored `output/radius-hybrid-20261028/`へ保持する。
再現gateは`python -m pytest -q tests/test_radius_cell_mass_cuda.py tests/test_radius_cell_mass_hybrid.py`。
完全optimizer step時間・allocated/reserved peakはこの候補では未測定であり、閾値2も未採用である。

**最新の設計指定：** [周期・完成response・階層別tile処理](#周期完成response階層別tile処理の最新指定)を参照。
直前の提案にあった全local階層共通の固定Kや部分responseの受け渡しを、確定規約とは扱わない。

**2026-10-28 追加方針：** 小幅profileの設計条件とコード上の名前は、下の
[support_widthとone-hot近似の設計方針](#support_widthとone-hot近似の設計方針)を正本とする。
これは合意した次の設計であり、既存profile・Plan・測定済み結果を変更したものではない。
現在の最小tile入力4／出力8 sitesではone-hot近似の条件を満たしていない。
さらに、次の公開APIは[半径を主語にする契約](#support_radiusを主語にする公開api方針)を採用する方向で設計する。
以下の既存σ規約とcheckpointを、単なる名前変更でその契約へ読み替えない。

**2026-10-27 目標の訂正：** ユーザーが指定した達成条件は、検査とメモリ条件を保ち、
完全step時間を同条件のdenseの2倍以下にすることである。以前の「denseの2倍の速度、
時間半分が最終目標」という記述は取り違えであり、以下の履歴もこの条件で判定する。
最新の独立再確認では128がdense比約1.26/1.23、1024が約1.67/1.67で、いずれも
同repeatのallocated/reservedはdense以下であり、このfixtureで目標を達成した。
全atomを含む3 step optimizer検査も通過した。異なる幅分布やGPUへの一般化はしない。
追加の補償phase試作は `kernel/cell-mass-phase-prototype`、commit `a3fe9da9` に保存し、
選択設定には採用していない。追加試験は回収・監査済みで、この達成判定の根拠に含めない。

2026-10-27。この文書を、sigmaからの階層分類・layout・Z/H/C・backwardを
つなぐ設計の正本とする。分類規約は下の
[sigmaから階層と周期を決める規約](#sigmaから階層と周期を決める規約)に置く。
他の実測ノートは、その時点の実装と測定の記録であり、この設計が全て実装済みという意味ではない。

数式は[sigmaと周期の規約](#sigmaから階層と周期を決める規約)、処理順は
[起動とkernel内関数の区別](#起動するものとkernel内で呼ぶものの区別)、
実測と選択kwargsは[最新の再確認と検証済み研究設定](#最新の再確認と検証済み研究設定)を参照する。

**階層版は既存ChebyshevSquareの式とsigmaの意味を維持し、Z→H→C→Yへ接続する。**
Zの二分集約木があることと、sigmaから異なるtile/周期を選ぶことを区別する。
既存の固定bank版は比較・復元用に維持する。以前追加した「sigma=全幅、I=sigma/2」は撤回する。

| 項目 | 現実装の設計（追加方針は下記） | 固定bank版 `cell_mass/`（source `7c1404c1`） |
| --- | --- | --- |
| sigmaの意味 | 既存の幅制御値。I=effective_radius(profile,sigma,P) | 同じ幅制御値・半径下限補正 |
| 階層分類 | T_minから2倍ずつ、sigma<=T_hの最小local bankを選ぶ | 未実装。軸ごとに一つのAxis設定 |
| 周期 | local bankごとにP_h=2L_h=3T_h | 軸ごとに固定P=K*T |
| sort | 入出力のbankと中心tileの辞書順 | 中心の(output owner,input owner)だけ |
| Z | bankの基底ごとに直接生成／一致modeの継承 | 固定Pの全modeを同じ基底で二分マージ |
| 小幅の直接計算／全周bankへの自動分類 | 全周bankは分類済み。小幅directは任意の高速化候補 | 未実装。宣言済みK=1の経路はあるが自動分類ではない |
| H/G・Parameter勾配 | 各threadが別atomを担当 | 実装済み、固定bank条件で検証済み |
| C→Y/dX | bankごとにCを作って展開後に加算 | owner Cの部分和と展開は実装済み。複数bankの合算は未実装 |
| 面積正規化・backward | 既存密度・面積式と解析微分を再利用。階層版backwardは独立参照へ照合済み | 固定bankの独立oracle照合完了 |

同じprofile式でも選ぶPが変わるので、固定bankの時間・勾配検証を階層版の結果へ読み替えない。
実装箇所は[`CellMassAxisSpec`](../src/torchcst/kernels/cell_mass.py)、
[`executor.py`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass/executor.py)、
[`kernels.cu`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass/kernels.cu)。
現状と測定条件の入口は[実測状態](cell-mass-status.ja.md)。

### 周期・完成response・階層別tile処理の最新指定

2026-10-28。ユーザーが繰り返し指定していた処理方針を、未確定事項として戻さない。
新しいradius契約について次を設計規約とし、既存CUDAや過去の実測条件とは区別する。

**周期の式：**

\[
 P=2L=\left(\left\lceil\frac{2R}{T}\right\rceil+1\right)T,
 \qquad K=\left\lceil\frac{2R}{T}\right\rceil+1.
\]

Rは`support_radius`、Tはその階層のtileの物理幅。
従来コードではFourier密度の一周を`P=2L`とし、`cos(2*pi*x/P)`を使う。
会話中の「どちらだったか」という確認に対し、この従来表記を説明した。
以降も一周をP（コードでは`period`）、半周期をLと区別して記述する。
ユーザーの訂正により、式は`period = (ceil(2 * support_radius / tile_width) + 1) * tile_width`である。
直前の`+1`なしの記述は撤回する。paddingの1 tileを落としたり、無条件に固定の`P=3T`へ置き換えたりしない。
以降の固定K=3や`(ceil(2I/T)+1)T`の式は、既存実装・先行案の条件として残す。
Lを一周と呼ぶ別の記法へ変更する場合も係数2の対応を明示し、P=2Lと混用しない。

Kはこの式から導かれる整数倍率であり、全atom・全階層について独立に固定したKを仮定しない。
同じ階層でもKが異なる場合、共有momentsの基底周期が一致する単位へさらに分ける。
中心／tile位置、chart端、K個のtile窓の配置と、全周rootの扱いはこの式に従う実装で検証する。
このceil則で得た周期に対するprofileの成立性・漏れ・正規化を確認する前に、
固定P=3dのCPU試算値を新しい則の検証済み結果として使わない。

CPU初版では最小の`T_h >= R`を選び、`K=ceil(2R/T_h)+1`を各atomについて計算する。
Kの偶奇によらず、中心に最も近いtile-alignedなK-tile窓を選ぶ。
原点を除いてchart内へwrapした中心をu、間隔をdとすると、開始tileは
`floor((u+d/2)/T_h - (K-1)/2)`（tieは大きい開始tile側）。
窓中心とatom中心の距離は最大T_h/2であり、`P/2 >= R+T_h/2`なので半径Rの区間を窓内に収められる。
Pがchart周期Qを超えても式を切り詰めず、一周Pのセル積分をsite index modulo Nへ加算する。
これはchartを一度だけ走査して切る処理とは異なり、繰り返し像も含めて確率和1を保つ。
hierarchy・K・開始tileはbackwardのsmooth branch内で固定する。
切替点そのものの微分や、近似された裾の窓切替が滑らかだとは保証しない。

**responseは必要範囲を全て加算して完成させてから使う。**

- `atom_response` Hは、そのatomの必要な全入力tile／siteについて足し終えた値とする。
- backwardのGも、そのatomの必要な全出力tile／siteについて足し終えた値とする。
- coeffs生成・出力への直接展開・Parameter勾配へ、未完成のresponseを渡さない。
  direct・局所的な経路も同じ規約を使う。
- threadやblock内で加算途中の値を持つことはあるが、それは内部の集計途中状態であり、
  公開するresponse bufferや段階間の受け渡しには使わない。

これは完成responseを必ずglobalに保存する規則ではない。
全寄与の集計後にregister／sharedからcoeffsやYへ使う融合経路も認める。

**今後のコード・説明で使う名前：**

| 数式 | forward | backwardの対応 | 意味 |
| --- | --- | --- | --- |
| Zx / Zy | `input_moments` | `grad_output_moments` | X／dYをtileの各基底へ集約 |
| H / G | `atom_response` | `grad_response` | X／dYを各atomのprofileへ投影した完成値 |
| C_out / C_in | `output_coeffs` | `grad_input_coeffs` | Y／dXへ展開する基底係数 |

H/Gはいずれも振幅を掛ける前の値。`grad_response` GはdYを出力profileで集約した値で、
H自体の勾配は`amp * G`である。振幅勾配は`sum_batch(H * G)`。
Parameter勾配では入力／出力のprofile微分への投影も使う。dXだけの逆向きpipelineと区別する。
既存σ版のsource名・native ABIを直ちに一括変更する規則ではない。

forward内で使うmoments・response・coeffsはregister／sharedでの局所完結を基本候補とする。
backwardへ残す値はgroupごとに、完成response保存／moments保存からresponse再計算／Xから再計算を選ぶ。
responseは振幅勾配・出力中心勾配で再利用するが、入力中心勾配にはresponseだけでは足りない。
同じmomentsを複数blockが生成する場合はglobalへの保存担当も定義する。
保存精度は現時点で固定しない。FP64で集計することとglobalへFP64保存することを区別する。
全momentsと全responseを常に両方保存する前提にはしない。

**担当はradiusから決まる階層とtileでまとめ、sortして割り振る。**

radiusの生値ごとに別々の小groupを作る意味ではない。階層・周期・入出力tileなど、
同じ基盤を共有できるキーでatomをまとめる。大きさ別に処理方式とデータ移動を分け、
完成responseの再利用を近づける。元atomの値は複製せず、sortされた元行indexとgroup範囲を使う。
周期を区別する必要がある場合のキー候補は`(h_out,h_in,K_out,K_in,q_out,q_in)`。
同じ入力窓のmoments共有と、同じatomの完成Hを複数の出力先で使うことは区別する。
異なるatomが同じH値になるとは仮定しない。
blockへ渡すtile群には、そのatomのresponseを完成させる全依存範囲を含める。
group全体が一つのblockに収まることは要求せず、各blockへatom／batchの担当を分けても
response完成前に段階をまたがない規約を守る。

### support_widthとone-hot近似の設計方針

2026-10-28合意。σという制御値と実際の集中幅を区別する。コードでは次の名前を使う。
半径の公開SpecとCPU参照を実装した。新しいCUDA経路とdirect／融合の実装状況は冒頭の節を参照する。

| 名前 | 意味 |
| --- | --- |
| `support_width` | 確率の`1 - support_eps`以上を含む、中心まわりの最小区間の全幅 |
| `support_radius` | `support_width / 2`。同じ物理座標単位の半幅 |
| `support_eps` | 選んだ幅の外に漏れる確率の総量。Mなどのprofile設定から導出し、必要ならその上限で報告する |

**2026-10-28の最新方針：Mを設定し、その設定で実現するsupport_epsを求める。**
99%／99.9%などのεを一律の必須条件として固定し、Mを自動的に増やす設計にはしない。
Mは計算量・保存量と集中度の調整項目とする。現在のprofileではM自体が数学的な関数を変えるため、
別Mは同じ固定関数の丸め精度だけを変える設定ではない。
最小幅では評価する区間幅を`support_width=spacing`とし、実現する中心セル確率を報告する。
Mだけで一意に決まるのではなく、σ・周期P・profile設定・評価幅を揃えて比較する。

非負のprofileを一周期Pで面積正規化した密度をf、中心をμとし、
中心からの周期距離で区間を取る。運用上はwを指定し、漏れを次で求める。

\[
 \varepsilon(M,\sigma,P,w)
 =1-\int_{-w/2}^{w/2}f(\mu+t;M,\sigma,P)\,dt.
\]

逆にεから対応する幅を調べる場合、`support_radius`は次の最小のrになる。

\[
 r_\varepsilon=\min\left\{r\in[0,P/2]:
 \int_{-r}^{r}f(\mu+t)\,dt\ge1-\varepsilon\right\},
 \qquad w_\varepsilon=2r_\varepsilon,
 \qquad\varepsilon=\texttt{support_eps}.
\]

区間外の**合計**を表す量であり、各siteにεずつ漏れるという意味ではない。
有限Fourierの非負profileでは周囲の区間を全て厳密な0にはできないため、
この名前のsupportは有効支持区間を意味する。セル積分と面積正規化は維持する。
次節のCPU調査に残す`ε=1e-3`の逆算は過去の試算であり、採用条件ではない。
Mの選択では、漏れに加えて振幅・入力・寄与の重なりと出力／勾配への影響も見る。公開既定値は未変更。
既存の`tail_bound=1e-3`とは同じ量ではない。
既存の`tail_bound`はprofile値の裾の条件で、区間外の確率積分を直接指定しない。
既存の集中半径`I=effective_radius(...)`とも区別し、Iをそのまま`support_radius`と改名しない。

#### 最小幅と階層・directの使い分け

- 最小幅では`support_width <= spacing`を満たせることを設計条件とする。
  中心がsite中央にあるとき、そのセルの確率は`1 - support_eps`以上となる。
  中心がセル境界に動いたときは隣接セルへ滑らかに分配する。全ての中心位置でone-hotとはしない。
- 表現・分割の最小単位は1 siteとし、合体数を指定できる階層を考える。
  最初の比較は2子ずつまとめる`1 -> 2 -> 4 -> 8 -> ...`を基準とする。
  全1-site nodeについてFourier基底表をglobalに生成することを要求しない。
  数学的なprofile周期と、blockが処理するtile／chunkの大きさも区別する。
- 集約が割に合わない範囲ではXから直接Hを作る。再利用が得になる範囲ではZからHを作る。
  どちらも`H[b,a] = sum_i X[b,i] * p[a,i]`を計算し、Hを出力への寄与に使う。
  Hを論理的に経由することは、全Hをglobal Tensorへ保存することを意味しない。
- 同じ入力窓を使うatomをまとめ、3タイルの集約からH生成までを同じblock内で処理する候補を持つ。
  集約値の共有はshared、各threadのHはregisterを基本候補とし、global Rxを必須としない。
  blockをまたぐ出力集計と、backward用に残すZ／X等の寿命は別途設計する。
- directと階層版は同じセル確率・正規化・微分を使う。同じprofileをdirect評価しても集中幅は変わらない。
  `support_eps`だけからY・dX・Parameter勾配の誤差上限が決まるとはしない。
  外側を0と近似する場合も、その近似を含む演算と勾配を明示して独立検証する。
- directへ切り替えるsite数／再利用量は、同じ数学条件の完全step時間とallocated／reservedで決める。
  batch・atom分布・group内のatom数も測り、σだけで固定的に担当thread数を決めない。

#### support_radiusを主語にする公開API方針

2026-10-28のユーザー指定。新しいsupportベースの契約では、公開の幅を`support_radius`とし、
`support_width = 2 * support_radius`を併せて扱う。σを全profile共通の物理半径とみなさない。
Mと半径・周期から得られる漏れを`support_eps`として報告する。εからMを自動的に増やさない。

| profile | 新APIで扱う幅と範囲 |
| --- | --- |
| Triweight等のcompact profile | `support_radius`は厳密に0となる境界までの半径 |
| 有限Fourier profile | `support_radius`は評価する有効支持区間の半幅。外の確率は`support_eps`で報告 |
| Gaussian | 標準偏差σを維持。有限半径を自動付与せず、今回の有限support階層最適化の対象外 |

Gaussianをライブラリから削除する方針ではない。既存の正確な参照経路を維持する。
Gaussianを有限範囲へ打ち切る別契約を将来追加する場合は、その近似と勾配を明示する。
compactでも旧σと半径の換算はprofileごとに確認する。現TriweightではR=σだが、
QuadraticBSplineではR=sqrt(6)*σであり、全profileで同じ数値へ改名できるわけではない。

新APIの設計名は`RadiusBounds(minimum, birth, maximum, upper_floor)`、
`input_radius_bounds`／`output_radius_bounds`とし、revision7のCPU APIへ実装した。
値は入力・出力／各軸の物理座標単位の半径。最小有効全幅をspacingにするなら
`radius_min = spacing / 2`であり、旧`σ_min=spacing`と同じ数値ではない。
幅の意味は数学的なSpecとcheckpointの識別情報へ残し、CUDA Recipeだけの設定にしない。
同じprofileのTorch/CUDA・direct/階層は、その半径の意味とセル確率を共有する。

**新しいCPU参照のFourier密度：**
旧`effective_radius(σ,P)`を通さず、Chebyshev式の集中半径IへRを直接渡す。

\[
 \beta=\sec(\pi R/P),\qquad
 g_R(t)=\left[\frac{T_M(\beta\cos(\pi t/P))}{T_M(\beta)}\right]^2,
 \qquad f_R(t)=\frac{g_R(t)}{\int_{-P/2}^{P/2}g_R(s)\,ds}.
\]

旧σからの集中半径下限補正はこの式には含めない。
新しい周期選択では`P>2R`。scaled Chebyshev recurrenceで密度を作り、最高harmonic Mの
密度を2M+1点からFFTすることで奇数Mにも対応した。旧偶数Mのprofile宣言は変更しない。
`support_eps = 1 - integral_{-R}^{R} f_R(t) dt`をM・R・Pから求める。
同じ有限Fourier密度なので、セル積分のatom係数／site基底への分離は維持できる。
`R=spacing/2,P=3*spacing`のCPU数値積分では、中心siteの確率は
M6/8/10で約0.988407/0.998506/0.999815になった。
M8でも約99.85%という結果は、以前の`σ=spacing`の結果とは異なる幅契約によるもので、
同じ演算の高速化や、既存CUDAの検証済み結果とは扱わない。
256/512点の独立Gauss–Legendre積分と3セルの確率和1を確認し、
式・条件・script hash・結果をignored `output/support-design-20261028/radius-candidate*`へ保存した。
固定P=3dの試算は、訂正後の周期選択そのものの検証ではない。
最新則で`R=d/2,T=d`なら`P=2d`となり、中心site確率はM3/5/8で
約0.981545725/0.999276680/0.999995417。これも厳密に指定条件を揃えて解釈する。

Polarの(s,t)から振幅・activityを出す更新則は維持する。
新契約ではactivityから決める幅の出力をRとして宣言し、そのboundsとprofileを同じ意味で接続する。
これは旧σの数値を黙って半径と読む変更ではなく、新しい数学的な契約／revisionである。
既存モデル・optimizer state・旧σ版は比較・再現用に残す。Parameter／momentの読み替えを伴う
checkpoint変換は、必要になった場合に明示的な変換として独立に検証する。
全σ名の一括置換や新API対応前の既存selectorへの自動採用は行わない。

半径宣言・checkpoint識別とTorchのセル確率・勾配参照を実装した。
公開READMEにはCPU参照としてのみ使用例を載せる。既存CUDA PlanとCUDA上の汎用Torch実行も
この初版の新契約を拒否する。次はCUDA対応・独立勾配gate・完全step／メモリ比較へ進める。

#### 合体数・Z開始階層・周期倍率の明示設定（2026-10-28合意）

以下の固定K設定例は先行提案である。
最新指定の`K=ceil(2R/T)+1`が優先し、固定倍率で同じ基底を共有する説明はKが一致するgroup内に適用する。
偶数KやK=1を旧実装だけの制約で新契約から除外せず、最新の窓規約として検討・検証する。

次の3項目は明示指定できる設計とする。以下は設定の擬似コードであり、新しい公開APIの実装ではない。
Mはprofile側で別に指定する。入力・出力それぞれに適用する設定として扱う。

```python
hierarchy = {
    "base_tile_sites": 1,
    "merge_factor": 2,       # 親を作る子タイル数。2に固定しない。
    "period_tiles": 3,       # local周期 P / tile幅 T。全local階層で同じ整数倍率。
}
compute = {
    "z_start_level": "auto", # 0以上の整数で固定もできる。例: 2。
}
```

各軸のsite間隔をd、最小site数をn_0、合体数をr、周期倍率をKとすると、

\[
 n_h=n_0 r^h,\qquad T_h=d\,n_h,\qquad
 P_h=2L_h=K T_h.
\]

`merge_factor`は2以上の整数、`period_tiles`は正の整数の定数倍率として扱う。
例としてr=2なら1/2/4/8 sites、r=3なら1/3/9/27 sitesとなる。
現在のwhole-tile窓で中心tileと左右同数を保持する初期実装では、localのKは3以上の奇数から対応する。
K=1は全周を覆うrootの特例とする。偶数Kを扱うなら先に窓の配置規約を定義する。
Kを変えても旧`σ<=T`分類をそのまま適用できるとはしない。profileの集中範囲と窓の対応を再確認する。
最後にlocal周期がchart周期Qへ収まらなくなる場合は、P=Qの全周rootを別に定義し、
周期だけを黙って切り詰めない。

**z_start_levelはZを使う演算を始める階層であり、下位階層のatom計算を省く指定ではない。**
例えばr=2・`z_start_level=2`なら、1/2-site階層のatomはXからdirectでHを作り、
4-site以上の階層では生成したZを利用する。各atomの数学的な所属・P・profile係数は同じままにする。
上位Zのために下位Zが必要な場合も、開始階層をXから直接生成する方法と、
必要な子だけを作る方法を選べる。未使用の全1-site Z表を常に確保する設計にはしない。
出力側の小幅profileも同じ規約のセル積分で直接展開し、入力側のdirect指定からCの保持方式を勝手に変えない。

自動選択の候補は、M、batch数、階層／ownerごとのatom数、対象site数、入出力での再利用、
Z／Cの保存量を使って見積もり、同条件の完全step実測で較正する。
Mが大きい小タイルでは基底生成・保存が割高になり得るが、Mだけでは再利用量は分からない。
`auto`が解決した開始階層は実行計画と測定結果へ残し、整数指定で再現・比較できるようにする。
既定selectorやGPU計測による自動探索を実装済みとは扱わない。

**周波数が一致するmodeだけ継承する。** Kが全階層で一定なら`P_parent=r*P_child`なので、
親harmonic kを子から継承できるのはkがrの倍数の場合で、子harmonicはk/rになる。DCも継承できる。
それ以外のmodeはその親区間について直接生成する。子タイルをr個足すだけで
全modeの親Zが完成するとはしない。原点・周期端の位相規約も揃える。

`merge_factor`やKを変えることでatomの所属周期が変わる場合、その設定はprofileを含む数学的宣言側に置く。
`z_start_level`やthread配分は実行Recipe／解決済み計画側に置き、変更してもYと正しいbackwardは同値とする。
これにより、GPUごとに開始階層を変えても学習対象のprofileを変えずに済む。

現CUDAはpower-of-two／2子マージ・K=3または全周root・既存presetの開始規則で動いており、
上の一般化は未実装。r=3などを指定しても黙ってr=2へ丸めない。
site数をrで割り切れない場合の端タイルは追加設計が必要で、対応前はその組合せを明示的に拒否する。
例えばN1024でr=3の多階層を使えると現時点で保証しない。

#### 現実装との違いと、次に決める順序

現M8・最小tile入力4／出力8 sites・σ_min=spacingでは、中心がsite上にある場合の
中心セル確率は入力約0.57284、出力約0.30901であり、one-hot近似を保証しない。
これはCPU FP64セル積分参照の確認で、GPUの性能測定ではない。
ignored `output/cell-mass-hc-20261028/profile-width-check.json`に全セル値・条件・source hashを保存した。

1. M・σ・Pと評価幅を明示し、導出される`support_eps`、中心位置の移動・周期端でのセル確率を調べる。
2. 新しい半径契約では`support_radius=spacing/2`を最小幅の基準に、
   Mごとの集中度・保存量・演算量を比較する。数学的な関数を変更する設定は明示する。
   旧σ契約の意味とPolarの更新則は保存し、新契約へ黙って読み替えない。
3. 指定合体数の階層と最新の`P=2L=(ceil(2R/T)+1)T`規約に従い、radius分類とtile窓の実装・検証を進める。
   最初の比較は合体数2。既存の固定`P=3T`と`σ<=T`を新しい規約へ混用しない。
4. 同じprofileのdirect／階層経路とRxをglobal保存しない融合候補を比較し、
   Y・dX・全atom勾配の検証後、batchを揃えた単層stepとMNIST完全stepで速度・メモリを測る。

名前と集中条件・実行方針は合意済み。次節にεと最小幅profileのCPU検証済み候補を示す。
階層への対応式・実行方式の切替点・新しい高速経路は未確定・未実装。
以降のσ分類式と実測表は、既存profile設定の実装・履歴として保持する。

#### 最小幅の具体候補とCPU検証（2026-10-28）

最初の試算では`ε=1e-3`を**各軸profile**に指定して逆算した。
最新方針はMからεを求める向きであり、この値を全Mへ要求しない。
既存ChebyshevSquareのσ・半径補正・面積正規化はそのまま使い、
最小の数学tileを入力／出力とも1 site、`T_min=d=spacing`、`P_min=3d`にする候補である。
`sigma_min=d`とPolar更新則は変更しない。

セル積分は既存Fourier係数から厳密に評価できる。生密度を
`g(t)=c_0 + 2 sum_{k=1}^M c_k cos(2 pi k t/P)`、正規化係数を`P*c_0`とすると、
中心まわりの幅wに含まれる確率は次になる。

\[
 F(w;\sigma,P,M)=\frac wP+
 \frac2\pi\sum_{k=1}^M\frac{c_k}{k c_0}
     \sin\left(\frac{\pi k w}{P}\right).
\]

主に報告する量は`support_eps = 1-F(support_width)`である。
逆算では`F(w)>=1-eps`となる最小wを求められる。
このCPU調査ではwを二分探索したが、各GPU threadが毎回二分探索する実装を提案するものではない。
既存式では`σ=T_min=d`のとき`I=d`、`beta=sec(pi/3)=2`。
そのためこの最小幅での分布は`tail_bound`によらず、Mと`P/d=3`で決まる。

最小幅`σ=d,P=3d,support_width=d`で得られる漏れは次のとおり。

| M | 中心セル確率 F(d) | 導出されるsupport_eps |
| --- | ---: | ---: |
| 8 | 0.983139494 | 0.016860506 |
| 10 | 0.992182317 | 0.007817683 |
| 12 | 0.996328216 | 0.003671784 |
| 14 | 0.998259056 | 0.001740944 |
| 16 | 0.999168538 | 0.000831462 |

次はε=1e-3を指定して幅を逆算した比較であり、採用の合否表ではない。

| M（密度の最高harmonic） | 中心セル確率 F(d) | support_width / d（ε=1e-3） | 試算上の99.9%条件 |
| --- | ---: | ---: | --- |
| 8 | 0.983139494 | 1.326919440 | 不成立 |
| 14 | 0.998259056 | 1.046529146 | 不成立 |
| **16** | **0.999168538** | **0.985760957** | **成立** |
| 18 | 0.999600612 | 0.934395316 | 成立。現nativeの上限16の外 |

M=4..64の偶数を調べた範囲では、ε=1e-2/1e-3/1e-4/1e-5を初めて満たすMは
それぞれ10/16/22/30だった。これは`σ=d,P=3d`という候補に対する結果である。
M16・1-siteはε=1e-3についてCPU検証した例であり、M16やM10を採用する決定ではない。
全階層をM16にして高速になるという判断や、現在のM8 presetへの自動置換は行わない。
input/outputの両profileに各εを指定した場合、中心位置が両方site上でも
2次元積の中心セル対の確率はこの候補で約0.998337767である。
積全体にε_totalを要求する場合は`(1-eps_in)*(1-eps_out)>=1-eps_total`を別途満たす必要がある。

独立検証では、係数／FFTを使わない生Chebyshev密度のGauss–Legendre積分と照合した。
128/1024 sites、spacing=1/0.125/0.37、原点の移動、中心のセル内移動・境界・torus端を含む21ケースで
確率和1・丸め誤差内の非負を確認。セル積分の最大差は約1.12e-15、積分次数256/512間の差は約3.78e-15。
中心がセル境界なら隣接セルの値は各0.5となる。中心微分のautogradと独立積分の有限差分は
spacingを掛けた絶対差で約1.03e-9以内。CUDAのY/dX/全atom勾配検証を完了したとは扱わない。
script・全条件・結果・source hashはignored `output/support-design-20261028/`に保存した。

**εはampを含む演算誤差でも評価する（2026-10-28追加）。** M16は99.9%という試算条件を
満たす候補であり、全階層の採用値ではない。同じ最小周期でM8/12/14の中心セル確率は
約98.314/99.633/99.826%。許容漏れを小さくする前に、その漏れが実際の振幅でどれだけ効くかを調べる。
one-hotへの入力投影の比較では、中心セル外の確率がε以下なら、各atomについて

\[
 |H_a-X_{i_a}|\le\varepsilon\max_i|X_i-X_{i_a}|.
\]

振幅を掛け、同じ非負・和1の出力profileへ展開する場合、この入力側差の出力L1ノルムは
`abs(amp_a)`倍以下になる。出力profile側の漏れと複数atomの集積は別途含める。
これは近似を導入した場合のtask勾配の精度や、別Mの二つのprofileの誤差を自動保証する式ではない。

同じN1024の凍結epoch3 checkpointをCPUで読み直すと、`abs(dense W)`の中央値／最大は
約0.028686／0.450348、RMSは約0.052644。一方`abs(atom amp)`は中央値約0.798768、
最大約1.137184、RMS約0.801817で、約31.4%のatomが絶対値1を超える。
面積正規化profileでは`W[o,i]=sum_a amp[a]*u[a,o]*v[a,i]`であり、atomのampとdenseの1要素は
同じ尺度ではない。今回のamp上限は既存初期化で実効WのRMSを1/sqrt(N)へ合わせて較正した値。
新しい狭いprofileで同じW尺度を比較するときは初期振幅の較正もやり直し、
旧profileのamp分布をそのまま将来の一般則にしない。Polar更新則自体は維持する。
checkpoint SHAと分位値はignored `output/support-design-20261028/amplitude-scale.json`に保存した。

**以下は先行する固定K案の診断である。最新のceil規約は冒頭の指定を使う。**
現行の`σ<=T`の境界を越えてPを2倍にすると、
同じMでもprofile自体が変わる。M16でσ/dが1の直前／直後の場合、
中心セル確率は約0.999169から0.972562へ、`support_width/d`は約0.985761から1.423331へ跳ぶ。
これはdirect／階層という同じ演算の実行方式の切替とは別であり、
`support_width`がσに滑らかに対応する設計として分類まで確定したとはしない。
新実装では最新のradius・ceil規約に対して成立性と値を検証し、この固定K案の診断をそのまま流用しない。

**保存量の制約：** 1-site階層をそのまま全global表にすると、N1024では各軸2045 nodes。
M16の33基底をB32・FP64で全保存するとZだけで各軸17,276,160 bytes（約16.476MiB）になる。
これは容量計算でありGPU peakの実測ではない。最下位はXそのものを使うdirect候補とし、
集約が得になる階層から必要なZだけを生成する。全node表を確保してmaskで演算を省くだけでは
この保存量は減らない。binary階層の概念と、実際に生成・保存するnodeを区別する。
最小の3セル窓をdirect評価してHを作る場合も、小さい裾を含めた同じ3セル確率を使えば面積和1を保てる。
εを理由に周辺セルを黙って切り捨てたり、Hを単純なXのコピーに置き換えたりしない。

### 実装の配置と、新しい階層版の段階

固定bank版の`cell_mass/`は移動・削除せず、既存テスト・結果の比較先に残す。
新しい実装は[`cell_mass_hierarchical/`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass_hierarchical/)
へ分ける。固定版executorへの暗黙のfallbackは使わず、同じprofile数学を明示的に再利用する。
同じ研究branch/Draft PRで進め、復元可能な旧sourceは`8f343c16`以前に残る。

| 新しい実装 | 処理と保存するもの | 状態 |
| --- | --- | --- |
| `plan.py` | 2倍ずつのlocal bank・全周bank・Z node位置。Tensorを持たない実行計画 | 実装済み、CPU分類検査済み |
| `executor.prepare_layout` | 分類とsortまたはcounting。元行番号、groupの(h_out,h_in,q_out,q_in)、offsets、任意の逆slot indexをcacheへ残す | CUDA実装済み、元行・group範囲を実GPU照合済み |
| `preparation.cu::build_hierarchy` | bank別の周期のZ。直接生成と偶数harmonicの子からの継承。完成Zはglobal FP64 | CUDA実装済み、direct/mergeを独立積分参照へ照合済み |
| `contraction.cu::compute_h` | 各threadが別atomのHをFP64計算し、指定dtypeで保存。係数と面積はregister | 実装済み、既存profileの独立参照へ照合済み |
| `compute_c` / `expand_c` | 32atom shared chunkからowner Cを作り、bank別の基底で展開後にYへ加算 | 実装済み、CとYを別々に照合済み |
| G/dX・Parameter勾配 | 同じprofile微分と選択中のlayoutを使う。各threadが別atomを担当 | 実装済み、全atom勾配とAdamWを実GPU照合済み |

新しい前半は空の雛形でなく、`prepare_layout(plan, atoms)`と
`build_hierarchy(signal, axis, method="merge")`を実行するCUDA入口。
直接生成だけの`method="direct"`も比較に使える。
研究用の`forward_only`はprepare→Z→H→C→Yを呼び、detached入力の診断用として残す。
新しい`hierarchical_linear`は明示的なautograd bindingを持つ。
PR #121の時点では公開CSTLinearへ未接続だった。現在の接続作業は次節を参照する。

### CSTLinearへの明示接続と学習確認（2026-10-27）

#### Polar接続（実装・GPU検証済み、幅範囲を訂正）

`presets.polar_cell_mass`は既存`PolarAmpWidthSpec`・初期activity指定・
`polar_activity_width`更新方針をそのまま使い、profileだけをセル積分/面積正規化へ置き換える。
atomの行は既存chart-pair規約の`[s,t,input_center,output_center]`。
Polar更新の実コード・gradient projection・AdamW moment輸送は変更せず再利用する。
CPUでは旧Polarに同じ勾配を渡した3 stepのParameter/moments/stepがbit一致した。

CUDAのAxis metadataに振幅上限・幅曲線のscalarを渡し、各threadのregisterで
振幅とinput/output sigmaを取り出す。元atomを直接読み、物理atom表や全係数表をglobalへ
複製しない。Polar decodeはFourier評価に合わせてFP64、特異原点のtinyは元dtypeを使う。
Zの二分階層・H/G/Cの保存方式・cell-mass profile自体は維持する。
parameter_gradは振幅勾配を(s,t)へ戻し、幅のtask gradientは既存どおり通さない。
center勾配は元のinput/output順に書く。旧物理sigma経路も比較用に保持する。

Polarのtask pullbackはZx/ZyからH/Gとcenter微分を同時にFP64 registerで再計算する。
Gだけを入力dtypeへ丸めてdX用に渡し、パラメータ勾配には丸め前の値を使う。
forwardのHはC生成後に解放し、backwardまで保存しない。FP64の全H/G表も追加しない。
幅stop-gradientなので、Polar専用pullbackは不要なbeta/sigma微分を計算しない。
物理sigma版の保存Hとwidth微分経路はそのまま残す。

各軸の幅上限はその軸のQ/3以下で宣言する。input/outputの別幅も使える。
幅が違う場合は共有sigmaを仮定するcompact bank省略を使わず、完全Cartesian countingまたは疎sortを使う。
shape係数の再利用は周期・profileに加え実際のsigmaが一致する場合だけ行う。

MNIST再確認は前回と同じ全データ・seed120・B32・3 epoch・128/1024の中央層置換。
Polarではsigmaを独立にUniform(2,40)へ置けないため、初期角度を乱数方向、
alpha=.5（q=2.5）、中心をUniform(0,128)と明示する。
幅の設定はminimum=2/birth=16/maximum=40/upper_floor=4、kappa=lower_kappa=3、
upper_decay_power=1。更新は既存の既定radial_regularization=.1/activity_gain=1/
finite_chord/dormant_expansion_rate=0。AdamWはlr1e-3/weight_decay0。
初期振幅の単位を調整するためamplitude_maxとw_c=.3*amplitude_maxを同率で拡大し、
実効WのRMSを1/sqrt(width)にする。w/w_c比と初期sigmaは丸めを除いて維持する。
これは既存Polarが受け付ける明示atom初期化とscalar設定であり、新しい更新則ではない。
denseにはこのPolarモデルの初期実効Wを渡し、外側の層・各epochのshuffleも一致させる。
以前の物理sigma版とは初期分布・学習則が違うので、時間の直接的な改善率には読み替えない。
この固定座標幅の先行条件ではCPU全2,047 passed/5,362 skipped。GPU結果と設定訂正を以下に分ける。

初回job `l4job-32efe4ab7d7c4fea97da7aeacf943fbb`は別幅8ケースで失敗した。
C生成のCartesian経路にも共有sigmaを仮定したbankペア省略が残っていたため、
別幅ではその省略を行わないよう修正した。次のjob `l4job-6f0f31bcb6884cb78312cd350c7455d5`で
全2,642 passed/12 skipped、racecheck 0 hazardsとなったが、全819atom/B32の勾配検査で
3/3,276要素がrtol3e-6/atol2e-7を超過（最大絶対差5.9084e-7）した。
許容誤差は維持し、保存FP32 H/Gを使うPolar pullbackを上記FP64 register再計算へ置き換えた。
いずれの試行もMNIST開始前に止まり、学習結果や速度の根拠には使わない。
両失敗runのsource/results/spec/receiptはignored `output/cell-mass-polar-integration-20261027/`
の`failed-split-preflight/`と`failed-rounded-gradient/`へ保存した。

#### σ範囲とsite単位の最小tile（2026-10-28訂正、現在の学習条件）

ユーザー指定は`σ_min/spacing <= 1`、`σ_max ≈ 100 σ_min`。
過去の[小幅混合の記録](research-history/local-product/20261004-subspacing-mixtures.md)にも
`σ_min < spacing`と0.25の例がある。上の固定幅2..40はこの条件を満たさなかった。
最新の指定により`σ_min/spacing = 1`を選ぶ。Qは座標空間の一周の長さで、Q=N*spacing。
現在の実装の許容上限Q/3を適用し、128では希望の100倍より小さくなる。
既存Polarの更新則・sigmaの意味を維持し、今回のMNIST harnessで次を明示設定する。
ライブラリの一般的なPolar既定値を変更するものではない。

```text
spacing = Q / N, Q = 128
σ_min = spacing
σ_max = min(100 * σ_min, Q/3)
σ_birth = 8 * σ_min, upper_floor = 2 * σ_min
input T_min = 4 * spacing, output T_min = 8 * spacing
```

| N | spacing | σ_min | σ_max | 入力/出力T_min（物理幅） |
| --- | ---: | ---: | ---: | ---: |
| 128 | 1 | 1 | 約42.666664 | 4 / 8 |
| 1024 | 0.125 | 0.125 | 12.5 | 0.5 / 1 |

FP32 scalarへの丸めでQ/3を超えないよう、非表現可能な上限は下向きに丸める。
0.25spacing版はsource `7448590f`、job `l4job-7eee0a834b874bb0952e92f49857890e`へ
固定投入した比較条件で、今回選択する1spacing版とは分ける。
そのrunは24 passed/8 failedで学習開始前に停止した。sort経路の入口でreverse指定を
prepareへ渡し忘れ、dXが必要とする逆indexが欠落したためである。今回その伝達を修正した。
0.25版の速度・学習結果は存在しない。source/results/receiptは
`output/cell-mass-polar-site-scaled-20261027/failed-missing-reverse/`へ保存した。
fixtureの実際の幅設定を保存し、再読込時は当時の`kernel_settings`から復元する。

T_h=2^h T_min、local bankのP_h=3T_h、σ<=T_hの最初のbankという分類は維持する。
1024でも最小tileを4/8 **sites**にする。以前の32/64sitesを残したままσだけ縮めると、
ほぼ全atomが最下位bankへ入り、profileの集中半径下限も相対的に大きくなるためである。
T_minはprofileの数学的宣言なので、この変更は同じ演算の高速化だけとは扱わない。
M8/tail1e-3の集中半径補正は変更しない。σ<spacingでも1siteへの完全な集中は保証しない。
診断に`σ/spacing`と`effective_radius/spacing`を併記する。

新しい明示Plan `SITE_SCALED`は疎sortと入力owner用の逆slot indexを使う。
旧`VALIDATED_128/1024`は前の条件の再現用に維持する。入力509/output253 nodesとなる
1024では完全Cartesian countingの4096group上限を超えるため、実在するgroupだけを保持する。
元atomは複製せず、順slot→元行、group範囲、逆slot/offsetだけがforward/backward用global cache。
分類keyと逆index構築中のnode配列は一時scratchでありcheckpointには入らない。
`atoms_per_split=256`のragged Cを使い、実測でメモリ・速度の両方を確認する。
修正後のsource `636c890c`、job `l4job-31211e59eb5c41b08ce8813e60ab7141`で
GPU検証とMNIST再学習が完了した。L4/Torch2.11.0+cu130/CUDA13.0、33+2,647 passed、
racecheck 0 hazards。B32の全819/52,428atomで独立FFT/autogradのY/dX/全4列勾配、
3stepのParameter/AdamW moments/stepを従来と同じ許容差で通過した。
CPUは2,050 passed/5,384 skipped/既存警告18件、PRのCPU CIも通過した。
wheel/sdistそれぞれの全346runtimeファイルはGPU snapshotとbyte一致した。

全MNIST60,000/10,000件、3epoch/B32/seed120、784→N→N→10の中央層のみ置換。
同じ初期実効W・外側層・data順、TF32/Graphなし、fused AdamW lr1e-3/wd0。
1条件につきfresh process一つであり、複数seedの独立反復ではない。
時間はgather/loss/forward/backward/実optimizerを含み、評価・診断・checkpointを除く。

| N | epoch3精度 CST/dense | 平均step CST/dense | 時間比 | allocated MiB CST/dense | reserved MiB CST/dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 97.02 / 97.11% | 12.022 / 1.912ms | 6.286 | 213.301 / 213.264 | 366 / 364 |
| 1024 | 97.36 / 97.84% | 20.935 / 1.926ms | 10.867 | 242.150 / 239.883 | 396 / 382 |

常駐MNISTデータ込みのTorch allocator peakで、GPU process全体の使用量ではない。
**この実学習条件では完全step時間・allocated/reservedの全目標が未達。**
1024のallocatedはdenseより2.267MiB、reservedは14MiB大きい。
B4合成fixtureでの達成や、以前の2..40幅の学習結果へ読み替えない。

σ/spacing中央値は128で4.793→1.494、1024で7.080→1.508。
1024の最小入力bankは0→44,993/52,428atom（85.82%）、
最小出力bankは30,319→50,710（96.72%）。全52,428atomが入力bankを移動した。
一方、単一ownerへの最大集中率は最終input0.414%/output0.866%であり、
最小階層に偏ることと単一tileに仕事が集中することは区別する。
最終の非空groupは30,702、group内最大9atom、同一shape再利用率は14.18%。

無task角度更新のradial減衰だけによる解析的予測のσ中央値は128で1.4993、1024で0.18804
（後者はspacing単位で1.50436）。実測1.4939/0.18844に近く、
狭幅化をデータ学習だけで獲得した偏りとは呼ばない。独立のablation学習ではなく、
初期角度・振幅を固定した正則化だけの解析controlで、FP32 step丸めは模擬していない。
profileの集中半径補正も維持したため、最終の実効半径中央値はinput約2.05sites、
output約3.82sites。σの中央値と混同しない。

source archive SHA256 `fb55f026381efb9944d63b6ed3e6f0fb92865e8700c679d0ecf3b0dec2faadda`。
source/tests/training491ファイルを照合し、source/results/receipt/checkpoints35ファイルを
`output/cell-mass-polar-unit-spacing-20261028/verified-run/`へ保存した。
同親の`analysis.json`と`sigma-distributions.png`にbank移動・分布と解析controlを残した。
所有L4とsupervisorは停止確認済み。追加学習・大型化は開始していない。
この学習run自体は未計装だった。続く段階別・配列寿命の監査を次節に記録する。

#### optimizerの固定layout検証のキャッシュ（2026-10-28）

前節の読み出し回数を減らすため、`operators/metadata.py::fixed_state_signature`を共用し、
`_backends/torch/kernels/execution.py::validate_layout`のcell-mass条件をキャッシュした。
chart/kernel/profileの所有module・spec/binding、全固定bufferのidentity/version/dtype/deviceを追う。
通常のatom Parameter更新はこのsignatureに入れず、現在のσ・中心による分類は毎forwardに行う。
キャッシュは固定layoutが検証済みという証明だけで、atom値や配置範囲の保存ではない。

scalarのin-place変更、buffer差し替え、profile変更、chart spacing/period変更、`.to()`/dtype変更、
checkpoint読込、新chart/deepcopyでは再検証する。失敗した検証はcacheしない。
versionを持たないinference bufferは再検証し続ける。CUDA Graph内で未検証設定をhostへ読む
ことは許可せず、変更後はcapture外で検証する。全LinearのGraph対応を追加したわけではない。
公開`KernelState.declaration()`自体は従来どおり現在値のsnapshotを返す。
Polarの式、gradient projection・geometry retraction・moment transport、非有限勾配の検査は変更していない。

計測source `f56af001`、job `l4job-1a9b42fc2b96442c8a5a5915e64bcc29`。
runtimeの変更は上の2 Pythonファイルだけで、残り344ファイル（CUDA/Polar更新則を含む）は同一。
旧比較群には`7d6caeb2`の未cache `validate_layout`をbyte固定して読み込み、他のruntime・
GPU・checkpoint・計測器を揃えた。新比較群は通常のcache経路を使う。
1024/A52,428/B32、同じ初期/epoch3状態、σ_min=.125/σ_max=12.5、同じSITE_SCALED Plan。
元の全MNISTデータは常駐させ、forward/backward/本物のCSTOptimizer+fused AdamWを含める。
前節と同じ4更新warmup、21step窓3回の中央値。条件ごとにfresh processを2回実行し、
2回目は旧/新の計測順を逆にした。窓内の21 sampleを独立run数には数えない。

| 状態 / fresh process | 旧完全step ms | cache後 ms | 短縮率 |
| --- | ---: | ---: | ---: |
| 初期 / 1 | 21.296 | 17.580 | 17.45% |
| 初期 / 2 | 21.255 | 17.543 | 17.47% |
| epoch3 / 1 | 22.431 | 17.885 | 20.26% |
| epoch3 / 2 | 21.758 | 17.845 | 17.98% |

同じGPUでのdense epoch3 controlは1.949/1.921ms。
全3epochを再学習した測定ではなく、保存された学習状態での完全step比較である。
旧の約21msという別jobの数字から引き算せず、各行の同一jobで旧/新を比較する。
改善後もdense比約9.2〜9.3倍であり、目標は未達。

同期を入れたoptimizer単独診断は、初期8.72/9.21→5.34/5.60ms、
epoch3 8.91/8.49→4.87/5.30ms。
実GPU作業のkernel数は160/stepのまま、copy208→24、synchronization204→20となった。
GPUのscalar化は155→19、CPU Torch2.13の補助監査は154→18である。
GPUのkernel本体は約.408msでほぼ変わらず、copyは約.246→.030msへ減った。
profilerのsync待ち時間には先行GPU作業の完了待ちが入るので、optimizer単独wallや
完全stepの短縮量と直接同一視しない。残るTorch更新の起動数・host処理も今後の対象となる。

メモリは旧/新で変わらなかった。初期peak allocated241.827MiB、epoch3 240.773MiB、
reserved396MiB。dense epoch3 controlはallocated239.551MiB、reserved402MiB。
reservedの差は前節のcheckpoint復元時の割当順を含み、元のepoch学習のdense382MiBを置き換えない。
G/Rx/Ry/C_in/部分和の寿命は今回変えていないため、allocatedの目標未達も残る。

初期/epoch3の全52,428atomのY/dX/4列勾配を独立FP64参照へ照合し、既存3e-6/2e-7を維持した。
両状態それぞれ4stepの旧/新optimizer比較では全model Parameter・moments・stepがbit一致。
4step目に固定のradial_regularizationを変更しても同じで、設定変更の再検証も通した。
全10processの計測器有無のbit一致、allocatorの全境界/peak一致も確認した。
GPU scoped suite66 passed（dirty capture拒否テストによるempty Graph警告1件）、
CPU2,062 passed/5,385 skipped/既存警告18件。Ruff/format/buildを通過し、
wheel/sdist両方の全346runtimeファイルをGPU snapshotへbyte照合した。

PolarのTriton融合更新は既存`cuda/algorithms/polar_update/kernels.py::apply_polar`と
`cuda_polar_update_fused/v1` Planに存在する。ただし対応はEuclidean中心2座標のFP32で、
FlatTorusの中心wrapは含まない。今回の中心更新は`flat_torus.retract`のremainderを保つ。
CSTOptimizerの既定update Planは`torch_atom_update`で、Linear Planとは独立している。
今回はそのTorch参照更新のままcacheの効果だけを分離した。
このcache比較を測った時点では周期座標への対応は未実装・未計測であり、
Euclidean用FUSEDをそのまま有効化しない。Triton数値計算やPolar則自体は今回変更していない。

source archive SHA256 `112d7ac51a29f6f508ef9b7e66c3579ad93e407a4cca0ce8cdfdd57cc05a3fe9`。
source/tests/training494ファイルを照合、source/results/receiptなど73ファイルを
`output/cell-mass-optimizer-cache-20261028/verified-run/`へ保存した。
同親`analysis.json`と`complete-step.png`に比較を残す。所有GPU/pool supervisorは停止確認済み。

#### FlatTorusのPolar更新を融合する（2026-10-28）

更新専用Plan `cuda_flat_torus_polar_update_fused/v1`（`FLAT_TORUS_FUSED`）を追加した。
既存のEuclidean `FUSED`の対応範囲と既定Torch経路は保ち、`CSTOptimizer`の
`update_selector=`で選ぶ。LinearのPlan、profile、Z/H/C/G/dXの数値カーネルは変えない。

AdamWが通常どおりproposalとmoments/stepを更新した後、Tritonの各laneが1atomを担当する。
128atomのblockで、旧Polar座標とproposalを読み、接方向の変位、活動量による半径更新、
半径正則化を計算し、2つの中心をそれぞれlive periodへ折り返してParameterへ直接書く。
中心は参照実装と同じ`old + (proposal - old)`を経て`fmod`と符号補正を使う。
`center - floor(center / period) * period`への置換や、幅への通常SGDは行わない。

書込前には旧中心と中心変位の有限性を専用kernelで同時に検査する。
blockごとの判定を4byteのint32 failure flagへまとめ、有効blockはatomic操作をしない。
flagだけをhostへ戻して異常時は書込前に拒否する。論理workspace上限は4byteで、
allocatorの割当単位とは区別する。旧座標snapshotは従来どおりatom数に比例する。
全model勾配の事前検査、射影勾配・更新結果・transportしたmomentsの検査はwrapperが保つ。
有限性検査の同期は残るため、このPlanもeager実行のみである。

設定変更は従来どおりidentity/versionにより再検証し、kernel scalarとperiodをliveに読む。
AtomUpdateContextにはgeometry bufferのdtype/device/numelも追加し、periodの配置が
非対応になった場合はAdam更新前のPlan選択で拒否する。
CPU全体検証は2,063 passed / 5,402 skipped / 既存警告18件。
計測runtimeは`54c53958`、job `l4job-54f18e9e8a74457596f0fd62669b57a6`。
比較元は固定layoutキャッシュ済みのTorch更新で、両群とも今回の共通metadata事前検査を使う。
N1024/A52,428/B32、Q128/spacing=.125、σ_min=.125/σ_max=12.5、SITE_SCALED Plan、
GPU常駐MNIST、同じ初期・epoch3 checkpointとoptimizer状態、同じAdamWを維持した。
L4 / Torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0、TF32/Graphなし。
3epochの学習をもう一度回した結果ではなく、保存した学習状態からの完全step比較である。
各群fresh process 2回、2回目は順序を反転し、21step窓3回の中央値を用いた。

| 状態 / fresh process | Torch完全step ms | 融合後 ms | 短縮率 |
| --- | ---: | ---: | ---: |
| initial / 1 | 17.931 | 16.090 | 10.26% |
| initial / 2 | 17.673 | 16.265 | 7.97% |
| epoch3 / 1 | 18.066 | 16.441 | 9.00% |
| epoch3 / 2 | 18.122 | 16.486 | 9.03% |

同期を入れたoptimizer単独診断では、epoch3で5.272/5.083→3.458/3.384ms。
GPU起動160→75（当初Runtime APIのみの集計）、copy24→18、同期20→15回/step。本体GPU時間は約.409→.219ms。
本体以外の差を純粋なPython時間と断定しない。profilerのsync待ちには先行計算の待ちも含むため、
その時間をoptimizer固有費用として加算しない。全Parameterの勾配検査やmoment transportの
検査は残っており、現行APIはその場で異常をhostへ報告するeager契約である。
更新式だけなら同一stream上のGPU kernelで順序を保てる。host同期をゼロにする場合は、
異常時にParameter/moments/clockをdevice側で更新しない処理と、異常報告時点を別途設計する。

peak allocatedは初期241.827MiB、epoch3 240.773MiBで旧/新同一。
epoch3 reservedは融合版394MiB、Torch396MiB、dense control402MiBだった。
dense allocated239.551MiB、完全step1.972/2.038msなので、速度・allocatedとも目標は未達。
reservedの差はこのcheckpoint復元・割当順の条件の観測であり、allocated削減と混同しない。

GPU scoped suite94 passed（既存のdirty Graph拒否テストの警告1件）。
初期・epoch3それぞれ全52,428atomのY/dX/全4列勾配を独立FP64参照へ照合し、
rtol3e-6/atol2e-7を維持した。両状態の同一勾配4更新ではParameterが既存Polar融合許容差
rtol2e-6/atol2e-6内、全modelのAdam moments/stepはbit一致。4更新目の設定変更も通過した。
全10processで計測器あり/なしの更新がbit一致し、allocatorの全境界・peakの照合も通過した。
wheel/sdistの全346runtimeファイルを現在のsourceへbyte照合した。

source SHA256 `07dbb154c222e9620ea61d907f57314bd2a00e74e642d360950cbf58618ff682`、
result archive SHA256 `36df93fcd27d818b393df12bff033330d5d13a2cb723b8c63e91577705035625`。
`output/cell-mass-flat-polar-20261028/verified-run/`へ73ファイルを保存し、
source/tests/training495ファイルが現在値と一致した。所有pool/GPUは停止確認済み。
同親`analysis.json`と`complete-step.png`に比較を保存した。

先行job `l4job-ba7e1d9c510b46e29614d5d48fcaf762`（metadata配置検証追加前）でも
92 test成功、epoch3完全step18.149/18.000→16.704/16.348msを観測したが、上の最終版の値と混ぜない。
それ以前の2job `l4job-ab79f0e5e99c435b9a0f1506dde9ba55` と
`l4job-80066d5062cf4000b36d142693e2c94d` は追加テストがperiodのみを変更し、
`period = site_count * spacing`を破ったため6件失敗した。数学的契約は変えず、
テストでspacingも合わせて変更した。これらは性能測定前に停止し、速度の証拠に含めない。
source/result/ログは同親`failed-live-period-check-v1/`と`v2/`に保全した。


#### optimizerの異常値検査を集約する（2026-10-28）

`_backends/torch/finite.py::first_nonfinite`で、CUDAのreal FP16/BF16/FP32/FP64を
同じdevice/dtypeごとにまとめて検査する。各tensorの最大絶対値を`_foreach_norm(..., inf)`で
計算し、その有限性の小さなvectorを一度にhostへ返す。大きなgradient/momentの結合や
activation-sized mask、二乗和、Parameterの書換え、数値結果のcacheは作らない。
空tensorは有限と扱い、complexは有限な実部・虚部でも絶対値がoverflowし得るため、
従来の`isfinite`判定へ戻す。CPU/その他layoutも参照検査を保つ。

`CSTOptimizer.step`は全Parameterのraw勾配をこの関数へ一括で渡し、不正ならAdamの
Parameter/moments/step更新前に拒否する。疎COOは従来どおりcoalesceしたvaluesを検査し、
一時参照はAdam前に解放する。Euclidean/FlatTorusのidentity transportも、旧点・新点・vectorを
一緒に検査する。transportの値、Polarの法則、Adam、projected/new/transport済みvectorの
追加検査、設定変更時の再検証、CUDA数値カーネル、σ/階層分類は変えない。

runtime source `479aec47`。主job `l4job-fb50907d69cf4bfb9b69c03a7b837ef0`では、
同じ最終コード内の検査関数だけをbenchmark用の逐次参照に差し替えた比較元と、通常の
集約版を比べた。両群とも前節の`FLAT_TORUS_FUSED`を使う。
N1024/A52,428/B32、Q128/spacing=.125、σ_min=.125/σ_max=12.5、同じSITE_SCALED Plan、
GPU常駐MNIST、同一の初期/epoch3 checkpointとoptimizer状態、同じAdamWを維持する。
L4 / Torch2.11.0+cu130 / CUDA13.0 / Triton3.6.0、TF32/Graphなし。
4更新warmup後、21step窓3回の中央値を採る。各群2 fresh processで2回目は順序を反転した。
小さい改善の確認として、独立job `l4job-a51fe92cd8af4e2c873e288fc62794a2`でもepoch3を
各群1 fresh processで再測定した。新たなepoch学習・精度比較ではない。

| 状態/独立process | 逐次検査の完全step ms | 集約後 ms | 短縮率 |
| --- | ---: | ---: | ---: |
| 主job initial/1 | 15.855 | 15.551 | 1.92% |
| 主job initial/2 | 16.138 | 15.806 | 2.06% |
| 主job epoch3/1 | 16.413 | 16.106 | 1.87% |
| 主job epoch3/2 | 16.357 | 16.114 | 1.49% |
| 独立確認job epoch3/1 | 17.231 | 16.979 | 1.47% |

完全stepは約1.5〜2.1%の小さい短縮で、同期回数の減少率と同一視しない。
optimizer内の`cudaStreamSynchronize`は15→7、kernel75→44（当初Runtime APIのみの集計）、copy18→10、memset5→3回/step。
主jobのepoch3単独同期診断は3.871/3.270→2.983/2.980ms、確認jobは3.825→3.605msだった。
GPU本体は約.218→.207msで、大幅な演算量削減ではない。
残る7回はraw勾配、射影勾配、中心/変位、更新結果、2軸のtransport、transport結果の検査。

`runtime-api-audit.json`にはCUDA runtime APIをstageへ直接対応付けた監査も保存した。
主job epoch3/1のprofilerで、逐次版のStreamSynchronize host時間は約6.46msだったが、
集約版ではMemcpyAsync host時間に約6.53msが現れる。転送APIも先行GPU処理を待つ場合がある。
StreamSynchronize時間だけを見て「CPU待ちが消えた」と呼ばず、同期診断や実GPU時間を
完全stepへ加算しない。前段のforward/backwardの計算時間もこの変更では残る。

peak allocatedは旧/新同一で、初期241.827MiB、epoch3 240.773MiB。
reservedも初期392MiB、epoch3 394MiBで旧/新同一、確認jobも394MiB。
主jobのdense epoch3 controlは1.957/1.971ms、allocated239.551MiB。
速度・allocatedともdense目標は引き続き未達。

CPU全体2,069 passed/5,409 skipped/既存警告18件、主GPU scoped suite161 passed
（既存のdirty Graph拒否警告1件）、独立確認のfinite suite13 passed。
初期/epoch3の全52,428atomのY/dX/全4列勾配を独立FP64参照へ、既存3e-6/2e-7で照合した。
同一勾配の4更新では全modelのParameter/moments/stepがbit一致し、4更新目の設定変更も通過。
NaN/±Inf、最大有限値、noncontiguous、空tensor、複素数、mixed dtypeを検査した。
全13 fresh processの計測器あり/なしの更新と、allocatorの全境界/peakが一致した。
wheel/sdistの全347runtimeファイルを計測sourceへbyte照合し、runtime commitのGitHub CPU CIも通過。

主source archive SHA256 `621fbb4b638cd8f58756093964125c2d166d39587df909b590ad112753b25881`、
result archive `7b254494921d57abaab3b734413afc00757946425689fdbefccc8f891a34cf07`。
確認source `dde1869b1d08d7810d1f61d7009e087e9fae7083afebebafe19ed489789d5ece`、
result `478a0166309cba238e16f69a256cea93bbde803e51b0043a7bcbb7d69dad125f`。
両jobともsource/tests/training497ファイルが現在値と一致。
`output/cell-mass-batched-checks-20261028/verified-run/`へ73ファイル、
同親`confirmation-run/`へ29ファイルを保全し、`analysis.json`、`confirmation-analysis.json`、
`complete-step.png`へ集計した。所有GPU/pool supervisorは停止確認済み。

#### PyTorch流の任意finite検査でoptimizerのCPU同期をなくす（2026-10-28）

ユーザー指定「PyTorchの流儀」に合わせ、`CSTOptimizer(..., check_finite=False)`を追加した。
通常のAdamWと同様にoptimizer独自のNaN/Inf検査を追加しない経路であり、異常時にGPUで
全更新をskipして後で報告する方式ではない。既存呼出しの挙動を保つため既定はTrueとする。
Falseでもshape/dtype/device・group・Plan・固定Kernel/Chart設定の検証は維持する。
有限入力のPolar更新則、Adam演算、moments/step、幅stop-gradientは一切変えない。
設定変更で必要な宣言の読出し・検証は残すため、同期ゼロの確認対象は通常の固定設定・
versioned buffer・closureなし・FP32/fused AdamW/FlatTorus FUSED Planの定常stepである。
CUDA GraphやAMP/GradScaler全体の同期ゼロを追加したものではない。

finiteチェックの有効性はContextVarでoptimizer呼出し内だけ切り替え、例外でも必ず元に戻す。
ユーザーclosureはそのscopeの外で一度だけ評価し、通常のgeometry validationを維持する。
外部のgeometry関数の検証を恒久的に変更せず、他threadへglobal flagを漏らさない。
デバッグ時は`check_finite=True`を使う。通常AdamWのstepとAMPの検査の区別は
[PyTorch 2.11 Adam実装](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/adam.py)と
[AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html)に対応する。

計測runtime `66593872`、job `l4job-122ccce94ed9491492423289b34e4696`。
両群とも同じ`FLAT_TORUS_FUSED`、比較元は前節のbatched検査あり、候補だけFalse。
N1024/A52,428/B32、Q128/spacing=.125、σ_min=.125/σ_max=12.5、SITE_SCALED Plan、
常駐MNIST、同じ初期/epoch3 checkpointとoptimizer状態、AdamW lr=.001/wd=0を維持。
L4/Torch2.11.0+cu130/CUDA13.0/Triton3.6.0、TF32/Graphなし。
各群2 fresh process、2回目は順序を反転、4更新warmup後の21step窓3回の中央値。
新しいepoch学習ではなく保存した学習状態での完全step比較である。

| 状態/独立process | 検査あり完全step ms | PyTorch流 ms | 短縮率 |
| --- | ---: | ---: | ---: |
| initial/1 | 15.704 | 11.831 | 24.66% |
| initial/2 | 15.749 | 11.653 | 26.01% |
| epoch3/1 | 16.136 | 12.058 | 25.27% |
| epoch3/2 | 16.187 | 12.032 | 25.67% |

4つのPyTorch流processすべてでoptimizer scopeの同期APIは0、GPU→CPU転送も0を確認した。
Runtime/Driver両APIを含むGPU kernel起動は46→5/step、copy10→3で、残る3copyはすべてD2D。
Polar自身は1kernel、ほかはAdamとgradient/momentの組立で、CPUへの結果受渡しはない。
実GPUのoptimizer kernel本体は約.214→.058ms。epoch3の同期を入れたoptimizer単独診断は
3.373/3.277→2.047/2.040msだった。CPUの配置・宣言・dispatch処理は残るが、値をCPUで
待たずにGPUへ並べられるため前段との重なりも変わる。段階診断の差を完全stepへ加算しない。

peak allocatedは旧/新同一で初期241.827MiB、epoch3 240.773MiB。
reservedも初期392MiB、epoch3 394MiBで旧/新同一。
dense epoch3 controlは1.899/1.961ms、allocated239.551MiBで、最終的な速度/allocated目標は未達。

CPU全体2,075 passed/5,411 skipped/既存警告18件、GPU scoped169 passed/警告2件。
同一勾配4更新・途中の設定変更を含め、初期/epoch3の全model Parameter/moments/stepはbit一致。
全52,428atomの独立FP64 Y/dX/全4列勾配は既存3e-6/2e-7を維持して通過。
全10processの計測器あり/なしの更新、allocator全境界/peakも一致した。
NaNを自動拒否しない通常AdamWとの一致、例外後のscope復旧、closure、設定検証も検査した。

**起動数の集計修正**：初期の`trace_stage_costs`はcuda_runtimeだけを見ており、Tritonの
`cuda_driver::cuLaunchKernelEx`がunassignedになっていた。postprocessing `7aa9a993`で
Driver APIを含め、CPU regression testを追加した。主計測とinstrumentationのASTは変更していない。
過去節のRuntime-only値75/44/4は、Driver API込みでは77/46/5である。
この訂正は同期ゼロ・完全step・メモリ・数値結果に影響しない。
`corrected-trace-costs.json`で全10traceを再集計し、4候補すべての0同期/0DtoHを再確認した。
旧集計も`analysis-runtime-only.json`に残し、`analysis.json`は訂正版とした。
source347runtimeの現在値一致と、postprocessing関数を除くbenchmark AST一致は
`postprocessing-audit.json`に保存した。postprocessing更新後のCPU全体は2,076 passed/5,411 skipped/既存警告18件。

source archive SHA256 `163ea0d77b279b3c50e8b323d0d686a69159a6e7d7415bea7eedcb8e0614a454`、
result archive `7cc047e8a3dbfc1bd7b2eaea8df45b00e39d419e132d00508c8a3ae02941cc06`。
source/tests/training498ファイルを提出sourceへ照合し、73証拠ファイルを
`output/cell-mass-pytorch-step-20261028/verified-run/`へ保全した。
wheel/sdistの全347runtimeファイルもbyte一致。所有GPU/pool supervisorは停止確認済み。

#### H/Cのbatch再利用と、担当量・作業配列の分離（2026-10-28）

Polarの最適化はここで区切り、同じN1024/A52,428/B32のinitial/epoch3保存状態で
Linearを比較した。FlatTorus fused更新と`check_finite=False`、σ範囲・spacing・M8・
profile・T_min・周期は直前の条件のまま。Zは二分階層を保つ。新たな学習epochは回していない。

**実装済み：** `compute_h_c_ragged_batched`でforwardのbatch tileを4/8から選ぶ。
各threadは別atomを担当し、入力profileとの積和Hをregisterで求め、そのFP32丸め後の同じ値を
shared chunkへ渡して出力Cを作る。atom係数表・packed atomコピー・dense Wは追加しない。
各sampleのharmonic順と各chunkのatom加算順は変えない。

```cpp
// 1 CTA = 出力ownerのatom部分区間 × 4または8batch、32threads
for (atom_chunk : assigned_owner_range) {
    // laneごとに別atom。1回の係数生成を担当batchへ再利用する。
    H[batch] = project_input_profile(Rx, atom, batch);
    shared_coeff[lane] = output_profile(atom);
    shared_amp_H[batch][lane] = amplitude(atom) * float(H[batch]);
    __syncthreads();
    C_partial += contract_32_atoms(shared_coeff, shared_amp_H);
    __syncthreads();
}
// 別kernelでsplit順に縮約、CからYを展開。Yへのatomicは使わない。
```

従来の`atom_threads=64`はこのCTAのthread数を指定する値ではない。H/C融合CTAは32threads。
`hc_batch_tile`は1threadが同時に処理するbatch数、`hc_atoms_per_split`はforward H/Cの
owner分割の目安。`atoms_per_split`は別のC_in側の目安として残す。各ownerの人数nに対し
`active_splits=min(c_splits,ceil(n/grain))`であり、常に128splitを起動する方式ではない。
実task数をhostへ読まず、安全上界`min(nodes*cap,ceil(A/grain)+nodes)`でscratchを確保する。

明示recipe `site_scaled_hc8_g64_cin512`はbatch8・H/C grain64・C_in grain512を表す。
`site_scaled_release`は従来の計算割当のまま早期解放だけを有効にする対照群。
古い`site_scaled`の挙動とpresetだけのPlan JSON形式は維持し、自動selectorへは追加しない。

主比較source `46df9d95`、L4 job `l4job-53b6443627bc4867abb7d4e1484f5827`。
20条件のH/C単体event比較で、両保存状態を合計した時間からtile4/8それぞれのgrainを選び、
各候補を別途完全stepで検査した。各armは2つのfresh processで順序を反転。
各process内の3つの21step窓の中央値であり、独立seed数を意味しない。

| 状態 | 従来完全step ms（2process） | batch8/grain64/C_in512 ms | 従来/候補 peak allocated MiB | dense allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| initial | 11.644 / 11.644 | 10.566 / 10.524 | 241.827 / 239.780 | 239.551 |
| epoch3 | 12.009 / 12.000 | 10.774 / 10.792 | 240.773 / 239.288 | 239.551 |

別のprofiler passでH/Cの実GPU作業はinitial約2.75–2.78→1.72–1.73ms、
epoch3約3.14–3.17→1.93–1.94ms。完全stepは約9–10%短縮した。
分割粒度だけをgrain64へ変えたtile4はepoch3 H/C約3.02–3.03ms、完全step約11.865ms。
主な効果はB32で8回だったprofile/phase等の再生成を4回へ減らしたことと整合する。
有効atomのC寄与総数`A*B*17`は減らない。命令stallや実効帯域のhardware counterは未測定。

σの偏りは、owner人数以外にも係数再利用条件を変える。初期では入出力bank周期が一致する
atomが100%だが、epoch3元checkpointでは14.181%になる。44,993atom（85.819%）は
入力最小bankのP=1.5、出力最小bankのP=3へ分類され、同じσでも`same_shape`がfalseとなる。
このatom群では出力profileの係数を別に生成する。追加計算の発生は分類とnative guardから
確認できるが、初期/epoch3の時間差の全てをこの原因へ帰する因果実験ではない。
profile周期とσの定義を維持して、batch間の再生成と不要な復号を減らす。

| コンパイル/runtime使用資源 | batch4 | batch8 |
| --- | ---: | ---: |
| register/thread | 126 | 130 |
| dynamic shared/CTA bytes | 6,088 | 7,688 |
| local storage/thread bytes | 40 | 40 |
| occupancy API予測 resident CTA/SM | 14 | 11 |

これは実測achieved occupancyではなく、L4の該当specialization・32threadsでの予測。
local bytesもspill trafficの測定値ではない。常駐CTA数が下がっても速くなるため、
単にthread/warpを増やす割合設定だけでは改善理由を説明できない。

メモリはRyをG/Parameter勾配完成後、GをC_in完成後に解放する。PolarのHもC_out後に解放する。
保存済みRxはretained backwardのため保持し、`ctx.saved_tensors`を勝手に消さない。
forwardのHC部分和はgrain64で1.901→4.453MiBと増えるが短命であり、最大peakは引き続きC_in内。
C_in grain512は部分和を2.963→2.540MiBへ減らす。早期解放だけではepoch3 239.711MiBで
まだdenseを超え、grain変更との組合せで239.288MiBになる。initialは0.229MiB超過が残った。
reservedはinitial候補392/dense382MiB、epoch3候補394/dense402MiBであり、
この時点で全てのmemory指標がdense以下とは主張しない。

921 GPU testsと、両保存状態の全52,428atom/B32のY・dX・全4列atom勾配を
独立CPU FP64 FFT/autogradへ照合して通過（rtol3e-6/atol2e-7）。全20processの計装有無は
logits/loss/全model勾配/更新後Parameter・momentsがbit一致し、allocator履歴の全境界・peakも一致。
CPU全体2,114 passed/5,420 skipped/既存警告18件、wheel/sdistは347runtimeファイルbyte一致。
source/tests/training499ファイルと135証拠ファイルを
ignored `output/cell-mass-hc-20261028/verified-run/`へ照合・保全した。
source archive SHA256 `f7d9e84cda0c0ca420963e7e2478e1c204c54436b8c450ef1aa6e62f2082b0ef`、
result archive `26960b53ddecf8aee53ee92f671a8784a6b508ff1beafeb3abafa206bee5053a`。

**46df9d95時点の既存関数内の候補（後述の4eabbd79で実装・GPU検証済み）：**
入出力のPolar幅復号法則が厳密に同じなら、
`atom_sigma`のsqrt/pow/log/expを2回評価せずσを再利用する案を検討した。
周期が異なる場合のprofile生成は分離したままにする。46df9d95ではinput/outputそれぞれで復号していた。
またouter loopで既知のinput bankを渡し、分類と同じwrap/floorでownerを再現できれば
per-atomのgroup二分探索を省けると考えた。境界とnextafter、原点、周期wrapの独立照合を要件とした。
どちらも上記46df9d95の速度結果には含めず、後述の4eabbd79で比較した。

**次のkernel候補（未実装）：** 同一output ownerのinput-bank区間を論理連結し、
担当境界を32atom chunkに合わせる。epoch3の実分布では現在1,920chunk/85.33%lane利用率から
1,721chunk/95.20%へ減らせる。active925taskとpartial容量は維持でき、range metadataは約17.2KB。
有効atomのprofile計算は減らないため、chunk10.36%減をH/C時間10.36%短縮とは呼ばない。
初期状態ではbank境界の連結だけの効果はなく、chunk境界整列だけが候補になる。

output最小bankへ50,710atomが入る一方、単一owner最大は454atomで128split上限には達していない。
最小階層への偏りと少数ownerへの集中を区別する。grain64ではtask925/安全枠1073、
従来grain256では320/458。小さいtaskの並列性と長いtaskの削減が、追加paddingを上回ったと
考えられるが、原因を分離したhardware-counter検証ではない。
複数warp/CTA化はsharedをwarp数分必要とし、独立warpの異なるloop長で現在の
`__syncthreads()`を使えないため、warp同期の別実装と資源検証を要する。優先順位は低い。

**46df9d95時点のZ側の候補（後述の4eabbd79で実装・GPU検証済み）：**
46df9d95のchunked builderは全bankの直接moment・reduce・
二分merge・3tile窓を作っていた。epoch3では入力consumer bank0/1/2、出力0/1しか使わない。
必要な子bankを含む依存閉包をGPU bitmaskで作り、未使用の上位bankを省く案を検討した。
直接評価するmode×siteの係数は入力82→33、出力74→25となる（いずれもB*N倍する仕事量で、
時間短縮率ではない）。window node数は入力509→448、出力253→192。
全周bankは独立に直接生成し、local bankのmerge祖先と混同しない。
ゼロ振幅でも勾配に必要なatomを含め、maskはforwardのlayoutに結び、更新時に作り直す。
全ownerを残すbank単位の省略から始めれば、隣接ownerと子ownerの複雑な個別閉包を避けられる。
raw momentの依存maskとH/Gのconsumer maskを区別し、windowが未初期化Zを読まないことが必要。
このmask gatingだけではscratchの確保量は減らない。4eabbd79での実装と実測は後述する。

**C_in scratch batch-slab（実装・独立GPU検証済み）：**
initialに残ったpeak超過へ対応するため、全Gと全C_inは保持し、部分和だけを16/8batchずつ
作って同じstreamで縮約後に使い回す。`cin_batch_slab`で指定し、recipe末尾`_slab16/8`で選べる。
atom和・split縮約順は変えず、追加のhost同期も入れない。以下の4eabbd79で16batch版を確認した。
先行source c3820945のjob `l4job-64f76715f1e443b6a6927c5cb82f2d85`は接続断で結果を
回収できず、全ての達成判定から除外する。source499ファイルはGit c3820945とbyte一致し、
source/archive/transport/lifecycleと回復記録だけを`failed-transport/`へ保存した。
所有sessionは停止してserver一覧から消えたことを確認し、pool recover後に別jobへ進んだ。

**係数共有とactive bankの選択設定（source 4eabbd79、GPU・完全step確認済み）：**
`site_scaled_hc8_g64_cin512_slab16_exact_reuse_active_z_cached32`は、前述のbatch8に加え、
厳密な同一幅復号の再利用・既知input bankからのowner再現・未使用bankのZ/R省略・
Polar pullback係数のshared再利用を有効にする。二分merge、profile、σ・周期・正規化は維持する。
consumer maskとraw Zの子依存maskはGPU int32[4]（16bytes）で、zero amplitudeのatomも含める。
forwardのmaskをrouting tensorsと一緒にsave_for_backwardし、retained backwardとversion検査を守る。

PolarのG/Parameter勾配では、各threadが別atomの17個のFP64係数を軸ごとにsharedの専用columnへ
一度保存する。B32で8回再生成していたstreamを1回にする。各threadは自分のcolumnしか読まない
ためbarrierを追加しない。中心微分は元の`omega*(-cs*zc+cc*zs)`、H/Gはパラメータ勾配用に
FP64のまま、dX向けGだけ従来どおりFP32へ丸める。sharedは32/64threadsで8,704/17,408bytes。
register/threadは180→145、occupancy API予測は32threadsで8→10CTA/SM。
これはachieved occupancyやspill trafficの測定ではない。

correctness job `l4job-a94c3853a437461f9da7ee9e5c4a7028`で1,116 testsが通過。
両保存状態・8recipeの全52,428atom/B32のY/dX/全4列勾配も独立FP64参照へ同じ3e-6/2e-7で通過した。
source500ファイルを照合し、native binary（SHA256
`c9f7dabc0c13cc993009caa02f01559d748872dfd1fad962f98006e05c8ab318`）を結果manifestへ保存した。
完全step job `l4job-c7a5466a0f064bb0a9391265590e1e1e`は、この検証済みbinaryを再利用する。
各fresh childでsource6ファイルの結合digest・binary hash・Python/Torch/CUDA・GPU名/能力を照合して
importし、JIT compileの再実行を避けた。runnerとbinaryもraw evidenceへ保存する。
数値runtimeの差し替えや古いsourceへの混在ではない。全source500ファイルはGit4eabbd79へ照合済み。

| 保存状態/独立process | 従来完全step ms | 上記選択設定 ms | peak allocated CST/dense MiB | reserved CST/dense MiB |
| --- | ---: | ---: | ---: | ---: |
| initial/0 | 11.628 | 8.413 | 238.712 / 239.551 | 392 / 382 |
| initial/1 | 11.607 | 8.411 | 238.712 / 239.551 | 392 / 382 |
| epoch3/0 | 12.013 | 8.231 | 239.170 / 239.551 | 394 / 402 |
| epoch3/1 | 11.966 | 8.225 | 239.170 / 239.551 | 394 / 402 |

epoch3の完全stepは約31.3–31.5%短縮。denseは同じprocess対照で1.946/1.950msなので、
時間目標は未達。allocatedは両状態でdense以下だが、initial reservedは未達。
reservedの初期差は計測開始前からあり、180MiB segment内に小さな生存Tensorと
大きな空き領域が同居している。allocated/reservedを混同せず、process総GPU使用量は未測定とする。
slab16後の最大allocatedはC_inからforward H/Cへ移り、H6.400MiBとHC部分和4.453MiBが重なる。

別passのepoch3実GPU作業はZx2.058→0.785ms、Zy1.056→0.319ms、H/C3.169→1.700ms、
G/Parameter勾配1.825→1.433ms。C_inはslab16で約1.650ms、Y/dX展開は約.365/.419ms。
これらを足して完全stepとは呼ばない。32/64thread cached版の完全step差は小さく、
速度優位を断定しない。32版は1CTAのsharedが半分であり、次のone-warp試作と揃える。
計装有無の全model出力/勾配/更新後Parameter・momentsは20processすべてbit一致し、
allocatorのpeak・全境界再構成も一致。optimizerの同期APIは全processで0を維持した。

CPU全体は2,211 passed/5,518 skipped/既存警告18件。
correctness source/result SHA256は`5591e2de8d619db3133e278e8b840a3a6e35f03293c2e29178dcaedff0cb199a` /
`df572685349dba7de8e90c25187279975f978354813142bbbebbba5de9a79cf6`。
完全step source/result SHA256は`661468a7e97de458681c296adf0bb971839dd75e8216d444753a9b1dab309c01` /
`6f2daad7ac2f00eef85fc13f9fb2d05a62c6d77e343e4eb1a02e2eba3dc6c7b3`。
ignored evidenceは`astra-gates/`と`astra-profile/`。現在worktreeの追加試作との差は、
測定前の500ファイル一致記録とGit4eabbd79への照合で区別する。

**追加試作（source 64471aa4、CPU・独立GPU・完全step確認済み）：**
`_node_z`はactive bankの二分mergeを保ち、既存のchunkなしnode builderを使う。
要求chunk32が実際には入力4/出力8へclampされるため、chunk版はinput16,384/output7,168CTAと
8.500/3.719MiBの一時配列を持っていた。両保存状態の必要な子依存閉包はtile32sites以下であり、
node版は4,072/2,024CTAとし、chunk scratchとその縮約kernelを作らない。
`_expand`は同じconsumer maskで、寄与のないbankのY/dX基底評価を省く。
Zにはraw依存mask、出力にはconsumer maskを使い、両方の意味を混同しない。
既存の完全hierarchy APIと旧recipeのdefaultは保持する。

job `l4job-08c840599a274910b900d36d75415e97`で1,157 GPU tests、両保存状態の
6recipe・全52,428atom/B32についてY/dX/全4列atom勾配を独立FP64参照へ照合して通過した。
閾値は従来と同じrtol3e-6/atol2e-7。source/tests/training501ファイルを照合済み。
Linear単体のforward/backward screeningは完全MNIST stepとは区別する。
先行job `l4job-10f0b00a37564f58907a000eff630b01`は実験コード開始前のCLI接続で停止し、
回収結果がないため全達成判定から除外した。source/transport/lifecycleを保存し所有sessionを回復した。
GPU gateのsource/result SHA256は`8763829122df26ba357357b7dd3edfe68347a2531c8279bdb7a1c18de969a310` / `20ee0ccf62c7dd4b10c162c4cfae3fd3025fb5595f61c432802120c17f939e1b`。
証拠はignored `output/cell-mass-hc-20261028/node-gates/`。

完全step job `l4job-d15e5ebd350a46ee94632c8ed269730c`は、同じ検証済みnative binaryを
全fresh childでsource digest/runtime/deviceと照合して使った。選択設定は
`site_scaled_hc8_g64_cin512_slab16_exact_reuse_node_z_cached32_expand`。

| 状態 | 従来完全step ms（2process） | node Z＋展開省略 ms | peak allocated CST/dense MiB | reserved CST/dense MiB |
| --- | ---: | ---: | ---: | ---: |
| initial | 11.613 / 11.649 | 7.294 / 7.320 | 238.712 / 239.551 | 392 / 382 |
| epoch3 | 12.022 / 12.002 | 7.396 / 7.390 | 239.170 / 239.551 | 394 / 402 |

別passのepoch3 GPU作業は、前選択設定に対してZx約.785→.501ms、Zy.319→.245ms、
Y展開.365→.112ms、dX展開.419→.165ms。H/C約1.674ms、G/Parameter勾配1.434ms、
C_in1.653msが残る。旧site_scaledから完全stepは約38.4%短縮したが、dense1.912/1.967msとの
速度差は残る。allocatedの最大は依然forwardのH/C内であり、reservedのinitial超過も残る。
全24processの計装bit一致・allocator境界/peak一致・optimizer同期API0を確認した。
追加試作後のworktree差とは別に、source/tests/training501ファイルをGit64471aa4へ照合した。
完全step source/result SHA256は`659f4032f30c36f24e8f3219a67c789b56c6c24f863e1015413a7c76731fd586` / `857973f29b3739ffbdb84a795ca92d3a6587a17bd495aedb0f68c791e2891e6b`、
証拠はignored `output/cell-mass-hc-20261028/node-profile/`。


**Polar経路のH保存省略と逆C融合（source 0b43283e、CPU・独立GPU・完全step確認済み）：**
Polar optimizer自体はHを受け取らず、Parameterの勾配と既存optimizer状態を使う。
論理的なHはforwardのC_out、およびLinear backwardの振幅勾配
`da[a] = sum_b H[b,a]*G[b,a]`・出力中心勾配に必要である。
ただしH配列をoptimizerへ渡す必要はなく、Polar backwardは保存RxからFP64 register内へHを再投影する。
forwardで一時的にglobal Hを書いていたのは、共通H/C APIがH配列も返すためであり、
optimizerの要件ではない。`_nostore_h`/`omit_unused_h=True`はこのbufferの確保・書込を省く。
C用の局所H計算とFP32丸めは残し、`release_workspaces`による確保後の解放とは区別する。
primitive `compute_h_c(..., store_h=True)`の既定と診断出力は維持する。
物理sigmaでatom勾配が必要な場合、非ragged、非融合H/Cでは従来のH保存へ戻る。

`_fused`/`fuse_polar_cin=True`は逆indexのinput owner部分区間を1CTA/32threadsで担当する。
各threadは別atomを全batchにわたって担当し、係数を2軸各17個のFP64 shared専用columnへ保持。
H/Gと中心微分からParameter勾配を作ると同時に、`amp*double(float(G))`を共有してC_inへ集約する。
Parameter勾配には丸め前H/G、C_inには従来どおり丸めたGを使い、global Gは作らない。
C_in partialは全B32を保持するため、eligible経路では`cin_batch_slab`を使わない。
FP32 Polar/M8/tail1e-3、値・勾配の両補償、reverse ragged、B<=32、両側勾配が条件。
条件外は既存経路へ戻り、cache/slab設定もそのfallback用に保持する。
B32のsharedは14,648bytes/CTA。これは資源予算で、実測ピークや速度改善の保証ではない。

CPU全体2,255 passed/5,596 skipped、既存警告18件。forward/apply/backwardは32引数で一致。
CUDA job `l4job-b31cf41d84f94791a99f78bbef6243bd`は1,225 tests、両保存状態・7recipeの
全52,428atom/B32についてY/dX/全4列atom勾配を同じ独立FP64参照・同じ閾値で通過した。
memcheck/racecheckはB5/A32のtail case各1件で0 errors/0 hazardsを確認した。
これは全GPUcaseのsanitizer検証ではない。native B32資源は146register/thread、
local storage40bytes/thread、dynamic shared14,648bytes、予測6CTA/SM（実効occupancy未測定）。
source/tests/training504ファイルとcompiled binaryを照合・保全した。
source/result SHA256は`b8bf916b086062fcc63fe9f3474ec52194a2761a97711138f74c1550532fe088` / `38ebd1542399248d713432e6acf8416d83ff7367582099421264618a64118df9`、
ignored evidenceは`output/cell-mass-hc-20261028/fused-gates/`。完全step/peakは次の別jobで測定した。
完全step job `l4job-245ad903780b4c969753bdbf9af2e4d5`はCLI接続断・840秒の外側timeoutとなり、
result/receiptを回収できなかったため、全ての速度・memory判定から除外した。後のtrace確認では
CLIの初期cwd設定RPC（execution.py:224）で接続断となり、本体remote.pyの実行前だった。
CUDA kernelの遅さや数値失敗の証拠とは扱わない。source504ファイルを
Git0b43283eへ照合し、source/spec/transport/lifecycleを`failed-fused-profile-transport/`へ保全した。
所有sessionは停止・server一覧からの消滅を確認しpool recover済み。
この未回収jobは以後も除外し、次の再測定と混同しない。自動selectorへの採用は別判断とする。


**通信修復後の融合版完全step再測定（2026-10-28、numeric source 0b43283e）：**
初期cwd設定を別のexecute requestで送る経路をなくし、実験用CLI内で最初の本体cellへ統合した。
その前にshell/IOPubのready handshakeを行い、接続開始も含む全経路を`runtime.stop()`のfinallyで閉じる。
旧経路ではsetup例外がfinallyの外へ出て、non-daemon WebSocket threadが残る場合を再現できた。
さらにjupyter-kernel-client 0.9.0の期限切れ後の空待機loopへ明示的TimeoutErrorを追加した。
close自体が止まる場合は実験用CLIをfailさせ、poolが所有VMを停止する。remote codeを自動再実行しない。
変更先はisolated `/tmp/torchcst-colab-075`のみで、共有CLI・poolの所有規則・数値runtimeは変えない。
実験用CLIはgoogle-colab-cli0.7.5/jupyter-kernel-client0.9.0/websocket-client1.9.2/requests2.34.2。
実ネットワーク切断の全原因を特定したという主張ではなく、setup・待機・後始末の修復と実機成功を記録する。

7件のlocal回帰検査で、旧deadlineの空待機・旧setup失敗後のprocess残留を再現し、修正後の
単一code request・接続/実行失敗時のprocess終了・stuck-closeのfail-closed終了を確認した。
共有L4のjob `l4job-6ee6ade9d3f7422eb06bbb018a66831a`と
`l4job-e91e43f69d3d42369bf09a08f4bd4a85`で、通常実行と65秒間無出力の実行を同じVMで連続検査し、
結果回収/hash照合/cleanup acknowledgementまで通過。その後の完全step jobも同じ修復CLIで完了した。
patch script・変更前後のsource・diff・依存版・local tests・probe source/result/receiptは
ignored `output/cell-mass-hc-20261028/transport-repair/`へ保全する。OSのVPN設定は変更していない。

完全step job `l4job-98bbb3b5e42a48c0a90f9b30b01f1eeb`は前回失敗時とbyte同一のdriver
（SHA256 `1184fe711051d5eb582b60784608b6e87dd9a015d94a36fb8e0fdb7803c825be`）を使った。
同じ検証済みnative binary（SHA256 `c7b9bcf848a1ad9880895cdf068c5aee6f7211592c7981eae94281ee4dfbbc1b`）を
各childでPython/Torch/CUDA/device/source digestと照合してloadする。新しいoracle試験を追加したのではなく、
前述の1,225 GPU tests/14全atom gateを通った同一binaryを測る。source/tests/training504ファイルはGit0b43283eとbyte一致。
初期値とCST/dense epoch3 checkpointのhash、MNIST70,000例resident、optimizer/model構成、TF32off、FP32、
N1024/A52,428/B32、4warm updates、各process内3×21step窓、各条件2 fresh processと逆順実行を維持した。
新しい学習epochは回していない。所要時間はwarm-up後の未計装なforward/backward/optimizer完全stepである。

| 保存状態 | 直前node版 ms（2process） | 融合grain64 ms | 融合grain128 ms | dense ms |
| --- | ---: | ---: | ---: | ---: |
| initial | 7.292 / 7.303 | 5.994 / 6.061 | 6.470 / 6.426 | 1.889 / 1.898 |
| epoch3 | 7.360 / 7.367 | 6.291 / 6.274 | 6.336 / 6.401 | 1.920 / 1.886 |

現在の速度候補は`site_scaled_hc8_g64_cin64_slab16_exact_reuse_node_z_cached32_expand_fused_nostore_h`。
initial平均6.028ms、epoch3平均6.282msで、直前node版より約17.4%/14.7%短縮。
旧site_scaled（同じjobで11.637/11.616、12.018/11.975ms）からは約48.2%/47.6%短縮した。
同時にCin grainも512から64へ変わるため、改善を融合単体の効果とは断定しない。
denseに対しては約3.18/3.30倍の時間がかかり、速度目標は未達。grain128はメモリを優先する別の候補である。

| 設定・状態 | peak allocated CST/dense MiB | peak reserved CST/dense MiB |
| --- | ---: | ---: |
| grain64 initial | 237.419 / 239.551 | 392 / 382 |
| grain64 epoch3 | 236.740 / 239.551 | 394 / 402 |
| grain128 initial | 235.717 / 239.551 | 392 / 382 |
| grain128 epoch3 | 235.698 / 239.551 | 394 / 402 |

表はfirst stepとsteady窓を合わせた上界でも同じ値。両融合設定のallocatedは両保存状態でdense以下だが、
initial reservedは10MiB上回る。よって全memory指標がdense以下とは主張しない。process総GPU使用量とGraph captureは未測定。
初回更新のwall時間は別記し、grain64でinitial1,269.6/1,274.0ms、epoch3は1,279.9/1,275.0ms、
denseは約40–42.5ms。これには初回準備費用が含まれ、表のsteady throughputと混同しない。初回費用の内訳は今回計測していない。

H保存だけを省いた非融合対照は速度が約7.3ms/7.37msのままで、epoch3 allocatedは239.170→238.018MiB。
6.4MiBのH配列をなくしても別段階がpeakになるため、実peakが6.4MiB減るわけではない。
融合64ではpeakが`G_parameter_grad_C_in`内へ移る。全24processで計装有無のlogits/loss/全model勾配/
更新Parameter・momentsがbit一致し、allocator全境界/peakの再構成も一致した。
融合8processすべてで5回の実測診断呼出が融合=5/旧Ggrad=0/別Cin=0となり、global H/G出力bufferも0bytesだった。
別passのepoch3 GPU作業はH/C約1.706ms、G/Parameter勾配/C_in融合約1.993ms、Zx約.500ms、Zy約.245ms。
これらを足して完全stepとは呼ばない。optimizerの同期APIは全processで0を維持した。

source archive SHA256 `c7bb3e5cee2f92dd1d3dca931f4911340c692f6c4f9f46aca09db691c4c3900f`、
result archive `4475f82c14f3f4cca7c08af5d4c5383e8801a29ae83036f464d52188e6c18ba6`。ignored evidenceは`output/cell-mass-hc-20261028/fused-profile-repaired/`。
source/result archives・binary・driver・生trace・allocator snapshot・recipe・per-process結果・照合JSONを保存した。
全3jobの後始末を完了し、所有sessionの停止とserver一覧からの消滅を確認済み。今回の通信修復・再測定ではAstraを使っていない。

再実行は同じdefault共有queueへ上記driverを`submit --timeout 600`し、
`pool.py --colab /tmp/torchcst-colab-075/bin/colab serve --workers 1`を使う。
実験用CLIを作り直す際の適用scriptは`transport-repair/apply_client_patch.py`、local回帰検査は
`/tmp/torchcst-colab-075/bin/python output/cell-mass-hc-20261028/transport-repair/test_client.py`。
元のCLIへ黙って戻したり、別queueや直接notebook実行で所有規則を回避しない。

**標本化による入力12基底へのfold（数式とCPU検査のみ・未実装）：**
最小input bankはT=4spacing、P=12spacing。Z基底のphaseは`omega*site*spacing`であり、
owner分類のhalf-cell shiftとは別なので、セル積分済み基底に次の恒等式がある。

\[
Z_{8,c}=\tfrac12Z_{4,c},\quad Z_{8,s}=-\tfrac12Z_{4,s},\quad
Z_{7,c}=\tfrac57Z_{5,c},\quad Z_{7,s}=-\tfrac57Z_{5,s},\quad Z_{6,s}=0.
\]

3tile窓の12サイトでは17実基底を12個へfoldできる。値用係数は
`q4c=p4c+.5*p8c, q4s=p4s-.5*p8s, q5c=p5c+(5/7)*p7c, q5s=p5s-(5/7)*p7s`。
中心微分には元の周波数が必要で、fold後に単純にomega4/5を掛ける方法は誤り。
ただし12個の値係数に`p4c-p8c, p4s+p8s, p5c-p7c, p5s+p7s, p6s`を加えれば、
元のomegaの微分を17個のcache値で保持できる。次の親が継承する子harmonic1–4は残る。
N64/1024・spacing3種・非ゼロ原点・全owner・torus seam・raw/windowの18CPUcaseで、
恒等式最大差2.17e-12、projection2.04e-13、中心微分1.03e-11を確認した。
これはGPUの全atom勾配や速度・memoryの証拠ではなく、採用設定にも含めない。

#### 1024の学習状態を使った段階別時間・メモリ監査（2026-10-28）

計測器は`benchmarks/training/profile_cell_mass_mnist.py`。runtimeのCUDAコードやPolar法則は
変更せず、直前の学習jobからinitializationとCST/denseのepoch3 checkpointをbyte固定して使用した。
checkpointのoptimizer状態も復元する。N1024/A52,428/B32、Q128、spacing=.125、
σ_min=.125/σ_max=12.5、最小tile4/8sites、同じSITE_SCALED Plan、全MNISTをGPU常駐。
実測はL4/Torch2.11.0+cu130/CUDA13.0。TF32/Graphなし、1caseにつきfresh process一つ。

まず初期・学習後それぞれ全52,428atomのY/dX/4列勾配を独立CPU FP64 FFT/autogradへ照合し、
rtol3e-6/atol2e-7を維持して通過した。runtime全346ファイルは先行検証sourceと一致する。
元の学習と同じ「backwardだけ」のwarmup後に1stepを記録し、さらに3stepでoptimizerをwarm化する。
その4更新後のParameter/momentsをCPUへ退避し、各計測passの直前に同じ状態へ戻した。
したがって以下の短窓は新たな学習epochの成績ではない。計時中の21stepでは通常どおり更新する。
計測器あり／なしは同一入力のlogits/loss・全model勾配・更新後Parameter/momentsがbit一致した。

計測なしの同期wallは21stepをまとめて測る3窓。同じprocess内の反復であり独立3seedではない。
初期CST中央値20.686ms / dense1.952ms、epoch3 CST21.084ms / dense1.888ms。
以前の全3epoch平均20.935/1.926msとは測定窓が違うので置き換えない。

次表は各段階の前後に同期した5step診断の平均wall ms。
**同期の追加で重なり方が変わるため、これらの和を計測なしの完全step時間とは呼ばない。**
別passの21step CUDA event区間でも近い値を確認した。

| 段階とコード | 初期 | epoch3 | epoch3の実GPU作業 ms（kernel/copy/memsetのみ） |
| --- | ---: | ---: | ---: |
| `prepare_layout` | 1.143 | 1.134 | 0.386 |
| `build_hierarchy`入力 + `build_window_moments` | 2.150 | 2.151 | 2.077 |
| `compute_h_c`（Hと出力C融合） | 2.872 | 3.191 | 3.183 |
| `expand_c`でY展開 | 0.388 | 0.393 | 0.364 |
| 出力側Zy + 窓集約Ry | 1.147 | 1.143 | 1.067 |
| `compute_polar_g_and_parameter_grad` | 1.911 | 1.888 | 1.842 |
| `compute_input_c` | 1.543 | 1.581 | 1.523 |
| `expand_c`でdX展開 | 0.444 | 0.442 | 0.419 |
| `CSTOptimizer.step` | 8.260 | 8.277 | 0.654 |

GPU作業列は別の3step profilerの実kernel/copy/memsetを、CUPTI correlation IDで
起動元CPU threadの最内側stageへ割り当てたもの。GPU annotationの区間や
`key_averages`の親子rangeを足すとidleや重複を含むため、その合計を使わない。
MLP外側の層・loss/gather/metrics・binding/dispatch等は別の残余であり、この表の対象外。
また非同期passのoptimizer host時間約16.6msには先行forward/backwardの完了待ちも入る。
それをoptimizer固有時間と解釈せず、前段を完了させた約8.3msと区別する。

optimizerでは1stepあたり160 kernel、208 copy、204 synchronizationを記録した。
GPU作業約0.65msに対し段階wall約8.3msで、更新則の変更より先にhost往復を調べる余地が大きい。
GPU profilerのoptimizer scopeには155回の`aten::item`と50回の`aten::_to_copy`がある。
CPU Torch2.13の補助的な呼出元監査ではscalar化154回中136回が
`KernelState.declaration()`（17 scalar × 8 snapshot）から発生した。
CPUの監査をGPU時間や同一call数の証拠とは扱わないが、runtime sourceの経路は一致する。
`polar_amp_width._split → parameter_dim → validate_layout`が
`cell_mass_axes(chart.declaration(), state.declaration())`を繰り返す。
`kernels/state.py`の`float(self.scalar(...))`がCUDAではhost読み出しになる。
固定設定の検証を変更時に維持しつつ、hot pathのsnapshot再構築をcacheすることが最初の候補。
実際の修正・速度再確認はまだ行っていない。

メモリも計測なしの21step窓と、別のallocator履歴passを分ける。

| 状態 | 計時前allocated CST/dense MiB | step peak CST/dense MiB | reserved CST/dense MiB |
| --- | ---: | ---: | ---: |
| 初期のwarm化後 | 223.434 / 232.221 | 241.827 / 239.551 | 396 / 382 |
| epoch3のwarm化後 | 223.434 / 232.221 | 240.773 / 239.551 | 396 / 402 |

CSTは平常時に約8.787MiB少ないが、epoch3のstep中増分は17.339MiBで、
denseの7.330MiBより約10.010MiB大きい。そのためpeakでは約1.223MiB超える。
reservedはcheckpoint復元／割当順にも依存し、この短窓のdense epoch3は402MiBになった。
元の全epoch測定のdense382MiBを置き換えない。全てTorch allocator値でありprocess総量ではない。

allocator履歴passの最大は`backward_total → compute_input_c`内。
同時に存在した主なglobal配列は次のとおり（値はTensorの論理容量）。

| 配列 | shape / dtype | 容量 MiB | いつまで必要か |
| --- | --- | ---: | --- |
| Rx（Zxの3tile窓を集約） | [32,509,17] FP64 | 2.113 | forwardから保存し、G/Parameter勾配生成後も現backward frame終了まで保持 |
| Ry（Zyの窓） | [32,253,17] FP64 | 1.050 | G/Parameter勾配生成後も現backward frame終了まで保持 |
| G | [32,52428] FP32 | 6.400 | dX用Cの全owner集約まで保持 |
| atom勾配 | [52428,4] FP32 | 0.800 | optimizerが使用、次のzero_gradまで |
| C_in | [32,509,17] FP64 | 2.113 | dX展開完了まで |
| C_in部分和 | [714,32,17] FP64 | 2.963 | C_in生成kernel内の縮約完了まで |
| layout indices/group/offsets | 元行・入力逆slot等 | 約1.31 | forward/backwardを結ぶ、backward完了まで |

Rx/Ryは同じ階層の3タイル窓の集約であり、Zの親を2子ずつ作る方式は変更していない。
Hはforwardで一時global [32,52428] FP32の6.400MiBを作り、C生成後に使い終わる。
現在の融合APIがHも返すための保存で、Polarのbackwardへは保存しない。
Hの書込み自体を省く候補はあるが、今回の最大peakでは既に解放済みなので、Hだけ消しても
このbackward peakの原因は残る。Z→R変換中は一時的に両方が存在することも履歴で確認した。
register/sharedはこのglobal表の全容量を置く場所ではなく、各CTA内のatom chunk計算だけに使う。

履歴のalloc/free entryのsizeは要求量であり、allocated counterのblock量とは異なる。
最初の要求量だけの再生では全ケースで不足したため、default native allocatorの512byte丸め・
大blockの非分割remainderを含めて再構成した。
[PyTorch v2.11.0の分割規則](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp#L2894-L2910)
に合わせ、4caseの全stage境界とpeak counterがbyte一致した。例としてepoch3のGは論理6.400MiBに対し
確保block約6.587MiB、Ryは1.050MiBに対し1.063MiBだった。
履歴passと計測なしpassのpeak差は1〜2KiBなので、その小差まで同一値とは呼ばない。
rawの要求量だけの再集計結果も改変せず残し、修正後を`derived/`に分けた。

分布とthread割り当ても一律ではない。H/Gは1thread=1atom、64threads/CTA・820CTAで、
M8/同じ32batchなのでσだけから大幅なper-atom軽量化は期待しない。
CのCTAは32threads、batch tile4で8個。owner人数からactive splitを生成しているが、
scratch/起動枠は安全上界のままである。

| 計時開始点 | 入力C active/cap | 出力HC active/cap | C_inのatom chunk lane利用率 |
| --- | ---: | ---: | ---: |
| 初期4更新後 | 263 / 714 | 265 / 458 | 94.2% |
| epoch3 + 4更新後 | 448 / 714 | 320 / 458 | 88.3% |

元の初期checkpointでは262/714と264/458であり、warmupの4更新で少数の境界移動がある。
最小階層への偏りと単一ownerへの集中は違う。元のepoch3では最小入力bank85.82%だが、
単一入力owner最大は0.414%、出力owner最大は0.866%。初期/epoch3でG+勾配はほぼ一定で、
H/Cは約0.32〜0.39ms増えた。分布だけを入れ替えた因果実験ではないのでσのみを原因とは断定しない。

次の実装候補は、(1)optimizerの固定宣言snapshotと同期を減らす、(2)G/勾配の後に不要なRyを
早期解放し、G/勾配/C_inを近づけて6.4MiBのG受け渡しを減らす、(3)prepareで得たowner人数を使い、実コストと
空きtask率からgrain/batch tileを決める順とする。階層Zとcell-mass/Polarの核は変えない。
Rxはautogradの保存とretained backwardの契約があるので、単なるdelで解放できるとは仮定しない。
Ryの早期解放だけなら論理1.050MiBの候補だが、どちらもまだ実装・再測定していない。
C_inの部分和をepoch3の実448taskだけにできれば論理容量で約1.10MiB減るが、
host同期・割当丸めも変わるので速度やpeak改善は再測定が必要。現段階では未実装。

source `bff1ee1a`、job `l4job-5432773eb82647e18a402c9868116fa8`、
source archive SHA256 `2ccb8eb3037d26b03a6fe0b341249832d9f6585b8cbd508e48af605d0d98fc20`。
raw/source/receipt/35 filesをignored `output/cell-mass-mnist-profile-20261028/verified-run/`へ保存。
修正後再集計は`derived/`、段階表と寿命図は同親`stage-and-lifetimes.png`、解析は`analysis.json`。
postprocess修正後もGPUで測った`Stages/step/cpu_template/restore/routing`のASTは一致する。
CUDAの追加変更・新たな学習epoch・allocator環境設定の変更はしていない。
CPU2,053 passed/5,384 skipped/既存警告18件、所有GPU/共有pool supervisor停止確認済み。

先行Polar runはsource `1e9a7cd4`、job `l4job-3b154a81ff6845b485d9a80332bb182e`。
L4/Torch2.11.0+cu130/CUDA13.0で22+2,621 passed/12 skipped、racecheck 0 hazards、
B32の全819/52,428atomについて独立FFT/autogradと3step optimizerの検査を通過した。
許容差はY/dX/全勾配3e-6/2e-7、Parameter2e-6/2e-7、moments4e-6/2e-8。
前述の2..40幅、同じMNIST全データ/B32/3epoch/seed120での結果は次のとおり。

| N | epoch3精度 CST/dense | 平均step CST/dense | allocated MiB | reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| 128 | 96.99 / 96.90% | 12.621 / 1.965ms | 213.540 / 213.264 | 366 / 364 |
| 1024 | 97.06 / 97.50% | 17.405 / 1.962ms | 235.929 / 239.883 | 396 / 382 |

この条件では完全step速度・両peakの目標は未達。幅範囲訂正後の結果には読み替えない。
σ中央値は128で6.708→2.951、1024で6.558→2.908へ縮小した。
無task角度更新の解析的radial減衰だけからの予測中央値2.951/2.906と近いため、
データ学習だけで獲得した偏りとは呼ばない。独立のablation学習ではない。
raw/source/checkpointsと491source照合は
`output/cell-mass-polar-integration-20261027/verified-run/`、解析と図は同親directoryに保存。
source archive SHA256 `3bf3adef38fb104c80189c39d667c5727211c1c8e77646ab706fa6c56c6ea75c`、
result archive `e7335004858acda21befa265d89c585454ca461c3b27071c4f2f5f4fbc26ebd9`。
所有runtimeは停止確認済み。

#### 物理sigmaによる接続（先行実験・Polarではない）

`presets.cell_mass`と`CellIntegratedProfileBinding`で面積方式を宣言し、
`PhysicalAmpWidthSpec`によりatomを`[amplitude,sigma,output_center,input_center]`のまま扱う。
旧Polar・点サンプル・L2の宣言にこの演算を混ぜない。チャートは入力・出力それぞれ
凍結1D周期RegularGridとし、T_minはprofileの数学的な境界としてcheckpointにも検査する。
新しい設計文書は増やさず、公開例はREADMEに置く。

Algorithm `cuda_cell_mass_hierarchical/v1`とRecipeの明示Planをcatalogへ登録する。
`VALIDATED_128`と`VALIDATED_1024`は既存の検証済みCUDA kwargsを選ぶ。
自動selectorとbenchmark DBへの採用は含めない。新しいprepareは不変のgeometryだけを
保持し、毎forwardに現atomから分類・countingを作る。atomの複製・全係数表は作らず、
layoutはstate_dictへ保存しない。Torch参照は同じ宣言からセル積分を独立に計算する。

`CSTOptimizer`の`cell_mass_physical`方針は物理sigmaへの提案だけを
`[sigma_floor,min(Q_in,Q_out)/3]`へ射影する。振幅と中心はEuclidean更新。
評価時にsigmaを丸めたりlog-widthへ変換したりしない。普通のoptimizerを使う場合は
呼び出し側が定義域を維持する。まだ二階微分・Graph学習を保証しない。

**この接続と以下のMNIST結果はPolarを使わない。** 既存Polarは`[s,t,入力中心,出力中心]`
から振幅とactivity由来の幅を取り出し、幅をtask lossからdetachする。今回の物理4列と
同じshapeでも意味・並び・backward・更新則が違い、現在のAlgorithmはPolarを拒否する。
以下のsigma直接学習の観測を、Polarでのsigma分布の検証として扱わない。

Polarの状態・更新則と今回のセル積分/面積正規化profileは分離して組み合わせられる。
後続で接続するならCUDA内でPolarから振幅・幅を取り出して分類・H/C評価へ渡し、
既存の幅stop-gradientとPolar更新・moment輸送を維持する必要がある。
物理atom表へのglobalコピーを前提にしない。この先行実験の時点ではPolar接続は未実装だった。
その後の実装・検証は直前のPolar接続節で追跡する。

接続後のGPU検査は128/1024のY・dX・全atom勾配と更新を確認する。
その後の最初の学習確認はMNIST全60,000/10,000件、3 epoch、B32、seed120。
`784→width→width→10`の中央正方層をCST/denseで比較し、widthは128/1024、
atom数はfloor(.05*width²)。同じ初期実効W・外側層・各epochのshuffle順を使う。
初期sigmaはUniform(2,40)、初期振幅だけを実効WのRMS=1/sqrt(width)へ調整する。
これは初期化のgainでありprofileの面積正規化を変える処理ではない。

実験harnessは`benchmarks/training/cell_mass_mnist.py`。
CSTは公開CSTLinear＋CSTOptimizer、denseは通常のfused AdamWを使う。
データgather・loss・全backward・実optimizer込みのepoch時間とallocated/reservedを測り、
評価・sigma診断・保存はその計時とpeakから除く。常駐データは両者のpeakに含む。
sigma分位点・初期値からの変化・bank人数・owner偏り・group人数をepochごとに残す。
1 seed/3 epochは動作と学習傾向の確認であり、汎用精度の証明ではない。
従来B4の合成完全step結果とB32の実学習結果は別の条件として記録する。
2048/4096/8192対応は、この結果を見てから進める。

#### 公開接続とMNISTの実測結果

source `bc39cb26` / snapshot SHA256
`4427c57562a01b96f552a37a6334cc9e5fc939aac031b8f0a05635eaf94411ae`、
job `l4job-186f9896e1094c1a8989758e90710752`。
NVIDIA L4、Torch 2.11.0+cu130 / CUDA 13.0 / Python 3.13.16。
CPU suite 2,043 passed/5,344 skipped、GPU新旧cell-mass suite 2,621 passed/12 skipped。
公開CSTLinearと直接executorのY/dX/全4列勾配はbit一致。
全819/52,428atomの3 step更新を、各stepの更新済み参照から独立CPU FFT/autogradへ照合した。
Y/dX/全4列勾配はrtol3e-6/atol2e-7、Parameterは2e-6/2e-7、
AdamW moments/stepは4e-6/2e-8を維持して通過。sigmaの射影は参照側にも明示的に適用した。
Graph・二階微分は未検証ではなく、この新Algorithmの非対応条件として明示する。

以下は全MNIST、3 epoch/B32/1 seedの結果。中央層だけを置き換え、外側はdenseのまま。
初期test loss/accuracyと各epochのデータ順hashは両方式で一致した。
初期化gainは128で4.34224833、1024で10.80467186。

| 中央層 | epoch3 test accuracy CST / dense | 平均step CST / dense | 3 epoch合計時間比 | peak allocated CST / dense | peak reserved CST / dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128×128 | 96.75 / 97.01% | 5.142 / 1.953ms | 2.633 | 213.633 / 213.264MiB | 366 / 364MiB |
| 1024×1024 | 96.70 / 97.59% | 8.353 / 1.947ms | 4.290 | 242.322 / 239.883MiB | 392 / 382MiB |

精度は最良epochの選び直しではなく固定epoch3。1024のCSTはepoch2が97.15%、
epoch3が96.70%で、3 epochだけから一般的な精度同等性や優劣は主張しない。
メモリはTorch allocatorの各epoch peakの最大、常駐MNISTデータを含む。
実GPU process全体の使用量ではなく、Graph captureも行っていない。
学習時間は実optimizer・gather/loss・外側dense層も含む同期wall。
**このB32実学習では時間2倍以内・dense以下のメモリは未達**。
前のB4単体合成stepの達成判定をこの条件へ拡張しない。

1024のepoch2直前allocatedはCST 238,960,128B、dense 251,535,872Bで、
CSTが11.993MiB少ない。一方、CSTの計時中peakはさらに14.432MiB増え、
最終的にdenseより2.439MiB多い。128もpeakは0.369MiB多い。
B32でH/Gはそれぞれ6.400MiBなので、H/G保存とC部分和の寿命が次の調査対象。
ただしこのrunに演算別のallocator trace/時間分解はなく、peak発生箇所や
CSTOptimizerの時間寄与を個別に確定したわけではない。

sigmaの変化量の中央値/P90/最大は128で0.07963/0.21008/0.74673、
1024で0.03507/0.09141/0.34975。1024のsigma中央値は21.04161→21.04075。
全atomは有限で、正のsigma定義域を維持した。
1024の入力bank人数（幅4/8/16/32/全周）は
`[2821,5413,11086,22003,11105] → [2812,5418,11080,22013,11105]`。
出力（幅8/16/32/全周）は
`[8234,11086,22003,11105] → [8230,11080,22013,11105]`。
階層が初期値から変わったatomは入力243/52,428、出力184/52,428、
階層またはownerが変わったものは408/284。人数が同じでも個体の移動は起きている。
128では階層変化が入力9/819、出力7/819、階層またはowner変化が12/14。

3 epochでsigmaが強く偏る傾向は未確認。一方、1024では初期から全周owner一つに
11,105atom、約21.18%が集中し、最終も同人数だった。最終の非空groupは847、
最大group人数11,105、最大/平均は179.407。階層別の仕事量の偏りは残っている。
この実学習fixtureでの保存領域と完全step時間を改善してから、大型化へ進む。
新しい課題・seed・hyperparameterの追加は、この観測から次に判断する。

source/result/spec/receipt/binary/初期fixture/最終model・optimizer checkpointを
ignored `output/cell-mass-linear-integration-20261027/verified-run/`へ保全した。
runtime 344ファイル＋tests/trainingを含む488sourceファイルをsnapshotへ照合し、
配布物2つの全runtime 344ファイルも照合した。結果はpoolでSHA256検証済み。
source取得・dataset checksum・raw epoch値・最終checkpointからのowner移動集計を残し、
所有L4停止・supervisor終了も確認した。新規DB結果の提出や自動dispatch採用は行っていない。

新しいlayout cacheにatom値や全atomキーは保持しない。基本のTensorは
`rows[A]`（int64）、`groups[G,4]`（int32）、`offsets[G+1]`（int64）。
countingでは任意で`reverse_slots[A]`（int32）、`reverse_offsets[input_nodes+1]`（int64）を加える。
groupがない場合もoffsetsは[0]。元Parameterへの参照とversionを持ち、更新後の古いcacheの使用を拒否する。
同じatom値・planなら異なるXで再利用できる。sort/uniqueのキー・countsはprepare中だけのglobal scratch。
一時領域の再利用は可能だが、block終了で消えるsharedだけで後続kernelへ順序を渡すことはしない。
現在のsort/uniqueは動的shapeとhostへのerror確認を含むため、layoutのCUDA Graph対応は主張しない。

各Z bankはその幅のtile nodeだけを持つ。全bankそれぞれに全幅の木を重複して作らない。
local node数はN/n_min、N/(2*n_min)、...と減り、最後に全周nodeを一つ置く。
1nodeは`[batch,1+2*M]`。direct kernelのsharedは128個のdouble部分和で1,024bytes/block。
`team_size=1,2,4,8,16,32`のthread集団が一つのZ要素のsite和を分担する。
Hのatomを複数threadで分担するという意味ではない。
最初は全宣言bankのnodeを準備し、使われないbankの省略は依存関係を含めた後続の改善とする。
cacheの`tensor_bytes`やZの容量はpayloadであり、完全stepのallocatorピークと混同しない。

### 階層版backwardの実装（検証済みの初版）

`hierarchical_linear(plan, x, atoms)`から次の順番で呼び出す。

```text
forward: prepare_layout → build_hierarchy(X) → H → C_out → Y
backward: build_hierarchy(dY) → G
                                ├→ C_in → dX
                                └→ parameter_grad(Zx,Zy,H,G) → d_atoms
```

H/Gは振幅を含まない`[batch,sorted_slot]`配列。autograd経路では入力dtypeで保存し、
H/G縮約・微分・Z/Cの計算はFP64を維持する。低水準診断の既定保存dtypeはFP64。
学習対象にatomがある場合だけ、
forwardのZx/Hをautogradの保存状態に残す。入力だけの勾配ならZx/Hを残さない。
backwardのZy/Gは同じ呼び出しのdXとparameter_gradで共用する。
C_out/C_inはbank別`[batch,node,mode]`で、展開後に解放する。
元X、密なW/dW、packed_atoms、全atom係数表、独立したdH配列は保存しない。
低メモリ用のH再計算切替は今後の実行方式であり、この初版ではZx/H保存を使う。

各threadは別atomを担当し、batch方向の和をregisterに持つ。
入力profileをv、出力profileをu、振幅をaとすると、H=Σ_i X_i v_i、G=Σ_o dY_o u_o。

```text
d_a    = sum_batch(H * G)
d_mu_i = sum_batch(a * G * dH/d_mu_i)
d_mu_o = sum_batch(a * H * dG/d_mu_o)
d_sigma= sum_batch(a * (G * dH/d_sigma + H * dG/d_sigma))
```

`backward.cu::parameter_grad`はZx/Zyから微分の縮約をregisterで計算し、元atom行へ一度だけ書く。
幅微分には既存effective_radiusの下限補正と、係数の面積正規化の商の微分を含める。
選択済みbank・ownerは固定して滑らかな枝を微分し、次のforwardで更新後の値を再分類する。
選択中のbankでは周期Pとsite基底が固定されるため、Z自身の幅微分は不要である。
atom側の係数だけを微分する。raw Fourier係数をr_kと書くと、

\[
A_k=\frac{2r_k(\beta)}{P r_0(\beta)},\qquad
\partial_\sigma A_k=
\frac{2\partial_\sigma\beta}{P r_0}
\left(\partial_\beta r_k-\frac{r_k}{r_0}\partial_\beta r_0\right).
\]

cos/sinの中心微分はそれぞれ-omega*sin、omega*cos。
定数係数1/Pの幅・中心微分は0。共通密度係数T_M(beta)^-2は商から消えている。
H/G作成は`contraction.cu::compute_h`の軸切替を共用し、dXは入力側Cを同じ展開kernelで評価する。

出力順layoutでは入力ownerの範囲が非連続なので、初版のC_inは非空groupのdescriptorを走査し、
該当する連続atom範囲だけを32atom shared chunkへ読み込む。atom値の転置コピーやdX atomicは使わない。
このgroup走査は高速化対象であり、完全stepの速度・メモリ達成は実測前に主張しない。
保存atomとlayout配列は`save_for_backward`で更新検査し、retained backwardに対応する。
二階微分とsortを含む全体Graph captureは未対応。

### 最適化の着手：元の版の処理別実測

ユーザーの指定により階層・sigmaの意味・セル積分と面積正規化・H/Gの意味を維持し、
完全step時間をdenseの2倍以下にする。T_min/Pを変えて別のWを測ったり、密Wを作って置換したりしない。
allocatedとreservedの両方をdense以下に保ち、Y/dX/全atom勾配の検査許容差も変えない。
GPT-6.1 Terraの3担当でC集約・atom数学・pipelineを調査し、GPUは単一の共有poolで測定する。

最初にsource `48d2d4a0`（runtime `1ec0f66b`と同じ）を凍結して診断した。
同一fixtureの各低水準stageをCUDA Graphへ一度captureし、21 replayのevent中央値を測った。
prepareはGraph未対応なので同期wall中央値。完全stepの採用時間へ足したり差し引いたりしない。
別のprofile traceでも1024のC kernel2回が約16.17msを占め、局所診断と対応した。

| 処理 | 128診断ms | 1024診断ms |
| --- | ---: | ---: |
| Zx | 0.237568 | 1.795872 |
| H | 0.088064 | 0.501536 |
| C_out | 0.156672 | 7.999072 |
| Y展開 | 0.197632 | 0.198432 |
| Zy | 0.227328 | 1.746048 |
| G | 0.080896 | 0.535552 |
| C_in | 0.242688 | 8.179552 |
| dX展開 | 0.256000 | 0.256640 |
| parameter_grad | 0.191488 | 1.250592 |
| prepare（同期wall） | 0.294586 | 0.513347 |

1024は849 group、1 ownerの最大atom数は入力・出力とも10,539。
約20%が全周bankの単一ownerへ集まり、従来は1warpが全chunkを順に集約していた。
C_outとC_inが主対象であり、次にZ生成の三角関数評価を削る。

job `l4job-67913d6841dc437ba6f17f3727c3676f`、L4/Torch2.11.0+cu130/CUDA13.0、driver239.16秒。
source SHA256 `7c51f473225632601327ee4ce7887c870ba9bf65a4e2ed528afa816e6918596d`、
result SHA256 `ea32824b0a88647ebe18f822512c9aee680dd6b36be1a5edd177e6a772c8b2c3`。
327 runtimeコードと結果11ファイルを照合し、ignored `output/cell-mass-opt-20261027/baseline-profile/`へ保全。
共通headerの別途byte照合も行う。profileは診断専用で、検証済み完全step/ピークの代替にしない。

候補source `8f129556`は次の変更を実装し、下記のCUDA検証を通過した。

- owner×固定split数のFP64部分Cを作って縮約。容量はatom数でなくowner数とsplit数で制限する。
  逆側の非一致groupにはblock同期を置かず、warp ballotで対応rangeを選ぶ。
- 既存の正の多項式をHorner評価し、値と解析微分をまとめる。全atom係数表は作らない。
  basisだけの定数表は444 doubles、3,552bytes/CUDA translation unit。
- Zのsite位相と出力のharmonic位相はFP64漸化式で再利用し、同じセル積分の基底を評価する。
  chart端の3tile周期回転は整数mod3のroots of unityで表し、全周bankは回転0。
- `fuse_g_grad=True`で、同じthreadがGとatom勾配を作る。Gを保存dtypeへ丸めてから勾配へ使う。
  `c_splits`と融合の有無は独立に比較し、採用前の既定は1/Falseのままとする。

CPU上の数式回帰や相互レビューはGPU検証・独立oracle・速度改善の証拠には数えない。
候補の全gateと完全stepピークが通るまで、下の検証済み版の結果と混在させない。

### 最適化第1段：C分割と位相再利用の実測

runtime `8f129556`、L4/Torch2.11.0+cu130/CUDA13.0。
階層版270件＋既存1816件で**2086 passed、failed/skipped 0**。
独立oracle/AdamW検査では、追加引数なしのautograd呼び出しもテストdriver内で
小さいatom数はC2分割、大きいatom数は16分割・G/勾配融合へ切り替え、候補を通した。
低水準stageの独立照合、分割数2/7/16・片側M16・空/端数batch・retained backwardも検査した。
M16の1024rootのZをteam1/32で独立Gauss積分へ照合した。
racecheckはsplit/fusedの8件で0 hazards/errors/warnings。

128/819atomと1024/52,428atomを、C分割1/2/8/16と融合の6設定で
既存FFTセル積分の独立CPU参照へ照合し、全Y/dX/全atom4列がPASS。
1024の最大絶対誤差は全設定でY1.7711e-8、dX1.99152e-8、atom勾配2.7011e-8。
profile式、sigma、T_min/P、許容差とH/G保存精度を変更していない。

同じ完全step条件で、各設定1 fresh process、warmup5/同期wall21標本。
以下は分割・融合を選んだ候補であり、公開dispatcherへの採用ではない。

| 設定 | 完全step中央値 | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、C1・非融合 | 1.376251ms | 166.5KiB | 2MiB |
| 128、C2・融合 | 1.182218ms | 242.5KiB | 2MiB |
| 同job dense128 | 0.666325ms | 264.5KiB | 2MiB |
| 1024、C1・非融合 | 14.312060ms | 5.332MiB | 8MiB |
| 1024、C8・非融合 | 4.925133ms | 5.332MiB | 8MiB |
| 1024、C16・非融合 | 4.416706ms | 5.332MiB | 8MiB |
| 1024、C16・融合 | 4.100332ms | 5.854MiB | 8MiB |
| 同job dense1024 | 0.664340ms | 16.063MiB | 22MiB |

128のC2融合はdenseの約1.77倍の時間で、時間とメモリ条件を満たす。
128のC8/C16は約1.2msでもallocatedがdenseを超えたため選ばない。
1024は約6.17倍で未達だが、元の23.21msから大きく短縮した。
別jobとの大きな差は初版改善の方向を示すが、採用性能は選択設定の独立再測定で確認する。

候補の1024診断（Graph event、完全stepへの加減算は禁止）は、
Zx/Zy=0.203/0.189ms、H/G=0.412/0.445ms、C_out/C_in=0.770/0.633ms、
Y/dX展開=0.055/0.070ms、非融合勾配0.956ms、G/勾配融合1.136ms。
次はatom側の基本位相・累乗・正規化逆数を再利用し、Zの大きなnodeを小区間のblockへ分割する。
sortを置換するカウント配置は同じgroup順序でindexだけを扱い、別途検証する。

job `l4job-41b59f34aa8c4c5aa02455316d9f5e79`（driver498.89秒）。

- source SHA256 `88faf220099f145a358ae4fe810bcdf64838d7ddc443a9fe9a1cab596eec810a`
- result SHA256 `6a63e8e37228bb6bcafbfa5850448010fbbfd05403471ad4d85750eb7d6e5cfa`
- driver SHA256 `a9dee371e0980c87d27da1ee304ab9b9e47a4cfbab9a5b935a0b5908f2dafdcd`
- binary SHA256 `bc0d8c91c0d942b6aec50aa447549708ad7c6e9f40e8937e2209c596d03702bf`

327 runtimeコードとmanifest58ファイルを照合し、ignored
`output/cell-mass-opt-20261027/first-candidate/`へsource/result/spec/receiptを保存。
queued状態でdriverのreport hook修正のため取り消したjob `l4job-c503f15621bb405e8d903f3aaee10117`は
GPU成功や実測へ数えない。検証前の候補・失敗・不採用設定も削除しない。

### 最適化第2段：harmonic stream・counting配置・Zの分割

runtime `0c4fc64d`、L4/Torch2.11.0+cu130/CUDA13.0。
階層版304件＋既存1816件の**2120 passed、failed/skipped 0**。
CPU全体2037 passed / 4837 skipped / 既存警告18件（38.76秒）、Ruff/format/build通過。
wheel/sdistの337ファイルをsourceへbyte照合した。

1 thread内で基本位相、betaの累乗、正規化逆数を再利用し、harmonic順に係数を生成する。
C_outとC_inでは、同じownerのatom列を均等なS区間へ分ける。
元の係数とprofile式は変えず、

\[
C_{h,q,b,k}=\sum_{s=0}^{S-1} C^{\mathrm{partial}}_{h,q,s,b,k}
\]

と最後に集約する。各atomはそのowner内で一つのsplit・一つのthreadへ割り当てる。
係数表を全atom分globalへ保存する方式ではない。

Zは大きなtileも32site以下の小区間へ分け、同じbank周期で基底を積分してから足す。

\[
Z_{h,t,b,k}=\sum_{j\,\subset\,t} Z^{\mathrm{partial}}_{h,j,b,k}.
\]

この小区間は実行上の分担であり、数学のT_hやP_hを変更しない。
1024/B4/M8のpartial Zは入力87,040bytes・出力69,632bytes、呼び出し中だけ保持する。
完成した実tileのZから、従来通り親の偶数harmonicを継承する。

counting配置は固定された辞書順のCartesian groupを使う。1024の現planでは1769 groupで、
空groupもoffsetsに残す。atom値のコピーは行わず、元row番号だけを並べる。
カウントとrow番号の予約にwarp集約atomicAddを使うが、Y/dX/atom勾配へatomicで加算しない。
一時keys/count/cursorはprepare内だけ、保存cacheはrows/groups/offsetsだけである。
4096 groupを超えるplanはcounting経路を拒否し、従来sort経路は残す。

独立FFT参照、空group/重複offset、空atom/batch、片側M16、32siteのZ分割、
全team・retained backward・幅/有限値拒否を検査した。128/1024の全atom勾配も全候補でPASS。
許容差は以前と同じ。racecheckとsource/result検証の詳細は保存artifactへ置く。

| 完全step設定 | 中央値 | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、C2/融合/sort/Z非分割 | 1.132184ms | 242.5KiB | 2MiB |
| 同job dense128 | 0.668032ms | 264.5KiB | 2MiB |
| 1024、C16/融合/sort/Z非分割 | 3.273565ms | 5.854MiB | 8MiB |
| 1024、C16/融合/sort/Z32分割 | 3.003025ms | 5.854MiB | 8MiB |
| 1024、C16/融合/counting/Z非分割 | 3.090996ms | 5.875MiB | 8MiB |
| 1024、C16/融合/counting/Z32分割 | 2.795126ms | 5.875MiB | 8MiB |
| 同job dense1024 | 0.667932ms | 16.063MiB | 22MiB |

1 fresh process/方式、warmup5・wall21標本、その他の条件は第1段と同じ。
128は約1.69倍、1024は約4.18倍。メモリ条件は両方で満たすが、1024の時間目標は未達。
1024の診断値はZx/Zy=0.050/0.048ms、H/G=0.313/0.313ms、C_out/C_in=0.467/0.403ms、
Y/dX展開=0.055/0.070ms、G/勾配融合=0.929ms。今後は係数の定数比とgradのregister保持が対象。

job `l4job-af3d11b157de415ca147825b7eb73a0d`（driver499.96秒）。

- source SHA256 `ef46a7db0977d21f02054fa29444d0945fada72c584f9146262aa3973cb4e536`
- result SHA256 `85b3b9c5092fa655a01182f788e1425b7068bff6503e0ea28ff30a9c74c4b09f`
- driver SHA256 `452575e1fbeb4f7c0d30c5ebc7924a374845c0db0b8899d48e6825f7b38bdc32`
- binary SHA256 `c093812a93223aa4dc7dd1a9764d87124f5ba056d2295110c8315109886fb701`

327 runtimeコードと結果manifest40ファイルを照合し、ignored
`output/cell-mass-opt-20261027/stream-candidate/`へsource/result/spec/receiptを保存した。

### 最適化第3段：定数比・partialページ・atom blockサイズ

runtime `8dc468cb`。basisだけの前係数72個を定数表へ加え、毎harmonicの定数比を
除算し直す処理を省いた。多項式444個と合わせ4,128bytes/CUDA translation unit。
β累乗の微分も逐次更新する。profileの近似次数やprecisionは変更しない。

C32/64分割は、単一partialが1MiBをわずかに超えて大きなallocator予約へ移ることを避けるため、
各partial Tensorが1MiB以下になるようsplit方向をページへ分ける。総partial容量は減らさず、
同じFP64値を同じ順番で縮約する。pointerはkernel引数として渡し、GPU pointer表は作らない。
元C一枚自体が1MiB超ならページを小さくできないため、そこで条件達成とは主張しない。

atom担当のblockを32/64/128/256 threadsへ変えても、1 threadが別atomを処理する。
同じlayoutを使った比較ではY/dX/各atom勾配がbit単位で一致することを検査した。
counting配置での3stepのAdamWと階層遷移も追加し、**2165 passed、failed/skipped 0**。
全52,428atomの独立Y/dX/4勾配は全候補でPASS。

| 1024、counting/Z32/融合 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| C16、128threads | 2.586894ms | 5.875MiB | 8MiB |
| C16、64threads | 2.554607ms | 5.875MiB | 8MiB |
| C32、128threads | 2.600685ms | 6.382MiB | 8MiB |
| C32、64threads | 2.561422ms | 6.382MiB | 8MiB |
| C64、64threads | 2.501205ms | 7.396MiB | 10MiB |
| 同job dense | 0.678842ms | 16.063MiB | 22MiB |

128のC2/128threadsは1.125077ms（dense0.686617ms）、allocated242.5KiB/dense264.5KiB、
reserved両方2MiB。両サイズでメモリ条件内、1024は約3.68倍で時間目標は未達。
単一fresh processずつ、wall21標本の診断であり、この小差を未再現の普遍的勝者にはしない。

job `l4job-3752c02b51b046988a9396aa7f5464fa`（driver502.43秒）。

- source SHA256 `94edf7e0793c90c284103d28b3f2135495adf9f177491706a4847808895b7b7e`
- result SHA256 `07d210687f848fe6ee54d1846c3a4cee18ae49ddd82e061eeea078c752c4d4cb`
- driver SHA256 `01db43bf9f3b475fd848fc64a3a8ab317062034f05abdd8e1b63a78eb3e44bff`
- binary SHA256 `ea6609d53591b1afaa6660b609fc79e7b9ce4c9853c2af75408330bc446677c9`

327 runtimeコードと結果43ファイルを照合し、ignored
`output/cell-mass-opt-20261027/paged-candidate/`へsource/result/spec/receiptを保存した。

次の第4段では、Cartesian groupの直接参照と加重微分を検証した。


### 最適化第4段：加重微分とgroup参照の簡略化

source `b0cb69a5`。Gとparameter勾配の融合で、batchごとの微分配列を作らず、
`sum_b amplitude * H_b * Z_b` と `sum_b amplitude * G_b * Z_b` を先に縮約する。
Gの保存dtypeへの丸めと振幅微分で使うGは従来通り。FP64加算順の変更は独立参照で検査した。
counting layoutでは既知のCartesian group番号を直接参照し、C_inで全groupを走査しない。
また、cacheを作らないcounting経路ではXとatomの有限値検査を同じerror wordへまとめる。
cache再利用時の新しいXの検査は残す。

同一の128/1024、B4、M8、初期sigma0.2..40、seed120、T_min/P不変。
各設定はfresh process、warm5、同期wall21回の中央値。prepare、全forward/backward、
fused AdamWと一時配列の作成・解放を含む。Graphなし、TF32無効、cuBLAS workspaceなし。

| 条件 | 完全step | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 128、C2/融合/sort/Z0/128threads | 1.099659ms | 242.5KiB | 2MiB |
| 128、matched dense | 0.679848ms | 264.5KiB | 2MiB |
| 1024、C16/融合/counting/Z32/64threads | 2.386650ms | 5.875MiB | 8MiB |
| 1024、C32/同上 | 2.391108ms | 6.382MiB | 8MiB |
| 1024、C64/同上 | 2.277464ms | 7.396MiB | 10MiB |
| 1024、matched dense | 0.654805ms | 16.063MiB | 22MiB |

1024初版23.211msから約10.2倍短縮したが、dense比3.48倍の時間を要する。
128はdense比1.62倍で時間条件を満たし、1024はこの時点では未達。
この1jobの結果を全幅分布・全GPUへ一般化しない。

低水準Graph-event診断は、1024 H/G各0.289792ms、融合G/勾配0.823296ms、
C_out0.409600ms、C_in0.325632ms、Zx0.050176ms、Zy0.048128ms、
Y展開0.055296ms、dX展開0.069632ms。診断値を加算して完全stepとしない。
独立Y/dX/全52,428atomの4列勾配、およびGPU・数学2171件が通過。
CPU全体2037 passed / 4888 skipped / 既存警告18件。
128のAdamW Parameter/moments/step照合はsort/counting両方を含む。
1024はこのjobで全勾配を照合し、optimizer状態の独立照合まで行ったとは主張しない。

job `l4job-78ac2e52bb064b3ca575953af131dde7`。
source archive SHA256 `425469c004887b648c1d17d537ae7b417092416a1386f0b61659218df28716d1`、
result archive `3be67c2a2f0e8a36722230b2556425c5b5fdb02048d7569ccec097900bf78f8c`、
frozen driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、
loaded binary `5075323aa51a7599dffc9f8450394e98f5c82c688f621181db0d619d28dadc87`。
ignored `output/cell-mass-opt-20261027/weighted-candidate/`にsource/results/spec/receiptと照合記録を保存。



### 最適化第5段：bankの定数と互換階層範囲

source `2a507b1b`。bankのinverse_periodとomegaをhost計画で一度計算する。
make_profileは同じsigma/scaleを使い、半径を経由する式を無次元のrootへ整理した。
Fourierセル積分のk=1..16の逆数を定数表へ追加し、表全体は4,264bytes/CUDA translation unit。
Cartesian Cの探索では、sigmaの分類区間に共通部分がないbank組を省く。
全周bankの上限はatom dtypeへ丸めた共通sigma上限であり、FP32端点を落とさない。

GPU・数学2171件、128/1024の独立Y/dX/全4列勾配が通過。
128 C2/融合/sort/Z0/128threadsは1.106592ms（dense0.666637ms）、
1024 C64/融合/counting/Z32/64threadsは2.201958ms（dense0.661160ms）。
allocated/reservedは第4段と同じでメモリ条件内。単一fresh processずつの診断値であり、
数%の差を普遍的な改善率としない。1024の時間条件はこの時点では未達。
1024の段階診断はH/G各0.278528ms、融合G/勾配0.741376ms、C_out0.408576ms、C_in0.343040ms。

job `l4job-0b0b8d63e3b044ed86a5b324e1ad0f90`、driver474.24秒。
source SHA256 `c865dc5e36ccfaa594487dd1f5e1e9953be123d98229fb84e0e17117999319dd`、
result `405bc3499a649efe7ecd94b920819d38880955dbf7fb3c51ba5ce48d7f731208`、
driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、
binary `766c6ab807d14cf4dced308f244be47bc578f69b0ed6d835abaded5f7dbced2c`。
ignored `output/cell-mass-opt-20261027/bank-candidate/`へ327 runtimeと結果34ファイルの照合を保存。

### 最適化第6段：shared配置、M8専用化、owner window

- source `7c610fcc`：Cの論理32atomは維持し、shared物理strideを33にする。
  M8/B4ではsharedは5,376から5,544bytes/block（+168bytes）。
  G/勾配のbatch和を先に縮約し、正規化のk=0値・微分も一度に計算する。
- source `468c858d`：M8のH/C/融合勾配を固定mode数で展開する。他のeven Mは汎用経路を残す。
  registerやcode sizeの増加で逆に遅くなる可能性を含め、同条件で比較する。
- source `760d4933`：全raw Z階層の完成後、ownerごとに3タイルの位相を補正して合算する。
  そのwindow ZだけをH/G/全勾配が一度読む。full-chartはraw値と同じ。
  `window_moments=True`の研究flagで切り替え、元のraw Z/低水準APIは維持する。

windowを元の階層Zと区別してRと書くと、各bank/owner qで

\[
R^{c,s}_{b,q,k}=\sum_{\delta=-1}^{1}
 R_{\mathrm{image}(q+\delta),k}\,Z^{c,s}_{b,\operatorname{wrap}(q+\delta),k}.
\]

**階層のマージは2つずつのまま。3枚の和は、同じ階層でH/Gが使う窓を先に作る別の処理である。**

| 値 | 担当する幅 | 作り方 |
| --- | --- | --- |
| raw Zの親node | 2T_h | 周波数が一致する親の偶数harmonicを左右2子から継承。奇数は直接生成 |
| R_h,q | 3T_h=P_h（local bank） | 同じ階層のZ_h,q-1・Z_h,q・Z_h,q+1を、chart image補正後に合計 |
| 全周bankのR | Q | 全周bankのraw Zと同じ |

Rを作ってもh、T_h、P_hやsigma分類は変えない。三分木へ変更する処理ではない。
この和はatomではなくownerにだけ依存するので、同じwindowを使うatom間で共有できる。
DCは単純和。中心の最後の半セルに対するmake_profileのimage補正は別の処理として残す。
windowはmergeへ戻さず、mergeが全て完了してから別bufferで作る。
rawとwindowは同shapeなので低水準APIには`windowed=True`を必ず渡す。
C/展開は変えない。入力windowは33,184bytes、出力は15,776bytes（1024/B4/M8）。
変換時のraw/window同時生存を含めた完全stepピークと、FP64再結合後の全勾配を検査する。

これらの単一jobのGPU検証結果は次の通り。全てsame physical T_min/P、同じ初期profileからのdense、
TF32なし、Graphなし、workspaceなし、warm5/同期wall21、fresh processずつ。

| runtime / 設定 | 128 CST / dense | 1024 CST / dense | GPU・数学 |
| --- | ---: | ---: | ---: |
| `7c610fcc` shared stride・加重回転 | 1.108148 / 0.673994ms | 2.169511 / 0.669591ms | 2171 passed |
| `468c858d` M8専用化 | 1.090994 / 0.660951ms | 2.094633 / 0.683576ms | 2171 passed |
| `760d4933` owner window R | 1.141979 / 0.663074ms | 1.840612 / 0.669264ms | 2217 passed |
| `833e68dc` Axisコピー回避・shape共用 | 1.123588 / 0.677137ms | 1.661862 / 0.666132ms | 2217 passed |

128はC2/sort/Z0/128threads、1024はC64/counting/Z32/64threads、G/勾配融合あり。
Rは後半2行で有効。allocated/reservedは全行、128=242.5KiB/2MiB、
1024=7.396MiB/10MiBで、各dense264.5KiB/2MiB、16.063MiB/22MiB以下。
全52,428atomの4列勾配、Y/dXを各jobの独立FFT参照へ照合した。
CPU回帰はwindow段階2037 passed / 4934 skipped / 既存警告18件。

Rは1024で有効だった一方、128では追加起動の費用を補えない結果だった。
全サイズ一律の有効化はせず、最終設定は別fresh processで再確認する。
source833の1024はdenseの約2.49倍の時間で、時間条件はこの時点では未達。
初版23.211msから約14倍短縮したが、固定bank版の別周期・別幅分布とは改善率を比較しない。

source833はAxisのmode値を書き換える代わりに、係数生成へ固定mode数をtemplateで渡す。
入力・出力でperiod/inverse_period/omega/modes/radius_ratio4が全て一致するときのみ、
同一sigmaのbeta・正規化形状と微分を共用する。中心・owner/imageは軸ごとに計算する。
このfixtureでは52,428中47,422atom（90.45%）が同じ形状を共有できる。
mode固定前後のresource出力は、H STACK648→40、C split688→80、融合勾配1256→40bytes/thread。
融合勾配のREGは248→244であり、register圧迫まで解消したとはしない。

1024の段階診断（Cは32分割）は、source833でH/G=0.152/0.151ms、
C_out/C_in=0.359/0.316ms、融合G/勾配0.360ms、Zx/R_x0.049ms、Zy/R_y0.047ms。
展開Y/dXは0.045/0.056ms。これらを加算して完全stepとはしない。



検証archive（各327 runtime＋共通header・検査ソースを照合、結果manifestも検証）:

- `7c610fcc` / `l4job-e08e148f295644ca9b7104dee4673bac`：source `211763d0560f20d0963f561bf0f7dd3438cba503222eeba4981234e4585bfebf`、result `8dd26e3cb25ba020cbc746ebe42ed1e38e24bb959bb171930c4b31fe4ccddfad`、driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、binary `1563457bb3882a602b00fda15b124928f55ba2cbec2111fa45df2f6b590da2b5`。
- `468c858d` / `l4job-cfbc5d6ff6594bedb152c4ab9185be77`：source `1166fbc96ac950c30b0920e5009fcad8c1684ae0ff8caf3e91cceb3e7162baf1`、result `682e0dac7b3120187504a347b5cfc93e0215a2792f2d18e9a1a0ed01188eec76`、driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、binary `bb640378186bbada9463cbb71e4080fc76b1d0c8e7f1d0037ca8cc3020b07fb8`。
- `760d4933` / `l4job-5cf94d223d37477d9a4caba176f5b706`：source `9571fed484351a86e5981da72c773782772a47c20e80ca47057abe3b892b0f0a`、result `d97011ad766039359779125bec3d253ca038cd6bc93a024a34ea0bab7a12317a`、driver `b95b332a67d11e49c0bdc2281e806da7fceabdb7dde353438c46656aa3e85bda`、binary `6b06002db885a8bbe68cb7a7160fd02af35b4067d42cce0607b6cdbc9db7668d`。
- `833e68dc` / `l4job-763872192be04cea9a0013015dcdc7bb`：source `2fa471732daa025db55a8417233393e5f822b0ab02967272faec894125e82973`、result `dc8ea5468866d76593b2f2076972b3c86e8c3e9683ea41f537b5bf20cd883c6d`、driver `b95b332a67d11e49c0bdc2281e806da7fceabdb7dde353438c46656aa3e85bda`、binary `f48edac099ee855bf583c75e2e192bbc16648b6fc55c2a221317611fc7029423`。

ignored `output/cell-mass-opt-20261027/` の各candidateにsource/results/spec/receiptを保全。

### 最適化第7段：G・Parameter勾配・C_in融合の負の結果

source `dcf48273`。研究flag `fuse_backward_cin=True`を追加した。
両方の勾配が必要で1<=batch<=4のとき、input owner/splitの各laneが別atomについて
G→保存dtypeへ丸め→全4勾配→input係数のshared保存を行い、C_inを縮約する。
Hの保存値と丸めたGを使う規約は維持する。C累積はsharedへ置き、重い勾配計算中のregister保持を減らす。
M8/B4のsharedは6,088bytes/block。Gはglobalへ保存し、全atom係数表は追加しない。
B0/B>4/片側だけの勾配は既存経路へ戻る。新header `projection.cuh`はJIT digestにも含める。
CPU全体2037 passed / 4995 skipped / 既存警告18件、Ruff/format/diffは通過。
GPUでは通常経路2278 passed、tuned経路2278 passed、failed/skipped 0。
全52,428atomの独立Y/dX/4列勾配、AdamW、fallback分岐のspy、racecheckも通過した。
しかし1024完全stepはC16=2.259520、C32=2.176199、C64=2.276359ms、dense0.671774ms。
128/C2は1.172671ms、dense0.674572ms。allocated/reservedは従来と同じでdense以下だが、
source833分離版1.661862msより遅いため**速度面では不採用、既定Falseを維持**する。
別sourceの小差を一般化せず、現在のsourceでflag False/Trueをfresh processずつ再確認する。

1024/C32のGPU段階診断は融合1.153024ms、分離G/勾配0.360448ms、C_in0.319488ms。
融合REG194、STACK40、LOCAL0であり、register増大が主因と断定できる証拠ではない。
同じfixtureのgroup/splitをCPUで列挙すると、逆方向S64の32lane枠の実atom比は30.97%、
S32は43.10%、S16は56.69%。S64は5291 chunkで、連続atomだけなら必要な1639 warpの約3.23倍。
これはGPU stall計測でなく、group区間ごとの末尾paddingの構造的な集計である。
重い勾配計算をその断片化したchunkへ移した影響は、今後の候補を設計する根拠になる。

次の未実装候補は、atom値をコピーせず`reverse_slots[A]`にforward sorted slotを並べ、
`reverse_offsets[input_nodes+1]`で入力ownerの連続範囲を渡す方法。
正規rowsは出力→入力順を維持し、H/Gも従来slot、勾配は元Parameter行に対応させる。
counting prepareでgroup内の既存scatter順位を再利用すれば、追加atomicなしにreverse配置を作れる。
int32 slotなら1024の恒常増分は209,712bytes＋offsets496bytes。
owner別の有効split数、追加prepare時間、group復元、保存状態の変更検出、全勾配とピークを検証するまで、
改善や採用は主張しない。現在はこの逆indexを実装していない。

job `l4job-6efb835e93b74ae48c03f8dbba502198`、source `23958a971ff86d7c328d24e3426f12c0795fa492e7e3c0fac04e7f17f249d72f`、result `161ab9561fc8af50862268ffbcf47fceacb3c7858f426bed7fa5833264932aec`、driver `ae0664bfc469ad720788d08f68341790f2f6118de254070cff79d63eace6075b`、binary `99cc6d5d0202d892753c71a584f619ee23795f82994e2854ddcdd0bd5f9cc14c`。
327 runtimeとcommon/projectionの両header、検査ソース、結果35ファイルを照合。
ignored `output/cell-mass-opt-20261027/fused-candidate/`へsource/results/spec/receiptを保存。
wheel/sdistの338 runtimeファイルもsourceへ照合した。



### 最適化第8段：逆方向indexとowner別の有効split数

source `a9995b07`。counting prepareに任意のreverse indexを追加した。
正規rows/groups/offsetsは出力→入力順のまま、atom値や係数はコピーしない。

```text
reverse position r → reverse_slots[r] = forward sorted slot s
                   → rows[s] = original atom a
H/G[:,s]、atoms[a,:]、gradient[a,:]を使用
```

`reverse_offsets[input_node:input_node+2]`がその入力ownerの連続範囲。
prepareの既存group内scatter順位を再利用して、追加atomicなしに逆slotを書く。
増分はint32 reverse_slots[A]とint64 reverse_offsets[input_nodes+1]のみで、
1024 fixtureは209,712+496bytes。group開始位置は一時int32 scratchで保存しない。
`save_for_backward`へ両配列を加え、in-place変更検出とretained backwardを維持する。

逆方向のowner範囲について、grain=atoms_per_splitが正なら
`active=min(max_splits,max(1,ceil(owner_atom_count/grain)))`とする。
active外blockは部分和をゼロにして全thread一斉に戻る。active内は区間を重複なく分担。
grain0は固定split数を維持する。Zの二分マージ・Rの3枚窓・sigmaの分類は変更しない。

同source、B4/FP32/M8/Q128/A52428/sigma初期.2..40、counting/Z32/64threads、C64、R有効。
通常/tuned各2369 passed / failed0 / skip12（unsupported sort+reverse combinations）。
全52,428atomのY/dX/4列勾配は全設定で独立FFT参照を通過した。

| 逆index / grain / Cin融合 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| なし / 0 / 分離 | 1.665969ms | 7.396MiB | 10MiB |
| あり / 128 / 分離 | 1.454741ms | 7.596MiB | 10MiB |
| あり / 0 / 融合 | 2.125936ms | 7.596MiB | 10MiB |
| あり / 64 / 融合 | 1.605266ms | 7.596MiB | 10MiB |
| あり / 128 / 融合 | 1.477567ms | 7.596MiB | 10MiB |
| あり / 256 / 融合 | 1.737101ms | 7.596MiB | 10MiB |
| matched dense | 0.688162ms | 16.063MiB | 22MiB |

各1 fresh process、warm5/wall21、全prepare/forward/backward/AdamWを含み、Graph/TF32なし。
逆index＋owner適応で分離版を約13%短縮したが、1024の時間条件はこの時点では未達。
128はreverse無し/C2/sort/Z0/128threadsで1.224288ms、dense0.674702ms、
allocated242.5KiB/264.5KiB、reserved両方2MiB。
段階診断はC32/grain128でCin0.218112ms、融合G/勾配/Cin0.743424ms、
Cout0.359424ms、G/勾配0.360448ms。stageを足して完全stepとはしない。
出力側Cが次の改善対象であり、grainをforwardへも適用する候補を検証する。

job `l4job-4e4975d749a54a6488085e238b3649e8`、source `95ebc10b5e9d3cba0a0c9db5882a626f9147cccfbcb25af0a9a23e511284c829`、result `b7048b1361741252c5f6eaeef39b72407c728dcec284c78395b584ec0d06d269`、driver `85a0cf3229f8f2e7d95106376a3a2fe719fa0817b6542b90d928e3a2c5639546`、binary `dfa70f7493160bebbf95f56fffddd24ae5ca4ff1cc846c3e47074e5c89534923`。
327 runtimeと両header・検査ソース、結果46ファイルを照合し、ignored
`output/cell-mass-opt-20261027/reverse-candidate/`へsource/results/spec/receiptを保存した。

出力Cにも適応splitを加えたsource `69a7806b`を別jobで検証した。
通常/tuned各2385 passed / failed0 / skip12、racecheckの対象caseも通過。
全52,428atomのY/dX/4列勾配は6設定全てで独立FFT参照を通過した。
最大絶対誤差はY 1.78e-8、dX 2.00e-8、全atom勾配2.71e-8未満。

| 条件 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、逆indexなし・固定C64・Cin分離 | 1.686256ms | 7.396MiB | 10MiB |
| 1024、逆indexあり・grain128・Cin分離 | 1.494954ms | 7.596MiB | 10MiB |
| 1024、逆indexあり・grain128・Cin融合 | 1.499301ms | 7.596MiB | 10MiB |
| matched dense1024 | 0.697796ms | 16.063MiB | 22MiB |
| 128、従来のC2/sort/128threads・Rなし | 1.211821ms | 242.5KiB | 2MiB |
| matched dense128 | 0.703213ms | 264.5KiB | 2MiB |

診断C32/grain128のCoutは0.348160ms、Cinは0.214016ms、G/勾配融合は0.360448ms。
前jobのCout0.359424msとの差は小さく、出力側への適応split追加だけが有意に速いとは
主張しない。逆indexを含む分離版の改善は再度観測したが、dense比は時間約2.14倍であり、
denseの2倍以内という時間条件はこの時点では未達。
準備段階の`c5796712`は旧input wrapperの引数渡し漏れでsuiteが154件失敗した。
これは性能採用対象外として保存し、修正後に上記の全suite/gateを再実行した。
数学/精度条件を緩める修正ではない。

job `l4job-70e3ce933e2949e1b6566b9506bf5901`、source `411b2095515fa2a67018dc27af719bb954570105aa28793cd9bec88261bfe9d0`、result `d0515dc338715206b8c7bb7f916c0fd9ccc5ab691ad652c38339869eb42af156`、driver `08170a732474b821da80bc17282ad52afbf12bafe8c8906dedeec0ecfa1adc39`、binary `a6e6baf0ab66ff7409386a21d3972e8fcbffc95484241dbb1fcdf90acecbffbc`。
327 runtimeと両header・検査ソース、結果46ファイルを照合し、ignored
`output/cell-mass-opt-20261027/adaptive-both/`へsource/results/spec/receiptを保存した。

### 最適化第9段：補償されたM8の値評価

source `0f08e84e`はM8多項式だけのdouble-single補償演算を実験用templateとして追加した。
既定Falseを維持し、現在の完全stepは全てDouble評価。定数表増分は352bytes、合計4616bytes/TU。
β・位相・正規化・Z/R/Cの縮約をDoubleのままにし、単独micro診断で価値を測った。
M8/A52428のGraph-event中央値はvalue 0.088064→0.078848ms、derivative
0.145408→0.116736ms。Doubleに対するchecksum差は最大1.46e-14だった。
registerはvalue 36→40、derivative 46→56へ増えた。
これは低registerの診断kernelであり、完全stepや独立全勾配の改善証拠ではない。
次に値計算H/Cだけの明示的opt-inで、完全step・正規化・全勾配を検証する。
勾配kernelの演算はDoubleのままにする。

micro job `l4job-6e5a58f5d60c4d42aa6107dc74c5eb8b`、source `9e2dded124f5b249547012920a4a5e4033e69dcd54ad05db2125c061b7acceab`、result `9ed48ae43fd9473dee83a449a187d51e34c74ca5af9ed403d5b1bdd2340a97cc`、driver `a22b607976e584a7dbdf0d54a0482ad9e9fc2a8ef2d3d3a50f58916c8fb351c4`、binary `63192d9779c5cd5f0906f9cfe68e5b07f3c524d7c69e20b3184171b5161e8cbf`。
このdriverはmain extensionを呼ばず、common.cuhを使う独立した診断kernelのみを実行した。
前述のc579 wrapper不具合を通るmain計算の合格とは扱わない。
327 runtime、両header、検査ソース、結果7ファイルを照合し、ignored
`output/cell-mass-opt-20261027/ds-micro/`へ保存した。

source `290e38b1`は`compensated_values=True`を研究入口へ接続した。
対象はFP32保存、M8/tail1e-3の両軸のみで、既定Falseを維持する。
H・Cout・単独G・分離Cinだけが補償演算を選べる。G/勾配融合とG/勾配/Cin融合は
従来のDoubleのまま。位相・正規化・縮約・Z/R/Cの保存もDoubleを維持する。
値評価の近似とDouble解析導関数の組合せを、独立oracleと同じ精度基準で検査する。
通常のCPU suiteは2037 passed / 5153 skipped / 既存警告18件。
wheel/sdistの338runtime byte照合とRuff/formatは通過した。
GPU job `l4job-2806b8176cb64be48be73b60f93f4ce2`で2424 passed / failed0 / skip12。
独立FFTのY/dX/全4列、unit mass、retained backward、128の3step AdamWも通過した。
1024は全52,428atomを各設定で照合し、最大誤差は従来の保存dtype丸めと同じ範囲だった。

| 条件 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、逆index/grain128、G/勾配融合、Double値評価 | 1.477870ms | 7.596MiB | 10MiB |
| 同上、補償値評価 | 1.428999ms | 7.596MiB | 10MiB |
| 補償値評価、G/勾配を分離 | 1.676313ms | 6.796MiB | 8MiB |
| matched dense1024 | 0.676794ms | 16.063MiB | 22MiB |
| 128、従来設定、Double / 補償値評価 | 1.158747 / 1.163540ms | 242.5KiB | 2MiB |
| matched dense128 | 0.693166ms | 264.5KiB | 2MiB |

1024の補償値評価はこの単一job内で約3.3%短縮したが、独立再確認前の条件付き候補とする。
128は改善せず、既定や128推奨を切り替えない。G/勾配を分けるとメモリは減る一方で遅くなる。
診断はH 0.150528→0.137216ms、Cout 0.348160→0.321536ms、
Cin 0.225280→0.202752ms。完全stepへの加減算はしない。
1024ではdenseの2倍以内という時間条件はこの時点では未達。
source `1d7a2c016e1b0d7d348d56588ced1d789d9f8df69658d0e0a6263c9405c31367`、result `34b5fa0c418ccdbba12f4dbffbdaec0e3f6736128885df2fd0349fbed03184ed`、driver `ebbffe58c47d4801601cf478a01b02dbd2b75011b38cfa0987fe101fa2c87857`、binary `b8ce9ffe5e0ba685796b4190e821d64c5f7f04d0d59210a7753e74c8c2fff142`。
327runtime、両header・検査ソース、結果39ファイルを照合し、ignored
`output/cell-mass-opt-20261027/compensated-values/`へsource/results/spec/receiptを保全した。

### 最適化第10段：HとCの融合と負の結果

source `e1d84c33`はHとCoutを同じatom担当threadで作る`fuse_h_c=True`を追加した。
両軸M8のみを対象とし、他modeは分離経路へ戻す。既定Falseである。
このfixtureのCPU分類では47,422/52,428 atom（90.4517%）で両軸のprofile形状が一致し、
その場合は一つの多項式streamから、中心差の位相回転で出力係数を生成する。
形状が異なるatomでは別の係数streamを使う。中心/imageの補正は各軸で独立に保つ。

```cpp
// owner・batch tile・splitごとのblock。各threadは異なるatom。
H_in_register, output_coefficients_in_shared = project_and_make_coefficients(atom, Zx);
H_saved = cast_to_atom_dtype(H_in_register);
amp_H_in_shared = amplitude * H_saved;  // 丸め前のHはCに使わない
sync_block();
C_partial = reduce_atom_chunk(amp_H_in_shared, output_coefficients_in_shared);
// 全chunkの後、従来の別kernelでsplit部分和をCへ合計する。
```

Hはbackward用に引き続き保存し、globalからの再読込と係数の重複生成を減らす。
Zの二分階層・Rの3枚窓・sigma分類は変えず、Y/dXやParameter勾配の担当も変えない。
M8/B4のsharedは6,088bytes/block。追加の全atom係数表やatom値コピーは作らない。
位相の再利用で丸め順序は変わるため、独立参照・AdamW・racecheckを同じ基準で検証する。
GPU job `l4job-5a1444db6d614fd2bea9289dbdf75369`は2438 passed / failed0 / skip12、
融合stageのracecheck、独立H/C/Y/dX/全4列、128の3step AdamWを通過した。
1024の全52,428atomも4設定全てで独立参照を通過したが、融合は完全stepで悪化した。

| 方式 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、Double値評価・H/C分離 | 1.479868ms | 7.596MiB | 10MiB |
| 1024、Double値評価・H/C融合 | 1.529752ms | 7.596MiB | 10MiB |
| 1024、補償値評価・H/C分離 | 1.427589ms | 7.596MiB | 10MiB |
| 1024、補償値評価・H/C融合 | 1.509354ms | 7.596MiB | 10MiB |
| matched dense1024 | 0.692989ms | 16.063MiB | 22MiB |
| 128、従来設定・H/C分離 | 1.183312ms | 242.5KiB | 2MiB |
| 128、Double値評価・H/C融合 | 1.161017ms | 242.5KiB | 2MiB |
| 128、補償値評価・H/C融合 | 1.135810ms | 242.5KiB | 2MiB |
| matched dense128 | 0.713953ms | 264.5KiB | 2MiB |

H/C融合単体の診断はDouble0.709632ms、補償0.679936ms。完全stepへの加減算はしない。
融合のregisterはFP32 atomで106/123（Double/補償）、分離Hは56、分離C splitは76。
融合はgroup参照・HとCの縮約を同じ32thread blockのchunk内にまとめるため、
係数共有だけで速くなるとは限らない。占有率低下などの因果を個別に実証したとはしない。
1024では不採用、既定Falseを維持する。128の小差は独立再確認前の候補に留める。
補償値評価の分離版は前jobと同じ約1.43msを再観測したが、1024の時間条件はこの時点では未達。

source `147ea8fac350307dbdb033d5c4a9351afbe04968d49296328c1d02a179673550`、result `f651f5f3e51da60c87447bc13376e3cf33c398d5eb21659d30c17091e095854d`、driver `080f4da97d4d76fac68c705d1c29447e3525f090b8d7b6b9dc2fbeb38e70e8e5`、binary `27d669e31aa81bf649c8428c213df52c3a85604c42a1b80f1adf1cb5c0ce65a9`。
327runtime、両header・検査ソース、結果45ファイルを照合し、ignored
`output/cell-mass-opt-20261027/fused-forward/`へsource/results/spec/receiptを保全した。

### 最適化第11段：Gと勾配の専用化と汎用化

source `7fd9132b`は、既存のG/Parameter勾配融合kernelのdispatchだけを切り替える
`specialize_g_grad=False`を追加した。axisのMは変えず、M8を既存の汎用kernelへ渡す。
低水準の同名操作では`specialize_modes=False`を指定する。
source `69a7806b`の資源診断ではM8専用版は244register/thread、汎用版は128だった。
演算の展開と占有率の関係を調べる候補であり、register数だけを速度改善の証拠にはしない。
既定Trueを維持し、G/勾配/Cin融合やParameter勾配単独には作用しない。
CPU2037 passed / 5176 skipped / 既存警告18件、Ruff/format/buildを通過。
GPU job `l4job-cfe3e961427946f38f40609358a7e549`で階層版suiteは631 passed /
failed0 / skip12、融合stage racecheckを通過した。このjobは階層版だけを再実行し、
前jobの固定bank/coefficients/referenceを含む2438件を再実行したとは扱わない。
128の全819atom、1024の全52,428atomについて、全設定のY/dX/4列勾配を独立参照へ照合した。

| 1024の条件 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 補償値評価・H/C分離・M8専用G/勾配 | 1.432525ms | 7.596MiB | 10MiB |
| 同上、汎用G/勾配 | 1.481324ms | 7.596MiB | 10MiB |
| 補償値評価・H/C融合・M8専用G/勾配 | 1.504512ms | 7.596MiB | 10MiB |
| 同上、汎用G/勾配 | 1.546615ms | 7.596MiB | 10MiB |
| matched dense | 0.683191ms | 16.063MiB | 22MiB |

G/勾配単体はM8専用0.360448ms、汎用0.404480ms。register数の削減だけでは改善せず、
`specialize_g_grad=True`を維持する。H/C分離・補償値評価の約1.43msを3job目でも再観測。
128も専用/汎用は分離1.153474/1.177473ms、融合1.139130/1.190129msで汎用が遅かった。
dense128は0.682831ms、allocated242.5/264.5KiB、reserved両方2MiB。
選んだ設定の別L4・2 fresh process再測定と、1024全atomの3step AdamW状態は
次の最新再確認節で検証した。

source `320a041d3801405afd673a93921e491a24333cbd2c814643db3ea6fdf5c10875`、result `678f035b3c2785c23401f81c26939449493d39d9988645d6dc5ff72f2f896f5a`、driver `494c38a190f82e01a88fad7a0d925a81ad964694a041a9ca9f4e5ea8285a16bf`、binary `ef0cdb0d72b3b55443f4f2667a6f19c76676a7ee1085cbe2bf92ab157bf26fc0`。
327runtime、両header・検査ソース、結果54ファイルを照合し、ignored
`output/cell-mass-opt-20261027/generic-grad/`へsource/results/spec/receiptを保全した。
wheel/sdistの338runtimeファイルもsourceへbyte照合した。

### 最適化第12段：Nsight Computeによる支配要因の確認

runtime `7fd9132b`、NVIDIA L4 / CC8.9 / Nsight Compute2025.3.1。
job `l4job-d5db8c1f79df484c90e5dc58d7b25ba8`で、同fixtureのCout、逆slot Cin、
G/勾配、H/C融合を各1launchだけ計装した。補償値評価、C64/grain128、R有効。
metricは実機のquery結果から11個を選び、各対象のCSVに一致するkernelと全11行を確認した。
これは計装による診断であり、時間・完全step・通常動作のメモリを測った結果ではない。

| kernel | FP64 pipe稼働率 | active warps | short scoreboard | shared load bank conflicts |
| --- | ---: | ---: | ---: | ---: |
| Cout | 80.36% | 19.29% | 74.76% | 169,932 |
| reverse Cin | 77.09% | 14.28% | 70.20% | 170,776 |
| G/勾配 | 86.75% | 15.73% | 67.45% | 0 |
| H/C融合 | 81.55% | 16.40% | 74.97% | 169,824 |

long scoreboardは順に2.62/3.28/5.52/4.36%、barrierは全対象0〜0.18%だった。
DFMA命令数はCout 8,404,724、Cin 8,010,740、G/勾配34,323,764、H/C融合16,935,740。
このケースでは、register数だけを減らすよりFP64演算量とshared参照を切り分ける方を
次の重点候補とする。bank conflictが観測されたことと、それが全体の律速であることは区別する。
G/勾配はshared conflictが0でもshort scoreboardが多い。
[NVIDIAの指標定義](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)でも
short scoreboardはMIO操作の依存待ちであり、shared memoryだけの待ち時間ではない。
指標は分母が異なるので足し合わせたり、壁時間の割合として読んだりしない。

source `62022658a62f8d05553fd9fc9f21849968d7ca1a63c03f7dec0acdb4e6baedf4`、result `c42040778ba7c7e79d6efedd10caa2af2136650239a41af057cabd619dc361eb`、driver `052ad21cba05b155971bddac7909e9fc50739ff91c5d6a9f729ae7cb01cb6193`。
338runtimeと検査ソース、結果37ファイルを照合し、ignored
`output/cell-mass-opt-20261027/ncu-owner/`へsource/results/spec/receipt・CSV・凍結driverを保存した。
loaded binaryのSHAはmetadataに記録したが、この診断jobではso本体をコピーしていない。
完全stepの採用根拠は、so本体も保全した別の確認jobである。

128の準備も次の候補。CPU分類でnonempty groupは224、現countingの全1769 groupに対し、
互換するbank対だけなら849 groupとなる。layout payloadは49,016→26,936bytesと算定した。
これはCPU metadataの見積りで、GPUピークや速度の実測ではない。compact countingは未実装。

### 最適化第13段：互換bank pairのcountingと、C積和の負の結果

runtime `e1a2caab`、job `l4job-3d4ee26009d34c4495de66fe7eceafc0`。
`counting_compact`はsigma区間が重なる入力・出力bankの組だけを列挙し、
その中を従来と同じ`(h_out,h_in,q_out,q_in)`順に配置する。
値のコピーやatomごとの係数表は作らず、元のatom行番号とgroup範囲だけを保持する。
128 fixtureではfull countingの1769 groupに対して849 groupとなる。
空groupも含むため、非空groupだけのsort配置224 groupとは異なる。
この初版は逆方向indexに未対応で、`reverse_layout=True`との組み合わせを拒否する。

| 128の配置、C積和は従来Double | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| sort | 1.123340ms | 248,320B | 2MiB |
| compact counting | 0.819594ms | 263,168B | 2MiB |
| full counting | 0.849601ms | 285,184B | 2MiB |
| matched dense | 0.705986ms | 270,848B | 2MiB |

同じ固定fixture・未計装の全検査/prepare/forward/backward/fused AdamWを含む。
各設定fresh process、warm5/同期wall21の中央値であり、21独立runではない。
compactはsort比約27%短縮したが、denseより約16%遅い。allocatedの余裕は7,680Bのみで、
再確認と全819atomの複数step optimizer照合までは研究候補として扱う。
full countingはdenseよりallocatedが多いため、この128条件では採用しない。

同時に試した`compensated_cdot=True`は、32atomのC積和をFP32の2数表現で計算し、
chunk後にDoubleへ戻す候補である。1024の逆index/grain128/C64では、
Double多項式値評価の完全stepが1.469052→1.632826ms、補償値評価では
1.418197→1.585851msとなり悪化した。allocated7,965,184B/reserved10MiBは同じ。
これは多項式値評価だけを補償する`compensated_values`とは別の選択肢で、既定はFalseを維持する。

初版Cdotのchunk局所誤差guardには、大きな既存Double累積との相殺を見落とす
CPU反例も見つかった。DC成分単独の反例と、全Yが旧経路だけ通過する反例を混同しない。
初版guardを採用根拠にはしない。後続`adf3908b`では、非有限または
`abs(amplitude*H/G)>16`が1つでもあればC全体を従来Doubleへ戻し、chunk判定にも
既存累積の大きさを含めた。非同期GPU上の1word flagを使い、host同期を増やさない。
この閾値は数値計算方式の選択でありsigmaを制限するものではない。
guardは工学的な判定であり、任意入力に対する誤差保証を証明したとは主張しない。
後続版job `l4job-d404ca089de64eafb5cba71d9065b1c3`はColab実行経路が1740秒で
timeoutし、検証済みresult archiveを回収できなかった。精度・速度・peakの判断から除外する。
poolの所有VM停止・server割当なしを確認し、正式recover後に次の検証へ進んだ。
source `1dd22ae0448f291b09949521275e26bcc6ffad6916b991ec16c38210cdbe2dbe`と
spec/transport logはignored `output/cell-mass-opt-20261027/cdot-preflight-interrupted/`へ保存した。
timeoutの原因をCUDA kernelやcompileに断定できるログはなく、そうした主張もしない。

初版jobは通常2487 passed / failed0 / skip12、選択したracecheck12件・hazard0。
128全819atomと1024全52,428atomのY/dX/全4列勾配は独立FFT/autograd oracleへ通過した。
この全atom gateはoptimizerの更新照合までは含まない。
sourceのruntime338ファイル、両header・検査ソース、結果56ファイル、driver/binaryを照合した。
wheel/sdistは`e1a2caab`、後続guardのwheel/sdistは`adf3908b`のruntime338ファイルと一致した。

source `b5510fe31023266f0079f32bf20023283d1af1ccbdee4d769084605f820a99cb`、result `b3136a98ef4f8abf53eee5a79733840ce75abe8ede721acc0c94a1c79d34467d`、driver `2dccd38018b0312a0da0f8241adfdc76eae18ea926ce915894b4de14a1287973`、binary `c646d5742372a2f70560d39be038e6378af632b31449151e80028de01ac9663e`。
source/results/spec/receiptはignored `output/cell-mass-opt-20261027/cdot-guard-initial/`へ保全した。

### 最適化第14段：GとParameter勾配の多項式評価（初回GPU検証）

runtime `7f337e80`。`compensated_grad=True`は、GとParameter勾配を一緒に計算する
kernelの多項式値とbeta微分だけをFP32の2数表現にする。
beta/beta_sigma、位相、正規化、Zとの縮約、batch和、勾配出力の累積はDoubleを維持する。
Gを保存dtypeへ丸めてから勾配とCinへ再利用する契約も変えない。
両軸M8/tail1e-3・FP32 atoms/H限定で、`fuse_g_grad=True`と
`fuse_backward_cin=False`を必要とする。融合Cinは従来Doubleのまま。
全atom係数表・追加global scratchは作らず、既定Falseとする。

job `l4job-61063af38bc24560b97bcffa547ccb5f`、NVIDIA L4 / FP32 / B4、
固定fixtureと前jobのatoms/X/dY/初期dense Wはbit一致した。
各設定1 fresh process、warm5/同期wall21。通常2518 passed / failed0 / skip12、
選択racecheck11件・hazard0、128全819atom/1024全52,428atomのY/dX/4列勾配は
独立FFT/autogradへrtol3e-6/atol2e-7で通過した。
全fixtureのoptimizer更新照合をこのgateで行ったとは扱わない。

| 設定 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、補償値あり・補償勾配なし | 1.425167ms | 7,965,184B | 10MiB |
| 1024、補償値あり・M8補償勾配あり | 1.388362ms | 7,965,184B | 10MiB |
| 1024、補償値あり・汎用補償勾配あり | 1.418376ms | 7,965,184B | 10MiB |
| 1024、Double値・M8補償勾配あり | 1.428369ms | 7,965,184B | 10MiB |
| matched dense1024 | 0.674118ms | 16,843,264B | 22MiB |
| 128、sort・Double値/勾配 | 1.147484ms | 248,320B | 2MiB |
| 128、compact・Double値/勾配 | 0.862185ms | 263,168B | 2MiB |
| 128、compact・Double値/M8補償勾配 | 0.845113ms | 263,168B | 2MiB |
| 128、compact・Double値/汎用補償勾配 | 0.861083ms | 263,168B | 2MiB |
| matched dense128 | 0.692992ms | 270,848B | 2MiB |

1024は同job baseline比約2.6%短縮したが、dense比約2.06倍の時間である。
別jobでの2 fresh process・測定順逆転・全atom optimizer確認までは採用確定にしない。
128のcompactは前jobに続きsort比約25%短縮し、メモリ条件も維持した。
128の補償勾配の小差は、引き続き独立再確認で判断する。

別Graph-event診断ではG/勾配がDouble0.360448→補償0.318464ms、
汎用補償0.353280ms。kernel資源はM8のREG244→254、STACK40は同じ、LOCAL0。
資源値だけからoccupancyや速度の原因は断定しない。
修正版Cdotも通常suiteで検査したが、今回の全fixture gate/完全stepはCdot=False。
別診断のCdotはCout0.335872/Cin0.245760msと、補償値だけの
0.225280/0.158720msより遅い。既定Falseを維持する。

source `e3f90cb5cd03e6d33b964ae3e7a3bf1bb326a95ca364e1286d74eb554413521f`、result `669ee9a256db85a0450423028b34f98c5bb865d8a80e8a69fe32c45e54599b9a`、driver `a16bfdbaaff02b0e7510616461b0df3242b32e150c24af490f862a180977ec8c`、binary `1a5cb4486883548e8624c96faccbd45eb8dce7ca9ae4d4063b6e753cea460790`。
runtime338・検査141ファイル、manifest50件とdriver/binaryを照合し、ignored
`output/cell-mass-opt-20261027/compensated-grad/`へsource/results/spec/receiptを保全した。

### 最適化第15段：owner windowの参照（後続sourceで検証）

runtime `bfe22c94`。Rを使うprojectionでは、ownerは既にcanonicalなnode番号なので、
`bank.node_offset+owner`を直接参照する。harmonicごとの剰余・image番号の整数除算と
`image_rotation`呼び出しを省き、既存のDouble回転式には同じ`rc=1,rs=0`を渡す。
raw Zの3枚参照、端のimage補正、profile center、二分Zのmerge、Rの構築は変えない。
H/G、融合G/Parameter勾配、独立Parameter勾配、任意の融合H/Cに適用する。
CUDA compilerが既に同等に最適化している可能性もあるため、GPU結果前に速度改善とはしない。
CPUの対応owner全件の整数検算と31 passed / 683 skippedを通過した。
全CPU suiteは2037 passed / 5247 skipped / 既存警告18件、wheel/sdist buildと
両配布物のruntime338ファイルのbyte照合も通過した。CUDA compileの代用にはしない。
後続driverでは、非計装step/両peakとは別区間でhost呼び出しのprofileも保存する。
profileに含まれる`.item()`等の待ち時間をCPUの演算時間と断定しない。
後続`180b2424`の通常suiteと全fixture gateで検証した。単独差分の速度採用とは区別する。

### 最適化第16段：compact配置の逆方向index（後続sourceで検証）

runtime `fca04158`。`counting_compact`でも`reverse_layout=True`を許可する。
互換bank pairの開始group番号を使い、input ownerごとに該当output bank/ownerをたどる。
group内の順位は既存のscatterでforward slotと対応づけ、既存のreverse Cinを再利用する。
reverse=Falseの配置・allocations・kernel列は維持する。atom値をコピーしない。

128/A819では保持する`reverse_slots[int32,A]`が3,276B、
`reverse_offsets[int64,input_nodes+1]`が496B。準備中だけの
`reverse_group_base[int32,G]`3,396Bも含めると追加論理量は7,168Bである。
これは実測peakではない。512B丸めを仮定すると追加7,680Bとなり、
第13段のcompactとdenseの差に等しいため、既存表のpeakへ単純加算して採用判断しない。
準備とbackwardのallocation共存、C splitページ、reservedの再利用を含む完全stepで確認する。

CPU indexモデルはseed741で1200のdyadic軸設定を生成し、4096 group上限で除外する33件を
除く1167件で、逆slotの一対一対応とowner/group順を照合した。うち33件はG0の強制検査である。
Python float64のindexモデルであり、CUDAやFP32丸めの検証とは扱わない。
再実行script/resultsはignored `output/cell-mass-opt-20261027/compact_reverse_cpu_model.*`に置く。
CUDA検査にはFP32/64・空atom・異なる軸長・full-chart-only・sigma境界、
保存buffer mutationとretained backward、S1/S7・grain0/32/128の独立FFT full routeを追加した。
CPU実行は31 passed / 714 skippedであり、追加CUDA検査は未実行である。
全CPU suiteは2037 passed / 5278 skipped / 既存警告18件、wheel/sdist buildと
runtime338ファイルのbyte照合も通過した。
後続`180b2424`で独立gateと実測を確認した。128 reverseありは267,264Bとなり
dense270,848B以下だったが、時間は0.836547〜0.862083msで、reverseなし0.830982msに
勝たなかった。128はreverseなしを選ぶ。1024は下の再確認でcompact reverseを選ぶ。

### 最適化第17段：幅係数とbeta微分のshared再利用（GPU検証・再確認）

runtime `180b2424`、任意の`reuse_shape_coefficients=True`。
G/Parameter勾配kernelは入力・出力profileの周期等が一致するとき、幅係数を二度評価していた。
出力projectionで位相を掛ける前の、harmonic1..8の正規化済み振幅とbeta微分を保存し、
入力projectionではその2種類を自身のcenter位相と組み合わせる。
振幅をcos/sinで割って復元したり、中心を共通化したりしない。
既存same_shape条件が不成立なら入力側は従来のHarmonicStreamへ戻す。

cacheはDoubleの`[16,blockDim.x]`をsharedに置く。64threadsで8192B、
32/128/256threadsでは4096/16384/32768B。各threadは自分の列だけを読み書きし、
Gを丸めて確定した後に読むため、block全体のbarrierは加えない。
batch tileごとに上書きし、global係数表やatom値コピーは作らない。
sharedによるSM資源とregister使用量への影響は実測する。

FP32 atoms/H・両軸M8・specialize_g_grad=True・fuse_g_grad=True・Cin分離に限定し、
既定False。compensated_gradは独立に切り替える。Gの保存丸め、振幅の適用回数、
位相・正規化・縮約のDouble、二分Z・同階層3枚のRは維持する。
CPU31 passed / 732 skipped、shared添字の一意/範囲・autograd入出力20個の静的検査を通過した。
全CPU2037 passed / 5296 skipped / 既存警告18件、wheel/sdist buildと338runtimeの
byte照合も通過した。これをCUDA検証の代用にはしない。
job `l4job-6ee2ad91ac524df0b8eb19742e01f19f`で、cache/補償勾配/compact reverseを
独立に切り替えた。通常2567 passed / failed0 / skip12、選択racecheck46件・hazard0。
128全819atomと1024全52,428atomのY/dX/4列勾配が独立FFT/autogradへ通過した。

| 1024、補償値あり・M8 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| counting逆index、Double勾配、shape再利用なし | 1.399503ms | 7,965,184B | 10MiB |
| 同上、補償勾配 | 1.362043ms | 7,965,184B | 10MiB |
| 同上、補償勾配＋shape再利用 | 1.334804ms | 7,965,184B | 10MiB |
| compact逆index、補償勾配、shape再利用なし | 1.334848ms | 7,943,168B | 10MiB |
| compact逆index、補償勾配＋shape再利用 | 1.311000ms | 7,943,168B | 10MiB |
| matched dense | 0.651329ms | 16,843,264B | 22MiB |

各設定1 fresh process、warm5/同期wall21。段階Graph-eventでは補償G/勾配が
0.306176→shape再利用0.282624ms、REG255→218、STACK40/LOCAL0は同じ。
dynamic sharedは64threadsで8192Bを別に使う。静的SHARED0を総shared量とは扱わない。
資源値のみでoccupancyや速度原因を断定しない。

fixtureのsigmaは全て一定ではなく、FP32 linspace(.2,40)である。
1024の入力bank別atom数はT=4/8/16/32/全周で5006/5269/10538/21076/10539、
出力側はT=8/16/32/全周で10275/10538/21076/10539。
このfixtureでは47,422atom、約90.45%で両側の幅shapeが一致する。
幅分布が変われば再利用率も変わる。集計はignored confirmationのsigma-distribution.jsonに置く。

source `5889ccdf842d8edba7af2ece6776aecd7cc88449a4eec78fe6bec110a24d28cc`、result `39234b31c3c2dbdd5bea1086f92f9ebdf312deaea8cebe7b58ac8c5496f95047`、driver `d2c011c685959ce3b9f2bbeb26f42c5ed859e6c6a6cbfdfcd1225343ccb831ba`、binary `75cb602b183ad15cb0561eee3b5172d3469b3d79f8d28ff4b035db033b882b25`。
runtime338・検査141、manifest81件、driver/binary、前jobとのfixture bit一致を照合し、
ignored `output/cell-mass-opt-20261027/shape-reuse/`へ保全した。

### 未採用の補償位相試作（回収済み）

`kernel/cell-mass-phase-prototype` / `a3fe9da9`ではG/パラメータ勾配の最初の
sin/cosだけを補償FP32で評価した。引数の縮約・位相漸化式・正規化・縮約はDoubleを維持し、
範囲外/非finite/極小remainderは従来sincosへ戻す。採用版のruntimeには含めない。
job `l4job-2f94b489b9d443e19fa28a529a319c6a`を回収し、通常2638 passed/12 skipped、
racecheck8件/hazard0、全819/52,428atomのY/dX/全4列勾配、両peak条件を確認した。
52,428角度のhelper診断で最大絶対差4.1078e-15、許容1e-12を通過した。
このjobは3step optimizer検査を含まない。

1024は同じ設定のDouble位相1.158649msに対し補償位相1.152899ms、差は約0.5%。
各1 fresh processの未再確認値である。別device-eventのG/勾配は0.281600→0.283648msで
短縮しなかった。allocatedは両方6,299,136B、reserved8MiB。
明確な高速化と判断せず、検証済みDouble位相を最終設定として残して終了する。
source338・検査141・manifest35・fixture bit・runtime/probe binaryを照合した。
source `568e428d4298f6a9c966a367b00e69f27cab19ddf6b1d7e7690c6dd96eda0a0b`、result `23654707fbb7b4365c07d673b7d1c42049e0c661e4ff1e4fc7efe33bea4dfd5b`、driver `43a3a5f4fedebd6a2c2935269d089f52810932f0e9a9256a86bfb0cf8ab1624e`、binary `4cd434ad9a341474917ef28cf7450007a807618fe0a8648fd71e8434c06c73bc`。
ignored `output/cell-mass-opt-20261027/phase-prototype/`へarchive/spec/receiptを保全した。
試作commitの復元bundleは`phase-prototype-recovery.bundle`で、base `edc90404`が必要。
共有poolの全slot停止とsupervisor終了を確認した。新しい実験は追加しない。

### 最適化第19段：有効なC部分和だけを保存する（初回GPU検証）

従来の`[split,batch,node,mode]`は、ownerのatom数が少ない場合にも
指定split数ぶんの領域と空blockを持つ。新しい任意経路`ragged_c_splits=True`では、
ownerごとのatom数`a_n`から` s_n=min(S,ceil(a_n/grain))`を求め、空ownerは0とする。
device上のprefixで有効taskを詰め、`[task,batch,mode]`だけを保存する。
taskの確保上限は`min(N_owner*S,ceil(A/grain)+N_owner)`で、hostへの件数転送はしない。
counts/prefixの生成kernelが一つ増えるため、メモリ減少だけで高速化とは判断しない。

各task内のatom区間は既存経路と同じ`[floor(a_n*s/s_n),floor(a_n*(s+1)/s_n))`。
最終Cの担当threadがtask番号順に加算し、出力atomicを使わない。
H/C融合でもHはinput dtypeへ丸めてからCへ使う。Cinは既存の逆indexを参照する。
atom値のコピーや全atom係数表は追加しない。Zの二分階層・同階層3枚のR・sigma分類は維持する。

最初の試作はFP32/M8/tail1e-3・補償値評価・正のgrain・逆indexを持つ1024経路だけで比較する。
既定値はFalse。CdotおよびG/勾配/Cin融合と併用しない。
独立Y/dX/全atom勾配、retained backward、空batch/atom、周期端、racecheckを確認してから、
同一fixtureの完全stepとallocated/reserved peakで採否を決める。

試作source `e4370477`。CPUで空ownerと重複prefixを含む2万条件のtask上限・逆引き・
非空split範囲を確認した。全CPU2037 passed/5342 skipped/既存警告18件、
Ruff/format/diff check、wheel/sdist buildと両配布物の338runtime byte照合を通過した。
これらはGPUの数値精度・同期・性能の証明ではない。
1024/B4/M8/A52428/S128/grain64では、Cinのlogical partialは4,247,552→479,264B、
Coutは2,019,328→461,856B。新しいcounts/prefixは各492B/236Bである。
この配列予算を完全stepのallocator peakへ読み替えない。

job `l4job-6004f86d063f4a009803a5509c49be59`で、通常2613 passed/12 skipped、
複数active split・sort/counting/compactとretained backwardのracecheck4件/hazard0を確認した。
全819/52,428atomのY/dX/全4列勾配をCPU FFT/autogradへ照合し、全9設定で通過した。
1024の最大絶対差はY1.7711e-8、dX1.9915e-8、atom勾配2.7011e-8。
rtol3e-6/atol2e-7は緩めていない。初回jobはoptimizer更新後の独立照合を含まない。

| 1024の同条件比較 | 従来完全step | ragged完全step | 従来 / ragged allocated | 従来 / ragged reserved |
| --- | ---: | ---: | ---: | ---: |
| S128/grain64、H/C融合 | 1.216969ms | 1.157024ms | 10,067,968 / 6,299,136B | 12 / 8MiB |
| S128/grain64、H/C分離 | 1.225079ms | 1.186063ms | 10,067,968 / 6,299,136B | 12 / 8MiB |
| S64/grain64、H/C分離 | 1.289284ms | 1.213230ms | 7,943,168 / 6,299,136B | 10 / 8MiB |
| S256/grain32、H/C分離 | 1.307852ms | 1.216390ms | 14,317,056 / 6,744,576B | 16 / 8MiB |

各設定1 fresh process、warm5/同期wall21。matched dense1024は0.669952ms、
16,843,264B/22MiB。128は既存設定だけを維持し0.852024ms、263,168B/2MiB、
matched dense128は0.669300ms、270,848B/2MiBだった。
全9設定で同jobのdense以下のallocated/reservedをassertし、step.jsonから再計算した。
S128/H/C融合では約4.9%短縮・allocated約37.4%減となったが、独立再確認前の初回結果である。
1024の選択候補はdense比約1.73で時間条件を満たすが、独立再確認はこの時点では未了。

別device-event診断ではS128/grain64のCout 0.166912→0.137216ms、
Cin 0.126976→0.113664ms、H/C 0.282624→0.228352ms。
これらはlaunch/待ちを含む段階診断で、足し合わせて完全stepや因果の証明にしない。
runtime338・検査141、manifest50、driver/binary、前確認fixtureのbit一致を照合した。
source `93b6790f946262a7787e72b87a27927ccc92d354aa274b739cdc0d1bb7c52e6b`、result `95abdb3814f3c97bc73136c250da646dbf78cbc4220a839da4bd7a1c47d5ca7d`、driver `44fd5716fd106835b255ddcada4ac6db63cb2efa66f03d6aaea4b7c653e397e3`、binary `9804e081f178e615db766aa14faacaf3cbe20b0be1c17e8019e693f430ea785e`。
raw/archive/spec/receiptはignored `output/cell-mass-opt-20261027/ragged-active-c/`へ保全した。
追加確認job `l4job-03dfde6ed0db4ecd81de335d14811210`はColab kernel接続503で中断し、
所有runtime停止・pool recover後に再投入した。未回収の試行を精度・性能根拠に含めない。

### 最新の再確認と検証済み研究設定

runtime `e4370477` / 文書込みsnapshot `2054143d`。
job `l4job-8ac49377a31246e3888e4574f5a9dcfe`で、S128/grain64・H/C融合の
ragged有無を各2 fresh process、2巡目の順番を逆転して比較した。
検査/prepare/forward/全backward/fused AdamWを含み、warm5/同期wall21、TF32/Graph/workspaceなし。
固定B4/FP32/M8/tail1e-3/Q128、物理T_min入力4/出力8、A=floor(.05*N*N)、seed120、
sigma linspace(.2,40)を維持した。atoms/X/dY/初期dense Wは初回jobとbit一致した。

| 設定 | 完全step（2 process） | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、既存compact、逆indexなし、Double値/勾配 | 0.872569 / 0.878817ms | 257KiB | 2MiB |
| matched dense128 | 0.690231 / 0.713909ms | 264.5KiB | 2MiB |
| 1024、S128/grain64・H/C融合、通常部分和 | 1.228739 / 1.242684ms | 9.602MiB | 12MiB |
| 同上、ragged部分和 | 1.160396 / 1.164707ms | 6.007MiB | 8MiB |
| matched dense1024 | 0.693952 / 0.698604ms | 16.063MiB | 22MiB |

raggedは完全stepを約5.6/6.3%短縮し、allocatedを10,067,968→6,299,136B、約37.4%減らした。
128の実行設定は変更しない。全6 memory gateで同repeatのdense以下のallocated/reservedを
step.jsonから独立に再計算した。1024はdense比約1.67の時間で、
**denseの2倍以内という時間条件を満たす**。
Torch allocator peakをGPU process全体の使用量とは扱わない。

全819/52,428atomを3step更新し、各stepの更新済み参照Parameterから独立CPU FFT/autogradで
Y/dX/4列勾配を再計算した。FP32 rtol3e-6/atol2e-7、Parameter rtol2e-6/atol2e-7、
AdamW moments/step rtol4e-6/atol2e-8を全stepで通過した。
1024の最大勾配絶対差は3.1573e-8、最大Parameter差1.5259e-5は許容差内でありbit一致ではない。
1024の3step gateはragged=True/HC=Trueだけで、HC=Falseや他splitへ一般化しない。
通常2613件/racecheck4件は直前の初回jobを根拠にし、確認jobで繰り返したとは扱わない。

```python
settings128 = dict(
    c_splits=2, fuse_g_grad=True, layout_method="counting_compact",
    z_chunk_sites=0, atom_threads=128, window_moments=False,
    reverse_layout=False, fuse_h_c=False, fuse_backward_cin=False,
    compensated_values=False, compensated_grad=False,
    reuse_shape_coefficients=False, compensated_cdot=False, ragged_c_splits=False,
)
settings1024 = dict(
    c_splits=128, fuse_g_grad=True, layout_method="counting_compact",
    z_chunk_sites=32, atom_threads=64, window_moments=True,
    reverse_layout=True, atoms_per_split=64,
    compensated_values=True, compensated_grad=True,
    reuse_shape_coefficients=True, specialize_g_grad=True,
    fuse_h_c=True, fuse_backward_cin=False, compensated_cdot=False,
    ragged_c_splits=True,
)
```

このfixtureの1024研究設定にraggedを選ぶ。公開dispatcherと研究APIの既定値は変更しない。
全CPU2037 passed/5342 skipped/既存警告18件、Ruff/format/build、両配布物338runtime照合は通過した。
確認snapshotのruntime338・検査141、manifest43、driver/binary、fixture bit一致を照合し、
ignored `output/cell-mass-opt-20261027/ragged-confirmation/`にraw/archive/spec/receiptを保全した。
source `cb8ab5a447d04982f1dade12d7e22cba78e824e162a765341cf3b74c48aa5845`、result `828254f5aa210735078d0f96fb0a35f1c30a3381133743065b16835f66530344`、driver `fd1b371d5787f1e301cc199895bcdb6d172531d01feba7dc24bb281430d0ec60`、binary `7d50b94353bf7dd3ec2337216833d3d517353c80908af744247196e03f73963f`。
共有poolのsucceeded・全slot停止・supervisor終了を確認した。

### 前回の再確認：S128の通常部分和

runtime `8bc27c2e` / 文書込みsnapshot `9e2c7204`。
job `l4job-cff5845718a6411fb10fb629b66ca143`で各設定2 fresh process、2巡目は順番を逆転した。
全検査/prepare/forward/backward/fused AdamWを含み、warm5/同期wall21、TF32/Graph/workspaceなし。
固定B4/FP32/M8/tail1e-3/Q128、物理T_min入力4/出力8、A=floor(.05*N*N)、seed120。
前jobとatoms/X/dY/初期dense Wのbit一致を確認した。

| 設定 | 完全step（2 process） | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、compact、逆indexなし、Double値/勾配 | 0.853524 / 0.842994ms | 257KiB | 2MiB |
| matched dense128 | 0.730834 / 0.694958ms | 264.5KiB | 2MiB |
| 1024、S128/grain64、H/C融合 | 1.233614 / 1.232073ms | 9.602MiB | 12MiB |
| 1024、S128/grain64、H/C分離 | 1.228041 / 1.235813ms | 9.602MiB | 12MiB |
| 1024、S64/grain64、H/C分離 | 1.277881 / 1.279383ms | 7.575MiB | 10MiB |
| 1024、S64/grain128、H/C分離 | 1.331440 / 1.324161ms | 7.575MiB | 10MiB |
| matched dense1024 | 0.695166 / 0.684253ms | 16.063MiB | 22MiB |

全10 memory gateで同repeatのdense以下のallocated/reservedをassertした。
S128/grain64の改善が再確認された。H/C融合の小差は順序で逆転し、明確な勝者とはしない。
以下のkwargsは全atom更新まで確認した融合ありの設定で、融合が普遍的に速いという主張ではない。
1024の時間比は約1.77/1.80。**denseの2倍以内という時間条件を満たす**。

全819/52,428atomを3step更新し、毎stepの更新済み参照ParameterからCPU FFT/autogradで
Y/dX/4列勾配を独立に再計算した。FP32 rtol3e-6/atol2e-7、Parameter rtol2e-6/atol2e-7、
AdamW exp_avg/exp_avg_sq/step rtol4e-6/atol2e-8で全step通過した。
最大Parameter差は1024で1.5259e-5、相対・絶対許容差内でありbit一致ではない。
1024の全3step gateはS128/grain64/HC=Trueだけで、分離版へこの検証範囲を読み替えない。
正確性用Tensorは測定processへ持ち込まず、上表のallocator peakと分離した。
確認jobでは広いsuiteを再実行せず、直前の2580件とracecheck12件を根拠にする。

```python
settings128 = dict(
    c_splits=2, fuse_g_grad=True, layout_method="counting_compact",
    z_chunk_sites=0, atom_threads=128, window_moments=False,
    reverse_layout=False, fuse_h_c=False, fuse_backward_cin=False,
    compensated_values=False, compensated_grad=False,
    reuse_shape_coefficients=False, compensated_cdot=False,
)
settings1024 = dict(
    c_splits=128, fuse_g_grad=True, layout_method="counting_compact",
    z_chunk_sites=32, atom_threads=64, window_moments=True,
    reverse_layout=True, atoms_per_split=64,
    compensated_values=True, compensated_grad=True,
    reuse_shape_coefficients=True, specialize_g_grad=True,
    fuse_h_c=True, fuse_backward_cin=False, compensated_cdot=False,
)
```

公開dispatcherや研究APIの既定値は変更しない。このfixtureでの研究選択であり、
別のsigma分布・多軸・他GPUへの勝者とはしない。
従来のS64/grain128/HC=Falseは1.320542/1.321303ms、7.575MiB/10MiBで、
job `l4job-ac58be3290c24995ad286a78db379a0a`で両サイズの全atom3stepを確認している。
そのsource/results/spec/receiptはignored `output/cell-mass-opt-20261027/compact-shape-confirmation/`に保存し、
今回のS128とメモリ・速度の交換を比較する基準として残す。

今回のsource `43ac2e19fb837c96f925ded93c7f88e4f9d78aa4995572d7d7087ad0572e99c3`、result `e8c92baa8fad541dd0fb707b7aea3e7a2744bd94d9dc2efb31c0066f5145809b`、driver `67b7fd257ae49268cf545d2175b45f91f0af6c63e16b9bd2ffa1a0391eae2ea0`、binary `0d935678a92506c89f46b3b59f48d6a9add555a199a4b0747873ca976dffc9fd`。
runtime338・検査141、manifest55件、driver/binaryとfixture bit一致を照合し、
ignored `output/cell-mass-opt-20261027/expanded-split-confirmation/`へ保全した。

### 最適化第18段：C分割数の拡張と融合forwardの位相（初回GPU検証）

runtime `8bc27c2e`。C splitを1..256まで許し、必要page数が64以下のときだけ実行する。
既存SplitPagesの64 pointer、CUDA引数とkernelの部分和レイアウトを拡大しない。
4 native入口でpage数を検査し、超過時はpointer書込みとpartial確保前に拒否する。
このfixtureのCin一枚は33,184Bで、S64/128/256の部分和は約2.025/4.051/8.102MiB。
128には大きすぎるため、性能候補は1024だけでS64/128/256×grain32/64/128を比較する。
logical scratchをdense peakへ単純加算して採用せず、完全stepの両peakを実測する。

同じsourceで任意のH/C融合も、入力位相を相対回転して出力係数を作る方式から、
正規化済み幅振幅に独立の出力位相を掛ける方式へ変えた。
入力H、丸め、shape不一致fallback、sigma/Pを保ち、余分なFP64回転を省く。
既定Falseとし、S/HC flagを独立に切り替える。
CPU31 passed / 745 skipped、全CPU2037 passed / 5309 skipped / 既存警告18件、
buildと両配布物338runtime照合は通過した。CUDA検証の代用にはしない。
job `l4job-c0a3e6edf7b64ec7a500f4846c10c8a6`は通常2580 passed / failed0 / skip12、
選択racecheck12件・hazard0。全819/52,428atomのY/dX/4列勾配も独立FFT/autogradへ通過した。
固定fixtureは前jobとbit一致。各1 fresh process、warm5/同期wall21の完全stepは次の通り。

| 1024、H/C分離 | grain32 | grain64 | grain128 | allocated / reserved |
| --- | ---: | ---: | ---: | ---: |
| S64 | 1.284454ms | 1.277864ms | 1.324204ms | 7,943,168B / 10MiB |
| S128 | 1.259195ms | 1.234461ms | 1.291558ms | 10,067,968B / 12MiB |
| S256 | 1.301561ms | 1.258579ms | 1.302658ms | 14,317,056B / 16MiB |
| matched dense | — | 0.677338ms | — | 16,843,264B / 22MiB |

H/C融合はS64/grain128で1.372390msと分離版より遅い一方、S128/grain64では
1.215154msとなった。peakは各Sの分離版と同じ。この小差は独立再確認前の候補とする。
128は従来のcompact/S2で0.834389ms、dense0.693687ms、両peakも従来と同じ。
全12 CST設定でdense以下のallocated/reservedを実測した。
採用判断はS/grain/HCを含む各step.jsonから再計算した。初版memory-candidate-gates.jsonには
S/HCの識別fieldがないため、単独では設定ごとの証拠として使わない。

別Graph-event診断ではCoutがS64/grain128の0.227328→S128/grain64の0.166912ms、
Cinが0.157696→0.126976msとなった。stageを足して完全stepとはしない。
S256の減速原因は未確定で、空splitのzero書込み・全split縮約・task配置を次に切り分ける。
frame/周期/σを変えた改善ではなく、同じ数式の仕事の分配だけを比較している。

source `00699dc1e3996d1478b0aadf66048794cf486bf16f12cd244ca4b66e5f4b3519`、result `3166c037d7c2d8701a7cbce35c276763c77345db12a29a6e50c3a53aa806ce9a`、driver `7a99c786dd4e1447cb8a16d58056fad8af6fb2b9be6f69a34f8c38bf700886e1`、binary `19d38183a8c16294e4574dafbe975c7ae5d79b9e86f4104cfa3ceee9f7d6fa0c`。
runtime338・検査141、manifest68件、driver/binaryとfixture bit一致を照合し、
ignored `output/cell-mass-opt-20261027/expanded-split/`へ保全した。
job `l4job-cff5845718a6411fb10fb629b66ca143`の再確認結果は上節に置く。
S128/grain64の改善は確認したが、H/C融合の上乗せは測定順で逆転した。

### 第12段までの再確認：完全stepと1024のAdamW状態

runtime `7fd9132b`。job `l4job-3d9f12cdfca6486fa532f515d7b04cda`で別の所有L4へ切り替え、
各設定2 fresh process、2巡目は順番を逆転した。warm5/同期wall21、全検査/prepare/
forward/backward/fused AdamWを含む。TF32/Graph/追加workspaceなし。
前jobとatom/X/dY/初期dense Wのbit一致を確認した。

| 設定 | 完全step（2 process） | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、保守的なH/C分離・Double値評価 | 1.182075 / 1.149119ms | 242.5KiB | 2MiB |
| 128、H/C融合・Double値評価 | 1.149945 / 1.146039ms | 242.5KiB | 2MiB |
| 128、H/C融合・補償値評価 | 1.164949 / 1.127448ms | 242.5KiB | 2MiB |
| matched dense128 | 0.690988 / 0.686519ms | 264.5KiB | 2MiB |
| 1024、逆index/grain128・H/C分離・補償値評価 | 1.434035 / 1.431054ms | 7.596MiB | 10MiB |
| 同上、Double値評価 | 1.476370 / 1.481064ms | 7.596MiB | 10MiB |
| matched dense1024 | 0.669389 / 0.688117ms | 16.063MiB | 22MiB |

全CST設定で、同repeatのdense以下のpeak allocated/reservedをassertした。
128の融合・補償の差は順序やjobで揺れがあるため、保守的な分離設定を維持する。
1024の補償値評価は独立再確認でも約3%短縮し、メモリ増加はなかった。
1024はdenseの2倍以内という時間条件にこの時点ではまだ届かない。

このjobでは1024の全52,428atomを3step更新し、各更新後の参照Parameterから
独立CPU FFT/autogradでY/dX/全4列勾配を再計算した。
Y/dX/勾配はrtol3e-6/atol2e-7、Parameterはrtol2e-6/atol2e-7、
AdamW momentsとstepはrtol4e-6/atol2e-8を維持して全step通過した。
最大勾配絶対誤差は3.16e-8未満、最大Parameter絶対差は1.53e-5未満で、
Parameterの相対・絶対誤差の複合基準を通過した。bit一致とは主張しない。
この正確性検査用の参照Tensorは計測に持ち込まず、時間/ピークは別fresh processで測った。
GPU suiteの再実行はこの確認jobでは行わず、直前の2438件/階層版631件を根拠とする。

source `0dc5210d24e864ce8dbde9abd32db44444899af3c62c2e5469382a6584c2c106`、result `9623c160c9fc06bd9ab7c8f770533324cae1d7475bcb620319fc810ca5440a26`、driver `71fc5970a7484a11bae226c954a4edee1d4e60fbec873e7b8b15efd07d09d67d`、binary `bf1bb7eec946cca2ccf7a246a1093bd94f4d37ec4379fb112801033518630c4a`。
327runtime、両header・検査ソース、結果54ファイルを照合し、ignored
`output/cell-mass-opt-20261027/final-confirmation/`へsource/results/spec/receiptを保全した。
CPU Actionsはsnapshot `9c972124`（runtimeは同じ）でsuccessを確認した。

### 第7段までの独立再確認と、現在使う研究設定

この節のkwargsは第7〜9段時点の記録である。現在の検証済み選択は
[最新の再確認と検証済み研究設定](#最新の再確認と検証済み研究設定)を参照する。

runtime `dcf48273` / snapshot `46b069e2`（差は文書のみ）。別の所有L4で各設定2 fresh process、
2巡目は測定順を逆転した。warm5/同期wall21、同じatom/X/dY/初期dense Wは前jobとbit一致。
検査/prepare/全forward/backward/fused AdamWを含む。JIT/oracle/初期W構築は除外。
TF32なし・Graphなし・cuBLAS workspaceなし。前jobの通常/tuned各2278 gateを根拠とし、
このjobで正確性suiteを再実行したとは扱わない。

| 条件 | 2 fresh processの完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、C2/sort/Z0/128threads、Rなし、Cin分離 | 1.205635 / 1.121764ms | 242.5KiB | 2MiB |
| 128、同上Rあり、Cin分離 | 1.148473 / 1.144093ms | 242.5KiB | 2MiB |
| 128、同上Rあり、Cin融合 | 1.121268 / 1.120112ms | 242.5KiB | 2MiB |
| dense128 | 0.699004 / 0.687199ms | 264.5KiB | 2MiB |
| 1024、C16/counting/Z32/64threads、Rあり、Cin分離 | 1.704220 / 1.723238ms | 5.875MiB | 8MiB |
| 1024、C64/同上、Cin分離 | 1.657808 / 1.655620ms | 7.396MiB | 10MiB |
| 1024、C16/同上、Cin融合 | 2.238962 / 2.236008ms | 5.875MiB | 8MiB |
| 1024、C64/同上、Cin融合 | 2.302701 / 2.263328ms | 7.396MiB | 10MiB |
| dense1024 | 0.668562 / 0.683754ms | 16.063MiB | 22MiB |

1024のCin融合は同sourceの比較でも悪化した。128のR/Cin融合はこのjobで小幅に速いが、
過去jobとのゆらぎが小差を上回るため、全サイズ一律の自動採用はしない。
通常のprivate API既定値は比較用に残し、公開dispatcherの既定は変更しない。
現在の保守的な128設定と、確認した1024設定は次のkwargsで明示する。
128でR/Cin融合を選ぶ場合も、同じ数学/メモリ契約の研究選択肢であり、普遍的勝者とはしない。

```python
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import hierarchical_linear

settings128 = dict(
    c_splits=2, fuse_g_grad=True, layout_method="sort",
    z_chunk_sites=0, atom_threads=128,
    window_moments=False, fuse_backward_cin=False,
)
settings1024 = dict(
    c_splits=64, fuse_g_grad=True, layout_method="counting",
    z_chunk_sites=32, atom_threads=64,
    window_moments=True, fuse_backward_cin=False,
    reverse_layout=True, atoms_per_split=128,
    compensated_values=True,
)
y = hierarchical_linear(plan, x, atoms, **settings1024)
```

上の1024 kwargsは第8段の逆index/grain128も含む。第7段の再確認表とは区別し、
第8段の逆indexで1.45〜1.49ms、補償値評価を加えると2jobで約1.43msを確認した。
H/C融合は1024で悪化したため、上の設定には含めない。

この設定はB4/FP32/M8/tail1e-3/Q128、A=floor(.05*N*N)、sigma初期linspace(.2,40)、
物理T_min入力4/出力8のfixtureで確認したもの。別の幅分布・多軸・他GPUの勝者へ一般化しない。
1024は初版23.211msから約14倍短縮したが、dense比約2.42〜2.48倍の時間であり、
denseの2倍以内という時間条件はこの時点では未達。

job `l4job-b3fdda00495f458d8706f9e0b74a4e13`、source `e87dda784b5b41dcf87505be5319242b002df6e8273659abdca9451d1a4eb872`、result `13aa1b7f83e6420a6948d79a564a596f0e56b5eab3561f0f2243645efdb38853`、driver `665b8c3f78fa1d0abcb383227891b188622f921f2cd70d122013182e0e8a7adf`、binary `b960d6fded186f6b49ddb96842bc02c25d88bd362616b1b75fb2318e04eb2219`。
327 runtime＋両header・検査ソース、結果69ファイルとfixture bit一致を照合し、
ignored `output/cell-mass-opt-20261027/confirmation/`へsource/result/spec/receiptを保全。
所有L4は共有poolで停止を確認した。source `46b069e2`のCPU validationもsuccess。


### 当時の物理sigma経路：入力dtypeでH/Gを保存するbackward

runtime source `1ec0f66b`。FP32入力ではH/GをFP32で完成保存し、途中の縮約・正規化係数・
微分・Z/Cは引き続きFP64で計算する。FP64入力は保存もFP64。
FP64の全Hを作ってからcastするのではなく、最終値を指定dtypeへ直接書く。
低水準`compute_h/compute_g`の診断既定はFP64のままで、`storage_dtype`で保存dtypeを指定できる。
元Parameter、profileの式、sigmaの意味、tile分類、許容誤差は変更していない。

階層版218件（CPU31/CUDA187）＋既存1816件の**2034 passed、failed/skipped 0**。
前段の全勾配・retained backward・mutation検査・gradcheck・3stepのAdamW比較を再実行し、
保存Hが入力dtype、保存ZがFP64であることも検査した。
racecheckはretained backwardの3件で0 hazards/errors/warnings。
CPU全体2037 passed / 4751 skipped / 既存警告18件（44.17秒）。
Ruff・format・差分検査・wheel/sdist buildを通過し、packageの337ファイルをsourceへbyte照合した。

1024/B4/A52,428でも全atom4列を512atomずつ独立FFTセル積分参照へ再照合し、PASS。
最大絶対誤差はY=1.7711e-8、dX=1.9915e-8、全atom勾配=2.7011e-8。
許容差は前段と同じFP32 rtol3e-6/atol2e-7である。1024のoptimizer状態の独立比較は未実施、
128の全atom/AdamW状態比較は再通過した。

完全stepの条件・処理範囲は前段と同じ。sigma0.2..40、Q128、M8、batch4、
物理T_minは入力4/出力8（128では4/8sites、1024では32/64sites）。
各方式1 fresh process、warmup5/同期wall20標本、入力/dY固定、TF32なし、cuBLAS workspace 0、Graphなし。
元atom profileからdense Wを作る時間とoracle/JITは計測から除外する。

| 完全layer step | CST中央値 | dense中央値 | CST allocated / dense | CST reserved / dense |
| --- | ---: | ---: | ---: | ---: |
| 128、819atom | 2.249732ms | 0.703702ms | 166.5 / 264.5KiB | 2 / 2MiB |
| 1024、52,428atom | 23.211011ms | 0.674475ms | 5.332 / 16.063MiB | 8 / 22MiB |

現在の両fixtureでallocated/reservedともdense以下。1024のCST allocatedは5,591,040bytes、
H/Gは各838,848bytes。FP64保存初版のreserved26MiBを8MiBへ減らした。
時間は128で約3.20倍、1024で約34.41倍で、**denseの2倍以内という時間条件は未達**。
保存量削減で速度も改善したとは判断しない。プロセス全体のGPUメモリ、全体Graph学習、
複数独立runでの性能再現は未検証。C_inのgroup走査とregister保持量は次の分解計測候補であり、
現時点でどちらが最大のボトルネックかを断定しない。

job `l4job-84892b903dae42c68640c7705652b8d2`、L4 / Torch2.11.0+cu130 / CUDA13.0。

- source SHA256 `882983d42d4985f106526bc9741027c3e74fad8e8df57d5255909b9410a5e1ac`
- result SHA256 `34a29566dfbdc89808b0408a2f70b2a8010eb6bad50be6e12de0c1caba71b614`
- driver SHA256 `fe8f555d2bff1cc648456b943c87e53635dbb5e15b8adffa8b3c2002b0eb59e3`
- binary SHA256 `4aaedbe3f72282d8cc514eb1198e917414c0fd9aeb1d7a8423250827ddbce3fb`

327 runtimeソース・共通header・検査ソース・埋込1024 driverを照合し、manifest42ファイルのhashを確認。
source/result/spec/receiptをignored `output/cell-mass-backward-20261027/storage-run/`へ保全した。
再検証は`python -m pytest -q tests/test_cell_mass_hierarchical.py`と既存3つのcell-mass検査。
同じGPU実行は保存済みdriverを共有poolのsubmit/serveへ渡す。
旧FP64保存型の結果と負のreserved結果も下に残し、現版の値と混在させない。

### 前段：FP64保存型backwardの検証と128の完全step

runtime source `33d7f9dd`、L4 / Torch2.11.0+cu130 / CUDA13.0。
階層版216件＋既存1816件の**2032 passed、failed/skipped 0**。
独立参照には既存profileのFFT係数から作る微分可能なセル積分を使い、
Y/dX/全atomの4列をFP32/64、direct/merge Z、local/full bank、chart端、
非ゼロ原点、異なる入出力サイズ、M16、batch1/3/4/5で照合した。
128/B4/A819は全atomを検査。1024siteの小さな35atom検査も含む。
幅・ownerの整数選択を固定した枝の解析微分であり、階層遷移の連続性の証明ではない。

FP32はrtol3e-6/atol2e-7、FP64はrtol3e-9/atol2e-10の許容差。
これは実測最大誤差の値ではない。数値gradcheck、空batch/atom、必要な勾配だけの呼び出し、
retained backward、非連続dY、保存atom/layoutのin-place更新検知も検査した。
128/819atomの3stepでは幅の階層とtorus ownerを移し、毎forwardの再分類を行い、
AdamWのParameter・exp_avg・exp_avg_sq・stepを独立勾配で更新した同じAdamWへ照合した。
これは物理atomへのAdamWであり、公開CSTOptimizer/Parameterizationとの統合検証ではない。
compute-sanitizer racecheckはretained backwardの3件で0 hazards/errors/warnings。
二階微分とsortを含む全体Graph学習は未対応で、backwardのGraph replayも未検証。

完全stepは128×128/A819/B4、FP32、M8/tail1e-3、Q128、入力T_min4・出力T_min8、
sigma=linspace(0.2,40)、seed120。入力/dY固定、grad初期化・検査・sort・全forward/backward・
fused AdamWを含む。JIT・oracle・初期dense Wの生成は含めない。
同じ初期profileからdense Wを作り、両者ともX勾配と重み側勾配を計算する。
TF32なし、cuBLAS workspace 0、Graphなし。各方式1 fresh process、warmup5/同期wall20標本。

| 完全layer step | 中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 階層版CST | 2.239738ms | 197,120bytes（192.5KiB） | 2MiB |
| dense、workspaceなし | 0.678630ms | 270,848bytes（264.5KiB） | 2MiB |

時間比約3.30倍で約2倍目標は未達。allocatedはdenseの約72.8%、reservedは同じ。
単一runの初版診断であり、採用性能として複数独立runの再確認はしていない。
全GPU process使用量は未測定。固定bank版の過去結果とは周期・幅分布が異なるため改善率にしない。

job `l4job-b59c3c9f88d64b239e13952d2bdc4cae`、driver実行411.20秒（JIT・検証込み、step時間ではない）。

- source SHA256 `d24ba011d7160a99eebc702b039ea342d4f26ea89343ce211c70c1954be33288`
- result SHA256 `75987af45be3f960e7e8febab64b659102ca7aed8394d1fddef082eaca7d4102`
- driver SHA256 `c0e9967b444aef36d57cc72d2b8b972e78d6b6bfeaf2b61c2ab926ed10ca2b44`
- binary SHA256 `5cad5b6ece85b5292f00680ac1e9559256dae15a6d9e4c2b141871617f2e69f5`

327 runtimeコード・共通header・検査ソースをcommitへbyte照合し、結果manifest22ファイルのhashを確認。
source/result/spec/receiptをignored `output/cell-mass-backward-20261027/first-run/`へ保全した。
CPU全体2037 passed / 4749 skipped / 既存警告18件（43.50秒）。Ruff・format・差分検査・build通過、
wheel/sdistの337ファイルを現sourceへbyte照合した。GPU/DBのskipは検証完了へ数えない。

### 前段：1024全atom検証とFP64保存の負の結果

同じruntime `33d7f9dd`を1024/B4/A52,428、Q128、spacing0.125、入力最小tile32sites・
出力64sites（物理T_min4/8）で検査した。sigmaは0.2..40の等間隔値、seed120。
512atomずつ既存FFTセル積分でCPU参照を作り、全atom4列の勾配を省略せずに照合した。
最大絶対誤差はY=1.4745e-8、dX=1.4850e-8、全atom勾配=2.7011e-8、全てPASS。
1024のoptimizer状態の独立照合はこのjobでは実施していない。

| 完全layer step、FP64 H/G保存 | 中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 階層版CST | 23.130443ms | 7,268,352bytes（6.932MiB） | 26MiB |
| dense、workspaceなし | 0.669730ms | 16,843,264bytes（16.063MiB） | 22MiB |

同じ初期profileからdense Wを作った。比較方法は128と同じ、方式ごとに1 fresh process。
約34.54倍で速度目標未達。allocatedはdense以下だが、**reservedはdenseを超える**。
このFP64保存型をメモリ条件を満たす採用値にはしない。各H/Gは1,677,696bytesであり、
FP32入力では入力dtype保存へ変更して再度全勾配・stepメモリを検査する。
allocatorの予約はTensor容量と異なるため、保存量だけで達成を判断しない。

job `l4job-558ff9aa21ec4ca7ac83e1d61db4eb46`（driver250.01秒、性能値とは区別）。

- source SHA256 `9ea59b717276f3ae5f0c18c573e8d91de2c9d8df16b8e1fa7455a731bf575785`
- result SHA256 `f281ec92603f5b88c02fcd5d5f3d3b1eab0b105076bc55d784eab46c46a72dd7`
- driver SHA256 `7ba7882c19f0acf568c4543c1fcc381c3c32f7a9619e2042692af8a1a06329e3`
- binary SHA256 `42634e664408a62a176ed0e28f548bf7fc3a4b702ee1a53457ccefd43cc20624`

327 runtimeソースと結果manifest20ファイルを照合し、
ignored `output/cell-mass-backward-20261027/run1024/`へsource/result/spec/receiptを保全した。
GPU process全体のメモリは測定していない。旧固定bank版とprofile周期・幅分布が異なり、
既存の1.354msからの退行率にはしない。

### 前段：H/C/Y接続とsigma整合の検証

runtime source `d09eba32`。既存`ChebyshevSquareSpec`の式・sigma・半径下限を維持し、
分類をsigma<=T_hへ修正した。I=sigma/2の前半prototypeから意味を読み替えない。
既存固定bank版のソースとpublic APIは変更していない。

新規188件（CPU31/CUDA157）＋既存1816件をL4で実行し、**2004 passed、failed/skipped 0**。
新しいHは既存profileのFFT係数からセル積分したCPU参照へ照合し、
Cは同じ既存係数を使う独立のowner集約へ、Yは直接profile行列を掛けた参照へ別々に照合した。
さらに既存生密度を48点Gaussでセル積分した参照ともCPU上で照合した。
新しいCUDAの正の多項式を参照側へコピーして正しさを判定していない。

FP32/64、local/full bank、QがPを割り切らないchart端、最後の半セルのowner=0、
非ゼロ原点、幅境界と上限のdtype cast、振幅0、空atom/batch、batch1/3/4/5、
128/1024site、M16を含むH/C/Y、direct/merge Z、出力担当幅8/32を検査した。
検査許容差はH/CとFP64 Yがrtol2e-10/atol5e-11、FP32 Yがrtol2e-6/atol2e-7。
これは測定した最大誤差を意味しない。unit XのH=1、Yの総和も照合した。
compute-sanitizer racecheckの小さなunit-mass/cache検査は1 passed、0 hazards/errors/warnings。
Z→H→C→Yの低水準Graph replayは新しいXで照合したが、atoms/layoutは不変である。
replayではPythonのcache version検査が再実行されないため、atom更新を伴うGraph学習の検証ではない。

CPU全体2037 passed / 4721 skipped / 既存警告18件（45.09秒）、Ruff・format・差分検査・buildを通過。
wheel/sdistの336ファイルを現srcへbyte照合した。GPU snapshotの326コードと共通headerもgit/sourceへ照合。
共通headerのSHA256は`ccf5f0073d70fc16b4b44703e18cffcaac7dbcf669ebcbb50673d657d7dd7b13`。

128×128/A819/B4、FP32、M8/tail1e-3、入力T_min4/出力T_min8、Q128、spacing1、
sigma=linspace(0.2,40)、中心seed120の**forwardのみ**をfresh processで診断した。
warmup5/同期wall20標本、Graph/TF32なし、X検査・分類/sort・Z/H/C/Yを含み、
backward・Parameter勾配・AdamWを含まない。1fresh processの中央値は1.113883ms、
peak allocated105,472bytes（103KiB）/ reserved2MiB、baseline allocated15,360bytes。
独立runを複数回確認した採用値でも、完全stepのdense以下を検証した値でもない。
固定bank版の完全stepとはsigma分布・周期・処理範囲が異なるので改善率へ使わない。

| 配列 | このcaseのpayloadと場所 | 用途 |
| --- | --- | --- |
| atoms / X | global FP32、13,104 / 2,048bytes | 元Parameterと入力。atomの複製なし |
| rows / groups / offsets | global、計11,936bytes、224 groups | 元atom行への対応とグループ範囲 |
| Zx | global FP64[4,61,17]、33,184bytes | 入力側の階層moment |
| H | global FP64[4,819]、26,208bytes | slot順の完成H、振幅前 |
| C | global FP64[4,29,17]、15,776bytes | 出力bank/ownerの完成C |
| Y | global FP32[4,128]、2,048bytes | siteへ展開し、bank寄与を合計した出力 |
| atom係数 / amp*H chunk | block shared、dynamic5,376bytes＋static16bytes | 現在の32atomだけ。chunk間の同期後に再利用 |

payload和とallocatorピークは丸め・寿命・scratchが違うため同一視しない。
cuobjdumpでもC kernelのstatic shared16bytesを確認した。register/sharedとCUDA moduleは
Torch allocatorの指標外であり、全GPU processの使用量は未測定。
係数・中心位相はregisterで生成してshared chunkへ書き、atomごとの係数表をglobalへ作らない。
H/Cのglobal保存はblock間で受け渡すための基準実装。完全stepの保存／再計算・融合は今後比較する。

job `l4job-95f9b6adde8c4ccb982cf1abad4fac12`:

- source SHA256 `92450b0fc1effcd20675d1df9bd110b5a90d224558b1779ef8927d1463161cc4`
- result SHA256 `f18d87b69a993190b6ab469e470e98f1241c93150a0128f02b4196bb4b02680a`
- driver SHA256 `2622ae687e9c672889126fb62608a219afe0f2adc346a73c1dabbd2f7f5ace2f`
- binary SHA256 `2c1a30a9fbdce6f90d949013b73a35c8b13f78e1ce676706a8631381fc2a81ac`

manifest16ファイル、driver/binaryを照合。raw/source/result/spec/receiptはignored
`output/cell-mass-hc-20261027/first-run/`に保全。所有L4 runtimeの停止も確認した。
再検証は`python -m pytest tests/test_cell_mass_hierarchical.py -q`。
新規profileの数式待ちではない。同じ式のG/dX/Parameter勾配を実装し、検証記録を冒頭に追記する。

### 前半prototypeの検証・診断（sigma整合修正前の履歴）

runtime source `44c93543`。この節は撤回したsigma=全幅の分類での履歴であり、現在のH/C/Yの検証結果ではない。
既存`cell_mass/`からのimport・ソース変更はなかった。
新規146件（CPU分類27件＋CUDA119件）と既存のCUDA・数学1816件を一緒に実行し、
**1962 passed / failed 0 / skipped 0**。L4 / Torch2.11.0+cu130 / CUDA13.0。
FP32/64、空/端数atom、幅の境界、周期端、全周のみのplan、空/端数batch、
128/1024site、M2/4/8、Zのthread team 1/4/32を含む。
Zはセルごとの24点Gauss積分から独立に作ったCPU参照へ照合した。
親が同じharmonic番号の子を足してしまう誤りも専用caseで検出する。
Z構築だけのCUDA Graph capture/replayは新しいXで照合したが、layoutや完全stepのGraph検証ではない。
新経路のY/dX/Parameter勾配はまだ存在しないので、既存1816件をその検証として数えない。

CPU全体は2033 passed / 4683 skipped / 既存警告18件（46.18秒）。Ruff・format・差分検査を通過。
wheel/sdistをbuildし、両packageの334ファイルを現srcとbyte照合した。

診断条件はB4、atom数floor(0.05N^2)、sigma=linspace(0.2,100,A)、中心seed120、
Q128、入力T_min=4/出力T_min=8、M8。128ではspacing1とmin_tile_sites=4/8、
1024ではspacing0.125とmin_tile_sites=32/64を使う。
独立な各2fresh process、2巡目は逆順、warmup5/同期wall20標本、Graphなし。
計測範囲は**分類・sort・group作成＋Zx/Zyのみ**。H/C/Y、dX、Parameter勾配、AdamWを含まない。

| site数 | 前半のみ ms run0 / run1 | peak allocated | peak reserved | layout / Zx / Zy payload bytes |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.7623 / 0.7605 | 78,848bytes（77KiB） | 2MiB | 11,192 / 33,184 / 15,776 |
| 1024 | 3.9127 / 3.9113 | 3,422,208bytes（約3.264MiB） | 4MiB | 439,808 / 33,184 / 15,776 |

baseline allocatedは各17,408 / 871,936bytes。peakには入力atoms/X/dYと前半の一時領域を含む。
新しい完全stepのメモリがdense以下だと検証した値ではない。
これは前半の初版の診断で、速度最適化・完全step比較・Graphピーク・全GPU processメモリは未完了。
新幅定義・幅分布・処理範囲も旧測定と異なるので、旧版に対する同条件の改善率にはしない。
128のgroup数193、1024では849。各入出力階層のatom数もraw JSONへ保存した。

検証job `l4job-dc3552d6f4744d2590e234a3d6e53133`:

- source SHA256 `15dd1879f53498ff779d072abc2b4be5a2ed4d438be66956edf57978edc04181`
- result SHA256 `c9e79dfbb559b2f2c858184cf1006c01b98393423cf9635303f5f8560e3c120c`
- driver SHA256 `302bb2881d278550d4e7c50dfd9937a3f6ddd9aed3f900a1283519576816bb35`
- binary SHA256 `4f667530443b20cf271248fc792074049760b41a39cbd15398dcd8dcaf1e000c`

fresh process診断job `l4job-64d587a39ee3453c8b2c952a02eea6b8`:

- source SHA256 `bfce127c57b7f7bd0c3d6607b8ea593894f2e320e895c4738608e7b01f18e619`
- result SHA256 `b2778b0fd239741b6ebbdd80e834f63db329767e1b3707967e7b7ab0205033e3`
- driver SHA256 `39f313349f286c3bff7e251a1e33613358d0433604b3f3d4dd372bc055a90faf`
- binary SHA256 `ac74a4ab841b4ead035d72c164e69ce35ed1d4d777b05b6b6e3e454288ca1133`

両jobの325 runtimeコード、manifestの8/31ファイル、driver/binaryをhash/byte照合。
raw/source/result/spec/receiptはignored `output/cell-mass-hierarchy-20261027/verified/`と
`fresh-process/`へ保全する。検証直後の同一process診断は約8MiBの残存割当があり、
新しい前半の代表ピークには使わず、上記fresh processの値を使う。
最初のjob `l4job-b9af7d6fa60d419ca101075438e51808`はNinja不足でコンパイル前に失敗。
成功件数・時間へ含めず、archiveとverificationを同じ親directoryの`first-run/`へ保全した。
依存はjob内へ追加し、共有poolが所有runtimeを停止する。停止状態は終了後にも確認する。

再検証は`python -m pytest tests/test_cell_mass_hierarchical.py -q`。
新旧一緒の検証には`tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py`も加える。
性能診断の再現は固定source archiveのdriverと空の`CST_JOB_OUTPUT`を使う。
共有L4ではpool submit/serve経由とし、新しいdriverを別profileの完全step測定へ流用しない。

現在のprofileは[1周期分のタイル窓](cell-mass-tile-window.ja.md)。
選んだwhole tileを共通のセル積分基底で処理し、支持境界での個別積分は行わない。
blockの担当site範囲は別の実行設定`sites_per_block`で分けられる。
固定bankの測定ではprofileのtile幅・周期を固定していた。
[担当site数の比較](cell-mass-block-sites.ja.md)は、その条件でのlaunch調整の記録。
現在の実コードには、Y/dXのatom集合を分担する`atom_splits`と、
H/G・Parameter勾配のblock幅を指定する`atom_threads_per_block`もある。
部分出力の寿命・合計kernel・係数sharedの具体的な実装は
[atom分担と位相共有](cell-mass-atom-splits.ja.md)を参照する。
実コードの投影微分を保存／再計算する分岐は
[projection cache](cell-mass-projection-cache.ja.md)、
メモリ制約の採否は[完全step比較](cell-mass-memory-frontier.ja.md)を参照。
現在の1bank runtimeでは、値検査とkey生成を融合しstable sortした行番号を保持する。
`owner_coefficient_reduction=True`は、その順序から(output owner,input owner)の
begin/endも作る。[owner C](cell-mass-owner-coefficients.ja.md)で係数を一度生成し、
[共有Z窓](cell-mass-owner-windows.ja.md)からH/Gを投影する。
最新の固定bank経路には[G/Parameter勾配とC_in生成の融合](cell-mass-cin-fusion.ja.md)、
[係数と微分のHorner評価](cell-mass-horner-and-sigma.ja.md)もある。
これらの算術・配列再利用は階層経路への移植候補であり、階層分類の代わりにはならない。

2026-10-26。[設計と数学契約](cell-integrated-profile.ja.md)に対応する、検討用の擬似コード。
以下の関数名・引数は一般化した擬似コード。明示的なChebyshevSquare密度から始めた
[初版の実コードと対応範囲](cell-mass-implementation.ja.md)は別に記録する。
全てのprofile族・複数bank分類・更新則まで実装済みとは扱わない。

最初はH/GとZx/Zyをglobalへ保存し、Parameter勾配まで再利用する構成で考える。
この保存量を基準に、HやZxの再計算・局所融合を後から比較する。
入力・出力128サイト、atom数128、batch 4から始める。thread数などは後で変更できる。

## 前準備と順序

```cpp
// forward
prepare_layout(atoms, spec, layout);
build_input_hierarchy(X, layout, Zx);
compute_h(Zx, atoms, layout, H);
compute_y(H, atoms, layout, Y);

// backward
build_output_hierarchy(dY, layout, Zy);
compute_g(Zy, atoms, layout, G);
compute_dx(G, atoms, layout, dX);
compute_parameter_grad(Zx, Zy, H, G, atoms, layout, grad);
```

同じstreamへ依存順に投入する。上の名前は論理段階であり、実際のlaunch数ではない。
owner方式のcompute_y/dxはC部分和を作るkernelと展開kernelに分かれる。
階層版の`fuse_g_grad=True`はGとParameter勾配を融合するが、forward Hはbackwardのため保存する。
旧固定bank版のH再計算型[融合経路](cell-mass-fused-pullback.ja.md)とは保存寿命が異なる。
階層構築は「signalから直接作る要素」と「子Zから継承する要素」に分ける。
前者はlevelをまたいで並列に作り、後者だけlevel順に起動して子の完成を待つ。
CPU側で毎回同期するという意味ではない。
以下ではgeometry・正規化設定などの引数を省略し、補助関数が同じspecを参照するものとする。
compute系ではソート済みの位置slotと、元のatom行番号original_idを区別する。
H/Gはslot順で保存する。atomの値は`atoms[layout.sorted_rows[slot]]`から読み、
Parameter勾配はoriginal_idの行へ書く。atom全体のソート済みglobalコピーは作らない。

## 起動するものとkernel内で呼ぶものの区別

| 名前 | 種類 | 役割 |
| --- | --- | --- |
| `prepare_layout` | host側の起動関数 | 以下の分類・ソート・配置・範囲作成を順に起動 |
| `classify_atom_keys` | CUDA kernel | 元のatom行から階層とソートキーを作る |
| `device_sort_pairs` | GPUライブラリ呼び出し | キーと元の行番号を一緒にソート |
| `mark_group_heads` | CUDA kernel | 同じキーの連続区間の先頭を検出 |
| `device_inclusive_scan` | GPUライブラリ呼び出し | 先頭フラグの累積和からグループ番号を作る |
| `write_group_begins` | CUDA kernel | 各グループのキー・beginとグループ数を書く |
| `write_group_ends` | CUDA kernel | 次のbeginから各グループのendを書く |
| `build_input_hierarchy` | host側の起動関数 | Xからの直接生成と、level順のマージを起動 |
| `build_output_hierarchy` | host側の起動関数 | dYからの直接生成と、level順のマージを起動 |
| `build_z_direct` | CUDA kernel | 全levelの直接生成要素を並列に作る |
| `merge_z_level` | CUDA kernel | 指定levelの継承要素を子Zから並列に作る |
| `compute_h` / `compute_y` | CUDA kernel | 以下の第1・第2節 |
| `compute_g` / `compute_dx` / `compute_parameter_grad` | CUDA kernel | 以下の第3〜第5節 |

各kernel節のコードブロックはkernelの本体を示す。
`assigned_atom`や`input_coefficient`などはkernel内のdevice補助関数で、追加のkernel起動ではない。
sort/scanの内部で使うkernel数はGPUライブラリに任せる。ライブラリ内部まで自作することは前提にしない。
host側でshapeから出力・scratch容量を確保し、同じstreamを全起動・ライブラリ呼び出しへ渡す。
以下の配列は明記しない限りglobalにある。

## prepare_layout：分類・並び替え・範囲の準備

目的はatomの意味を変えることではなく、実行用の順序と検索情報を作ること。
atom数A=128の例では、次の作業配列を最大A要素分用意できる。
これは一般化したsort/scanの説明用で、全配列を常設する要求ではない。
新しい初版は上記のライブラリsort/uniqueを使い、後続へ渡す順序・範囲だけを残す。
sort/scan用scratchはライブラリが要求する容量を別途数える。

| 配列 | 内容 |
| --- | --- |
| `keys[A]`, `source_rows[A]` | ソート前のキーと元のParameter行番号 |
| `sorted_keys[A]`, `sorted_rows[A]` | ソート結果 |
| `head_flags[A]`, `group_numbers[A]` | グループ先頭とその累積和 |
| `group_key[A]`, `group_begin[A]`, `group_end[A]` | 有効なグループだけを先頭から格納 |
| `group_count[1]` | 有効グループ数。GPU上で後続kernelが読む |

ここでsorted_rowsとgroup関連の出力はlayoutが参照するbufferであり、
擬似コード中の短い配列名はその別名とする。head_flagsやscan結果はprepareだけのscratch。
atomのsigma・中心・振幅は元のParameter配列にだけ置き、prepareでは複製しない。

```cpp
// host側。同じstreamへ順に起動する。
classify_atom_keys<<<...>>>(atoms, spec, keys, source_rows);
device_sort_pairs(keys, source_rows, sorted_keys, sorted_rows);
mark_group_heads<<<...>>>(sorted_keys, head_flags);
device_inclusive_scan(head_flags, group_numbers);
write_group_begins<<<...>>>(sorted_keys, head_flags, group_numbers, layout);
write_group_ends<<<...>>>(layout);
```

A=0ならhost側でprepareの空経路を選び、group_count=0をstream順に設定する。
以下のprepare kernelはA>0を対象とする。空のatom集合でも後段はY/dXへ必要な0を書く。
その場合はH/G/Parameter勾配のatom用kernelを起動せず、Y/dXの所有blockで0を保存する。
keysなどのscratchは最後の利用が済んだ後に再利用できるが、sorted_rowsと
グループ範囲などbackwardで使う状態はforward呼び出しごとに保持する。

### classify_atom_keys：各atomの所属とキーを作る

各threadが元の一つのatomを読む。ここではprofile値やHを計算しない。

```cpp
a = blockIdx.x * blockDim.x + threadIdx.x;
if (a >= atom_count) return;

level = classify_hierarchy(atoms[a], hierarchy_spec); // 入出力のbank番号の組
tile_shape = hierarchy_tile_shape(level);
q_out = center_tile_coordinates(atoms[a].output_center, tile_shape.output);
q_in  = center_tile_coordinates(atoms[a].input_center,  tile_shape.input);

keys[a] = make_lexicographic_key(level, q_out, q_in);
source_rows[a] = a;
```

キーは`(h_out[0], ..., h_out[Dout-1], h_in[0], ..., h_in[Din-1],
q_out[0], ..., q_out[Dout-1], q_in[0], ..., q_in[Din-1])`の辞書順。
全周bankには下記の専用番号を使う。tile座標はsite中心ではなくセル端を境界にする。
周期軸の中心のwrapとspacing/2の補正は、下記のqの式に従う。
多次元のキーの符号化はこの辞書順を保ち、packed整数のbit数超過を黙って切り捨てない。
同じキーのatomの順序は演算を変えない。元の行番号は対応付け用で、必須の数学的な分類軸ではない。

階層分類に使う幅・周期L・タイル寸法の関係は次節の規約を使う。
ここは従来未確定だった`classify_hierarchy`の設計。
新ディレクトリのprepareとH/C/Yへ接続済み。backwardも同じforwardの分類結果を使う。
中心の所属キーと、支持が重なる全タイルの集合は別である。

### sigmaから階層と周期を決める規約

以下は測定済みの既存profile設定に対する規約である。
次の[one-hot近似の設計方針](#support_widthとone-hot近似の設計方針)を実装済みとは扱わない。

#### 既存profileとsigmaを維持する

この実装は既存の`ChebyshevSquareSpec`を使う。profileの式や係数が未定だったわけではない。
`torch/profiles/fourier.py`の`effective_radius`、`kernel_coefficients`と、固定bank CUDAの
正の係数多項式を基準とする。profile族を選び直したり、sigmaを別の物理量へ変換したりしない。
Kaiser/DPSS等も既存の式はあるが、このCUDA初版の対象はChebyshevSquareだけである。

sigmaは従来通り**幅制御パラメータ**であり、全幅・標準偏差・支持半幅そのものではない。
先の文書／前半prototypeへ追加した`I=sigma/2`は撤回し、分類も修正する。
Iは既存式による集中半径とする。P=2L、s=P/3、epsilon=tail_bound、M=modesとして、

\[
a=\operatorname{acosh}(\epsilon^{-1/2})/M,\quad
r_0=\frac P\pi\arccos(1/\cosh a),
\]
\[
I(\sigma,P)=\left[r_0^4+\left(1-(r_0/s)^4\right)\sigma^4\right]^{1/4},
\quad\beta=\sec(\pi I/P),
\]
\[
g(t;\sigma,P)=\left[\frac{T_M(\beta\cos(\pi t/P))}{T_M(\beta)}\right]^2.
\]

半径下限を含めて既存式をそのまま使う。これは非負な有限Fourier密度そのものであり、
新しいcompact目標profileを作ってfitする処理は今回追加しない。
Iの外の裾と、選んだwhole-tile窓の外をゼロにする支持マスクは区別する。
以前のrho=sigma/(2*d)は幅制御値の比率として保持できるが、I/dと同じ量ではない。
rho<1だけで常に1siteになるとも仮定しない。

#### T_minからbankを選ぶ式

各軸はN sites、間隔d、chart周期Q=N*dを持つ。最小tileのsite数n_minとN/n_minは
2のべき乗とし、T_min=n_min*d、T_h=2^h*T_minとする。
local bankではP_h=2L_h=3T_hを固定する。
既存profileの実現可能性条件r_0<T_hの下では、

\[
I(\sigma,3T_h)\le T_h\quad\Longleftrightarrow\quad\sigma\le T_h.
\]

実際、I^4=r_0^4+(1-(r_0/T_h)^4)*sigma^4なので、I=T_hのときsigma=T_h。
したがって分類のために全atomのprofileを評価する必要はなく、

\[
h_a^{\rm raw}=\min\{h\ge0:\sigma_a\le T_h\}
=\max\left(0,\left\lceil\log_2\frac{\sigma_a}{T_{\min}}\right\rceil\right),
\quad T_a=2^{h_a}T_{\min},\quad L_a=\frac32T_a.
\]

等号は小さい階層側。h=0は0<sigma<=T_min、h>=1はT_h/2<sigma<=T_hを受け持つ。
実装はlog2丸めの代わりにsigma>Tの2倍更新で決める。sigmaをTへ丸めず、
係数生成へ元の値を渡す。T_min=4ならsigma範囲(0,4]/(4,8]/(8,16]は
h=0/1/2、T=4/8/16、L=6/12/24となる。

以前の式はbankの集中半径上限I_h_max=T_hに対して
2L_h=(ceil(2*I_h_max/T_h)+1)*T_h=3T_hとなる。同じbankでPと基底を共有する。
T_minは`sites_per_block`等のlaunch幅とは別。既存native profileはPも数学的な引数なので、
T_minを変えることは同じWの単なるメモリ配置変更ではない。異なる設定を同一条件の速度比較へ混ぜない。

#### 中心tile、全周bank、有効幅範囲

3T_h<=Qなら中心tileと左右1tileを使う。u=(mu-origin) mod Q、N_h=Q/T_hとして、

\[
q_{\rm raw}=\left\lfloor\frac{u+d/2}{T_h}\right\rfloor,\qquad
q=q_{\rm raw}\bmod N_h.
\]

窓のunwrappedなセル範囲は[(q_raw-1)*T_h-d/2,(q_raw+2)*T_h-d/2)。
I<=T_hなので中心がtile端へ寄っても集中領域を覆う。窓内は全セルを同じ基底で積分し、
Iの境界に合わせたcellごとのclipは入れない。

3T_h>Qなら切り詰めず全周bankへ移す。専用level H=log2(N/n_min)、T_full=Q、
K_full=1、P_full=2L_full=Q、q=0とする。全siteを一度だけ扱う。
この場合もprofile内部のsはP_full/3であり、tile幅Qに置き換えない。
既存profileの有効範囲に揃え、0<sigma<=min(Q_in,Q_out)/3を受け付ける。
上端の比較は既存Torch/固定CUDAと同じく入力dtypeへboundをcastしてから行う。
以前の前半prototypeが許したsigma<Qまでの範囲は撤回し、暗黙にclipしない。

```cpp
classify_axis(sigma, center, axis) {
    h = 0;
    T = axis.T_min;
    while (sigma > T) { T *= 2; h += 1; }
    if (3 * T > axis.Q)
        return full_bank(axis.H, T=axis.Q, L=axis.Q/2, q=0);
    u = positive_mod(center - axis.origin, axis.Q);
    q_raw = floor((u + axis.spacing/2) / T);
    return local_bank(h, T, L=1.5*T, q=positive_mod(q_raw, axis.Q/T));
}

// finiteと0<sigma<=min(Qin,Qout)/3を検査してから分類する。
out = classify_axis(atom.sigma, atom.output_center, output_axis);
in  = classify_axis(atom.sigma, atom.input_center, input_axis);
key = (out.h, in.h, out.q, in.q);
```

原点・spacing・T_minが異なれば入出力のhも異なってよい。多次元では各軸へ同じ規則を適用する
設計だが、現CUDAは1入力軸/1出力軸のみ。small sigmaも同じprofileの半径下限を維持する。
直接計算はその同じprofileのセル積分を評価する高速化候補で、別の狭い密度へ差し替える意味ではない。

#### 既存係数のセル積分、面積正規化、H/C

既存密度の複素Fourier係数をc_kとすると、

\[
g(t)=c_0+2\sum_{k=1}^{M}c_k\cos(\omega_k t),\quad
\omega_k=2\pi k/P,\quad S=P c_0,
\]
\[
F(t)=c_0t+2\sum_{k=1}^{M}\frac{c_k}{\omega_k}\sin(\omega_k t),\qquad
\phi_a(i)=\frac{F(b_i-\mu_a)-F(a_i-\mu_a)}{P c_0}.
\]

係数は既に定義済みであり、新しく学習・fitするものではない。通常実行では同じ係数を
正の多項式からregister内で生成し、中心位相とSによる正規化を適用する。
T_M(beta)^-2は全c_kに共通なので、c_k/(P*c_0)から消去できる。
全atomの係数表やL2計算passは作らない。

セル積分のJ_k=2*sin(omega_k*d/2)/omega_kはZと展開基底に含める。
atom側の係数ベクトルはp_0=1/P、p_cos=2*c_k/(P*c_0)*cos(omega_k*mu)、
p_sin=2*c_k/(P*c_0)*sin(omega_k*mu)。そのため、

```cpp
H[b,slot] = sum(retained_tile,mode, input_coeff(atom,mode) * Zx[b,tile,mode]);
C[b,owner,mode] = sum(atom_in_owner, amplitude[atom] * H[b,slot] * output_coeff(atom,mode));
Y[b,site] = sum(bank,covering_owner,mode, C[b,owner,mode] * integrated_basis(bank,mode,site));
```

となる。amplitude=0でもHを省略しない。C同士は異なるPのまま混ぜず、展開したsite値を加算する。
初版のHはglobal FP64[batch,A]でslot順。Cはglobal FP64[batch,output_nodes,1+2M]の小さなowner表。
C blockは入力bankごとにgroup prefixを二分探索し、対象atom区間を取得する。
32atomの係数とamplitude*Hだけをsharedに置き、使用後の同期を挟んで次chunkへ再利用する。
一つのC要素・Y要素に書き手を一つ割り当て、出力atomicやblock間spin待ちは使わない。
Z/H/Cをglobalに渡す初版は検証用の基準。H/C融合や再計算は、同じ式のまま後で比較する。

chart端では窓と中心を同じ座標像へ移す。owner=0で中心が最後の半セルにある場合はmuを-Qずらす。
Zを読むtileの像がx+jQならcos/sin momentをomega*jQだけ回転する。
Cの展開も同じ像で評価する。QがPを割り切るとは仮定せず、同じsiteを二重計上しない。

#### 周期が変わるZの継承と、残る検証

local bankでP_h=2P_(h-1)なので、共通原点・spacingでは

\[
\omega_{h,2j}=\omega_{h-1,j},\qquad
Z_{h,t,b,2j}^{\cos/\sin}
=Z_{h-1,2t,b,j}^{\cos/\sin}+Z_{h-1,2t+1,b,j}^{\cos/\sin}.
\]

親の偶数harmonic（j=1,...,floor(M/2)）と定数modeを継承し、奇数harmonicは直接生成する。
全周bankはP=Qの別基底なので初版は直接生成する。atomがいないbankの子要素も親が使うなら準備する。

- 実装: 既存profileとsigmaの分類、group offsets、bank別Z、H/G、C、Y/dX、Parameter勾配。検証範囲は冒頭の表と記録で追跡する。
- PR #122で接続: 公開CSTLinearの明示Plan。上の接続節で検証状態を追跡する。
- 未実装: 階層版の保存／再計算切替、自動dispatchへの採用。
- 継続検証: branch境界の誤差、完全step時間とメモリ、直接計算/融合等の速度比較。

整数の階層・owner選択は微分せず、選択中のbankのprofileを微分する既存方針を維持する。
同じ式でもPを変えるとprofile値は変わり、階層遷移で関数が連続とは限らない。
forwardのlayoutをbackwardへ引き継ぎ、optimizer後の次forwardで再分類する。
新しいprofile式を待つことをH/C実装の条件にせず、既存密度の独立oracleで各段階を検証する。
完全stepでのY/dX/全atom勾配・更新と、dense以下のallocated/reserved、速度目標を同条件で確認する。

### device_sort_pairs：キーと元の行番号を一緒に並べ替える

```cpp
// GPU sortライブラリへ渡す意味。内部kernelはライブラリに任せる。
(sorted_keys, sorted_rows) = sort_by_key(keys, source_rows);
```

atomのParameter本体やoptimizerのmomentsを並べ替える操作ではない。
多wordキーを使う場合も辞書順を維持できるsort経路を選ぶ。

### mark_group_heads：同じキーの区間の先頭に印を付ける

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

if (s == 0)
    head_flags[s] = 1;
else
    head_flags[s] = (sorted_keys[s] != sorted_keys[s - 1]);
```

その後にinclusive scanを行う。例えば`head_flags=[1,0,1,0]`なら
`group_numbers=[1,1,2,2]`となり、0始まりのgroup番号は`group_numbers[s]-1`。

```cpp
group_numbers = inclusive_sum(head_flags);  // GPU scanライブラリ呼び出し。
```

### write_group_begins：groupごとのbeginを記録する

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

if (head_flags[s]) {
    g = group_numbers[s] - 1;
    layout.group_key[g] = sorted_keys[s];
    layout.group_begin[g] = s;
}
if (s == atom_count - 1)
    layout.group_count = group_numbers[s];
```

beginとgroup_countの書き手はそれぞれ一意。全beginが完成するのをkernel境界で待つ。

### write_group_ends：次のgroupのbeginからendを決める

```cpp
g = blockIdx.x * blockDim.x + threadIdx.x;
if (g >= layout.group_count) return;

if (g + 1 < layout.group_count)
    layout.group_end[g] = layout.group_begin[g + 1];
else
    layout.group_end[g] = atom_count;
```

`[begin,end)`は同じキーのatomの連続区間。全区間を合わせるとソート済みatomを一度ずつ覆う。
gridは最大Aグループ分で起動でき、group_countをCPUへコピーして起動数を決める必要はない。

### blockが使うrangeはどう取得するか

出力タイルや入力タイルに寄与するatomが、一つの連続区間に全て収まるとは限らない。
各blockは必要なgroupの`[begin,end)`を順に読む。この範囲はsorted_rows上の位置であり、
元のatomパラメータがglobalで連続していることは要求しない。
最初の128atomの説明形では、groupを走査するdevice補助関数から始められる。

```cpp
for (g = 0; g < layout.group_count; ++g) {
    if (!group_can_contribute(layout.group_key[g], target_tile, side))
        continue;
    consume_range(layout.group_begin[g], layout.group_end[g]);
}
```

これは追加のkernelではない。`group_can_contribute`は階層ごとの最大支持幅を用いる
保守的な候補判定で、中心が対象タイルの外にあるgroupも必要なら含める。
支持幅の上限を保証できないgroupは、この判定で除外しない。
支持が周期境界をまたいでも、同じgroupの区間を二重に消費しない。
候補に含まれるが実際の寄与が0のatomを含めて処理してよい。範囲を狭めるためのatom別判定は必須にしない。
group検索の索引は後で改善できるが、タイルごとのatomパラメータ全体の複製は前提にしない。

### load_atom_chunk_together：必要なatomだけsharedへ集める

追加のkernelではなく、Y/dXなどのblock内で呼ぶdevice補助関数。
同じatomを複数threadで再利用するときに、そのblockが必要な小さなchunkだけを集める。

```cpp
// rangeとchunkのループはblock内で共通。全threadが参加する。
for (base = range.begin; base < range.end; base += ATOM_CHUNK) {
    count = min(ATOM_CHUNK, range.end - base);
    for (j = threadIdx.x; j < count; j += blockDim.x) {
        original_id = layout.sorted_rows[base + j];
        shared_atoms[j] = atoms[original_id];
    }
    __syncthreads();

    consume_atom_chunk(shared_atoms, count, base, H_or_G);
    __syncthreads();  // 全員が使い終わってから次のchunkで上書き。
}
```

shared_atoms[j]はソート順に並び、対応するH/Gの行は`H_or_G[b,base+j]`。
実装では必要なparameter成分だけをロードしてよい。shared全量はATOM_CHUNKと成分数で制限する。
Cの担当がないthreadもchunkロードと全barrierへ参加する。
Cを保持するshared領域とは、同時に必要な間は別の領域を使い、容量を両方数える。
globalからの最初のgatherは飛び飛びになり得る。sharedでの再利用とロード・同期の費用を測る。
block間でsharedを共有することはできず、別blockはそれぞれ必要なchunkをロードする。

一つのthreadが一つのatomだけを使うH/G・Parameter勾配では、sorted_rowsを介して元のatomを
直接registerへ読む。共有する相手がいなければsharedを経由する必要はない。
いずれの場合も、Fourier係数やその微分はsigma・中心・bankから利用時に計算する。
全atom分の係数表や係数微分表をglobalへ用意しない。

## build_input_hierarchy：XからZxを作る起動関数

以前の`build_z_level`には、signalを読む処理と子Zを読む処理が一つに入っていた。
ここを分ける。二分マージを出発点にする。
nodesはタイルのsite範囲・bank・mode・子ノードを記述する実行設定で、atom値を含むZとは別。
最初は宣言済みのbankとlevelの全ノードを作る単純な構成とし、必要ノードだけへの限定は後で考える。

### 並列な部分と、依存で順序が必要な部分

| 処理 | 並列にできる範囲 | 待つ相手 |
| --- | --- | --- |
| signalからZを直接生成 | 異なるlevel・タイル・batch・mode | 入力XまたはdYが利用可能になることだけ |
| 一つのZへのsiteの加算 | siteを複数threadに分担し、部分和を作る | 部分和を合わせるときの担当thread |
| 子Zから親Zを継承 | 同じlevelの異なる親・batch・mode | その要素で使う子Zの完成 |

**親が子を使う依存は残るが、全levelの全処理が直列という意味ではない。**
直接生成するmodeは、子Zがなくても作れる。最初に全level分をまとめて起動できる。
マージのlevel単位の完了待ちは、必要な子ごとの待機より粗いが、安全で単純な基準になる。

基底設定から、各`(node,mode)`を直接生成か継承のどちらかへ一度だけ割り当てる。
`direct_jobs`は直接生成するmodeをnodeごとに束ねた記述、`merge_jobs[level]`は
継承する要素と子modeの対応を記述する。GPUで空きthreadを監視するjob管理ではなく、固定の担当表である。
同じZ要素を両方のkernelから書かない。これはatomの支持分類とは別の、基底間の対応である。

```cpp
// 並列部分：直接生成する要素は全level分をまとめて処理。
if (has_input_direct_jobs)
    build_z_direct<<<input_direct_grid, threads>>>(X, input_direct_jobs, Zx);

// 依存部分：子の完成後に次のlevelを起動。各起動の内部は並列。
for (level = 1; level < input_level_count; ++level) {
    if (has_input_merge_jobs(level))
        merge_z_level<<<input_merge_grid(level), threads>>>(
            input_merge_jobs[level], Zx);
}
```

forはCPU側から起動順序を記述するもの。タイルをCPUの一つのthreadで計算しているわけではない。
各起動を同じstreamへ投入し、CPU側で毎levelのGPU完了を待つ呼び出しは入れない。
例えば128サイト、最小タイル幅16なら、二分マージのノード数は8→4→2→1。
マージでは4親を並列→2親を並列→1親へ進む。どのmodeを直接生成するかは基底の継承条件による。
これは構造の例で、最小幅16や各levelの周波数・mode数を固定する決定ではない。

## build_output_hierarchy：dYからZyを作る起動関数

同じ処理にdYと出力側のタイル・基底設定を渡す。
入力側と出力側で次元、spacing、タイル寸法、bankが異なっていてよい。

```cpp
if (has_output_direct_jobs)
    build_z_direct<<<output_direct_grid, threads>>>(dY, output_direct_jobs, Zy);

for (level = 1; level < output_level_count; ++level) {
    if (has_output_merge_jobs(level))
        merge_z_level<<<output_merge_grid(level), threads>>>(
            output_merge_jobs[level], Zy);
}
```

Zyはbackwardで新しく作る。Zxを入力にしてZyを作るという意味ではない。

## build_z_direct：site方向も複数threadで並列に足す

以前のコードも、タイル・batch・modeは並列だった。ただし、各Z要素のsite方向の加算は
一つのthreadに任せていた。ここでは一つのZをP個のthreadで分担する案を示す。
これはatomを複数threadで処理する話ではなく、複数atomが再利用するZの生成である。

一つのblockは一つの直接生成nodeとbatch範囲を担当する。
Pは調整可能な2のべき乗で、blockDim.xを割り切る値とする。warp単位に固定しない。
例として128thread、P=4なら、32個のZ要素を同時に処理できる。

| thread | 担当 | 分担するsiteの番号 |
| --- | --- | --- |
| 0〜3 | 一つ目の`(batch,mode)` | 0,4,8,… / 1,5,9,… / 2,6,10,… / 3,7,11,… |
| 4〜7 | 二つ目の`(batch,mode)` | 同じ分割を次のZ要素に適用 |
| … | … | … |
| 124〜127 | 32個目の`(batch,mode)` | 同じ分割を次のZ要素に適用 |

batch 4・直接生成mode 8なら32個のZなので、この例では全128threadに担当がある。
要素が32個より多い場合は、次の波でまた32個を並列に処理する。
下のteamはP個のthreadをまとめるソフトウェア上の番号であり、新しいCUDAの実行単位ではない。

```cpp
job = direct_jobs[blockIdx.x];  // 異なるlevelのnodeも同じgridに含む。
team = threadIdx.x / P;
lane = threadIdx.x % P;
team_count = blockDim.x / P;
work_count = batch_size * job.direct_mode_count;
__shared__ Scalar partials[THREADS];

// 全threadが同じ回数繰り返す。各wave内の全teamは並列。
for (wave = 0; wave < ceil_div(work_count, team_count); ++wave) {
    q = wave * team_count + team;
    (b, mode) = decode_direct_job(q, job, blockIdx.y);
    valid = (q < work_count && b < total_batch_size);

    partial = 0;  // 各threadのregister。
    if (valid) {
        for (i = lane; i < job.site_count; i += P) {
            site = job.site_at(i);
            partial += signal[b, site] * integrated_basis(job.node, mode, site);
        }
    }
    partials[threadIdx.x] = partial;
    __syncthreads();

    // teamごとの部分和を並列reduction。P=4なら2段。
    for (offset = P / 2; offset > 0; offset /= 2) {
        if (lane < offset)
            partials[threadIdx.x] += partials[threadIdx.x + offset];
        __syncthreads();
    }
    if (valid && lane == 0)
        Z[job.node, b, mode] = partials[threadIdx.x];
    __syncthreads();  // 次のwaveでsharedを再利用する前の同期。
}
```

`decode_direct_job`は範囲外qでも不正な配列を読まず、valid=falseとなる番兵を返す。
担当のないthreadもpartial=0を書き、全barrierへ参加する。途中return/continueはしない。
siteのループはthreadごとにsite数/P程度の加算をするものであり、全siteを一人で走査しない。
足し算の依存は各thread内の部分和と、log2(P)段のreductionに残る。

`signal`は入力側ならX、出力側ならdY。signalはglobal、部分和はregister/shared、完成Zはglobal。
shared容量はこの例でTHREADS個のScalar分。128thread・FP32なら512 byteであり、
タイル全体や全levelのZをsharedに置くという意味ではない。
同じsignalをmode間で共有ロードする追加案もあるが、この擬似コードではまだ導入しない。
P=1なら以前と同様に一人で一つのZを作る割り当てになる。短いタイルやmode数が多い場合は
Pを増やさない方がよい可能性もあり、reductionと同期の費用を含めて比較する。

## merge_z_level：子Zの完成後、親の各要素を並列に作る

ここではsiteを走査しない。二つの子の完成値を足すだけなので、一つの親Z要素を
一つのthreadへ割り当てる。他の親・batch・modeは別threadで並列に進む。

```cpp
q = blockIdx.x * blockDim.x + threadIdx.x;
if (q >= merge_jobs.count * total_batch_size) return;
(job, b) = decode_merge_job(q, merge_jobs);

left = Z[job.left_child, b, job.left_mode];
right = job.has_right_child ? Z[job.right_child, b, job.right_mode] : 0;
Z[job.parent, b, job.parent_mode] = left + right;
```

子Zは先に起動したkernelで完成済みなので、ここにblock内barrierは不要。
親要素の書き手も一つ。次のlevelはこのkernelの完了後に読む。
rootが一つになってもbatch/modeは並列にできるが、上位ほど仕事が少なくなる点は残る。
この縮小した並列度や起動費用は、後でlevel融合や複数nodeの同一block処理と比較する。

## セル積分基底と継承条件

`integrated_basis`はsiteの担当セル全体を積分した基底値。例えば共通原点のcos基底なら、

```cpp
if (omega == 0) return spacing;
return 2 * sin(omega * spacing / 2) / omega
       * cos(omega * (site_position - common_origin));
```

sin基底も同じ積分係数にsinを掛ける。多次元の積基底なら軸ごとのセル積分を掛ける。
ここにはatomの中心・幅・振幅を混ぜない。選択する領域はwhole tileのみなので、セル境界の補正は不要。

`can_inherit_same_basis`は、子タイルのsite集合が親を重複なく覆い、該当modeの周波数、
sin/cosの種類、座標原点、spacing、セル積分係数が一致するときだけtrueにする。
葉ノードや一方の子で該当modeがない場合はfalse。原点が異なる場合の位相変換は、この単純な継承には含めない。
親を二つの子から作ることだけでは、基底まで一致するとは限らない。
一致しないmodeはdirect_jobsへ、一致するmodeだけmerge_jobsへ入れる。
全levelの直接生成を先に行っても、継承できる要素までsignalから作り直すわけではない。
必要なZの総要素数も変えない。signalの再読込は残り、処理順によるcacheの効き方や
同時に扱う作業範囲は変わり得る。速さは未測定であり、元のlevel単位の混合kernelも比較候補として残す。

## 1. compute_h：入力側の全寄与をatomごとにまとめる

blockはatom群とbatch範囲を担当する。各threadが別々のatomを担当し、
そのatomに必要な全入力タイルを足す。最初の例では128thread、各threadに1atom、batch_size=4。
Hは振幅を掛ける前の値。Zxはglobalから読み、Hの累積値はregisterへ置く。

```cpp
slot = assigned_atom(blockIdx.x, threadIdx.x);
if (slot >= atom_count) return;
original_id = layout.sorted_rows[slot];
atom = atoms[original_id];  // 必要なparameterをregisterへ読む。
h[batch_size] = {0};

for (tile : full_input_tiles(atom)) {
    for (mode : tile.modes) {
        c = input_coefficient(atom, tile, mode);
        for (b : assigned_batches(blockIdx.y, batch_size))
            h[local_batch(b)] += c * Zx[b, tile, mode];
    }
}
for (b : assigned_batches(blockIdx.y, batch_size)) {
    H[b, slot] = h[local_batch(b)];
}
```

`full_input_tiles`は選択した1周期分の窓を重複なく覆うwhole tile集合。親とその子を二重に含めない。
modeのループは`sum_k input_coefficient(atom,tile,k) * Zx[b,tile,k]`を計算する。
係数には中心・幅と面積による正規化を含め、必要なmodeごとにregisterで評価し、batch間で再利用する。
窓の外はタイル単位でゼロ扱いする。窓内は共通の積分基底を使い、端セルの直接積分は行わない。

この案では異なるthreadのHを足す必要はない。thread自身が必要な入力寄与を全て足す。
sharedへの共同ロードやblock内同期はまだ入れず、完成したHを一度globalへ書く。
後に共同ロードを導入する場合は、担当atomがないthreadも必要な同期へ参加させる。

## 2. compute_y：担当出力タイルのCを作り、Yまで完成させる

blockは出力タイルとbatch範囲を所有する。Yを0から累積し、最後に一度globalへ書く。
一つのY要素に対する全グループの寄与をこのblockが集める。

```cpp
output_tile = assigned_output_tile(blockIdx.x);
init_output_accumulator(y);  // 各threadの担当site/batchは重ならない。

for (bank : contributing_output_banks(output_tile)) {
    build_output_c_together(bank, output_tile, H, atoms, layout, shared_c);
    __syncthreads();
    expand_output_c_into_y(bank, output_tile, shared_c, y);
    __syncthreads();  // 全員が読み終えてからshared_cを再利用する。
}
store_output_tile(output_tile, y, Y);
```

内部部分でのCは`C[b,k] = sum_a alpha[a] * H[b,a] * output_coefficient(a,tile,k)`。
`build_output_c_together`はCの各batch/modeに書き手を一つ決め、対応atomを走査して完成値を
sharedへ書く方式を出発点にする。その走査中、必要なatomを上記のchunkロードでsharedへ集める。
係数を担当するthreadはchunk内のatomから係数をその場で算出する。
atomごとの部分Cを並列生成してreductionする案は後で比較する。
係数を作るthread配置と、Yのsiteを展開するthread配置は異なってよい。

bankごとに基底が異なるため、C同士を混ぜず、展開したsite値をyへ足す。
窓内の全寄与をCで表すので、境界セルの追加処理はない。
bankのループと二つのbarrierには全threadが参加し、担当要素がないthreadも途中returnしない。
globalのHは読み取り専用、Cはshared、yは各threadのregisterを基本とする。
各threadが持つyの個数は担当タイル寸法とbatch範囲で決まり、容量に応じて変更する。

## 3. compute_g：dY側でHと対称の集約を行う

atom群とbatch範囲を担当し、各threadが別々のatomを処理する。
Zyを読み、振幅を掛ける前のGを完成させてglobalへ書く。

```cpp
slot = assigned_atom(blockIdx.x, threadIdx.x);
if (slot >= atom_count) return;
original_id = layout.sorted_rows[slot];
atom = atoms[original_id];
g[batch_size] = {0};

for (tile : full_output_tiles(atom)) {
    for (mode : tile.modes) {
        c = output_coefficient(atom, tile, mode);
        for (b : assigned_batches(blockIdx.y, batch_size))
            g[local_batch(b)] += c * Zy[b, tile, mode];
    }
}
for (b : assigned_batches(blockIdx.y, batch_size)) {
    G[b, slot] = g[local_batch(b)];
}
```

`compute_h`と入出力の役割が逆になる。選ぶタイル窓と係数の定義は、
forwardで使う同じprofileに対応させる。G全体のほかにdH全体を保存することは前提にしない。

## 4. compute_dx：担当入力タイルの係数を作り、dXまで完成させる

blockは入力タイルとbatch範囲を所有する。`compute_y`の入力側版として構成する。

```cpp
input_tile = assigned_input_tile(blockIdx.x);
init_input_gradient_accumulator(dx);

for (bank : contributing_input_banks(input_tile)) {
    build_input_c_together(bank, input_tile, G, atoms, layout, shared_c);
    __syncthreads();
    expand_input_c_into_dx(bank, input_tile, shared_c, dx);
    __syncthreads();
}
store_input_tile(input_tile, dx, dX);
```

内部部分の係数は`C_in[b,k] = sum_a alpha[a] * G[b,a] * input_coefficient(a,tile,k)`。
振幅は一度だけ掛ける。bankごとに展開して担当dXを完成させる。境界セル補正は不要。
Gはglobal、集約したC_inはshared、dxはregisterを基本とする。barrierの規則は`compute_y`と同じ。
atomのロードもYと同様に、元配列から必要なchunkだけをsharedへ集める。
Y/dXの各要素を一つのblockが所有するため、block間の出力加算atomicは不要にできる。

## 5. compute_parameter_grad：H/Gと微分を組み合わせる

各threadが一つのatomの全batchを担当し、そのatomのParameter勾配を完成させる基準案。
batch分割で複数blockが同じ勾配を書く方式は、追加reductionを伴う別候補にする。

```cpp
slot = assigned_atom(blockIdx.x, threadIdx.x);
if (slot >= atom_count) return;
original_id = layout.sorted_rows[slot];
atom = atoms[original_id];

grad_amp = 0;
for (b : all_batches) {
    grad_amp += H[b, slot] * G[b, slot];
}
store_original_amp_grad(original_id, grad_amp, grad);

for (parameter : profile_parameters(atom)) {
    grad_parameter = 0;
    for (b : all_batches) {
        dh = input_parameter_derivative(atom, parameter, b, Zx);
        dg = output_parameter_derivative(atom, parameter, b, Zy);
        grad_parameter += atom.amplitude *
            (G[b, slot] * dh + H[b, slot] * dg);
    }
    store_original_profile_grad(original_id, parameter, grad_parameter, grad);
}
```

ここで`dh/dg`はそれぞれ`partial H / partial parameter`と`partial G / partial parameter`。
逆伝播の中間勾配`dH = alpha * G`とは別である。
入力側だけに効くparameterならdg=0、出力側だけならdh=0、両側に効くなら両方を含める。

微分関数は窓内の全セルについて`sum partial_coefficient * Z`を計算する。
支持境界をさらう補正やraw X/dYの追加読込みはない。
面積正規化の微分も含める。H/Gだけからこの情報を復元できるとはしない。
Zx/Zyは読み取り専用のglobal、gradの累積値はregisterへ置く。
元のParameter行への対応はlayoutで戻し、optimizerによる更新はここでは行わない。

このコードはparameterごとにZを読む単純な説明形である。同じZを読んだときに複数parameterの
微分を同時に累積すれば読み取りを共有できるが、register使用量との交換になる。
physical sigmaの解析微分は実装済み。bank/ownerの離散的な選択は固定した枝として微分し、
次のforwardで再分類する。階層遷移が滑らかだと保証するものではない。

## 確認すること

各段階の保存値、配列のshape、書き手、寿命を先に確定し、独立oracleで
Y/dX/全atom勾配を検証してから速度を比較する。上記は実機での正しさ・性能の結果ではない。
H/Gの保存を省く案では、同じforward状態での再計算と、全利用先の終了を明示する。
