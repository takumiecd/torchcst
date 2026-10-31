# 区間確率方式：prepareからbackwardまでのkernelの流れ

## 設計の正本と、現在の実装範囲

**実装の区切り（2026-10-28）：** 既存σ版と検証済み最適化はPR #122から
`main`へmerge済み（`0047e29c`）。次期半径APIと以下の設計は、そこから切った
`kernel/support-radius-pipeline`で進める。
**現在（2026-10-31）：** 半径APIのCPU参照、N<=128/A<=4096/48KiBの小型native・block-local経路、
N<=1024/A<=65536の明示global経路（専用native host版を含む）、階層momentsと完成responseの再利用、
shared C/site展開とbounded owner partials、既存則を保つFP32 Polar更新を実装した。自動dispatchは変更していない。

| 検証条件 | 採用・検証状況 | 完全stepとメモリの結論 |
| --- | --- | --- |
| N128/B4/A819/M8単体 | native cached Planを採用 | 2fresh runでdense比約1.94倍、allocated/reservedともdense未満 |
| N128/B32/A819/M8単体 | 明示batched/grad4 Planを採用 | 2fresh runで約1.41〜1.43ms、dense比2倍以内、両ピークもdense未満 |
| N1024/B4/A52428/M8単体 | 明示owner response/C融合Planを追加 | 約3.13ms、両ピークはdense未満、速度2倍以内は未達 |
| MNIST N128/B4・3epoch | 同じ初期W/外側層/data orderで実学習 | 速度は約1.60〜1.76倍、allocatedは下回るがreserved同値 |
| MNIST N128/B32・1epoch | 同条件の別batch検証 | 速度約2.07〜2.12倍、allocated超過、同時目標は未達 |
| MNIST N128/B32・local tile候補3epoch | 別branch、初回コストも含める | 再compileを含む別runで全体平均1.74倍、allocatedは下回るがreserved同値、両ピーク厳密未満は未達 |
| MNIST N1024/B4・3epoch | 同じ初期化を使う明示owner融合Planで再測定 | 約2.89〜2.97ms対dense1.30〜1.32ms、両ピークは下回る、速度2.19〜2.29倍で未達 |
| active moments省略 | grad4との統合GPU884件通過、明示Planに採用 | 未使用bank計算を省く。配列shapeとメモリ量は同じ |

MNIST N1024の最終test accuracyは97.37%対97.87%（1seed/3epoch）。精度同等・一般化は主張しない。
単体denseの小型ピーク差にはcuBLAS workspace約17.04MBが大きく寄与する。MNISTでは両モデルに共通であり、
単体の総ピーク差をCSTのParameterや中間配列そのものの削減量と読み替えない。末尾に因果実験を記録する。
現在の統合CPU検査は3198 passed/6771 skipped。追加ごとの検証範囲は末尾に記録する。
CPU参照修正時の配布wheel/sdistはruntime388ファイルがbyte一致。owner融合追加後の再buildは別途記録する。
owner融合統合時のruntime/benchmarkは実GPU qualificationの5a snapshotとbyte一致した。
Graph、autocast、高階微分、自動dispatch、N1024より大きい形状の達成は未検証である。
末尾の各実測節にjob ID・source/archive hash・採用範囲と失敗結果を残す。
以下の初版・候補・時系列の記述は、それぞれの段階の制約として読む。

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

#### 対称な融合computeと、保存値の契約（2026-10-28の先行版）

2026-10-29の共有投影版では、以下の入出力・保存契約を維持し、1方向のcomputeを
「moments/responseを一度作るkernel」と「C/site展開kernel」に分ける。起動数と結果は後述する。

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

初回job `l4job-f9fdc9945af34aa3b2549105eb2b3d45`では209テストと6回のrunner測定が完了したが、
最後の広いmemcheck subprocessが240秒の制限を超え、job全体は失敗した。
性能JSON・source・binary・失敗logは保全し、このjobだけで採用確定とはしない。
sanitizerは数値参照の重い処理と分けたnative経路の検査へ切り分け、後続jobで完了させる。

次の範囲取得候補では、ownerのhistogram/cumsumを作る代わりに、ソート済みcomposite keyの
固定境界を`searchsorted(..., right=False)`で直接引く。
forwardのkey `o*N_in+i`は`[o*N_in,(o+1)*N_in)`に入り、reverseは役割を入れ替える。
このlower boundがownerのbegin/endになり、空ownerは同じoffsetを繰り返す。
stable sortの元atom行順を維持し、0atom・重複key・異なる入出力node数もCPU6ケースで照合した。
GPU最大値の読戻しを伴う可能性があるbincountを避ける候補であり、同期削減は別のtraceで確認する。
sorted keysと境界の一時領域も完全stepのpeakへ含める。
CPU全体は2367 passed／5768 skipped／既存警告18件。

**完了した再確認：** job `l4job-02934bee8794488288d10c6753195594`で215テスト成功。
native経路だけに分けた7ケース（M3 chunk tail、M8/A819、M16の33modeとshared fallback、
FP32/64・P>Qを含む）はmemcheck 0 errors、racecheck 0 hazardsだった。
これは前jobの時間切れを成功と読み替えたものではなく、同じbinaryをsource照合して別jobで確認した結果である。

同一L4、旧版→shared＋位相＋key保存→固定境界検索、逆順の計6 fresh processで測定した。
条件は前節のN128/B4/A819/M8/seed41/eager AdamW、warmup3・21サンプルのまま。

| 経路 | fresh process 1 ms | fresh process 2 ms | peak allocated / reserved bytes |
| --- | ---: | ---: | ---: |
| 前段の多項式版 | 10.244239 | 10.225017 | 291328 / 2097152 |
| shared C・位相・分類key保存 | 5.231266 | 5.218902 | 297984 / 2097152 |
| 上記＋固定境界検索 | 4.456594 | 4.463809 | 297984 / 2097152 |

denseは同じ6実行で0.695353〜0.710390ms、allocated/reservedは17313280/23068672 bytes。
固定境界検索までを次の検証済み候補とする。約2.29倍の改善だが、denseの約6.4倍の時間であり目標未達。
allocatedの6656 bytes増加は保存keyのallocator丸めと対応し、予算値6552 bytesと区別する。
shared Cの追加17952 bytes/CTA（この条件では4 warp分）はSM内の資源で、global表の追加ではない。
全GPU processメモリとGraphは未測定。

別のinstrumented traceではbincountのpageable DtoH 2回が消え、`aten::item`は3回から1回になった。
残る1回はforward分類の異常値確認で、同期ゼロを意味しない。
transform kernel 2回の合計は約2.567ms、atom勾配は約0.244msだった。
これらのdevice時間や計装phase時間を完全stepへ足したり差し引いたりしない。
Python profileでは3 stepで`AxisPlan.banks`が75回呼ばれ、BankPlanを1200回構築していた。
この不変な階層metadataの再構築も次の削減対象とする。

committed source=`f5ba4332`、snapshot SHA256=
`10caee22bb99f238ecbd8dd1e0cfdf430e08ac47a3daf198f92d316ca1162a70`、
result archive SHA256=`c32ca516525abf15945ea186a585ee39cbf8b7e552d07d4c47bf22eb67bd9f8b`。
native module=`torchcst_radius_cell_mass_dd670b6e96a5`、binary SHA256=
`4aec159dfc2582f9c0109790ec54dfe6a7c40b7fef15e2067ef2ee93335fb14e`。
前jobの未採用な切り分け結果も含め、raw結果・source照合・trace・Python profile・sanitizer・receiptを
同じignored `output/radius-optimize-20261028/`へ保持する。

#### 不変bankのcacheと、moment内のサイト位相

次の候補は、frozenなAxisPlanのbank tupleを初期検証時に一度だけ作り、以後再利用する。
保持するのは変更不能な整数／floatのmetadataだけで、atom値・分類・Tensorはcacheしない。
`dataclasses.replace`で設定を変えれば新しいbankを作り、`native_config()`は引き続き独立したlistを返す。

GPU側では旧`cell_mass_hierarchical/preparation.cu`と同様に、momentの入力site走査の位相を回転で進める。
半径版はtile相対座標なので初期位相を正確に(cos,sin)=(1,0)とし、DCは三角関数なし、1siteも専用処理とする。
セル積分filter、入力の加算順、子から継承するmerge枝は変更しない。
host実runtime抽出の1179648条件では最大絶対差3.84e-13、重み付きL1比3.07e-15、merge枝の差は0。
bank cacheのみと、site位相も変えた版を別々に完全stepで比較してから採否を決める。

**後続の検証・採用結果：** job `l4job-bfe086a49b634c59bc971da46fddfd11`で217テスト成功
（CPU/宣言45、CUDA実行172）。上と同じnative経路の7ケースでmemcheck 0 errors、racecheck 0 hazards。
CPU全体は2369 passed／5768 skipped／既存警告18件。

同一L4、固定境界検索版→bank cache→moment回転、逆順の6 fresh processで測定した。
前節と同じN128/B4/A819/M8/seed41/eager AdamW、warmup3・21サンプル。
Parameter・X・target・初期半径のfingerprintは6実行で一致し、更新後のlevelも全atomが2だった。

| 経路 | fresh process 1 ms | fresh process 2 ms | peak allocated / reserved bytes |
| --- | ---: | ---: | ---: |
| 前段の固定境界検索版 | 4.509981 | 4.551533 | 297984 / 2097152 |
| bank cache | 3.884308 | 3.895434 | 297984 / 2097152 |
| bank cache＋moment回転 | 3.606726 | 3.583001 | 297984 / 2097152 |

denseは同じ6実行で0.704189〜0.769767ms、allocated/reservedは17313280/23068672 bytes。
bank cacheとmoment回転を採用する。最初の多項式版約10.24msからは約2.85倍速いが、
denseの約5倍の時間を要し、完全stepの速度目標は未達。N1024やMNISTでの結果ではない。
allocated/reservedは不変で、cached bankはCPUの不変metadataだけを保持する。
全GPU process量とGraphは未測定であり、denseのlibrary workspaceを除いた比較へ読み替えない。

別のdevice traceではtransform 2回の合計が約2.567→1.588ms、atom勾配は約0.244msのまま。
block kernelのregister数は151→152、stackは1376 bytesのまま。atom勾配は110 register／1112 bytes。
別のPython profileで3stepあたりのBankPlan構築は1200→96回、bank生成は75→6回へ減った。
新しい宣言を検証する分は残し、同じAxisPlanの読み取りでは作り直さない。
計装phaseはforward/loss約1.71〜1.76ms、backward約1.58〜1.59ms、optimizer約1.85〜1.89ms。
これらを未計装の完全stepへ足したり差し引いたりせず、残るCPU処理・GPU処理の診断として扱う。

残る候補は、旧版のbatch間の係数共有とactive-bank masksである。
現在のBLOCK_LOCALは各出力CTAがmoments/Hを再生成し、使われないbankのmomentsも構築する。
これらは未移植。maskを使う場合は返すmoment cacheの有効範囲とconsumer側の契約も明示する。
旧版の補償FP32演算も自動的には持ち込まず、精度と全勾配を別候補として検証する。
今回の変更はFP64のmoments/H/G保存、半径の定義、Polarの更新則を維持する。

committed source=`d36032fc`、snapshot SHA256=
`b092d3b553813506efa850b754cf862c01e889f062a39e61a0e6381aa970f8ab`、
result archive SHA256=`a209f4cc38fb7cc8fd9122d5a37b76a617c57ce7857f4e45db55b0c57c580dd9`。
native module=`torchcst_radius_cell_mass_cf34d733bd80`、binary SHA256=
`1dca007d8fcd382f408debe396b08189da668d157462f7794a86d7347f467896`。
raw結果、source/binary照合、trace、profile、sanitizer、host oracleとreceiptを同じignored保全先に保持する。

#### 深掘りprofile：host区間・融合kernel内部・hardware counterを分ける

条件は引き続きN128/B4/A819/M8・公開初期化・明示BLOCK_LOCAL・eagerの完全step。
最適化の追加前に、以下の3種類を別に測った。本番sourceは変更していない。
速度の正本は前節の未計装3.606726/3.583001msである。

**1. 軽いhost区間計測。** 各phaseの前後のhost時計だけを読み、途中にCUDA同期を入れず、
stepの最後だけ同期する。warmup5＋21step。これはCPU計算・enqueue・CUDA API内の待ちを含み、
GPU kernel時間ではない。各step内の区間は時系列を分割するが、中央値同士を足し合わせない。
CPUはCUDAへ処理をenqueueすると、GPUの完了前に関数から戻れる。
従ってbackwardのhost呼出し0.610msと、別traceでのGPU処理1.109msは矛盾しない。
「host呼出し」は関数の入口からCPUへ戻るまでであり、CUDA完了までの時間を意味しない。
CUDA API内で待つ場合も含むため、host区間のすべてをPython計算時間と断定もしない。
この測定でもoptimizerは明示した`RADIUS_FLAT_TORUS_FUSED`を通り、前に最適化した
`apply_polar`を実行している。約1.775msのwrapper呼出しと約2.784µsのPolar kernelを区別する。

| host区間 | CST中央値 ms | dense中央値 ms |
| --- | ---: | ---: |
| zero_grad | 0.053194 | 0.043297 |
| forward/loss呼出し | 1.124650 | 0.144128 |
| backward呼出し | 0.609777 | 0.327906 |
| optimizer呼出し | 1.775268 | 0.166492 |
| 最後のGPU完了待ち | 0.030354 | 0.023493 |
| 診断step全体 | 3.614349 | 0.717620 |

hostの最大区間は`CSTOptimizer._step_impl`の呼出しである。
group/Parameter確認、gradient projection、old pointの準備、更新context/Planの構築とdispatch、
AdamW呼出し、state transportを含む。Polarの更新式そのもののdevice時間とは分ける。
前節のcProfileでは全体で1stepあたり13回の`fixed_state_signature`巡回があり、
`operators/metadata.py`のmodule/spec/buffer/version確認が繰り返されている。
これはCPU整理の候補を特定するcall-count evidenceであり、cProfileの重複する累積時間を足して損失時間としない。

**2. 実際のGPU kernelだけのtrace集計。** user annotationのdevice帰属を重複計上せず、
`cat=kernel`の29 launchをCPU launch correlationで対応付けた。

| 実GPU処理 | kernel数 | device時間 µs |
| --- | ---: | ---: |
| forward/loss範囲 | 10 | 830.595 |
| backward範囲 | 14 | 1109.476 |
| optimizer範囲 | 5 | 13.825 |
| kernel合計 | 29 | 1953.896 |

個別には融合forward768.897µs、融合reverse820.514µs、atom勾配244.481µs、
分類23.936µs、Polar更新`apply_polar`2.784µsだった。GPU copyはこの表へ含めない。
GPU処理はhost区間と重なるので、この表とhost表を足して完全step時間を作らない。
以前のprofilerでは2.087msの大きな空白が2.083msのActivity Buffer Requestと重なっていた。
profiler自身のbuffer確保を、実プログラムのCPU待ちや遅い除算と解釈しない。

**3. 融合kernel内部のCTA別clock64。** checkoutと別のsource copyだけに6つの境界を挿入し、
moments、globalへのmoments保存、H/G、C、site展開を区切った。公開return値は維持する。
5step×16CTA＝各方向80サンプル。clockは同じCTA内の差分だけを使い、異なるSMの絶対clockを比較しない。
各境界に追加barrierとstoreがある診断版であり、cyclesをmsへ換算したり本番速度へ使ったりしない。
表は各方向の全CTA・反復で合計したstage cycleを、全stageの合計cycleで割った比率である。

| CTA内cycle比率 | forward | reverse |
| --- | ---: | ---: |
| moments構築 | 30.00% | 28.62% |
| momentsをglobalへ保存する区間 | 0.02% | 0.02% |
| 完成H/Gの生成 | 32.80% | 31.31% |
| Cの生成 | 27.73% | 31.02% |
| site展開・保存 | 9.46% | 9.03% |

GPU内部はmoments/H/G/Cの3つがそれぞれ重い。moments保存の区間は小さい。
H/Gのglobal書込みはresponse区間内に含まれるため、保存全般のコストを分離したとは言わない。
診断版も全819atomのY/dX/dPを公開FP64参照へ照合してPASSした。

profile job=`l4job-e03194dcc9274260b3f89e95adf933c8`、production source=`cb25bc3c`、
snapshot SHA256=`2fb53666c10dc048a38dbf80e5eca5c729c4bc9a4ca11dcb00ef639ce4cde3e5`、
result archive SHA256=`7b9c99378966c4e903bcb716b9df33f6f622964c2c4379a1e6815a9b7068bed2`。
通常版binaryは前節の`1dca007d…`をsource/runtime照合して使用した。
診断sourceの生成器SHA256=`32c3f0fbe4af0cb3ebf9a377d4b9c49d0f337a5428f0117e9bdcd3118504a775`。
生成後native source digestは`c65421bda029e20e77026847d006b1e39c537b0f6bb38a461b9864a61108497c`。
生成器・変更前後source hash・raw clock・trace・host sample・導出解析を同じignored保全先に保持する。

**Nsight Computeによる確認。** 本番binaryの融合kernelを1 launchに限定し、16 replay passで調べた。
cache/clock制御はnoneで、以下は1kernelの診断値であり、別の完全step性能値とはしない。

| hardware診断 | 値 |
| --- | ---: |
| SM数 / launch block数 | 58 / 16 |
| threads/block / registers/thread | 128 / 152 |
| dynamic shared/block | 41640 bytes |
| shared使用量によるresident block上限 | 2 blocks/SM |
| waves/SM | 0.14 |
| achieved occupancy（active-cycle基準） | 7.744% |
| FP64 pipe activity（active / elapsed） | 74.498% / 20.476% |
| DRAM throughput（peak比） | 0.000885% |
| eligible warps/scheduler/cycle | 0.034646 |

この小さいB4条件ではblock数が少なく、動いているSM内でFP64計算と依存待ちが目立つ。
DRAM帯域が飽和しているという観測ではない。Short Scoreboardも大きいが、
具体的なshared命令やmath命令を原因と断定するsource/SASSの対応付けは未実施。
cache/clockが未制御であること、補助Pythonにsitecustomizeのtorch欠落警告があったことをraw logに残す。
対象processの16pass、ncu-rep、CSV、正常終了を確認した。補助Pythonの警告は未解消である。

NCU job=`l4job-95b9d249ebe04db9aba72b1ba15d3f30`、snapshot SHA256=
`fa41efab5ed4bf5cd166fedb9fcd87f3f8cc3c23bb8b0648e18134b915af1c57`、
result archive SHA256=`64b4ef61c2038349303d1a79d63adab0f49d10002f65e7b8616071fe2d7a6abf`。
production binaryは同じ`1dca007d…`。report/CSV・command・警告・導出解析を保持する。

次の優先候補はCPU側の重複宣言検証・dispatch整理、GPU側の未使用bank省略と仕事の分割である。
またH/Cで値だけが必要な呼出しにも`profile_coefficients_m8`が微分係数を返しているため、
値だけの評価経路が候補になる。ただし現binaryでのdead-code eliminationやその独立コストは未測定である。
今回のprofile段階では、これらの最適化やPolar更新則の変更は行っていない。

#### CPU wrapperの重複検証を削減（2026-10-29）

前回のGPU Polarを引き続き明示選択したまま、CPU側の重複だけを減らした。
`Algorithm.run`は準備済みstateを確認した直後の同じ確認を省き、cold時のprepare後検証は保持する。
Polarのgradient projectionとvector transportは、隣接する複数tableを`_split_many`で分割し、
同じlive layoutを1回だけ確認する。各tableのrank/列幅検査・slice位置・計算式は維持する。
宣言や数値の新しいcacheは追加せず、次の呼出しでは設定変更を検知する。
base optimizerのhookを挟む前後のContext/Plan検証は省略しない。

比較は同じL4、N128/B4/A819/M8、FP32 IEEE、公開初期化、eager AdamW+Polar、
`check_finite=False`。既存Linear runnerをbaseline/candidate/candidate/baselineの順に
別processで実行した。2回ずつの未計装完全step中央値は以下の通り。

| 条件 | 完全step ms | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| 変更前 | 3.504480 / 3.553039 | 297984 | 2097152 |
| CPU重複削減後 | 3.229141 / 3.244558 | 297984 | 2097152 |
| 各runのdense対照 | 0.704169〜0.726363 | 17313280 | 23068672 |

2回の中央値の平均で所要時間約8.3%減。denseとの速度差はまだ残る。
全比較の初期Parameter/X/targetのfingerprint一致、独立Y/dX/全atom勾配・matched-state更新を通過。
別途、同じ外部勾配と途中のregularization変更を与えた5stepで、ParameterとAdamWの
exp_avg/exp_avg_sq/stepの計20tensorが変更前後でbit一致した。

未同期のhost区間診断ではoptimizer呼出し1.793064→1.473819ms、
診断step全体3.575346→3.235677ms。これは完全stepの採用測定とは別のprocessであり、
host区間とGPU時間を足したり、中央値を引いて純Python時間を算出したりしない。
別のcProfileのcall countでは`fixed_state_signature`が39→27回/3step（13→9回/step）、
`operator_declaration`が24→21回/3step。計装下の累積時間は速度の根拠には使わない。
別traceのoptimizer scope内は両方とも同期CUDA API 0回、`aten::item` 0回。
`apply_polar`は両方とも2.816µsで、GPU更新式の高速化としては扱わない。

CPU全体2383 passed／5768 skipped／既存警告18件。L4の選択gateは337 passed。
hot/cold state、失敗prepare、live配置・設定変更、不正tableのshapeを確認した。
CUDAのnative sourceとbinaryは不変で、前段のsanitizerを今回再実行したとは主張しない。
wheel/sdist buildも成功し、全361package fileのsource一致を確認した。

job=`l4job-851744f7057a4f83bce384a158282ce0`、snapshot SHA256=
`b6e9b270c4ee9bdb0dd916bd30b53976f893c35e81481b311fc6173a139a8c65`、
result archive SHA256=`797a9fe05d5612f3f84cfd2f070fec14172df00a0b5bae782a7348cb4a6adb13`。
同じproduction binary `1dca007d…`をsource/runtime/hash照合して使った。
source snapshot・正逆順のrunner JSON・host区間・trace・5step state・テストlogを
ignored `output/radius-optimize-20261028/l4job-851744f7057a4f83bce384a158282ce0/`に保持する。
再現driverは同じoutput下の`optimizer-host-confirm-driver.py`。

#### 共有投影：moments/H/Gを1回だけ構築（2026-10-29）

source `d0324fb4`。明示`BLOCK_LOCAL`のcomputeを以下へ変更した。
旧`block_local_*` native入口は比較用に保持し、自動dispatchの採用は変更しない。

```text
forward:
  shared_projection_kernel(X) -> input_moments, atom_response   [1 kernel]
  shared_expand_kernel(H)     -> sharedのC -> Y                [1 kernel]
backward:
  shared_projection_kernel(dY)-> grad_output_moments, G         [1 kernel]
  shared_expand_kernel(G)     -> sharedのC_in -> dX             [1 kernel]
  hybrid_parameter_grad(保存moments/H, 同じ逆moments/G) -> grad_atom [1 kernel]
```

nativeの`shared_forward`／`shared_backward`は上の2 kernelを同じcurrent streamへenqueueし、
従来と同じ3値を返す。先行projection完了が次のlaunchの依存境界で、CPU同期・block間spin waitはない。
atom-onlyは`shared_project`の1 kernel＋atom勾配だけ、dX-onlyは逆向き2 kernelだけを実行する。
X側moments/Hと逆向きmoments/Gは実際に保存した同じstorageを使い、再構築しない。

projectionはbatchの1行を1 CTAが担当する。必要なmomentsをsharedに構築し、
各threadが別atomの全K tileを足し終えてからglobalのH/Gへ1回だけ保存する。
後段の出力CTAはこの完成responseを読み、Cとsite展開を融合する。
N128/B4/A819・出力32siteでは、H/Gの投影回数が方向ごとに13,104→3,276になる。
4batch間の係数生成共有は今回まだ復活させていない。

必要な入力bank（階層とKの組）は現在のatom分類からsharedのbitmaskへ集約する。
mergeでは同じKの`bank-2`をstart levelまで閉包し、子→親のbarrierを維持する。
直接生成では子bankを要求しない。出力bankの有効性は既存のsorted offsets両端の差で判定し、
C生成と`hybrid_site`の両方でinactive bankを読む前に除外する。
新しいglobal mask、packed atom、全atom Fourier係数表、global C表は追加しない。

現在のfixtureは全819atomがlevel2/K3なので、start2では12 retained bank中1 bank、
126 node中32 nodeだけを数値計算する。保存配列は従来の固定compact shapeを維持し、
未使用slotはゼロに定義する。したがって**階層の演算省略であり、global確保容量の圧縮ではない**。
各forwardが独立した保存領域を持ち、設定・atomが変われば次の呼出しで分類し直す。

L4で350件成功。必要bankとmergeの子依存、inactive momentsのzero、direct/merge、
非対称軸、odd M、M16、all-direct、空atom/batch、retained backward、複数forwardの保存独立性、
独立Y/dX/全atom勾配と公開optimizerを確認した。
同じ外部勾配と途中のregularization変更を与えた5stepは、Parameter/AdamW moment/stepの
計20tensorが変更前後でbit一致した。nativeのみの7ケースでmemcheck 0 errors、racecheck 0 hazards。
CPU全体2383 passed／5781 skipped／既存警告18件。

最初の比較は同じresident L4、N128/B4/A819/M8、公開初期化・FP32 IEEE・eager AdamW+Polar、
warmup3/21step、baseline/candidate/candidate/baselineの4 fresh process。
初期Parameter/X/target fingerprintは全runで一致した。

| 初回比較 | 完全step中央値 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| 出力CTAごとに再投影する旧版 | 3.220690 / 3.308590 | 297984 / 2097152 |
| 共有投影＋inactive bank省略 | 3.352655 / 3.332686 | 297984 / 2097152 |
| 各runのdense対照 | 0.710179〜0.735280 | 17313280 / 23068672 |

初回完全stepは短縮せず、2中央値の平均では共有版が約2.4%遅かった。
メモリは増えておらず、この条件のdense対照以下を維持する。
これらはTorch allocatorのeager peakであり、GraphとGPU process全体は未測定。

別の未同期host診断は総計3.256506→3.251686ms、forward呼出し1.089134→1.084366ms、
backward呼出し0.605583→0.598335ms、optimizer呼出し1.480270→1.487666ms。
最後のGPU完了待ちは0.028973→0.028153msだった。
別traceの実kernelのみではforward transform768.416→585.154µs、
reverse transform821.057→638.402µs。atom勾配は244.321→243.521µs。
forward/backwardのlaunchは各1回増え、stepの実kernel総数29→31、
実kernel時間合計1954.945→1588.131µs。host区間と重なるため足してstep時間を作らない。
GPU処理の削減だけでは、host呼出しが占めるこのfixtureの完全step改善を示せなかった。
projectionは4 CTA、C/site展開は16 CTAなので、batch共有と仕事の分割は引き続き検討対象である。

job=`l4job-bc66a7dc50684300ba1d4e082d0a1e75`、snapshot SHA256=
`76be2a55595836f453942ab12fcf7e7cd43b2a10a98483a5f7abc5e2995c9f52`、
result archive SHA256=`6eac3a1dc63d197a5af8ebb347ba9c341fe69a60c1e043da334cfb53f82b4db9`。
新native=`torchcst_radius_cell_mass_a47d30b38549`、binary SHA256=
`15c0c76596ba74784b24bac6cea74b519edb98ea073f69aa5fb9fb2c17fd81db`。
baselineは検証済み旧binary `1dca007d…`をsource/runtime/hash照合して使用した。
source snapshot・runner JSON・raw trace・state・sanitizer log・新binaryを
ignored `output/radius-optimize-20261028/l4job-bc66a7dc50684300ba1d4e082d0a1e75/`へ保持する。
再現driverは同じoutput下の`shared-projection-confirm-driver.py`。

**同sourceの再確認：** 別のL4割当で、各比較内のhardwareを固定し、
candidate/baseline/baseline/candidateの順に2 fresh processずつ再測定した。
fixture・3 warmup/21step・dtype・optimizer・native binaryは変更していない。
source/バイナリhashをロード時に照合し、全runの初期Parameter/X/targetの一致を確認した。

| 再確認 | 完全step中央値 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| 旧版 | 3.316176 / 3.247524 | 297984 / 2097152 |
| 共有投影版 | 3.283822 / 3.321738 | 297984 / 2097152 |
| 各runのdense対照 | 0.732475〜0.801015 | 17313280 / 23068672 |

再確認でも完全stepの短縮は示せず、2中央値の平均で共有版が約0.6%遅い。
初回の約2.4%を普遍的な退化率とは扱わず、**GPUの重複削減は検証済みだが、
完全stepでは横ばい〜微増で、高速化の採用条件は未達**とする。
各runner内の独立勾配・更新gateは再確認でも通過した。350件suite/sanitizerの再実行ではない。
保存配列の固定容量を維持しているため、global allocator peakの減少も今回の成果ではない。

再確認job=`l4job-050cb06549ab41f8834c51b17ab91db7`、snapshot SHA256=
`bd43ac112bf98c1a8a8945ea5ba7431e0b4fc5d551c94295745e9019890d9613`、
result archive SHA256=`3a93640608f3628e41c6a61b45e05a9cf3fffa4d0e53971802d956dc73c9a99c`。
raw source/spec/receipt/runner JSONは同じignored output配下のjob directoryへ保持する。
再現driverは`shared-projection-reconfirm-driver.py`。新たなCUDA compileは行わず、
初回に生成した`15c0c765…`と旧binaryをsource/runtime/hash照合して使った。

#### 通常stepの同期除去と、固定Planの準備再利用（2026-10-29）

source `a471b585`。共有投影・数学的profile・Polar更新則は維持して、次の2点を変更した。

- 通常の`_classify_layout_keys`は`classify_atoms_async`を呼ぶ。異常値を検出したthreadは
  keyを0に定義してからdevice assertionを発生させ、bank indexの計算へ進まない。
  CPUでerror wordを読む`.item()`は通常経路にない。異常値は後続CUDA APIまたは同期点で
  RuntimeErrorとして報告される。即時ValueErrorが必要な診断用`prepare_layout_checked`は
  旧checked classifierを使い、通常Linearからは呼ばれない。
- 明示した固定Planはbindingに1件だけprepared metadataを持ち、同じContext・配置・所有者・
  layout世代・live設定signatureなら選択済みAlgorithm/Stateとsupport/workspaceの判定を再利用する。
  custom selector、mutable Context/Recipe、対応を明示しないbindingは従来の選択経路へ進む。

**毎stepの検査を全廃した実装ではない。** Inputsとlive Contextの構築、型・shape・device・
Plan schema/Recipe等の宣言検証、AlgorithmStateの鮮度確認は維持する。
重い選択・対応判定を再利用する一方、設定・入力条件・配置が変われば失効させる。
Planの比較はidentityで行い、偽の`__eq__`で型/schema/Recipe検証を通過できないことも確認した。
通常のatom値更新は失効条件にせず、radius/centerの分類はGPUで毎forward更新する。
cacheはbinding所有で、Dispatcher/RegistryがTensorやStateを保持する設計にはしない。
checkpoint/deepcopyへprepared cacheや統計を残さない。

optimizerのpreflightも同じ`Dispatcher.prepare`へまとめ、base optimizerのhookを挟んだ後に
fresh Inputs/Contextから条件を確認する。先行検証を理由にhook後の検証を省略しない。
既存の固定設定signatureを借用するので、cache確認のためのmodule巡回は追加しない。
`torch.compile`やCUDA Graphは導入しておらず、この段階もeagerの比較である。

**同条件の完全step：** 1台のL4内で変更前→同期除去のみ→同期除去＋prepared→逆順に
各2 fresh process、3 warmup/21stepを測った。N128/B4/A819/M8・seed41・公開初期化、
FP32 IEEE、dX/全atom勾配・loss・fused AdamW+Polarを含む。
`check_finite=False`・AdamW lr1e-4/wd.01/capturable=Falseは従来と同じ。
初期Parameter/X/target fingerprintは全6 runで一致した。

| 条件 | 完全step中央値 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| 共有投影版・変更前 | 3.282665 / 3.250319 | 297984 / 2097152 |
| 同期除去のみ | 3.294300 / 3.233293 | 297984 / 2097152 |
| 同期除去＋固定Plan準備再利用 | 3.067269 / 3.068885 | 297984 / 2097152 |
| 各runのdense対照 | 0.706788〜0.721539 | 17313280 / 23068672 |

2中央値の平均で完全stepは約6.1%短縮した。同期除去だけでは明確な改善を示しておらず、
主な改善は固定Planのprepared経路による。denseとの速度差は残る。
メモリは増えずdense対照以下。この値はTorchのeager allocator peakで、Graph/process総量ではない。

**通常stepの同期確認：** 別の1step traceで、変更前はforwardに同期API 1回／`aten::item` 1回。
同期除去のみ・preparedの両方はforward/loss、backward、optimizerの全scopeで同期API 0回、
`aten::item` 0回、GPU→CPU copyも0回だった。計測末尾の完了待ちはscope外へ分離した。
初回準備、設定変更時の再検証、`check_finite=True`の任意数値検査、診断用checked経路まで
無同期化したという意味ではない。

別の未同期host区間は変更前→preparedで、forward1.080564→0.842046ms、
backward0.603019→0.612988ms、optimizer1.480919→1.521235ms、総計3.258469→3.087118ms。
これらは完全stepの正本と別processの診断であり、GPU時間と足し合わせない。
optimizerは改善しておらず、残るCPU側処理の大きな候補である。
30stepの診断を通したcache統計は、Linear miss1/hit29、AtomUpdate miss1/hit59。
別cProfileの固定設定巡回は27→24回/3step（9→8回/step）で、入力・設定確認はなお残る。
Registryの軽い宣言検証もhot時に継続し、cProfile累積時間は純CPU損失や速度の根拠に使わない。

CPU全体2406 passed／5819 skipped／既存警告18件。L4の選択gateは411 passed。
22個の隔離子processで不正CUDA値の非同期assertionを確認し、正常/空keysのchecked版一致、
通常経路のscalar抽出禁止、設定変更・配置/shape/flags変更・optimizer hook・serialization・
偽Plan/Recipeの拒否を確認した。Y/dX/全atom勾配・matched-state更新gateも通過。
同じ外部勾配と途中のregularization変更を使う5stepでは、20個のParameter/AdamW moment/stepが
変更前・同期除去のみ・preparedの3経路でbit一致した。
新nativeの7ケースでmemcheck 0 errors、racecheck 0 hazards。

job=`l4job-61656d9fa3ea4ec1a7d7fdddf7e711af`、snapshot SHA256=
`421457c3c21037d56a50c16f6e269945c519f3b957ecdc83e6ab05e293808ba0`、
result archive SHA256=`77b3a9dcc30f814dd2205a8975467ad02065ac5babcfd2a1382ee381aeac17d8`。
新native=`torchcst_radius_cell_mass_a6de79d2683a`、binary SHA256=
`536cdb38e21c857af909ceeabc8741c1009b950093a2458a2bacd95b99bc7624`。
変更前の比較は旧shared native `15c0c765…`をsource/runtime/hash照合して使った。
source snapshot・6個のrunner JSON・各scopeの同期集計・host区間・trace・cache統計・state・
新binary・sanitizer logはignored `output/radius-optimize-20261028/l4job-61656d9fa3ea4ec1a7d7fdddf7e711af/`へ保持する。
再現driverは同じoutput下の`prepared-dispatch-confirm-driver.py`。

#### Optimizerのhost区間の切り分け（2026-10-29、実装変更なし）

`4999d405`のoptimizerを、同じN128/B4/A819/M8・FP32 IEEE・lr1e-4・
weight_decay=.01・fused AdamW・`check_finite=False`で診断した。
本番ソースは変更せず、ignored driver内で`_step_impl`のstatement境界へ時計だけを挿入した。
途中にCUDA同期を入れず、8 warmup/41 sample、2 fresh processで区間を測った。
計測用wrapperのオーバーヘッドを含むため、以下は完全stepの採用判断用の時間ではない。

| 処理 | 2 processの区間中央値 ms（通常の未fence経路） |
| --- | ---: |
| group検証 | .0448 / .0579 |
| active Parameterとgroupの対応 | .0124 / .0126 |
| finite policy用の勾配列作成 | .0085 / .0091 |
| Polarの勾配射影＋検証 | .2368 / .2282 |
| old point snapshot | .0345 / .0336 |
| 更新Planのpreflight | .1972 / .1909 |
| 射影勾配のcopy | .0253 / .0239 |
| base AdamW＋hooks | .1552 / .1566 |
| baseとのPython参照の再接続 | .0030 / .0034 |
| 更新dispatch＋Polar CUDA起動 | .4016 / .4143 |
| optimizer vector state transport＋copy | .2971 / .2817 |
| 計装したoptimizer呼び出し全体 | 1.5287 / 1.5081 |

区間中央値の和は全体中央値と一致する必要はなく、表外にはpublic wrapper、import、return等もある。
時計を挿入しない元のoptimizer呼び出しは1.4896 / 1.4801 msだった。
同じworker内で、直前のbackwardをCUDA同期で完了させてから元のoptimizerを呼ぶと
1.5055 / 1.4779 msで、待たせない条件に対する大きな短縮はない。
**直前のbackwardのGPU待ちがoptimizerの約1.5 msを作っている、という説明は支持されない。**
このfenceは診断用の比較条件だけであり、本番経路には追加していない。

別のprofiler passで、optimizer scope内の同期API・`aten::item`・DtoHは全て0。
correlationで対応したGPU kernelは5個、合計14.305 / 14.272 µs、
別にDtoD 3回が合計3.584 / 3.520 µsだった。`apply_polar`自体は約2.8 µs。
profilerのactivity buffer準備が最初のcat等へ乗るため、profilerのCPU累積時間を
上表のhost区間の代わりに使わない。GPU実行時間とhost時間を加算しない。
`_sync_from_base()`はparam_groups/state/defaultsのPython参照を揃えるだけで、CUDA同期ではない。

**重複している仕事：** 1 optimizer stepで`fixed_state_signature`を5回、
`AtomUpdateBinding.build_context`を2回、Plan宣言検証を4回行っている。
fixtureのsignature対象は8 module/23 buffer、Contextのscalar metadataは19項目。
各Contextで20個のDeviceInfoを作る。prepared cacheはsupport/選択を省くが、この構築までは省かない。

さらに別のlight nested計測では、signature 5回の合計が.2533 / .2527 ms、
Context 2回がinclusive .2838 / .2911 ms、内側で計測したsignatureを除く部分が
.1755 / .1834 msだった。Context以外も含む`_split_many`2回はinclusive .2806 / .2756 ms、
内側signature等を除く部分が.1793 / .1773 ms。Plan宣言検証4回自体は.0095 / .0096 ms。
これらは包含関係のある別計測なので、全行を加えてhost全体に対する割合を作らない。
この別L4割当では未計装の呼び出しも約1.56〜1.65 msと変動しており、割当間の差を改善率にしない。

FlatTorusの勾配射影とvector transportは数学的にidentityだが、現在は両profileの検査を行い、
sliceをcatして、元のgrad/exp_avgへcopyしている。検査やcustom hookまで消す必要はない。
優先する候補は、(1)検査後に両profileの結果が入力sliceそのものなら元Tensorを返し、
同一objectへのcopyを省く、(2)fresh declaration/config guardを残した上でContextの不変metadataを
binding所有で再利用すること。preflightとbase optimizer hook後のlive検証は維持する。
**今回は原因の切り分けまでで、この候補の実装・高速化・メモリ削減は未検証。**

初回stage診断job=`l4job-4c6c2ee3eba1497ba09d579042b5bf7a`、snapshot SHA256=
`9bdc0bd815ecd65d448d86bb544d14832d94564b24a97eacc778d37e9f181d9f`、
result archive SHA256=`d66ddb835cbf4da916125e8f8e5de064851a8481ea08e68cab3b37029c7d190e`。
追加nested診断job=`l4job-7cf56ca98e354e868f652e2b0f25253c`、snapshot SHA256=
`b00a214acae5498a21413b4f0ce6a5c317eccd3c47fdcc208ef365fe31d9f449`、
result archive SHA256=`610c93c3907503efadf98c2e1d8eaa531490de7fb366d168ed7c8b289f37bf48`。
途中job `l4job-86549d14bb1241798cb7596a2c21b3ea`のnested集計は、sample辞書を共有したため
次stepのforwardの3 signatureが混入しており不採用。辞書をコピーして上記jobで再計測した。
全raw証拠はignored outputへ保持する。再現driverは
`output/radius-optimize-20261028/optimizer-diagnose-driver.py`。
nativeは前節と同じ`a6de79d2683a`をsource/runtime/binary hash照合して再使用した。
本番の演算・optimizer設定・保存配列を変えておらず、新たな完全step高速化やpeak memoryの結果はない。

**AdamW weight decayの作用先：** 現在のfixtureの`.01`はamp専用ではない。
`radius_cell_mass.optimizer()`はatom Parameter全体をAdamWへ渡すため、
raw `(s,t,mu_input,mu_output)`全体にdecayが入る。
公開READMEはCST側groupの`weight_decay=0`を例示しており、このbenchmark設定とは区別する。

現行Polarの振幅は`amp=amplitude_max*s/sqrt(s*s+t*t)`なので、s/tの同率縮小は
振幅の減衰ではない。中心座標のdecayはそのまま更新に入る。
ゼロ勾配・lr1e-4のCPU診断では、`.01`でcenterが30.26513671875から30.265106201171875へ動き、
全centerの変化は`old_center*(1-lr*weight_decay)`とbit一致した。
`weight_decay=0`ではcenterは不変だった。raw結果は
`output/radius-optimize-20261028/zero-gradient-weight-decay-audit.json`。
Polarのradial_regularizationは活動半径/幅の制御で、amp用AdamW decayとは別物である。
今後の「中心等にdecayを掛けない」条件ではraw atomのAdamW decayを0とし、ampだけの減衰を
採る場合は別の数式/APIとして扱う。既存の`.01`条件の速度結果を、その新条件の結果へ読み替えない。

#### amp専用weight decay（2026-10-29）

前節で確認したraw座標へのdecayを避けるため、radius fixtureのAdamW設定を
`weight_decay=0, amplitude_weight_decay=case.optimizer.weight_decay`へ変更した。
通常Dense対照は従来のAdamW `weight_decay`を使う。case上の`.01`はCSTではamp専用係数、
Denseでは通常の重み係数となる。**これ以前のraw decay `.01`の学習則とは異なるため、
旧timingに対する高速化率はこの変更から算出しない。**

公開APIはAdamW groupの`amplitude_weight_decay`。raw decayとの併用、非有限/負の係数、
`lr*係数>1`、対応しないoptimizerは更新前に拒否する。optionを省略した既存groupの
raw座標の更新は互換性のため維持する。amp-only条件では必ず新optionとraw decay=0を指定する。
通常ParameterのgroupやAdamWのmoment/stepは変更しない。

既存Polar更新で得た正規化振幅を`u_task`、更新前を`u_old`として、
`u_new=clamp(u_task-lr*λ*u_old,-1,1)`を適用する。更新後の`q=s²+t²`を保って
`s,t`を再構成し、`t`の符号側を`copysign`で保持する。減衰係数0では再構成を省き旧経路と
bit一致させる。中心座標にはこの減衰を掛けない。ampからprofileのR-boundsを決める元の契約は
変えず、そのdecodeによるradius変化と活動状態qの保存は区別する。
式・clamp・group設定の詳細は[CSTOptimizer](cst-optimizer.ja.md#polarのampだけにweight-decayを掛ける)を正本とする。

baseの最後のpre-hookで、ユーザーhook後のgrad/lr/係数を使って検証・射影・snapshot・Plan準備を行う。
元のclosureは一度だけ評価し、baseへの再提示の差し替えは拒否する。一時hookはfinallyで除去する。
post-hookによる係数変更は次stepへ効かせ、現在stepは実proposal前に捕捉した値を使う。
通常stepにCPUのTensor値読出しを追加しない。明示CUDA融合Planでは既存のPolar更新kernel内へ減衰を融合し、
追加kernel・常駐arrayを作らない。係数はruntime scalarであり、係数値ごとのconstexpr化もしない。

対応するTorch汎用/Polar参照とEuclidean/FlatTorus/radius融合の更新Planをv2へ上げた。
実装未対応の旧Sphere/Torus専用v1 Planは正のamp decayを拒否し、黙って無視しない。
非PolarのContextでamp decayを要求した場合も汎用Algorithmのsupports/実行前検証で拒否する。

**検証：** CPU全体2,438 passed／5,825 skipped／既存警告18件。新しい独立law・ゼロ勾配・
符号付きt=0・clamp・moments/step・通常group・checkpoint・pre/post hook・closureの検証は
CPU32件通過。L4の関連suite403件通過。別の確認jobで新modeとdispatchの61件も通過した。
CUDAの新6ケースでは0/7/257 atom、liveなlr/係数、ゼロ勾配の周期端center不変、
5stepのTorch参照・独立law/q・AdamW state照合、係数0のlegacy bit一致を確認した。
runnerのY/dX/全atom勾配・独立FP64小ケースquadrature・3 matched-state更新gateも通過した。

同じN128/B4/A819/M8・seed41・FP32 IEEE・3 warmup/21step、2 fresh processで、
新しいamp-only条件の完全stepを測定した。最後の確認jobのBLOCK_LOCALは
3.314006 / 3.180609 ms、dense対照は.711969 / .739176 ms。
その前のL4割当でも3.066594 / 3.030523 ms（dense .733258 / .809294 ms）だった。
別割当間のCPU/実行時間の変動があり、旧raw-decay条件との改善率は主張しない。
BLOCK_LOCALのeager peakは全runでallocated 297984 / reserved 2097152 bytes、
denseは17313280 / 23068672 bytes。Graph/process全体のメモリは未測定。

別traceのoptimizer内は同期API0・aten::item0・DtoH0。GPU kernelは従来どおり5個で
合計13.985 µs、そのうちamp decayを融合したapply_polarは2.912 µsだった。
別の軽いhost診断はoptimizer呼び出し1.610864 ms。設定走査・Tensor組立等の重さを
解消した結果ではなく、まず更新対象を正した段階である。

suite/初回runner job=`l4job-cf097dcd94a44f268884baaa38013613`、snapshot SHA256=
`3712b217a81cba0c5f3a49c4e902fa42b636c83c136c10783d1dcde51bb98ebf`、
result archive SHA256=`b59350aafd83dbb9a2cb8706c4fd8dbe309a686bde168bb7c7a47ac258a7c697`。
このjobは403 suite/2 runner通過後、診断用profile.pyがstdlib profileをshadowして最終traceだけ失敗した。
診断名を修正し、同じruntime/tests/benchmarkの全source hash一致を確認して再実行した。
確認job=`l4job-2ea072cf0e2540e697b599e237022b6e`は全体PASS、snapshot SHA256=
`dbd9a640fca1a00068d0548807351d84bef3bfea097a688955b5991f55fba1e9`、
result archive SHA256=`f677d6d31e26609dc926260ca19e8d1047108f7ac39d6d567c1b344e883b7010`。
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0/Python3.13.16、実機は各jobともNVIDIA L4。
CST nativeは前節と同じa6de79d2683aを照合して再使用し、今回変更したのは別のTriton Polar更新等である。
source/spec/receipt/suite/4 runner JSON/traceは各job名のignored output配下へ保持する。
再現driverは`output/radius-optimize-20261028/amplitude-decay-confirm-driver.py`。
変更PythonのRuffとwheel/sdist buildも確認した。全srcのRuffで報告される既存の
未変更torus_profile_product実装15件は、この変更のlint結果と区別する。
計測後、runnerの返却metadataへ`optimizer_effective_groups`を追加し、caseの要求係数と
実際のraw/amp係数を区別した。時間計測・更新処理・runtime sourceは変えていない。
fixtureがraw=0/amp=要求係数である追加回帰を含む対象CPUテスト37件も通過した。


#### 恒等変換の組立・自己copy削除と更新metadata再利用（2026-10-29）

amp-onlyの更新則を保ったまま、`802799c6`に対して次の2点を変更した。

- Polarの両profile callbackでshape/layout/finite等を従来どおり検証し、結果がそれぞれの
  入力viewそのものなら、元のgradient/vector Tensorを返す。optimizerとstate adapterは
  同じobjectへのcopyを省く。Sphereや値を変えるcustom callbackは従来のcatへ進む。
  値が等しいことだけでは省略しない。Tensorのaliasとgrad/vectorのversion増分は変わるが、
  Parameter・Adam stateの数値更新は変えない。
- AtomUpdateBindingは毎回freshなexecution_declarationを確認した後、scalar placement metadataを
  1件だけ再利用する。検証済み宣言entryのidentityとbufferのname/id/numel topologyをkeyにする。
  bufferのrenameは固定設定signatureが同じでも検出する。parameter shape/device/contiguity、
  capture、workspace、amp-decay flagを含むContext本体は毎回構築する。
  signatureがNoneのinference Tensorや、返却宣言に対応する検証済みentryがない場合は再利用しない。
  cacheはTensorやold pointを保持せず、pickle/deepcopyでは消去する。

設定signatureの走査や、base hook後のlive検証自体を消した変更ではない。
raw atom decay=0／amp decay=.01、既存Polar activity/radial則、profile、階層、H/G保存量は同じ。
CUDA/Triton kernelのコードは変更していない。

**完全step：** 一つのNVIDIA L4割当内でbaseline→candidate→candidate→baselineの順に
各2 fresh process、3 warmup/21 samplesを実行した。N128/B4/A819/M8・seed41・base1/merge2、
FP32 IEEE、loss/dX/全atom勾配/fused AdamW+Polar、check_finite=False。
4 runの初期Parameter/X/target fingerprintは一致した。

| 条件 | 完全step中央値 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| amp-only変更前 | 3.055410 / 3.075820 | 297984 / 2097152 |
| 恒等変換省略＋metadata再利用 | 2.869463 / 2.802360 | 297984 / 2097152 |
| 各runのdense対照 | .698100〜.739150 | 17313280 / 23068672 |

2中央値の平均は3.065615→2.835912 ms、**完全stepは約7.5%短縮**。
メモリは維持してdense以下。これはeager allocator peakで、Graph/process総量は未測定。
denseに対する速度目標は未達で、N1024/MNIST等の別条件へこの改善率を外挿しない。

**optimizer内部の別診断：** host呼び出し区間は1.711464→1.309260 ms。
これはCPU準備/起動/API blockingを含む別計測で、完全stepの採用時間の代わりにせず、
GPU時間と加算しない。

| optimizer trace / metadata計数 | 変更前 | 変更後 |
| --- | ---: | ---: |
| aten::cat | 2 | 0 |
| GPU kernel数 | 5 | 3 |
| GPU kernel合計 µs | 14.208 | 9.600 |
| GPU DtoD copy数 | 3 | 1 |
| DeviceInfo構築数 / optimizer step | 40 | 2 |
| 同期API / aten::item / DtoH | 0 / 0 / 0 | 0 / 0 / 0 |

残るcopyは旧座標のsnapshot。metadata計数はtimingとは別passでfactoryを包んで行い、
各3stepとも同じ回数だった。Context構築2回は維持し、scalar metadata分38個の生成を省いた。

**検証：** CPU全体2,474 passed／5,825 skipped／既存警告18件。新規35件でidentity/fallback、
型/shape/finite、設定・buffer変更、inference、serialization等を確認した。L4関連suite226件通過。
各runnerのY/dX/全atom勾配・独立小ケースquadrature・matched-state更新gateも通過した。
同じ外部勾配で5step、途中にradial_regularizationとamp decay係数を変更した別比較では、
20個のParameter/AdamW moment/step Tensorがbaseline/candidateでbit一致した。
変更PythonのRuffも通過。既存の全src Ruffの未変更ファイルの指摘は前節と同じ範囲である。

job=`l4job-cab68a7dd994451bbc3782538f5ac5be`、snapshot SHA256=
`31f0c216172cdd570f513a9b4461d02d91b0a9f6daea17b72f88244db2b28036`、
result archive SHA256=`f6bef603d7a3b10ee827618c4712f54217b4a2e9ded5361265ec4dcf81f71640`。
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0/Python3.13.16、実機NVIDIA L4。
両経路は同じCST native a6de79d2683aをsource/runtime/binary hash照合して使った。
比較元4ファイルはcommit802799c6の内容へ戻して隔離し、それ以外のsourceは共通。
source/spec/receipt/4 runner JSON/trace/metadata計数/20 Tensor比較をignored
`output/radius-optimize-20261028/l4job-cab68a7dd994451bbc3782538f5ac5be/`へ保持する。
再現driverは同じoutput下の`identity-metadata-confirm-driver.py`。

#### 固定設定signatureとAtomUpdateContextの再利用（2026-10-29）

`5cfc651b`を比較元に、設定検査の境界を維持してCPU側の組立を削減した。

- 固定設定の確認はcallback/hookの前後で毎回行う。通常のModule・Tensor bufferでは、
  PyTorchのCPU metadata guardで登録構造・spec/binding identity・buffer identity/version・
  dtype/device/shape/layoutとclass変更を確認し、未変更なら同じsignature tupleを返す。
  Pythonの`modules()`再帰走査とsignature再組立を省く。学習するatom Parameterの数値更新は
  固定設定の変更に含めない。bufferの同値write・rename・所有者移動も再検証する。
- AtomUpdateContextはfreshな宣言確認とscalar metadata guardの後で再利用する。
  検証済みentry/metadataのidentity、geometry identity、Parameterのshape/dtype/device、
  current/previousのcontiguity、capture mode、workspace、amp-decay有無がkey。
  lr・amp decay係数の大きさ・previous TensorはContextへ保存せず、毎回Inputsとして検証する。
  設定変更後の再選択、custom Algorithmの検証、base optimizerのpre/post hook境界は維持する。
- privateなPyTorch guard APIはfeature検出し、構築直後のcheckで利用可能性も確認する。
  PythonのDynamo/tracing stackをこの機能からimportせず、PyTorchが既にロードしたAPIだけ使う。
  custom walker/属性provider、Tensor subclass、inference buffer、未対応API等はfreshな
  Python検証へ戻す。guardやContextのcacheは各ownerの直近1件に限定し、pickle/deepcopyで消去する。
  cacheは入力、旧座標、Parameter、buffer自体を保持しない。

CPUだけのPython比較cache案は、signature確認が約12→16µsへ悪化したため不採用。
その後のネイティブ案の初回L4検証は17 failed/274 passedで性能採用していない。
Torch2.11/Python3.13ではimmutableな組込み型のTypeDictGuardAccessorがnullになることが原因だった。
修正版はtype/MRO identityを確認したうえで変更不能な型の辞書検査を省く。
変更可能なModule/Tensor classの辞書検査、canonical callbackのcode/defaults変更検査は残し、
L4の短いAPI確認jobでguard.check=Trueを確認した。
CUDA/Triton kernel、Polar activity/radial/amp-decay則、profile、階層や中間配列は変更していない。

**完全stepと切り分け：** 同じN128/B4/A819/M8・seed41・base1/merge2、FP32 IEEE、
raw decay=0／amp decay=.01、loss/dX/全atom勾配/fused AdamW+Polar、check_finite=False。
初回は一つのL4でbaseline→candidate→candidate→baseline、3 warmup/21 samples。
変更前2.743986 / 2.803769 ms、両方を再利用した版2.788669 / 2.821477 msで、
この初回測定では改善を示せなかった。メモリは全runで同じだった。

続けて別の一つのL4で、baseline→Contextのみ→両方→両方→Contextのみ→baseline、
各fresh process、10 warmup/101 samplesへ増やして切り分けた。
配列shape・初期値・更新則・Planは同じで、6 runのParameter/X/target fingerprintが一致した。

| 条件 | 完全step中央値2回 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| 比較元5cfc651b | 2.773119 / 2.814898 | 297984 / 2097152 |
| Context再利用のみ | 2.774644 / 2.743653 | 297984 / 2097152 |
| native metadata guard＋Context再利用 | 2.730094 / 2.735081 | 297984 / 2097152 |
| 6 runのdense対照 | .677001〜.724073 | 17313280 / 23068672 |

2中央値の平均は2.794009→2.732587 ms、**追加比較では約2.2%短縮**。
Contextのみは2.759148 msだった。初回の結果も含め、完全stepの改善は小さく、
大幅高速化や別条件での改善を示した結果ではない。denseより依然遅い。
メモリは維持してdense以下。eager allocator peakでありGraph/process総量は未測定。

| optimizerの別診断 | 比較元 | Contextのみ | 両方 |
| --- | ---: | ---: | ---: |
| host呼び出し区間 ms | 1.296328 | 1.276697 | 1.258649 |
| Context実生成回数 / step | 2 | 0 | 0 |
| DeviceInfo実生成回数 / step | 2 | 0 | 0 |
| fresh signature確認回数 / step | 5 | 5 | 5 |
| CPU上のsignature単体中央値 µs | 32.630 | 32.703 | 21.773 |

host区間は別の21 sample診断で、enqueue/API blockingを含み、GPU時間へ加算しない。
signature単体は実L4ホスト上のCPU fixture、warm確認50回後、3000 calls×7回の別診断。
両方を再利用する版ではoptimizer内の`_flatten_modules`呼び出しが3回の観測すべて0回だった。
比較元とContextのみにはこのhelper自体がなく、raw counterの0は走査省略を意味しない。
両経路は従来の関数内で毎回full scanする。fresh signature確認5回はcallback境界として残る。
GPU kernel数3、DtoD copy数1、同期API/item/DtoH=0は全3条件で維持した。

**検証：** CPU全体2,540 passed／5,825 skipped／既存警告18件、両方を変更する版のL4関連suite
292件、Contextのみの切り分けsuite114件が成功（coverageは重なる）。
各runnerのY/dX/全atom勾配・独立quadrature・matched-state gateも成功した。
同じ外部勾配で5step、途中でradial_regularizationとamp decay係数を変える比較で、
各20個のParameter/Adam moment/step Tensorが3条件でbit一致した。
設定・Tensor配置/shape・登録名/所有者・custom callback・hook・invalid Inputs・
inference・GC非保持・serializationと、利用不能guardのbounded fallbackを回帰検証した。
変更PythonのRuffとisolated wheel/sdist build・内容照合も通過した。

初回確認job=`l4job-f1a08ebfb8264659b63b2fbaa7a4d391`、snapshot SHA256=
`e5f1384f8eb045d40992cf62aab5b0a8feb9942c65f25d3567d462f5c7914bc0`、
result archive SHA256=`c620fc2f067ac02f1669b88bb58d365dd3c00aa8482a076f5ade23adfe910da5`。
切り分けjob=`l4job-09d43c66259e448193211989c879fc5c`、snapshot SHA256=
`0cbaa9b4490b2c928ff8b11ae1e788707698a16a9548953222ce6dbfe62ff4be`、
result archive SHA256=`f2d46519bd83b85c129731331b6696dbefaebd9d8099af6725d1765adc2954ec`。
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0/Python3.13.16、実機NVIDIA L4。
両jobとも比較元は4ファイルを5cfc651bへ戻し、Contextのみはmetadata.pyだけを戻した。
native CUDA source/binaryは同一a6de79d2683aをSHA照合して再利用し、実装Pythonもsnapshotと照合した。
raw evidenceはignored `output/radius-optimize-20261028/<job>/`、driverは同outputの
`native-guard-confirm-driver.py`と`native-guard-isolation-driver.py`へ保存した。
先行失敗job3732f076・API診断jobc4cc0f80/28770cb2も同outputへsource/receipt付きで残した。


#### optimizerのCPU呼び出し経路の追加診断（2026-10-29）

runtimeはb944a283のまま、N128/B4/A819/M8 fixtureのoptimizerだけを追加計測した。
一定の外部gradientを設定し、CST・denseともfused AdamW、20 warmup後101 updates。
各updateの直前直後を同期した中央値はCST 1.279729 ms、dense .104418 ms、
同期前のhost呼び出し区間はそれぞれ1.250760 / .089359 msだった。
これはforward/backwardを走らせないoptimizer-only診断で、前節の完全stepや、
学習stepから切り出したphase診断（CST約1.35 / dense約.18 ms）とは同じ測定区間ではない。
新たな完全stepの速度改善・メモリ測定を示す値ではない。

別passで41 updatesに軽いnested timerを付けると次の中央値になった。
inclusive行は包含する子の値と加算しない。wrapper自身の計測負担も含む。

| 処理 | 回数 / update | inclusive ms |
| --- | ---: | ---: |
| group/所有者検証 | 2 | .080354 |
| 勾配射影 | 1 | .182267 |
| AtomUpdateContext確認 | 2 | .179434 |
| Dispatcher._prepare | 2 | .282620 |
| Registry.validate_plan | 4 | .008236 |
| AlgorithmState signature | 1 | .053420 |
| Algorithm.run | 1 | .203274 |
| state移送adapter | 1 | .223328 |
| その内部のKernel transport | 1 | .207329 |

単一の重いGPU演算というより、更新前後の構造確認・汎用呼び出しが積み重なっている。
既測のGPU更新kernel合計約9.5µs／Polar約2.9µsと合わせると、denseに近いoptimizer時間を
狙う余地はある。ただしdense同等は未達で、この診断だけで到達を保証しない。
次の大きな候補は、設定が同じ間の更新前後処理を検証済み実行経路へまとめること。
型/shape/配置、lr/amp decay、設定変更とpre/post hook境界の確認は維持し、
customなcallback・Algorithm・state adapterは元の汎用経路を使う設計を検討する。
この診断時点では未実装だった。下の「標準Polar更新経路の事前準備」で実装・測定結果を記録する。

clean診断job=`l4job-96eb5c6adaf34975a30993d82d623d77`、snapshot SHA256=
`18f089f9abd769a2e281a4e18dc72b85d6ad9ce9cbee104591359c15fae97dc9`、
result archive SHA256=`df7d6e367ebb143beb4161b8682b607e72d332293241522d5ea5eee3e01a1493`。
Torch2.11.0+cu130/CUDA13.0/Python3.13.16、NVIDIA L4。
一定gradient・nested timer・cProfile・runtime・receipt・sourceをignored
`output/radius-optimize-20261028/<job>/`、driverを同outputの
`optimizer-current-profile-simple-driver.py`へ保持する。採用値はplain/nested timer。
cProfileはnative guard内のPython callbackでcall treeの時間帰属が崩れるため参考扱いにした。
先行job3571b06eは不要なnative loaderのmetadata不足をstderrに出したため採用せず、
このclean jobではそのloaderを使わず、optimizer自身のTriton実装を実行した。


#### 標準Polar更新経路の事前準備（2026-10-29）

前節の診断を受け、radius/FlatTorusの標準callback・明示的な固定Plan・
標準AdamW state adapterを使うamp-only更新に、binding所有の準備済み経路を加えた。
対象は`check_finite=False`の通常eager step。公開既定の`check_finite=True`は変えない。

```text
初回／設定変更時：従来の射影 → Dispatcher検証 → fused AdamW → Polar → state輸送
                 → 条件が合えば、この経路を1件だけ事前準備
通常step：user pre-hooks → 設定・shape・storage・callback確認 → 旧座標snapshot
         → 通常のfused AdamW → user post-hooks → 再確認 → 同じPolar kernel
         → state輸送がidentityであることを確認
```

- `optim/optimizer.py`は標準hookの実行順と、このstep用に捕捉したlr/amp decayを維持する。
  係数・shape・device・stride・storage・Plan・Registry・設定bufferは毎step確認する。
  不正な設定は従来どおりAdam更新前に検査する。数値異常検査の有無は従来のpolicyに従う。
- `polar_update/optimizer_program.py`が、標準の恒等射影／恒等state輸送と、
  再構築不要なContext/選択済みAlgorithmの呼び出しをまとめる。custom callback、
  state adapter、Tensor subclass、TorchFunction/Dispatch mode等は汎用経路へ戻す。
  数値更新のたびに変わるParameter/Adam stateのversionは固定設定の変更とはしないが、
  shape/dtype/device/layout/stride/data pointerとstate辞書の差し替えは検査する。
- 元の関数のidentity/code/defaults/globals/closureとclass/moduleの変更も確認する。
  `functools.wraps`で注釈をコピーした関数は標準関数として扱わない。
  共通の`_builtin_callbacks.py`はこの識別情報だけを持ち、backend選択やTensor計算はしない。
- `compiled_launch.py`は、通常のTriton実行が返した**同じCompiledKernel**を再利用する。
  同じTensor配置・constexpr・lr/amp decay specialization・device/cacheのときに、
  同じTriton runnerへそのstepのTensorを渡す。JITの引数binding/hash準備を省く。
  未対応API・変更時は元のAlgorithm.executeを使い、JIT pre-run hookも省略しない。
  launch hookは同じrunnerで実行し、その後にcallback/状態を再確認する。
  Polar/Triton数式やCUDA source、階層、profile、AdamW更新則は変更していない。
- guardはCPU metadataだけを扱う。Tensorのshape/stride/dtype/device検査は、対応時には
  native TensorMatch guardへまとめる。private APIは構築時に適合を確認し、未対応時は
  従来の検査へ戻る。通常stepへ同期・`.item()`・device readbackは追加しない。
- cacheは各bindingで直近1件。model所有のParameter/固定scalar/stateへの参照と
  コンパイル済みcodeを持ち、snapshot・入力・gradientのコピーは保持しない。
  旧座標のcloneは従来と同じ1回で、更新後に破棄する。GPU配列は追加しない。
  pickle/deepcopyでは準備済みprogramを除去する。

**最終比較：** aeec9360（runtime b944a283）を比較元に、同じ一つのL4で
baseline→candidate→candidate→baseline、各fresh process、10 warmup/101 samples。
N128/B4/A819/M8・seed41・base1/merge2・FP32 IEEE・raw decay0/amp decay.01・
loss/dX/全atom勾配/fused AdamW+Polarを含む。4 runの初期Parameter/X/targetは同じ。

| 条件 | 未計装の完全step中央値2回 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| 比較元 | 2.741230 / 2.781407 | 297984 / 2097152 |
| 準備済み更新経路 | 2.407448 / 2.439732 | 297984 / 2097152 |
| 同じ4 runのdense対照 | .682137〜.709832 | 17313280 / 23068672 |

2中央値の平均は**2.761319→2.423590 ms、約12.2%短縮**。
メモリは変わらずdense以下。ここではeager allocator peakを測っており、
Graph/process総量や別batch/1024/MNISTでの改善は未確認。
このN128のdense allocatedにはGEMM用のruntime作業領域も含まれるため、
全量を重みやoptimizer stateのサイズと解釈しない。

別の同期入りphase診断ではoptimizerが1.311849/1.315004→1.000770/1.008032 ms、
denseは.172945〜.179607 ms。**optimizerのdense同等は未達**。
同じjobの別host診断は1.270034→.929226 msだった。host区間はenqueue/API blockingも
含むため純粋なPython時間とはせず、phaseやGPU時間と足したり完全stepから引いたりしない。
GPU更新kernelは3個、合計9.568→9.632 µs（Polarは2.944→2.976 µs）、
旧座標copyは1個で両方1.120 µs。同期API/item/DtoHは0を維持した。
準備済み経路のpreflight/実行が3 updatesすべて再利用されたことも別passで確認した。

一定の外部gradientを与えるoptimizer-onlyの別jobでは、20 warmup/101 updates、
CST .921456 ms／dense .108966 ms（各update前後を同期）、hostは.892584/.093661 ms。
forward/backwardを含まない診断であり、完全stepや上のphaseとは測定範囲が異なる。

**検証と限界：** CPU全体2551 passed／5861 skipped／既存警告18件。
最終比較jobのL4関連337件と各runnerのY/dX/全atom勾配・独立quadrature・matched-state gateが成功。
同じ外部勾配で5step、途中でradial設定とamp decayを変更する比較でも、
Parameter/Adam moment/stepの20 Tensorがbit一致した。
両Polar modeの6step比較では、係数変更・lr pre/post hook・custom射影/validator/transport/adapter・
state差し替え・有限検査切替・gradなし・lr0・JIT hookを検査した。
非連続gradは標準fused AdamWも拒否するため、その例外と例外時stateを両経路で比較する。
GC非保持、serialization、native guardでのcallback/code/Plan変更検出も回帰検証した。
追加のL4 47件ではlaunch-exit hook内でstate輸送を変更し、更新後の再確認で
その変更も取り込むことを両Polar modeで確認した（上の337件とcoverageは重なる）。
Python Ruff、isolated wheel/sdist buildと全367 package sourceの内容照合が成功。

先行のprepared経路（compiled runner追加前）も完全step2.739727→2.396118 ms、約12.5%短縮だった。
今回の追加launch変更による上乗せ効果は、この別job間の数値からは主張しない。
初回launch適合確認ではTritonの空HookChainをNoneと区別できず汎用経路へ戻っていた。
診断後に同じrunnerでhookを実行する形へ直し、bit一致を確認した。
非連続gradの先行テストは「成功する」というfixture側の想定が誤って失敗したもので、
最終比較では標準AdamWと同じ拒否を確認している。失敗jobも原本を保持する。

最終比較job=`l4job-3a610851c9ff4530b29fa67579389d3f`、snapshot SHA256=
`192d033dc19659769b6771747ab3de0976b943993be4fe1d77bfaa873b98c459`、
result archive SHA256=`49bdcc142d291a7dded9777e6b65871498736e31fd81a40d5b45e312a5786b64`。
optimizer-only job=`l4job-bae8cda9e469471e9913911b92b181d9`、snapshot SHA256=
`0777bff6286aa3014b72dc6ad2da6a76f1dcef9b94509e32079e38c6a9e109a8`、
result archive SHA256=`327b670054d275902be40f1415f6361f1f8f1666527b081cffac08f2da79a472`。
追加hook検証job=`l4job-4e8e9c9afbb3448286862d28abece189`、snapshot SHA256=
`24d358a8f55b26185d1e621ed032acc5c21c35640802a8acdb06ed31d6f4fd78`、
result archive SHA256=`97af4f28d277f56c66cfce8688590883b18158172f0f145202a45d18479853f4`。
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0/Python3.13.16、NVIDIA L4。
先行prepared比較job=`l4job-facbaab2ff6e463e900690ee11374e35`、snapshot SHA256=
`fb788cf1523fe3e8cf6716e9aa59758ca5c5cc87083d4d2d3463803a512f25ca`、
result archive SHA256=`0a388f4947cbb4895d0ba751118174316c662ab1515ffdddae515e9035edfe16`。
raw logs/trace/state Tensor/receipt/sourceはignored `output/radius-optimize-20261028/<job>/`、
driverは同outputの`prepared-optimizer-confirm-driver.py`と
`prepared-optimizer-only-final-driver.py`。最終比較は全変更済みruntimeファイルを
比較元へ戻す別sourceを作り、native source/binaryは従来と同じa6de79d2683aをSHA照合して再利用した。
再現検証は`python -m pytest -q tests/test_prepared_optimizer_update.py`、
完全stepは既存Linear runnerと`radius-cell-mass-128-initial.json`を使う。


#### optimizerの残存CPU処理とguard統合（2026-10-29）

直前の96825723を比較元に、同じN128/B4/A819/M8・amp-only条件で残る呼び出しを計測した。
一定gradient、20 warmup/101 updatesのoptimizer-onlyはhost .902892 ms／同期込み.929280 ms。
41 updatesの別nested timerでは次の費用だった。包含する行どうしは加算しない。

| 96825723での処理 | 回数 / update | 合計中央値 ms |
| --- | ---: | ---: |
| prepared program guard | 3 | .242321 |
| 固定設定guard | 3 | .110754 |
| `_validate_groups` | 2 | .076422 |
| `CompiledPolarLaunch.run` | 1 | .041913 |
| `_sync_from_base` | 1 | .000982 |

全体同期や大きなoptimizer stateのcopyが主因ではなかった。
`hooks_active`がTritonの空HookChainを常に有効と扱い、更新後にも全guardを実行していた。
`_matches`も設定確認のために`execution_declaration`を経由し、別のguardへ入り直していた。

**採用した変更：**

- 空の標準HookChainはuser codeを実行しないので、追加の更新後確認を省く。
  標準loopのcode/defaults・Genericの継承・属性方式を限定して適合確認する。
  実hook、変更された実装、未対応runtimeでは従来の更新後確認を残す。
- 固定設定の検査を、同じ`_build_guard`でprogram guardへ組み込む。
  通常はAdam前後の2回、それぞれ登録・buffer version・配置・callbackを検査する。
  検証項目を減らさず、`execution_declaration`経由の二重確認をなくす。
  derived geometry dimensions、OperatorのKernel/Chart、siteの登録先identityも明示的に検査する。
- 標準ModuleのParameter/子Moduleは、登録辞書をnativeで確認する。
  native guard内からPythonの`nn.Module.__getattr__`へ戻る回数を減らす。
  同じMROの辞書検査は共有し、関数code/defaultsの変更検出にはnative accessorを使う。
- 固定設定guard自身のcallback確認もnative accessorへ移す。
  Python callbackは固定function tupleを返すだけにし、関数ごとのPython比較loopを省く。
  不適合なprivate APIは従来のfresh validationへ戻る。

Polar/AdamWの更新則、amp-only decay、階層、profile、GPU kernelと配列は変えない。
base pre/post hook、lr/decayの値検証、設定変更時の再構築、有限検査policyも維持する。
名前だけでなくcallback code変更、数値が同じbuffer write、登録先/Parameter storage変更も検出する。

**採用版の完全step：** 一つのNVIDIA L4、baseline→candidate→candidate→baseline、
各fresh process、10 warmup/101 samples。loss/dX/全atom勾配/fused AdamW+Polarを含む。

| 条件 | 未計装の完全step中央値2回 ms | allocated / reserved bytes |
| --- | ---: | ---: |
| 96825723 | 2.717420 / 2.549276 | 297984 / 2097152 |
| guard統合・callback確認のnative化 | 2.476592 / 2.412705 | 297984 / 2097152 |

2中央値の平均は**2.633348→2.444649 ms、約7.2%短縮**。
メモリは増えずdense以下。eager allocator peakで、Graph/process総量は未測定。
このjobのdense対照も.712793〜.758565 msへ揺れており、改善率の精度には限界がある。
絶対時間を別jobと直接比較せず、対照のある測定として扱う。
別optimizer-only診断はCST 0.754732 ms、dense 0.108973 ms。
これはforward/backwardを含まず、完全stepやphase診断と同じ区間ではない。
**dense同等は未達**で、別batch/1024/MNISTでの速度はこの変更では測っていない。

別nested timerで、残るprogram+固定設定の一体guardは2回合計
0.195205 ms。
base AdamW呼び出しのうちtemporary pre-hookを除く時間は
0.155103 ms、
temporary pre-hook自身の残りは0.090981 ms。
後者は旧座標snapshot、入力/gradient確認、groupsの対応付け等を含む。
包含関係と計測wrapperの負担があるため、これらをplain timeから差し引かない。
GPU更新は引き続き3 kernelsとsnapshot copy1回、同期API/item/DtoHは0。

**不採用の試行：** `named_parameters`による所有者走査のnative cacheも試した。
最初の完全step比較では小さな上乗せが見えたが、独立比較とgroup単体では安定しなかった。
最終の3条件比較でも統合のみ2.246631/2.269500 ms、cache込み2.222015/2.304347 msで、
cache込みの2中央値平均がわずかに悪化した。今回はhelperと関連APIを除去し、
`_validate_groups`の値と所有者の検証はそのまま残す。
先行のdefault accessor適合確認はNoneを扱えずfallbackし、空HookChainの初回適合も
Generic継承を除外していた。これらは速度採用しておらず、修正後に実経路の再利用を検証した。

CPU全体2558 passed／5861 skipped／既存警告18件。採用版L4関連346件と各runnerの
Y/dX/全atom勾配・独立quadrature・matched-state gateが成功し、
Parameter/Adam moment/stepの20 Tensorは5 updatesでbit一致した。
両Polar modeの実hookによるstate transport変更、custom callback、layout/登録/shape変更も回帰検証した。
変更PythonのRuff、isolated wheel/sdist buildと全367 package sourceの内容照合を通過した。

採用版job=`l4job-2d298c84ccec4a3f9da59554f9b3590f`、snapshot SHA256=`c7cad9babd2a7011a079705bb5d40189443937309bd073b0802f42efef1fd5d8`、
result archive SHA256=`f43a4f8e25050dc74458ad8ecef5c3b0feabbd54a5c8daf09514f0ac965aa040`。
最終単体・nested診断job=`l4job-30a3521a5a3f42249eca1046bbde4935`。
初期診断job=`l4job-d1fefb309c0c429fa05cedc23160a420`、
3条件比較job=`l4job-d1b167d9f64d47a89bb0424c2ef42ab5`、
所有者cache再比較job=`l4job-4a0a7bdd9a3643c69cb352e9fdca9089`。
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0/Python3.13.16、NVIDIA L4。
原本・失敗job・不採用source・receipt・traceはignored `output/optimizer-host-20261029/<job>/`へ保全する。
driverは同outputの`final-comparison-driver.py`と`adopted-breakdown-driver.py`。
比較元は変更runtimeファイルをすべて96825723へ戻す別sourceで、同じnative source/binary
(a6de79d2683a)をSHA照合して使う。新規CUDA/Triton数式やcustom CUDA optimizerは追加していない。


#### optimizerのstep内一時情報の再利用（2026-10-29）

2e4da5cdを比較元に、前節で残ったPython周辺処理を整理した。
所有者走査のstep間cacheは追加しない。`_validate_groups`の所有者・重複検査で作る辞書を、
その呼び出し内のParameter→group対応にも使い、検証直後の別走査をなくす。
user pre-hook後は必ず新しく検証・作成し、最初のgroup対応を流用しない。
`check_finite=False`では使用されないgradient一覧の作成とsparse coalesceを省く。
標準callbackが認証されたprepared経路だけは、検証済みのfrozen `AtomUpdateInputs`を
同じstep内で再利用する。Adam後のnative guardと入力検証は残し、設定・callbackが変われば
generic経路で入力を再作成する。独自の入力constructorをpost-hookで変更した場合も追従する。
前の座標snapshotは従来の1個で、step終了後に残す入力・gradient・新しいGPU配列はない。
更新則、amp-only decay、既定`check_finite=True`、PyTorch hookの実行順序は変えない。

**最終sourceの測定：** NVIDIA L4、Torch2.11.0+cu130/CUDA13.0/Triton3.6.0。
同じN128/B4/A819/M8/seed41/base1/merge2、FP32 IEEE、raw decay0/amp decay.01。
各fresh processでbaseline→candidate→candidate→baseline、完全stepは10 warmup/101 samples。
別optimizer-only測定は20 warmup/101 updates、一定gradientで各updateの前後を同期した。

| 条件 | 完全step中央値2回 ms | optimizer-only中央値2回 ms |
| --- | ---: | ---: |
| 2e4da5cd | 2.196579 / 2.189587 | .726460 / .724193 |
| step内情報の再利用 | 2.196450 / 2.185664 | .721080 / .746037 |

完全stepの2中央値平均は**2.193083→2.191057 msで実質横ばい**。
optimizer-onlyも安定した短縮を確認できなかった。最終確認前の別snapshotでは
完全step2.208732→2.203368 ms、optimizer-only .733681→.723080 msと小さな差だったが、
最終sourceの再測定では短縮が安定しないため、速度向上の達成とは扱わない。
今回の採用理由は不要な準備処理の除去であり、dense並みのoptimizer速度は未達。
matched denseは完全step.689904〜.703607 ms、optimizer-only .101894〜.104673 msだった。
異なるjobの絶対時間から改善率を出さない。

allocated/reserved peakは全4条件で**297984 / 2097152 B**のまま。
denseの**17313280 / 23068672 B**以下を維持する。eager allocator peakで、
Graph capture/replayとGPU process総量は今回未測定。
別nested timerではtemporary pre-hookのexclusive時間が約.088→.081 msへ減った一方、
2回のcombined guard約.183〜.185 ms、group検証約.075〜.077 ms、
pre-hookを除くbase AdamW約.147〜.149 msが残る。包含する行は加算せず、plain timeから差し引かない。
GPU更新は引き続き3 kernels合計約9.6 µsとsnapshot copy1回、sync API/item/DtoHは0。
小さなPython重複の除去だけでは、この残存費用を解決できなかった。

CPU全体**2561 passed / 5863 skipped**（警告19件、うち追加sparse fixtureのbeta警告1件）。
最終L4関連**351 passed**、Y/dX/全atom勾配・quadrature・matched-state gateが成功し、
Parameter/Adam moment/stepの20 Tensorは5 updatesでbit一致。
post-Adam constructor変更を両Polar modeで追加検証した。
最終runtime全367ファイルは計測snapshotとbyte一致した。

最終job=`l4job-efd8d595ae6a4a3b86f8161f99c17bb1`、
snapshot SHA256=`a54c29db58a75c1b187fda513686470dfee325e8b59a43290ad61aefb9ec00d6`、
result archive SHA256=`f8295c1ade64721d4242f71d7e47be3195d0d225347c596cf4b896f408c78e86`。
先行比較=`l4job-b42b9efd33e646539c6a2a86e0f3d158`、
最初の試作比較=`l4job-42a101a4f4fc4e8683bc9ea439ba7b88`、
constructor回帰gate=`l4job-9b71d0b600b14e0e9701a932ff044b97`。
原本・source・receipt・traceはignored `output/optimizer-wrapper-20261029/<job>/`、
再現driverは同outputの`final-driver.py`、集計は`verified-summary.json`に保全する。
既存Linear runnerと同じnative source/binary a6de79d2683aをSHA照合して使う。
新規CUDA/Triton kernel、PolarとAdamWの数式変更はない。


#### group検証全体のcontract再利用（2026-10-29）

前節の小さなstep内整理とは別に、`_validate_groups`全体の成功結果をnative guardで
再利用する経路を追加した。比較元は5711978e、runtime commitは5a37271b。
以前不採用だった所有者名だけのcacheでは、group値のPython検証が残っていた。
今回はParameter→group対応と、その妥当性を決める設定・登録構造を一緒に検査する。

- 既定`check_finite=True`、custom walker/site、Tensor mode、未対応private APIはfresh検証を保つ。
  最適化は`check_finite=False`、標準CSTLinear/RegularGrid等の適合を確認できた場合に限定する。
- 通常の各検証境界ではgroupの辞書version、paramsの長さ/要素identity、modelの
  Parameter/子Module登録、site/chart/spec、class/MRO、関数code/defaults/globalsを確認する。
  変更があればPythonで全検証し直してからguardを作り直す。invalid値ならAdam前に失敗する。
  pre-hook後も必ず再確認し、最初の検証結果を無条件には渡さない。
- custom callbackをcache適合確認のためだけに追加実行しない。登録を直接検査して判定する。
  未登録属性による登録先のshadowing、Parameter差し替え、trainable条件の変更も検出する。
- group contractはIDと既存groupへの参照だけを保持する。数値更新で変わるTensor versionや
  kernel bufferはこのgroup検査に含めないが、従来の更新program guardによる数式設定・
  shape/stride/storage等のAdam前後検証は引き続き実行する。
- snapshotは従来の1個。新しいGPU配列や保持activationはない。pickle/deepcopyでは
  optional guardを捨て、optimizer/modelを解放した後の参照残存も検査した。

**最終sourceのL4比較：** N128/B4/A819/M8/seed41/base1/merge2、FP32 IEEE、
raw decay0/amp decay.01、fused AdamW+Polar、check_finite=False。
同一L4上でbaseline→candidate→candidate→baselineを各fresh processで実行した。
完全stepは10 warmup/101 samples、別optimizer-onlyは20 warmup/101 updates。

| 条件 | 完全step中央値2回 ms | optimizer-only中央値2回 ms |
| --- | ---: | ---: |
| 5711978e | 2.195570 / 2.229130 | .734726 / .704923 |
| group contract再利用 | 2.198661 / 2.197335 | .689533 / .691138 |

optimizer-onlyの2中央値平均は**.719825→.690336 ms、約4.1%短縮**。
完全stepは**2.212350→2.197998 ms、約.65%の小さな差**で、hostの揺れも残る。
この完全step比率を他条件へ一般化しない。matched denseの完全stepは.692913〜.708078 ms、
optimizer-onlyは.104338〜.107420 msで、dense並みには届いていない。

差が小さいため、補助測定では同じ既存`train()`を使い、同一process内の2モデルを
交互順で実行した。独立した2 process、20 warmup/101 pairs、各step前後を同期する。
各pairのbaseline−candidate差の中央値は**.035924 / .046582 ms**。
それぞれの通常中央値は2.217337→2.175725 ms、2.311045→2.254237 msだった。
両processで121 updates後のParameterがbit一致し、学習中も同じgroup contractを再利用した。
これは小さな短縮を補足する診断で、2モデル常駐時のメモリをdense比較に使わない。

正規の別process測定のallocated/reservedは全4回**297984 / 2097152 B**で増加なし。
denseの**17313280 / 23068672 B**以下を維持する。eager allocator peakで、
Graph capture/replay・process総量と1024/MNISTの再測定は今回未実施。
別nested timerのgroup検査は従来2回合計.078688/.075483 msから、native検査2回合計
.043673/.044152 msへ短縮した。測定wrapperで`_validate_groups`自体を置き換えると
cacheを無効化するため、candidateはnative guardを直接計時した。
包含する診断値は足し合わせず、未計装の完全stepから差し引かない。
GPU更新は引き続き3 kernels合計約9.5 µs、snapshot copy1回、sync API/item/DtoHは0。

CPU全体**2594 passed / 5863 skipped**（警告19件）。最終L4関連**384 passed**。
独立Y/dX/全atom勾配・quadrature・matched-state gate、Parameter/Adam moment/stepの
20 Tensorの5 updates bit一致が成功した。新しいgroup contract関連33件では、
設定変更/不正値、pre-hook、所属/登録/属性shadow、callback code、Tensor mode、
custom callbackの実行回数、mutable option、serializationと解放を検証した。
最終runtime全368ファイルは計測snapshotとbyte一致した。

最終job=`l4job-7f3e815039ee48af829e5045c4bd8237`、
snapshot SHA256=`215839d5377c0baef8b516d6fd8b86f69d93526adf933cc4ea71a1bb55aab956`、
result archive SHA256=`1e27fa9c5ab8b15a1c627b0eeda43ec95370634c84387e7be9567b0a3ad7277a`。
先行比較は`l4job-e346a67b1f06486ebe2a1e4e54309c9a`と
`l4job-449cb3b6b0c342bba92a8b1e887331d7`。pair-onlyの
`l4job-315cf494544c468f92e86a489e168256`はqueuedのまま取り消し、最終jobへ統合した。
原本・source・receipt・trace・driverはignored `output/optimizer-group-20261029/`へ保全する。
再現は同outputの`verified-driver.py`。比較元ではoptimizer.pyを5711978eへ戻し、
新helperを取り除く。既存Linear runnerと同じnative source/binary a6de79d2683aをSHA照合して使う。
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0、NVIDIA L4。Polar/AdamWやCST数式とGPU kernelは変更しない。

#### 関数metadataの一括検査（2026-10-30）

runtime commit `27b9da42`では、prepared optimizerが確認するPython関数のcode/defaults/
kwdefaults/globals/closure/dictionary identityを、GILを保持したCPython C++ helperへまとめた。
dictionaryの内容変更は従来のnative dictionary-version guardで検出する。
Python callback、Tensor値の読出し、CUDA処理はhelper内にない。
非CPython/free-threaded Python、compiler/API非対応では従来のguardへ戻る。
helperのimport/build後は適合条件を再検査し、cold JIT hookが変更したtransportも再検査する。
設定変更時の検証、Polar/AdamW則、amp-only decay、GPU配列・snapshot数は変えない。

同じN128/B4/A819/M8/seed41/base1/merge2、FP32 IEEE、raw decay0/amp decay.01、
check_finite=Falseで比較元35edaf98とbaseline/candidate/candidate/baselineのfresh process測定。
完全stepは10 warmup/101 samples、optimizer-onlyは20 warmup/101 updates。

| 条件 | 完全step中央値2回 ms | optimizer-only中央値2回 ms |
| --- | ---: | ---: |
| 35edaf98 | 2.208162 / 2.254113 | .704499 / .689706 |
| 一括検査 | 2.204973 / 2.232900 | .637537 / .648932 |

optimizer-onlyの平均中央値は**.697103→.643234 ms、約7.7%短縮**。
完全stepは**2.231137→2.218937 ms、約.55%の小さな差**で、別processの揺れが残る。
既存train()を同一process内で交互順に測った補助比較は20 warmup/101 pairsを独立2回実施し、
pair差の中央値は.051469/.048981 ms。121 updates後のParameterはbit一致。
2モデル常駐の補助測定のメモリはdense比較には使わない。
正規の全4回のallocated/reservedは**297984 / 2097152 B**で変化なし。
denseの17313280 / 23068672 B以下。eager peakでありGraph/process総量は未測定。
denseの完全stepは.700780〜.726952 ms、optimizer-onlyは.102710〜.103852 ms。
dense並みのoptimizer速度には依然として届いていない。

別nested timerの2回合計program guardは.186841/.177463→.147696/.151672 ms。
GPU更新は3 kernels合計約9.6 µs、snapshot copy1回、sync API/item/DtoH0。
包含する診断値は完全stepに加算・減算しない。L4関連402件成功、独立Y/dX/全atom勾配・
quadrature・matched-state gate成功。5 updatesのParameter/Adam moment/step20 Tensorはbit一致。
host helperのcold build/importは7.432秒であり、steady timingには含めない。
Python3.13.16、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、NVIDIA L4。

job=`l4job-b92ad68d94044f3dbc7b6f6d6fb2a82a`、snapshot SHA256=
`2f718173cf512f7333a9770a3763c23857099270b560e756d4f42c93e51b2acb`、
result archive SHA256=`964469377dc387dcde1d0bafe925aa08d573b1dfe341f6126d49617c5ca9225b`。
linear source/binaryは従来a6de79d2683aのままSHA照合した。
初回はNinja欠如、次回は記録scriptのmodule取得で失敗し、速度根拠から除外した。
最終driverはNinjaをjob-local --targetへ導入し、system packageを変更しない。
失敗原本を含むsource/result/receipt/trace/stateと`qualified-driver.py`、集計はignored
`output/optimizer-native-20261029/`へ保全する。新しいGPU並列化の結果とは混ぜない。

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

### 追加比較: atomを担当するprojectionブロック

`radius-block-local-parallel`は既存`radius-block-local`との明示的な比較用Planである。
momentsは従来どおりbatch行ごとのCTAで一度だけ作る。そのlaunchが完了した後、
32threadのCTAへatomを分配し、各threadが同じatomの最大4batch行の完全responseを作る。
profile係数と位相をbatch間で使い回すが、各行の加算順・FP64中間値は維持する。
parameter勾配も同じ数式のまま32thread/CTAで実行する。

N128/B4/A819ではresponseのCTA数は4から26、parameter勾配は7から26になる。
momentsの生成とresponseを別launchにするため、起動回数との交換がある。
globalのmoments/response、sharedのcoeffs、autogradの保存寿命は既存Planと同じ。
runtime `e20acd57`をL4で検証し、関連258件が成功した。非対称軸、direct/merge、
混在/空/全direct、batch1/4/5と端数、M1/8/16、FP32/64、saved-cacheのpointer再利用、
dX-only/atom-only/両方のbackward、別stream寿命を含む。N128/B4/A819/M8/seed41、
base1/merge2、amp-only、FP32 IEEEで、独立Y/dX/全atom勾配・quadrature・
matched-state optimizer gateも成功した。

同じ新規compiled binary内の旧Planと新Planを、fresh processで旧/新/新/旧の順に測った。
各10 warmup/101 samples。両方とも前節の一括optimizer metadata検査を使う。

| 条件 | 完全step中央値2回 ms | allocated / reserved peak B |
| --- | ---: | ---: |
| radius-block-local | 2.265875 / 2.433527 | 297984 / 2097152 |
| radius-block-local-parallel | 2.380437 / 2.256816 | 297984 / 2097152 |

平均中央値2.349701→2.318626 msの差は約1.3%だが、ばらつきが大きく、
この比較だけで安定した完全step改善とは判断しない。matched denseは.696273〜.733030 ms、
allocated/reservedは17313280 / 23068672 B。Graph/process総量は未測定。

別traceのGPU時間は、forward/reverse合計のmoments+responseが744.704→308.194 µs
（内訳: moments251.233、response56.961 µs）、parameter勾配243.712→128.929 µs。
coeffs/site展開は477.249→478.112 µsで横ばい。kernel自身は短縮したが、
これらを完全step時間へ加算・減算しない。自動dispatchや既存Planの意味は変更しない。

job=`l4job-41eab8ef158e4a0d80602a6f69d6b228`、snapshot SHA256=
`f7bc33884d0f473b0c4b23685850eec7b68802a2a583c4a5cdacf8bc9fdb34e4`、
result archive SHA256=`093fdce897db647a5ade5ca2cfa0ec3c89ce66c9bc5dbe01c12e2ff18f4aa0cf`。
NVIDIA L4、Torch2.11.0+cu130/CUDA13.0、Triton3.6.0、nvcc13.0、job-local Ninja。
source/result/receipt/trace/driver/集計はignored `output/radius-parallel-20261030/`。
CPU全体2598 passed / 5928 skipped（既存警告19件）、変更PythonのRuffとdiff検査、
wheel/sdistのruntime全371ファイルのbyte照合が成功した。

**完全stepの補助交互比較と残るhost区間:** 既存train()を2モデル交互順、
20 warmup/101 pairs、独立2 processで測った。通常中央値は2.304201→2.277955 msと
2.220142→2.187477 ms、pair差の中央値は**.031355/.039651 ms**。
両processで121 updates後のParameterはbit一致。小さな改善を補足する結果であり、
GPU kernel時間の短縮率を完全stepへ読み替えない。2モデル常駐のメモリはdense比較に使わない。

別cProfileでは新Planのforward呼出し内が約1.11〜1.12 ms、
そのうちruntimeの_prepare約.32〜.34 ms、Linear build_context約.18〜.21 ms、
forward/backward合計の_ranges_from_layout_keys約.32 msだった。
3回のexecution_declaration約.24〜.28 ms、protect_tensor約.18〜.19 msも観測した。
包含する行が重なり、profiling overheadやCUDA driverの待ちも含み得るため、
純Python時間の内訳として加算しない。次の調査対象はLinear側の宣言/Contextの再準備と
layout範囲取得。Tensor・設定変更時の検証や現在のatomの再分類を無条件に省略する案ではない。

補助job=`l4job-7428cb8d7cfd487f903f23bb7435637c`、snapshot SHA256=
`e7a7e1557416983c55c6f70455cb6de72b6fb9b2de981b668d15608412566127`、
result archive SHA256=`20941f1d76963c992ce9d17e2dc1eeea9d0828fc23a03ce5717d5b05deeaf0d1`。
両実験のCUDA source群をSHA照合し、最終runtime371ファイルも主測定snapshotと一致した。
raw pstats、source、binary、receipt、`paired-driver.py`は同じignored output配下へ保全した。
この変更の1024/MNIST、他GPU世代、CUDA Graph、sanitizerは未測定。

### Linear側の宣言共有とnative layout範囲計算（2026-10-30）

不採用の試作ではDispatcherにoptional `prepare_context(inputs)`を追加した。
標準CSTLinearの変更されていないcallbackとplain Tensorを確認し、一度読んだ現在の宣言と
Parameterを入力検証とContext構築で共有した。Contextのgrad/precision/deterministic等は
毎回取得し、入力TensorもContextも保存しない案だった。custom callback、関数codeの変更、
subclass、Tensor mode、未対応private APIでは従来の2 callbackへ戻した。最終sourceには残さない。

radius layoutは現在のatomから毎forward再分類する。CUDAでのforward/reverse key変換・
安定ソート・全ownerのlower_boundをnative呼出し1回へまとめる。ATenのstable sortは維持し、
boundary配列は生成せずthreadのregisterで計算する。空ownerのbegin=end、同じkey内の元の
atom順、保存したforward keyのbackwardでの再利用を変えない。CPU診断は従来のTorch式。
moments/H/G/coeffsとPolar/AdamWの数式は変更しない。

**宣言共有の試作は不採用:** runtime `3efb7634`で3条件を別process、
変更前→Contextのみ→両方→両方→Contextのみ→変更前の順に測定した。
N128/B4/A819/M8/seed41、explicit parallel Plan、FP32 IEEE、base1/merge2、amp-only、
10 warmup/101 samples。完全step中央値は変更前2.200744/2.165796 ms、
Contextのみ2.304337/2.309520 ms、両方2.051531/2.117500 ms。
Contextのみの平均中央値は約5.7%遅く、callbackの適合確認を増やす案は採用しない。
このoptional Dispatcher/CSTLinear入口と専用testは次のcommitで取り除き、
従来のvalidate_inputs→build_contextと各宣言検証を維持する。
試作はcommit `3efb7634`とjobのfrozen sourceから復元できる。

両方では約4.5%短縮したが、この差をnative単独の結果としない。
全条件のallocated/reservedは297984 / 2097152 B、独立Y/dX/全atom勾配・quadrature・
matched-state gate、関連595件が成功。3条件のParameter/Adam moment/step20 Tensorは
5 updatesでbit一致。GPU更新は3 kernels、snapshot copy1回、sync/item/DtoH0。
native layoutだけの最終sourceを別途測定する。

試作job=`l4job-6538791929604b268cd0a2734b4065ae`、snapshot SHA256=
`1233e5a9dbeba2d7e61b4cb813a588d4bafc5cbcbee87b3e4a1ba7132aaef357`、
result archive SHA256=`58eb727a80cfe225d4c2d3f226be04b4cc0660d7512ad4cdc18eb490b53f0ee6`。
source/result/receipt/traceと復元用driverはignored `output/linear-host-20261030/`。

**最終採用はnative layoutだけ:** runtime `88eb8008`で、通常の入力/宣言/Context検証は
1c232dcbのままにした。比較元も1c232dcb。両方で同じexplicit parallel Planを使い、
上記と同じN128/B4/A819/M8/seed41・FP32 IEEE・amp-only条件で測定した。
各fresh processで変更前→変更後→変更後→変更前、10 warmup/101 samples。

| 条件 | 完全step中央値2回 ms | allocated / reserved peak B |
| --- | ---: | ---: |
| 1c232dcb | 2.144717 / 2.123830 | 297984 / 2097152 |
| native layout | 2.011921 / 2.028915 | 297984 / 2097152 |

平均中央値は**2.134274→2.020418 ms、約5.3%短縮**。全4回のメモリ増加なし。
matched denseは.693543〜.729819 ms、allocated/reservedは17313280 / 23068672 Bで、
dense並みの完全stepには未達。eager allocator peakでありGraph/process総量ではない。
別traceのkernel数は30→25、合計約1.030→1.022 ms。moments/H/G・coeffs/site展開・
parameter勾配はほぼ横ばい。Python経由のlayout Tensor操作とlaunch/割当の入口を
減らした結果と整合するが、kernel時間を完全stepから差し引いて純Python費用としない。

関連584件成功。独立Y/dX/全atom勾配・quadrature・matched-state gate、
Parameter/Adam moment/step20 Tensorの5 updates bit一致が成功した。
空/偏った/混在key、同一key内のstable順序、最大510 ownerと4096 atom、
forward/reverse、invalid metadata、retained backward、別stream寿命を含む。
optimizerはprepared経路のまま、3 kernels、snapshot copy1回、sync/item/DtoH0。
最終CPU全体2598 passed / 5942 skipped（既存警告19件）、変更PythonのRuff/diff検査が成功。
runtime全372ファイルを最終測定snapshotへbyte照合した。CUDA source/binaryは両条件でSHA照合し、
試作の検証済みbinaryを再利用した。入力/精度/Polar/AdamW則は変更しない。

最終job=`l4job-87c20e61371d45ee910ed3d3795c1855`、snapshot SHA256=
`5deae9ae766e7247d4dff2a0ed85d0d23281e19f930fbb6d7303119acf0f2b85`、
result archive SHA256=`fa6790bf90d969cb47a44a808a4b5dfb332fcedca120e9c12177ec85448d1ba7`。
NVIDIA L4、Python3.13.16、Torch2.11.0+cu130/CUDA13.0、Triton3.6.0。
source/result/receipt/trace/state/binaryと`final-driver.py`、`final-verified-summary.json`は
同じignored output配下へ保全した。所有L4は停止済み。
この変更の1024/MNIST、他GPU世代、Graph/process peak、新規sanitizerは未測定。

### Optimizerのhook登録とgroup関数metadataの再利用の試作（2026-10-30）

試作では最後のCST pre-hookの登録を再利用し、各stepでuser pre-hookの後へ移した。
step固有callbackを終了時に消し、wrapper・古いhook辞書を弱参照にした。
別の試作ではgroup検証の関数metadataを更新programと同じhost C++ helperで一括確認した。
どちらもPolar/AdamW則、amp-only decay、設定変更時の検証とuser hook順序は維持した。

4条件の比較では、group一括化のoptimizer単独の平均中央値は.654720→.645784 ms、
完全stepは2.030540→2.031258 msで横ばいだった。両方でも上積みを確認できず、
group一括化とhelperの共通ディレクトリ移動は不採用。hookのみはoptimizer単独
.654720→.638258 msだが、完全stepは2.030540→2.067027 msとばらついた。
4条件すべてでParameter/Adam moment/step20 Tensorは5 updatesでbit一致。
N128/B4/A819/M8、FP32 IEEE、amp-only、parallel Planで10 warmup/101 full steps、
optimizer-onlyは20 warmup/101 updates。全条件のallocated/reservedは297984/2097152 B。
関連513件と独立Y/dX/全atom勾配・quadrature・matched-state gateが成功した。

比較job=`l4job-27b98dbe6b61436ea7913eb4ee1d9ee7`、snapshot SHA256=
`69145dcc8ecabbb6c0c3e4616cc387d700940c4448e4ea0095a7a7ea71343fbf`、
result archive SHA256=`3e275b3a838195dfb870093a7f3779bee8c1deca579f92e030c4432bbb73938e`。
最初のjob `l4job-e15662cf554a44ec88f755f2ad3aaeb2`はoptimizer-only scriptのPlan指定で失敗。
失敗原本も含めignored `output/optimizer-hook-20261030/`へ保全した。

hook単独の最終再測定（runtime `2cb70c60`）も通常測定では短縮を再現しなかった。
変更前→変更後→変更後→変更前のfresh processで、完全stepの平均中央値は
2.037018→2.041250 ms、optimizer単独は.646428→.647283 ms。
補助の交互ペア測定では約.032/.038 msの差があったが、通常のoptimizer単独と
完全stepで再現しなかったためhookの常設化も不採用とし、`f174cec2`で取り除いた。
現在のhook登録・group検証は`efe2984b`のまま。試作は
`e91b7b72`/`f6fef2b5`/`2cb70c60`とfrozen sourceから復元できる。

最終hook比較は関連217件・独立oracleが成功、20 state Tensorが5 updatesでbit一致。
補助ペア測定も121 updates後のParameter/Adam stateがbit一致した。
通常測定のallocated/reservedは297984/2097152 B、matched denseは
17313280/23068672 B。dense完全stepは.683〜.703 ms、optimizer単独は.103〜.106 ms。
GPU更新3 kernels、snapshot copy1回、sync/item/DtoH0。ペア測定の2モデル同居時の
メモリは比較に使わない。eager allocator peakでありGraph/process総量ではない。
job=`l4job-b28a5a647b4d43a6837d54320d9e8502`、snapshot SHA256=
`97cc28d1d81128a0f5c88fd13c13aa67783dc42e06986f1dda847cdd64366398`、
result archive SHA256=`17730c547c921a005bb4ed5dda10d88f1483a3d7bb69d6baf26512d0f9b4ec4f`。

### Optimizerの保存先metadata検証の一括化の試作（2026-10-30）

更新programでParameter・Adam stateの`data_ptr()`を個別のguard accessorから呼ぶ処理を、
任意のhost C++ helperで一括確認する。Tensorは弱参照のみで保持し、値やversionは読まない。
各Tensorの`data_ptr`が準備時と同じbuiltinで同じTensorに結び付いていることを確かめてから、
そのbuiltinのC入口で保存先metadataを取得する。独自accessorや未対応環境は従来の
個別accessorへ戻す。後からaccessorが変われば独自callbackを実行せずguardを失効させる。
shape/dtype/device/stride、設定変更、Adam前後の検証境界は維持する。
数式・CUDA kernel・GPU配列とその寿命を変えない。CPUの実native helperで保存先変更、
数値更新、弱参照の解放、accessor差し替えを確認済み。最初の個別builtin lookup版（9e49caf8）は完全step
2.025678→2.030235 ms、optimizer単独.658112→.664299 msで改善なし。
job=`l4job-e42aaa4ab1cf452886f746c71c9c2e82`、snapshot SHA256=
`757ff48a64e814f290c6beee8f7621dffa452ef4bb24bc885ca7a96236d06bf6`、
result SHA256=`4a4bb4bc85fbb4e1cc89f651eed052e23cfc9341a8e6774d5f5834252d164bc1`。

descriptorとinstance辞書を直接確認する最終版（cb81643b）は、同条件のfresh process
変更前→変更後→変更後→変更前で、完全step中央値2.009271/2.114675→1.972790/1.989867 ms、
optimizer単独.726479/.657449→.645377/.651079 ms。平均中央値は2.061973→1.981329 ms、
.691964→.648228 ms。ただしoptimizerの変更前1回が高いため、この6.3%を一般的な改善率と
しない。全条件のallocated/reservedは297984/2097152 B、GPU更新3 kernels、snapshot copy1、
sync/item/DtoH0。関連212件、独立Y/dX/全atom勾配・quadrature・matched-state gateと
20 state Tensorの5 updates bit一致が成功。最終sourceのCPU全体2598 passed/5950 skipped、
追加の実native helper検査23 passed/1 skipped。
job=`l4job-42981aba9ff54a529f5526e82172446d`、snapshot SHA256=
`621d52c56a8fa422014b6b02f58bc67759a4c3c0918fb1924e6c357709a72410`、
result SHA256=`1313e815d26f89bc601d0c25c55a6280ff5f181a21db551b5c75da4781375aa1`。
raw source/result/receipt/trace/stateは同じignored output配下へ保存した。

### CSTAdamWの単一stepへの構造変更（2026-10-30）

標準AdamW用にCSTAdamWを追加する。1つのOptimizerがgroups/state/hooks/scheduler接続を持ち、
CST準備→PyTorch functional AdamW→CST retraction/transportをstep本体で実行する。
一時base pre-hook、closure replay、base/wrapper間のstate同期は使わない。
既存CSTOptimizerは任意optimizerとの互換経路として残す。
atom準備とfinishの数式を_coordinate_step.pyへ抽出して両方で共有する。
構造・所有の全model走査は構築/add-group/loadへ寄せ、毎stepは現在のgroup scalarと
Parameterの対応を確認する。Polar更新、amp-only decay、snapshot寿命、
現在のKernel/Geometry/adapter callbackを尊重する。PyTorchのstate-dict hookでmanifestと
atom-owned stateの再接続を扱う。native checkpointと明示的な旧wrapper checkpoint移行を分ける。
PyTorch 2.6+、check_finite=Falseをnative既定とする。
AMPではGradScalerの通常unscale/skip経路を使い、overflowならCSTを含む全stepを止める。
実測runtime `31169b9c`。N128/B4/A819/M8/seed41、base1/merge2、FP32 IEEE、
explicit parallel Plan、raw decay0/amp decay.01、fused AdamW、check_finite=Falseを共通にした。
比較元は公開済み`efe2984b`のwrapper。新経路は上記native保存先検証も含むため、
差のすべてを二重wrapper除去だけへ帰属させない。変更前→native→native→変更前の
fresh processで各10 warmup/101 complete steps、別に20 warmup/101 optimizer-only updates。

| 条件 | 完全step中央値2回 ms | optimizer単独中央値2回 ms | allocated / reserved peak B |
| --- | ---: | ---: | ---: |
| efe2984b wrapper | 2.041999 / 2.016069 | .625522 / .623704 | 297984 / 2097152 |
| native CSTAdamW | 1.888717 / 1.923996 | .527303 / .529000 | 297984 / 2097152 |

平均中央値は完全step **2.029034→1.906357 ms（約6.0%短縮）**、optimizer単独
**.624613→.528152 ms（約15.4%短縮）**。matched denseは完全step.672〜.703 ms、
optimizer単独.102〜.104 msで、dense並みには未達。dense allocator peakは
17313280/23068672 B。CSTのGPU配列の追加なし。通常eagerのallocator peakであり、
Graph/process全体のメモリではない。

関連247件が成功。CPUのforeach/AMSGrad/maximize/lr0/grad=None/通常Parameter混在、
closure・step/state-dict hooks・scheduler・live lr/decay・checkpoint再開/明示移行、
CUDA GradScalerのfused/unfusedと明示/自動unscale・overflow全step skipを含む。
独立Y/dX/全atom勾配・quadrature・matched-state gateが成功。別5 updatesの
Parameter/Adam moment/step20 Tensorがbit一致。別traceでnativeのGPU更新3 kernels、
合計約9.50 us、snapshot copy1回。sync/item/DtoH0、prepared programの再利用も確認した。
新経路にgroup-validation cacheはなく、検証のための全model走査もない。
CPU全体2629 passed/5954 skipped（既存警告19件）、Ruff/diff検査が成功。
wheel/sdistのruntime全374ファイルを現在のsourceとbyte照合した。
CUDA source/binaryを両条件でSHA照合し、検証済みbinaryを再利用した。

job=`l4job-2d33d531877844bcb3b13c851c070292`、snapshot SHA256=
`9dfc7fa53d3d50ea2a9f84a556e11e9cdfb5e4123bab88e72116c737d95386eb`、
result archive SHA256=`81a8888537ddb08728b82b396921a4b9236137fb0505b209d560fa292290ce23`。
最初のjob `l4job-db026f8564c84d9e9906950deb2fdd32`はAMPテストのupdate Planが
CPU参照のままで4件失敗し、速度測定前に停止した。テストへ対応CUDA Planを明示して修正した。
失敗原本も含め、source/result/receipt/trace/state・driverと復元用sourceは
ignored `output/optimizer-hook-20261030/`へ保全した。所有L4は停止済み。
このnative optimizerの1024/MNIST、他GPU世代、Torch 2.6〜2.10、Graph/process peak、
新規sanitizerは未測定。標準ライフサイクルの継承をすべてのPyTorch拡張への対応と
同一視しない。数値比較は上記有限入力・固定条件の結果であり任意学習軌道の保証ではない。

### Native optimizerのmetadata費用の診断と共有化の試作（2026-10-30）

b50901b8のnative CSTAdamWを同じN128/B4/A819/M8条件で診断した。
未計装optimizer単独中央値は.640895 ms。別cProfileでは_matches/runに含まれる
大きなlive guardと2回のsnapshot入力検証が目立つ。cProfileによる負荷とinclusive時間の
重複があるため、各行を足して未計装時間の内訳とはしない。
job=`l4job-56791d726c4d47998a80f4f9c529ab18`、snapshot SHA256=
`9f3ac5ce57094487ba510d4919e6afcf0014910b998f10f2ff190947a303dffd`、
result SHA256=`9652a4be043ecfb48dc3a76389d3c80b6d511c49ccc57a31beda8a081a7b9d4a`。
source/result/receipt/pstatsはignored `output/optimizer-native-profile-20261030/`に保全する。

試作Aはcallback classの不変なMRO tupleをhost C++ helperが保持して一括確認し、
各baseの重複したtype/id guardを減らす。mutable class namespaceのversion確認は残す。
固定state側とcallback側で同じclass/MRO/namespaceを確認する部分も共有する。
MRO tupleを強参照で保持し、__bases__変更後のアドレス再利用による誤判定を防ぐ。
Tensorや入力snapshotは保持しない。未対応native helperは従来のguardへ戻す。

試作Bはsnapshotのshape/stride/dtype/device/type/detachednessをnative TensorMatchで
確認する。現在のParameter・Kernel設定・callbackは従来のlive guardで確認済みの範囲だけを
使う。Python scalarのlr/decayとPolar対応・non-aliasは引き続き検証し、変更前後の
2つの境界を維持する。未対応APIは元のvalidator、異なるstride等は元の検証を通した上で
通常dispatchへ戻す。旧snapshotやGPU配列を常駐させない。

実native helperを使ったCPU検査92 passed/45 skipped。
class namespace/MRO変更、snapshotのshape/dtype/stride/sparse/alias/requires_grad、
lr/decayの不正値と解放を含む。Aのみ/Bのみ/両方/変更前をfresh processで比較し、
完全step・optimizer単独・memory・state一致で採否を決める。

runtime `0eeb2a9e`の4条件比較では、全条件を同じnative CSTAdamW・N128/B4/A819/M8、
FP32 IEEE・explicit parallel Plan・base1/merge2・amp-only .01・fused AdamWにした。
変更前→A→B→両方→両方→B→A→変更前のfresh processで、
各10 warmup/101 full steps、別に20 warmup/101 optimizer-only updates。

| 条件 | 完全stepの平均中央値 ms | optimizer単独の平均中央値 ms | allocated / reserved peak B |
| --- | ---: | ---: | ---: |
| b50901b8 | 1.980261 | .541695 | 297984 / 2097152 |
| A: MRO共有のみ | 1.939362 | .532462 | 297984 / 2097152 |
| B: snapshot検証のみ | 1.950688 | .534873 | 297984 / 2097152 |
| 両方 | 1.961873 | .516395 | 297984 / 2097152 |

267件のGPU関連検査、独立Y/dX/全atom勾配・quadrature・matched-state gateが成功。
4条件のParameter/Adam moment/step20 Tensorは5 updatesでbit一致。
job=`l4job-72648a62e07840088186a485d4403058`、snapshot SHA256=
`9816b7d9caa96d51e550f0cf8aaf4a4a289fc8f57016a8b20b465a9c87a427bc`、
result SHA256=`59b84da1552a5948397bd1462bfb58aee4431b1f72ff334751d3932a7dd749f5`。

両方を入れた同じruntime全374ファイルをbyte照合して最終再測定した。
変更前→両方→両方→変更前のfresh process、warmup/samplesは同じ。
完全step中央値は変更前1.953512/1.907893→変更後1.939686/1.894630 ms、
optimizer単独は.538338/.534543→.517331/.508414 ms。
平均中央値は **1.930703→1.917158 ms（約.7%短縮）**、
**.536441→.512873 ms（約4.4%短縮）**。改善を再現したため両方を採用する。
メモリは全4回で297984/2097152 Bのまま。matched denseは完全step.697〜.714 ms、
optimizer単独.102〜.105 ms、allocator peak17313280/23068672 B。
いずれも通常eagerのallocator peakであり、Graph/process全体のメモリではない。

最終job=`l4job-ce094532b18f4e6094bf4cc785468080`、snapshot SHA256=
`a172977347484bccfabafa689ba1649298fb1215c7a2b6f6289b00333b2eebc0`、
result SHA256=`a536e95aa5fd33ae47bec13bae53d0e9bb564a371ebe10e1e2e7610b939ec4bb`。
新しいsource変更はなく267件の検査を再利用し、各runnerの独立oracleと
別5 updatesの20 state Tensor bit一致を再確認した。
別traceのGPU更新3 kernels・合計約9.54 us、snapshot copy1回、sync/item/DtoH0。
全CPU suite2644 passed/5959 skipped（既存警告19件）、Ruff/diffとwheel/sdist照合が成功。
CUDA source/binaryを各条件で照合し、従来の検証済みbinaryを再利用した。

最終の別instrumented stepでは、大きなprogram guardが2回で約.133 ms、
PyTorch AdamW proposalが約.089 ms、snapshot入力検証2回が約.028 ms、
compiled launch入口が約.037 ms。これはtimer負荷を含む診断で、inclusive中央値は足さない。
各stepのpreflight/run成功と同じprogramの再利用を確認し、計装でgeneric経路へ落ちた
値ではないことを確かめた。単純なgroup走査は約.008 msであり主因ではない。
今後の大幅削減には、残るlive guard・snapshot・launchを含む実行方式の再設計が必要になる。
optimizerはここで区切り、CSTLinear側の完全step費用へ対象を移す。
この変更の1024/MNIST、他GPU世代、Torch 2.6〜2.10、Graph/process peak、新規sanitizerは未測定。
source/result/receipt/trace/state/pstatsと4条件・最終driverは同じignored output配下へ保存し、
所有L4は停止済み。更新則・精度・階層タイルとamp-only decayは変えない。

### coeffs生成とsite展開の分離計測（2026-10-30）

対象をoptimizerからCSTLinearへ移し、同じN128/B4/A819/M8/seed41、FP32 IEEE、
base1/merge2、direct levels2、32-site CTA、explicit `radius-block-local-parallel`、
native CSTAdamW（fused AdamW、raw decay0/amp decay.01、check_finite=False）で診断する。
階層・profile・Polar更新則は変えず、入力moments/Hと逆向きmoments/Gは各方向1回だけ作る。
通常のPlan・dispatcherは変更しない。

```text
既存のshared_expand_kernel（Y側/dX側とも同じ構成）
  completed H / G + atom ranges
    → block_local_coefficient_tile → shared coeffs
    → CTA barrier
    → hybrid_site → Y / dX
```

ここでいうcoeffs生成は、atomのFourier係数を求めるだけではなく、
`amplitude × completed response`を掛けてタイルの17係数へ集約する処理である。
`block_local.cuh`の`block_local_coefficient_tile`は各nodeの寄与atomに対して、
Polar radiusの復元、中心のwrap・分類、profile係数とradius微分、位相の計算、
sharedのpadded列への書き込み、atom順序を保った加算を行う。
`common.cuh`の`profile_coefficients_m8`は`__noinline__`であり、C側で使わない
radius微分も求める。今回はその費用も含む現行実装を測る。
`hybrid.cuh`の`hybrid_site`は、retained bankではcoeffsをセル積分した基底で展開し、
direct bankではatomのprofileをsiteごとに評価する。後者が混ざる条件では
「coeffsからの展開」だけの時間とは呼ばず、site/direct展開とする。

計測用の`expansion_probe.cuh`には以下の入口を追加した。通常のforward/backwardからは
呼ばれず、global C・timestamp用の配列も通常経路には追加しない。

- 同じ既存kernelをcaller-owned出力へ反復起動する未計装の基準。
- 同じCTA内でsetup/coeffs/siteの境界を記録する融合probe。
- 同じCTA所有・node重複・atom順序でCを作り、必要なslotだけglobalへ書き出すkernel。
- そのslotをsharedへ読み戻して同じsite/direct本体を実行するkernel。

分離kernelの時間はglobalへの書き出し/読み戻しも含む。CTA単位の時刻は足し合わせず、
CTA中央値・最後に完了したCTA・device全体のspanを別々に報告する。
融合probeには最後のbarrierと時刻記録が増えるため、未計装版との時間と
register/local/shared使用量も併記する。PTX `%globaltimer`はこのL4でのtool用計測であり、
他GPUで同じ性質を保証しない（[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)）。

benchmark側の`SplitExpansionNative`だけがprojection→C→siteの分離を呼ぶ。
H/Gを再計算・コピーせず、Cは同一streamのconsumer投入後に解放する。
公開Planへの登録や高速化候補の採用ではなく、追加起動・一時メモリも含む診断である。
同じCTAのCを保持するので、粗いnodeをCTA間で1回だけ作る別方式には変えていない。

実L4（Torch2.11.0+cu130/CUDA13.0、Python3.13.16、Triton3.6.0）で検査後、
変更前融合→追加後融合→分割→分割→追加後融合→変更前融合のfresh process順に、
各10 warmup/101 complete stepsを実行した。各回は独立Y/dX/全atom勾配、
quadratureとmatched-state検査も通す。分割版は上記benchmark注入を明示した診断であり、
通常Planの性能観測としてDBやdispatcherへ提出しない。

| 条件 | 完全step中央値2回 ms | 平均中央値 ms | allocated / reserved peak B |
| --- | ---: | ---: | ---: |
| 1a06c11dの変更前融合 | 1.982342 / 1.906974 | 1.944658 | 297984 / 2097152 |
| 計測入口追加後・通常融合 | 1.923351 / 1.965219 | 1.944285 | 297984 / 2097152 |
| 診断用C/site分割 | 2.225497 / 2.214679 | 2.220088 | 560128 / 2097152 |
| matched dense（全6回） | .698692〜.728067 | — | 17313280 / 23068672 |

追加後の通常融合に速度・メモリの実質的な変化は認めない。
分割は約14.2%遅く、通常経路には採用しない。これはC export/reload、追加launch、
host proxyの追加処理を含む差であり、kernel時間の差だけに帰属させない。
メモリは通常融合.284MiB、分割.534MiB、dense16.511MiBのallocated peakで、
全回でallocated/reservedともdense以下。model/optimizer/activation/workspaceを含む
通常eagerのPyTorch allocatorピークであり、bare weight量、Graph peak、
GPUプロセス全体の使用量を示す数値ではない。

別microprobeは初期状態と111回の実更新後を固定し、各方向のmoments/responseを
1回だけ作って再利用した。全5 modeを4 warmup後に64反復×31 samplesのCUDA Eventで測り、
別7反復のGPU traceと、9回のCTA時刻記録も保存した。表は1反復あたりの中央値us。

| 状態・方向 | 未計装融合全体 | 時刻計装融合 | C生成＋export | reload＋site展開 | split pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初期 Y | 213.903 | 222.160 | 222.048 | 14.640 | 237.072 |
| 初期 dX | 265.958 | 272.985 | 272.107 | 14.705 | 286.959 |
| 111更新後 Y | 213.152 | 221.355 | 221.032 | 14.715 | 236.194 |
| 111更新後 dX | 266.134 | 272.791 | 272.146 | 14.704 | 287.153 |

GPU traceも同じ傾向を示した。111更新後の融合probeで最後に完了したCTAの
phase中央値は、Yがsetup1.184/C211.936/site6.656us、
dXがsetup1.152/C262.400/site6.720usだった。
CTAごとの中央値とwall timeは異なるため、全CTAの時間は合計しない。
Cとsiteの計装区間の約97%をC生成が占め、**この融合kernel内の主因はsite展開ではなくC生成**。
profile係数の生成・微分・radius復元・shared集約の個別の寄与率までは分離していない。

| kernel | registers / thread | local B / thread（compiler属性） |
| --- | ---: | ---: |
| 未計装融合 | 152 | 1104 |
| 時刻計装融合 | 162 | 3408 |
| C＋export | 158 | 2256 |
| reload＋site | 125 | 840 |

いずれも128 threads/CTA、dynamic shared35088B（C17136B＋warp列17952B）、
static shared属性16B。計装融合は未計装より約2.5〜3.9%遅い。
したがって上記Cの時刻を未計装kernelの厳密な内訳とは断言しない。
分割probeのCは`[B,4,126×17]`のFP64で274176B、mask64B、出力2048B、
timestamp512B、payload計276800B。これは診断用の一時容量であり、
通常経路にはない。別probeの最大allocatedは588288B/reserved2097152Bで、dense以下。
Torch allocatorは上表のcompiler local領域を個別Tensorとして数えるものではない。

分布は均等な階層混合ではなかった。GPUの実分類では、入力・出力とも初期と111更新後の
819 atomsすべてがlevel2/T4/K3/P12に属し、direct bankは0だった。
runnerでの初期radiusは3.985522〜4.000000、111更新後は3.942180〜3.999791。
各方向のC生成で`4 batch ×819 atoms ×3 tiles = 9828`回のprofile評価がある。
B4を共有するresponse計算と異なり、C側はbatch行ごと・寄与nodeごとに同じatomの
profile/radiusを再評価している。

初期中心から、GPUで確認済みのK3/T4分類のfirst tileをCPUでも再構成した。
Yのfirst-tile atom数は23〜26、dXは16〜35。
`block_local_coefficient_tile`は各rangeを32 atomsずつ処理するため、
1 batchのnode全体のwarp chunk数はY96→dX111、各CTAで最大のwarp担当は
Y6→dX8 chunksとなる。dXの最終CTAも主にその重い区間に対応した。
これは同じ9828評価でも逆向きの負荷が揃わない機構であり、時間差の全量を
この偏りだけで説明した比較実験ではない。
次の候補は、Cで使わないradius微分の計算除去、profile/radiusの再評価削減、
32個の境界で増えるchunk処理の配分改善。階層と完成responseの再利用は維持する。
この分布での結果をMNIST/1024や広いradius混合へ一般化しない。

実GPUホスト上の関連検査171件が成功し、混合階層・direct・torus・空atom・retained backwardを含む。
変更前/追加後通常融合/分割の別5 updatesでY/dX/全atom勾配/Parameter/Adam stateの
35 Tensorがbit一致。probe前後のatoms/moments/H/GのSHAも不変だった。
CPU全体2671 passed/5947 skipped（既存警告19件）、Ruff/diff検査、
wheel/sdistのruntime全375ファイルのbyte照合が成功した。
最初のjobでは新規テストがstart0/M8/N128の既存48KiB上限を超える設定を指定して
4件失敗した。対応内のstart1へテストを修正し、上記171件と実測を再実行した。
製品kernelの数式・処理・上限は変更していない。

成功job=`l4job-fbbd0228e8dc4da6a9934970708828d2`、snapshot SHA256=
`614a1c77bc4218a24fc427d99a87310ca597caf4410f9b20a4e8f31e14188a29`、
result archive SHA256=`b723fdc18450a4996618683a9e2d792fd8567094bcaa97ac21f9900e07cefb80`。
CUDA binary=`torchcst_radius_cell_mass_a7525eeb93a3`（初回jobでbuild成功後、
同一native source digestとbinary SHAを照合して再利用）。
初回job=`l4job-c27a7e21ccc7459d807b7bf7c0e18e2a`、snapshot SHA256=
`1a3c29bee6aa6ddea06b43b535248c0032054e12f66205fd08d591b29e422732`。
再現用driver/probe worker、2 jobのsource/result/receipt、GPU traces、
state tensors、初期負荷カウントと集計JSONをignored
`output/radius-expansion-20261030/`へ保全した。

### block内のatom profile共有（2026-10-30）

coeffs生成の重複評価を減らすため、明示Plan
`BLOCK_LOCAL_CACHED_PARALLEL`（benchmark alias
`radius-block-local-cached-parallel`、schedule=`block_local_cached_parallel`）を追加する。
既存のPlanと自動selectorは変更しない。入力moments/Hと逆向きmoments/Gは従来の
parallel projectionで各方向1回だけ作り、atom勾配にも同じ完成済みcacheを渡す。
変更するのは、完成responseからcoeffs/siteへ進むdestination kernelである。

```text
1 CTA = 1 batch行 × 1出力区間（今回32 sites）
  active bankごとに、区間へ寄与し得るfirst-tile ownerを列挙
  周期wrapを重複しない最大2区間に分ける
  各区間を最大128 atomsずつ処理:
    threadごとに別atomのradius・分類・profile係数を1回だけ計算
    sharedへprofile係数・wrapped center・amp×response・first tileを公開
    CTA同期
    retained bank: 担当warpが同じcacheから各寄与タイルのcoeffsを集約
    direct bank: 担当siteのthreadが同じcacheからセル質量を積算
    CTA同期 → shared cacheを次のchunkに再利用
  retained coeffsを展開し、direct寄与と合わせてY/dXを保存
```

`cached_coefficients.cuh`のproducerはradiusをregisterで一度復元し、そこから求めた
profileをsharedへ置く。consumerはradius/profileを再計算しない。
同じatomでも入力・出力で異なるradius bounds/周期を持てるため、
方向を超えてprofileが同一だとは仮定しない。今回もbatch行が異なるCTA間の共有はしない。
profile式とradius微分を含む既存の`profile_coefficients`自体は変更せず、共有効果を分離する。

bankのtile数をN、Kをperiodのtile倍数、出力区間と重なるdestination tile数をDとすると、
必要なfirst-tile ownerは連続した`min(N,D+K−1)`個である。
KがNより大きい場合も全ownerを一度だけ読み、同じatomの複数のunwrapped像へcacheを使う。
atomのParameter行は並べ替えず、既存のrows/offsetsだけで必要範囲を取る。

shared cacheはatom数に比例して保持せず、128/64/32 slotsの中から
48KiBへ収まる容量とconsumer warp数をhostの固定metadataで決める。
今回M8/start2では、coeffs17136B、4 warpの列17952B、
128 slotsのprofile/center/value11264B、first tile512B、合計46864B。
通常経路にはglobalのprofile表・packed atom・counter配列を追加しない。
検査専用の別specializationだけが`[batch, destination CTA, atom]`のproducer回数を数える。

chunkごとに寄与を集約するため、異なるfirst-tile範囲のFP64加算順が既存kernelと
異なる場合がある。数学的な和とセル質量の定義は同じであり、独立Y/dX/全atom勾配、
retained backward、実更新のstate比較で誤差を検査する。

実L4、Torch2.11.0+cu130/CUDA13.0、Python3.13.16、Triton3.6.0で、前節と同じ
N128/B4/A819/M8/seed41、base1/merge2、direct levels2、32-site CTA、FP32 IEEE、
native CSTAdamW（fused AdamW、raw decay0/amp decay.01、check_finite=False）を測った。
比較元は`bbf0cb3a`の`radius-block-local-parallel`、候補は新しいcached Plan。
baseline→cached→cached→baselineのfresh process順に各10 warmup/101 samplesを使い、
各runnerで独立Y/dX/全atom勾配・quadrature・matched-state検査を通した。

| 条件 | 完全step中央値2回 ms | 平均中央値 ms | allocated / reserved peak B |
| --- | ---: | ---: | ---: |
| 変更前parallel | 1.954870 / 1.953213 | 1.954042 | 297984 / 2097152 |
| shared profile cache | 1.984635 / 1.962660 | 1.973647 | 297984 / 2097152 |
| matched dense（4回） | .705771〜.727790 | — | 17313280 / 23068672 |

今回の完全stepでは約1.0%遅く、**完全stepの高速化は確認できなかった**。
候補は明示Planとして比較可能に残し、既存Planの置換や自動採用は行わない。
allocated/reservedは全回同じでdense以下。通常eagerのPyTorch allocator peakであり、
GraphやGPUプロセス全体のpeakではない。新しいglobal Tensorは通常経路に追加していない。

一方、111更新と別profile stepの後の同じ状態で、同一candidate binary内の
旧/新destination kernelを11回ずつtraceした結果は以下だった。
projectionは両方で同じparallel方式を使い、その時間を表には含めない。

| 方向 | 旧coeffs＋site us | cached coeffs＋site us | profile生成回数 旧→新 |
| --- | ---: | ---: | ---: |
| Y | 213.057 | 129.600 | 9828→4072 |
| dX | 266.144 | 130.209 | 9828→4092 |

旧の生成回数は、この状態でもGPUで分類された819 atomsすべてがT4/K3のため
`B×A×K`から求める。新の回数は別のcounted specializationで実際に数えた。
`[batch, CTA, atom]`の最大countは両方向とも1で、周期の複数像でもproducerを
重複させていない。1 batch行の各出力CTAでYは252/257/257/252 atoms、
dXは257/247/264/255 atomsをcacheした。counted出力と旧出力の最大絶対差は両方向0。
GPU kernelだけではY約39%、dX約51%短縮したが、これは完全stepの短縮率ではない。

別の1 complete-step traceでも、destination kernel合計479.809→260.097us、
全25 kernelsのbusy時間1022.528→802.785usを確認した。
Z構築251.712→251.328us、responses56.896→56.896us、
atom勾配128.352→128.960usで、再利用するprojection/atom勾配はほぼ同じ。
GPU上ではZ構築とdestinationが同程度の費用になった。
新kernelは140 registers/thread、shared46864B、旧は152 registers/thread、
shared属性35104B（dynamic35088B＋static16B）。128 threads/CTA、grid4×4は同じ。

計装traceにはkernel間のCPU投入待ちがあるが、profilerの大きな追加費用もある。
そのspanやkernel busy時間を未計装の完全stepから引いて純CPU時間とはしない。
**GPUの短縮が完全stepへ現れない主因の特定は未完了**であり、
CPU投入・dispatch・検証等を未計装条件と対応させて別途調べる必要がある。
共有化だけで学習stepが高速になったとは報告しない。

実GPUホスト上の関連190件が成功。混合階層、direct/all-direct、torusの繰り返し像、
空atom、FP32/FP64、merge/direct moment、batch端、retained backwardを含む。
新cached経路のVJPはM3/M8/M16、既存Planの回帰検査はM1〜16を含む。
別5 updatesのY/dX/全atom勾配/Parameter/Adam state35 Tensorは全てbit一致し、
今回のFP32条件で最大差0だった。FP64の加算順差が一般に0になるという保証ではない。
CPU全体2672 passed/5965 skipped（既存警告19件）、Ruff/diffが成功。
wheel/sdistの現在runtime全376ファイルをbyte照合した。

job=`l4job-5a667546ad3d44b4a84ef93edea5c339`、snapshot SHA256=
`f73996d069feee4c86be834fe44ddda9333c6be934b0427d9f423cb8e3af4fb2`、
result archive SHA256=`dee61ed4064d909100749b712ca938a38fb29f2dd7d5cac98863cfba588be0ec`。
新CUDA binary=`torchcst_radius_cell_mass_b1d35927a8ff`、binary SHA256=
`7fd5cf591cf8fe45d47fc08f2f6f9ef353fb64af947a25b628246960810f65a8`。
旧binaryもnative source digestとSHAを照合して再利用した。
再現driver、source/result/receipt、traces、counter結果、state tensors、
CPU/build logsはignored `output/radius-cache-20261030/`へ保全する。
1024/MNIST、他GPU/Torch版、新規Graph/sanitizer/process peakは今回未検証。

### N128完全stepのCPU呼び出し・同期・dispatch診断（2026-10-30）

前節の「GPUは短縮したが完全stepは約1%遅い」を、runtime `9412cb7a`の同じsourceと
CUDA binaryで再検査した。製品code・検証・callback・Polar則は変更していない。
N128/B4/A819/M8/seed41、FP32 IEEE、base1/merge2、direct levels2、32-site CTA、
native CSTAdamW（fused AdamW、raw decay0/amp decay.01、check_finite=False）を維持する。
L4、Torch2.11.0+cu130/CUDA13.0、Python3.13.16、Triton3.6.0。

primaryは旧/共有/共有/旧/共有/旧/旧/共有の8 fresh processesで、各10 warmup/101 samples。
両Planとも同じ現在のPython sourceと検証済みbinary
`torchcst_radius_cell_mass_b1d35927a8ff`を使い、source/binary SHAを照合した。
各runnerの独立Y/dX/全atom勾配、quadrature、matched-state検査が成功した。

| 方式 | 完全step中央値4回 ms | 平均中央値 ms |
| --- | ---: | ---: |
| 旧parallel | 1.918028 / 1.873163 / 1.969099 / 1.865652 | 1.906485 |
| shared profile cache | 1.899762 / 1.907743 / 1.911712 / 1.903319 | 1.905634 |

平均中央値差は約−.000852ms（−.045%）で実質同じ。前回の約1%遅延は再現せず、
定常的な性能回帰とは確認できない。一方、この通常のfresh-process測定で
大きな完全step改善が得られていないという判断も変わらない。
matched denseは.676165〜.708533ms。

次に2 fresh processesで、それぞれ旧/共有の2 modelを同居させ、20 warmup後に
101 pairsをAB/BA交互で測った。2 process目は最初の順序も逆にする。
両modelの実更新数は揃え、測定中には内部timerやfunction差し替えを入れず、
step前後のCUDA同期とhost時計だけを使う。2 model同居はtimingの補助実験であり、
そのmemoryをsingle-modelのprimary memoryと混ぜない。

| process・方式 | 完全step中央値 ms | train()のhost呼び出し中央値 ms | 呼び出し後の同期中央値 ms |
| --- | ---: | ---: | ---: |
| 0 旧 | 1.867235 | 1.839948 | .026430 |
| 0 共有 | 1.836529 | 1.810183 | .026461 |
| 1 旧 | 1.913601 | 1.886928 | .026497 |
| 1 共有 | 1.888954 | 1.860597 | .026544 |

同じpair内の共有−旧の中央値は−.031757/−.027195msで、両processとも101中88 pairsで
共有版が短かった。小さな改善が見えるが、前記fresh-process primaryの大きな速度改善を
立証するものではない。host呼び出し中にもGPUは非同期に進むため、
**host時間とGPU kernel時間を足して完全stepにはしない**。
呼び出し後の同期時間もCUDA APIの費用を含み、純粋なGPU残り時間とは限らない。

別の軽い区間計測は51 stepsずつ、zero-grad/forward-loss/backward/optimizerの
CPU呼び出し境界だけに時計を入れ、途中のCUDA同期は追加しなかった。
共有版の各区間の中央値（2 processes）は以下。これもprimaryの内訳へ機械的に加算しない。

| host側の区間 | process0 / process1 ms |
| --- | ---: |
| zero_grad・x.grad解放 | .046897 / .047068 |
| forward・loss投入 | .727357 / .732309 |
| backward呼び出し | .484367 / .566106 |
| optimizer呼び出し | .534684 / .544752 |
| 最後の同期 | .025887 / .025924 |

**同期の有無**は、別のphase付きTorch profiler traceで確認した。
2 processes×8 train scopes＝16 scopesのすべてで、内部のCUDA同期API、
`aten::item`/`aten::_local_scalar_dense`、DtoHコピー、cudaMalloc/cudaFreeは0。
意図的なstep末尾の`boundary_sync`はtrain scopeの外に置いて区別した。
毎step25 kernels、capture状態の問い合わせ5回を確認した。
`cudaStreamIsCapturing`等の問い合わせはGPU完了待ちではない。
`executor.py::prepare_layout_checked`の`error.item()`は診断専用で、
通常の非同期classificationからは呼ばれていない。

**CPUの場所**はcProfileでcall数を確認し、さらに独立L4 runtimeの2 fresh processesで
Python3.13の[sys.monitoring](https://docs.python.org/3.13/library/sys.monitoring.html)を使って
対象code objectを1つずつ局所計測した。functionや`__code__`を交換せず、
PY_START/PY_RETURNだけを観測する。各対象の前に同じ20更新後のParameter/Adam状態へ
内容を戻し、10 warm monitored steps＋51 recorded stepsを使う。
prepared object/programのidentity、miss数不変を毎対象で確認した。
下表は共有版の1 stepあたりの呼び出し時間合計の中央値（2 runsの範囲、us）。
local timerの負荷とruntime間のCPU実行時間差を含む目安であり、
primaryの1.906msの厳密な内訳ではない。親子のinclusive値は足さない。

| 場所 | calls/step | 局所計測 us | codeで行うこと |
| --- | ---: | ---: | --- |
| `CSTLinear.forward` | 1 | 676.88〜687.74 | flatten、dispatch、出力reshape/保護。下記forward処理を含む |
| `Dispatcher._prepare` | 1 | 147.05〜151.06 | 入力検証、Context構築、配置・Registry・cache照合 |
| `CSTLinear.build_context` | 1 | 63.62〜64.34 | 現在の宣言とTensor metadataからContextを作る。_prepare内 |
| `operator_declaration` | 3 | 91.85〜94.88 | 入力検証・Context・実行入口で固定状態guardを確認 |
| `AlgorithmState.is_current` | 1 | 24.40〜24.55 | owner/layout/storage/state対応を確認 |
| `Registry.validate_plan` | 2 | 9.56〜9.99 | 登録先とrecipe型/内容の確認 |
| `RadiusCellMassAlgorithm.validate_recipe` | 4 | 10.88〜11.95 | Registry2回、Algorithm.run、radius_linear。Registryと重複する包含値 |
| `radius_linear` | 1 | 250.44〜277.74 | metadata検証からforward native起動まで。layout処理を含む |
| `_ranges_from_layout_keys` | 2 | 179.18〜183.79 | forward/reverseのnative layout_ranges。stable ATen sortとoffset kernelを起動 |
| `_classify_layout_keys` | 1 | 37.28〜37.95 | deviceで現在atomのownerを分類して非同期検証 |
| `AxisPlan.native_config` | 8 | 13.51〜13.86 | 2軸configを各native入口へ渡すlist生成 |
| `prepare_atom_steps` | 1 | 185.31〜188.89 | 更新guard、snapshot、入力検証。下位の_matches/validateを含む |
| `finish_atom_steps` | 1 | 158.80〜178.16 | 更新直前の確認、compiled Polar launch等 |
| optimizer `_matches` | 2 | 111.07〜116.30 | preflight/runでlive設定と実行条件を確認。prepare/finish内 |
| optimizer `validate_inputs` | 2 | 27.63〜27.74 | snapshot metadata、scalar、non-alias確認。prepare/finish内 |
| `_adamw_proposal` | 1 | 93.92〜97.52 | PyTorch functional AdamWへのhost入口 |

`operator_declaration`の3回は`validate_linear_inputs`、`CSTLinear.build_context`、
`linear_execution`から来る。宣言を作り直す3回ではなく、現行固定状態guardを確認して
同じcacheを返す3回である。`layout_ranges`の約.18msもGPU sortの実行時間ではなく、
ATen/allocator/launchを含むCPU呼び出し時間である。GPU活動との重なりを含む。

**Planの探索や階層metadataの作り直しが主因ではない。**
交互測定では両modelともprepared dispatchのmissは初回1のまま、hit19→120。
同じalgorithm stateとcompiled optimizer programを使い続けた。
cProfileのsteady区間でも`Dispatcher.select`/`select_plan`やAlgorithm.prepareは0回。
RootGuardManager、native function metadata、CompiledPolarLaunchの使用も確認し、
generic fallbackへ落ちた計測ではない。
共有版で追加されたshared容量検証によりactive_nodesは2→4 calls/stepで、
局所計測は旧約8.1us→共有約12.2us。小さい追加費用はあるが、
GPUの約220us削減と同じ規模の新しいCPU処理が増えたわけではない。

全面cProfileはsnapshot validate_inputsを約.38〜.39ms/stepと表示したが、
局所計測は約.028msだった。このcProfile値を未計装の純CPU費用とは読まない。
profilerありのtrain scopeも約2.5〜10msへ増え、primaryとは異なる。
呼び出し回数と経路の確認、局所timer、未計装完全stepを使い分ける。

結論は、前回の1%遅延は再現せず、この128条件の完全stepは主にhost側の
検証・投入・autograd等の経路に制約されている、というもの。
GPUのcoeffs/site短縮の大部分が他のCPU呼び出しと重なるため、完全stepへ同じ量は現れない。
次に調べる優先箇所はnative layout/sortのCPU入口と、forwardの重複した宣言・Context確認。
設定変更時の検証とcustom callbackを維持する必要があり、全面的な検証停止は行わない。
recipe検証だけの約11usや、内部には見つからなかった同期を主因とはしない。
optimizerの約.54msもhost費用が中心で、PolarのGPU更新則が重くなったという結果ではない。

primaryのallocated/reservedは全8回297984/2097152B、dense17313280/23068672Bで不変。
診断用の2 model同居やprofilerのmemoryはこの比較に含めない。
入力と初期Parameterのhashも全回同じだった。交互101 pairs後（計121更新）の
P/dP/dX/Adam step/exp_avg/exp_avg_sqの6 Tensorは2 processesともbit一致。
製品source全376ファイルは測定snapshotとbyte照合済み。既存190件のGPU gateと
CPU2672件の検査・package照合を再利用し、今回はruntime codeを変更していない。

primary/trace job=`l4job-4cc1ec1deef4473cb4669bdde7fe6c4d`、snapshot SHA256=
`afb89dc668e86d1c6842618333cbffb146b3111e7755150ba56a9fb25853c0c5`、
result archive SHA256=`cdcf3c188dfcb4489e303d1c8632de9587225d9c77245798e82541ce5fff446e`。
局所計測job=`l4job-7ce88b44eabd457694dc48ba2c1569a1`、snapshot SHA256=
`8ed54d4b179442db6425d100d75bb067b1028d85bfa362589ae63cf8c3ad4f89`、
result archive SHA256=`41c810f7fbae76b9efa179f7581dd8262542117b24b22dac4d32859e8e8d1867`。
2件目は既存の共有pool supervisorへqueue投入し、別supervisorやGPUを増やさず完了した。
再現driver/worker、source/result/receipt、8 fresh records、paired/phases JSON、
pstats、全trace、monitor結果・集計scriptはignored
`output/radius-host-diagnosis-20261030/`へ保全した。
1024/MNIST、他GPU/Torch版、Graph/process peakはこの診断の対象外。

### 小さいatom表のlayout/sortを単一起動へまとめる（2026-10-30）

前節で約.18ms/stepだった`_ranges_from_layout_keys`のCPU入口を対象にする。
`layout.cuh::layout_ranges`は1024 atoms以下で、キーの向きの変換、stable sort、
全ownerのlower-bound offsetを1 CTAで実行する。
1024を超える既存上限4096までの表は、従来のATen経路を保つ。
階層、profile、moments、完成済みH/G、atom勾配、Polar更新則、dispatchの検証は変更しない。

```text
旧 forward: ATen stable sort → offset生成
旧 backward: 逆向きkey表を生成 → ATen stable sort → offset生成

新（A <= 1024、両方向共通）:
  保存された元keyをsharedへ読む（backwardならここで向きを変換）
  → 元atom番号をtie-breakにしたbitonic sort
  → 元atom番号のrowsと全ownerのoffsetsを保存
```

元keyは`q_out * input_nodes + q_in`、逆向きは
`q_in * output_nodes + q_out`である。両軸のnode数は既存契約どおり510以下なので、
owner keyは18 bits以内。元atom番号を下位12 bitsへ付けたunsigned key
`(owner_key << 12) | atom_row`を使うと30 bits以内となり、
同値owner keyの順序は元atom番号の昇順、つまり従来のstable sortと一致する。
次の2冪までUINT_MAXでpaddingし、paddingを除いたA個だけをrowsへ戻す。
owner境界は`(owner * stride) << 12`のlower-boundで、空ownerと最後のend=Aも定義する。

1 CTAは256 threads、dynamic sharedは`4 * next_power_of_two(max(A,1))`B。
今回A819では4096B。並べ替え対象はshared内の一時キーだけで、
元のforward key・atom Parameter・response表を変更しない。
新しい永続cacheはなく、ni/noの変更は各呼び出しで読む。
通常の小さい経路で確保するglobal Tensorは返り値のrowsとoffsetsだけで、
sorted-key表・reverse-key表・ATen sort workspaceは作らない。

CUDA/連続性/dtype/shape/A上限/ni/noの既存host検証を維持する。
小さい経路のキー内容は`[0,ni*no)`をdevice assertionで検証し、
CPUへの値読み出しや同期待ちは追加しない。通常経路はdeviceで検証済みの
classification keyだけを渡す。保存済みキーのversion検査、現在atomの再分類、
Kernel/Chart設定変更時の検証と再準備もそのままである。
従来の`layout_ranges_aten`は大きい表へのfallbackと検査・計測用の比較入口として残す。

#### 完全stepとメモリの採用判定

比較元は`ef4c027a`のlayout、候補は`bff96eb1`のCUDA実装と
`b2c0b18c`の検査。runtime全376ファイルの差は`layout.cuh`と
診断用bindingを追加した`pipeline.cu`だけで、Python入口やoptimizerは同一である。
L4/Torch2.11.0+cu130/CUDA13.0/Python3.13.16、N128/B4/A819/M8/seed41、
FP32 IEEE、spacing1、base1/merge2、direct levels2、32-site CTA、
native CSTAdamW/fused AdamW、raw decay0/amp decay.01/check_finite=Falseを固定した。
旧/新×通常parallel/shared-profileの4条件を各2 fresh processes、
順番を入れ替えて計8回測った。各回10 warmups/101 synchronized complete stepsで、
入力・target・初期Parameter hashも同じである。

| Plan | 旧layoutのstep中央値2回 ms | 新layoutのstep中央値2回 ms | 中央値の平均の変化 |
| --- | --- | --- | --- |
| 通常parallel | 1.996941 / 2.067898 | 1.842697 / 1.801634 | 2.032420 → 1.822166、10.35%短縮 |
| shared-profile | 1.985200 / 1.984384 | 1.863882 / 1.862510 | 1.984792 → 1.863196、6.13%短縮 |
| matched dense（全8回） | .696102–.741802 | — | denseとの速度同等は未達 |

| eager allocatorのpeak（全8回同じ） | CST | matched dense |
| --- | ---: | ---: |
| allocated | 297984 B = .284180 MiB | 17313280 B = 16.511230 MiB |
| reserved | 2097152 B = 2 MiB | 23068672 B = 22 MiB |

globalのlayout一時表は減ったが、学習全体のpeakは旧版から変わらない。
この計測だけからpeakを作る配列・時点は特定しない。
新しい永続保存を追加せず、今回の両Planでdense以下を維持した。

同じ実学習を別途code-eventで観察すると、`_ranges_from_layout_keys`の
2呼び出し合計は通常189.596→47.879us、shared-profile182.414→46.864usになった。
Python入口の回数はforward/backward各1回のままで、内部のATen呼び出し・
allocation・kernel起動を削った効果である。prepared/program identityは同じで、
dispatch cacheはhits19/miss1からhits80/miss1へ進み、再prepareは起きなかった。
計16 training trace scopesで内部sync/item/DtoHは0、stepのGPU kernelは25→20。
同一binaryのlayout単体traceでは旧forward3 kernels約29.30us、旧reverse4 kernels
約30.48usから、各1 kernel約21.29/21.60usへ変わった。
新kernelは32 registers/thread、shared4096B、grid1/block256だった。
別計測のinclusive host時間やprofilerのbusy時間は、primary stepと加減算しない。

今回のGPU gateは412 passed。stable rows/offsetsは独立CPU oracleと完全一致し、
A0–4096、padding境界、同値key、空owner、両方向、owner数変更、
1024を超えるfallback、deviceの不正keyを含む。
独立Y/dX/全atom勾配・quadrature検査も各測定前に通過した。
5回実更新後の35 Tensor（Parameter/勾配/Adam状態等）は、両Planとも旧新bit一致。
CPU suiteは2672 passed/6162 skipped（既存warning19件）、Ruff/diff検査と
wheel/sdistの全376 runtimeファイル照合も通過した。

最初のjob `l4job-de5d857f719e490c802bec73d988df5e`は411 passed/1 failed。
owner数を変えても偶然同順になる入力を、異順になると期待した検査fixtureの誤りで、
実装の出力は両設定ともCPU oracleと一致していた。入力へkey4を足して修正し、
runtimeを変更せず、ソース・binary SHA照合済みのビルドを再利用して全gateを再実行した。
失敗jobの性能値は採用していない。

成功job=`l4job-490403651edb4e668858c17a32b3dbae`、snapshot SHA256=
`446f662e4947882051887bf6f2bab790424a37ef3ee1fa205bc57ccdeecd9200`、
result archive SHA256=`6aa6dbf11292c47b7c3fa70521ba78069132afd4de633111910f6a635a63d4a9`。
候補binary=`torchcst_radius_cell_mass_37a00de5c036`、SHA256=
`7c81476250b4ea1651a0e36f4afb8edb9ab5ebb88d3f77d4ad29969855ac70eb`。
比較元binary=`torchcst_radius_cell_mass_b1d35927a8ff`、SHA256=
`7fd5cf591cf8fe45d47fc08f2f6f9ef353fb64af947a25b628246960810f65a8`。
driver/worker/source/results/receipt、raw timing/state/trace、集計と失敗jobも
ignored `output/radius-block-layout-20261030/`へ保全した。
共有poolは結果取得後にstoppedを確認した。

この範囲では完全stepが両Plan・両反復とも短縮したため、A<=1024のlayoutを採用する。
Plan/自動selector/数式・許容誤差・階層とresponseの再利用は変更しない。
1024行列/MNIST、他GPU/Torch版、Graph/process memory、新たなsanitizer検査は対象外である。

### forwardの不変Contextをbinding内で再利用する（2026-10-30）

`28a15c2c`は`CSTLinear.build_context`とlive `LinearBinding.build_context`で、
検証済み宣言と実行メタデータが同じときだけ、直前の`LinearContext`を再利用する。
Contextに含まれるmetadataの型・範囲検査と、DeviceInfo/RequiredGrads/PrecisionPolicy等の
生成を毎step繰り返さない。Dispatcherの選択・配置検証と数値処理は変更しない。

```text
毎回 validate_inputs（現在のx/p、宣言の確認）
  → build_context（現在の宣言を再確認）
      → x/pのdevice・dtype・parameter次元を確認
      → 検証済み宣言のtokenと現在のmetadata keyを比較
      → 一致: 直前のimmutable Contextを返す
        不一致: 通常のContext構築・metadata検証、cacheを1件だけ置き換える
  → 既存のprepared dispatch / Algorithm.run / 実行時の宣言確認
```

keyは入力shape/stride/dtype/device、Parameter shape、現在の入力/Parameterの勾配要否、
grad-enabled/capture/決定性/autocast/TF32、workspace制約から得る。
grad-enabledは必要勾配のboolへ反映し、CPUでは従来どおりautocast/TF32を使わない。
atomの数値、xの値とstorageはContextの条件ではなく、毎回の実入力と既存の
placement guard/AtomState経由で扱う。数値や所有storageの正しさをcacheで証明しない。

cacheはbinding所有の`_linear_context_cache`に1件だけ置き、Tensor・Parameter・
input/response/momentsを保持しない。保存・deepcopyでは消し、checkpointへ追加しない。
現在の`execution_declaration`が検証済みentryと同一であり、固定stateのsignatureが有効で、
通常Tensor/Parameter・整数shape/stride・整数またはNoneのworkspaceの場合だけ使う。
未検証の宣言、固定stateのinference Tensor、Tensor subclassやsymbolic metadata等は
通常構築へ戻して古いcacheを消す。OperatorSpecだけのdirect bindingも通常構築を保つ。

入力検証・build_context・execution_declarationのcallback入口は省略しない。
宣言の確認にはカスタムcallbackが介在できるため、最初の案のように
入力検証で得た宣言を無条件に次の境界へ渡す方式は採らなかった。
従来の約5.7%悪化したContext共有案も踏まえ、今回も完全stepで判定する。
有効/無効な設定変更と同じ値のbuffer書き込み、現在のshape/stride/勾配要否/精度設定、
callback内の設定変更をcacheより前で観察する。

#### 同じN128条件での完全step比較

比較元は`049d4f62`、候補は`28a15c2c`。前節のblock sort、階層moments、
完成済みH/Gとresponseの再利用、coeffs/site展開、Polar更新則は共通である。
runtime全376ファイルの差は`operators/context.py`、`operators/execution.py`、
`nn/linear.py`、`nn/module.py`の4ファイルだけ。
両者は同じSHA検証済みnative binary `torchcst_radius_cell_mass_37a00de5c036`を使う。

N128/B4/A819/M8/seed41、spacing1、base1/merge2、direct levels2、32-site CTA、
FP32 IEEE、native CSTAdamW/fused AdamW、raw decay0/amp decay.01、
check_finite=Falseは前節から固定。L4、Torch2.11.0+cu130/CUDA13.0、
Python3.13.16/Triton3.6.0、CPU1 Torch threadで測った。
独立した2 GPU jobsで、各jobが旧/新×2 Plansを2 fresh processesずつ、順序を入れ替えて
計8回実行した。各processは10 warmups/101 synchronized complete stepsで、
毎回独立Y/dX/全atom勾配・quadrature・matched-state検査を先に通す。

| job / Plan | 旧版の中央値2回 ms | 候補の中央値2回 ms | 中央値の平均の短縮率 |
| --- | --- | --- | --- |
| 1 / 通常parallel | 1.842211 / 1.801759 | 1.765542 / 1.804278 | 2.03% |
| 1 / shared-profile | 1.783319 / 1.787599 | 1.785040 / 1.728544 | 1.61% |
| 2 / 通常parallel | 1.817390 / 1.823728 | 1.787191 / 1.740829 | 3.11% |
| 2 / shared-profile | 1.853002 / 1.848081 | 1.783540 / 1.789890 | 3.45% |

2 jobsの4 process中央値の平均は、通常parallel **1.821272→1.774460ms（2.57%）**、
shared-profile **1.818000→1.771753ms（2.54%）**。
job1にはわずかに遅い反復もあり、全runが必ず速いという結果ではない。
ただし両独立jobの平均で両Planとも短縮が再現したため、このContext再利用を採用する。
異なるjobsやPlans同士の順位を、この集計から一般化しない。
matched denseは全16回で.692145–.739654msで、速度同等には未達である。

| eager allocator peak（全16回同じ） | CST | matched dense |
| --- | ---: | ---: |
| allocated | 297984 B = .284180 MiB | 17313280 B = 16.511230 MiB |
| reserved | 2097152 B = 2 MiB | 23068672 B = 22 MiB |

設定が変わるcold経路では通常のContext生成・検証を行い、metadata keyの生成費用もある。
今回の主測定は既定のwarmup後であり、cold開始・設定変更stepの速度改善は主張しない。

#### 呼び出しとの対応と検証

job1の別code-event計測では、定常stepの`context_from_metadata`と
`LinearContext.__post_init__`は各1→0 callsになった。
`context_from_tensors`は通常parallel61.613→30.912us、shared-profile61.219→30.590us、
親の`CSTLinear.build_context`は106.847→83.906us / 107.300→84.316usだった。
`Dispatcher._prepare`は236.675→204.958us / 234.579→207.125us。
monitorのcallback費用を含むinclusive値であり、親子を足したり、未計装の完全stepと
差し引いて純CPU時間を求めたりしない。

`validate_linear_inputs`/`build_context`/`context_from_tensors`は各1回、
`operator_declaration`/`fixed_state_signature`は各3回のまま。
入力検証の設定確認、Context構築前の設定確認、Algorithm実行時の確認を省いた結果ではない。
prepared/program identityは変わらず、cache statsはhits19/miss1→hits80/miss1。
4条件×4 training trace scopesでGPU kernelは旧新とも20、
内部sync/item/local_scalar_dense/DtoHは0。数値GPU処理の短縮ではなく、
同じmetadataからContextを作り直すhost処理を減らした変更である。

新しい検査を含むGPU-host suiteは400 passed。
CUDA Contextの現在のautocast/TF32/captureフラグ、通常/逆向きlayout、saved-key寿命、
prepared dispatchとoptimizerのcallback・変更検証も対象とする。
captureフラグの検査は実際のGraph capture/replay性能検証ではない。
job1の5実更新後の35 Tensorは、両Planとも旧新bit一致。
CPU全体は2715 passed/6163 skipped、既存warning19件。
Ruff/diff、wheel/sdistの376 runtimeファイルbyte照合も通過した。
両jobsの同variant全runtimeとfactory hashは一致し、候補は現ソース全376ファイルと一致する。
入力・target・初期Parameterも全16回同一である。

job1=`l4job-0d02c86147a34c0f8d21e2d156513373`、snapshot SHA256=
`2b83df05ccbe9859805f26613756c374b9912daeaed6c6334b1e890437a66e40`、
result archive SHA256=`5e19ce3b476985a285c303ffdb3c8df34504ca4f827d7a2564a535021c9d2372`。
job2=`l4job-036f7ebee62643eca6cc4f5b9a8e3782`、snapshot SHA256=
`d82726accc9fd352ead5eedb429b8a6352f73cc79c4e63b070e27c849ef2e34f`、
result archive SHA256=`eb3fe89fc4b420c3d64072bf956582ff9ce2fcf30fd9db8029d50507c06a7c98`。
GPU UUIDは別だが、いずれもNVIDIA L4/driver580.82.07である。
job2は同runtimeとbinaryの独立再測定であり、400件suiteと35 Tensor比較はjob1の記録を使う。
各benchmark自体の独立correctness gateはjob2も全回実行した。

driver/worker、2件のsource/result/receipt、16 fresh records、状態、全trace、
集計・manifest照合、CPU/build/package記録はignored
`output/radius-forward-context-20261030/`へ保全した。
1024/MNIST、他GPU世代/Torch版、Graph/process memoryは今回の検証範囲に含めない。

### radius階層の配置確認からoptimizerだけの依存を外す（比較・不採用、2026-10-30）

このState変更は局所時間を減らしたが、通常parallelの完全stepを短縮できなかった。
`eb3fbe4b`で専用Stateとその専用テストを戻した。最終runtimeには残さず、
候補の復元先は`e0f44e5b`と以下のfrozen source/resultsである。

次のCPU候補は`AlgorithmState._signature()`のoptimizer field走査である。
汎用Stateは`owner.optimizer_state.fields`をsortし、各Tensorの
device/dtype/shape/stride/data_ptrまで調べる。optimizer状態を使うAlgorithmには必要だが、
radius Linearの準備はimmutableな階層幾何であり、Adamのmoment/clockを読まない。
optimizer reloadでmoment storageだけが替わっても、その階層の再準備は必要ない。

`e0f44e5b`は具体的な`RadiusCellMassAlgorithm`だけが作る
`RadiusCellMassState`を追加する。共通のAlgorithmStateやoptimizerの更新・検証は変更しない。

```text
旧Radius Stateの配置signature:
  owner ID + Parameter配置 + 全optimizer field配置 + Context + binding callback

新Radius Stateの配置signature:
  owner ID + Parameter配置 + Context + binding callback

layout_versionと保存済みsignatureの照合は既存AlgorithmState.is_currentを使う。
```

Parameter配置は引き続きidentity/device/dtype/shape/stride/data_ptrを含む。
Chart/Kernelの固定設定と入力・実行条件は、既存の宣言検証/Context/prepared dispatchから
渡す。現在のatom値とroutingは毎実行の入力から作る。
任意の`binding.state_signature` callbackも毎回呼ぶ。
カスタムRadius Algorithm subclassは追加依存を持ち得るため、汎用Stateへ戻す。
数値GPUコード・階層moments・完成済みresponseとその再利用・保存寿命は変更しない。
新たなTensor表や永続数値cacheは追加しない。

CPUではoptimizer reloadやmoment storage変更が階層を無効化しない一方、
Parameterのstorage/identity/dtype、relayout世代、入力Context、設定の変更を検出すること、
不正設定がContext構築で拒否されること、custom callback/subclassとserializationの
従来境界を確認する。追加GPU検査は両parallel Plansと全/dX-only/atom-onlyの3勾配条件で、
optimizer field storageが変わってもretained backwardのread leaseが解除されず、
通常backward完了後のrelayoutがStateを無効化することを対象にする。

同時に`6a8c98d1`でnative metadata guardの重複subtreeも比較する。
`guard_callable_metadata`は元からdict fast pathの外側でModule walker関数の
code/default/kwdefault identityを確認しているため、class namespace側で同じ3 fieldsを
再確認しない。kwdefault辞書の内容変更は別途dict-version guardを残す。
bufferについては、親`_buffers`辞書のversion guardが登録objectの差し替えを検出し、
TensorMatchがその現在typeとmetadataを検査する。この場合だけ重複object ID/typeノードを省き、
Tensor class/MRO/namespaceのsubtreeはclassごとに1件残す。
TensorMatch APIが使えなければ、以前のID/type/field guardへ戻す。
bufferの`_version`チェック、設定変更時のPython再走査と検証、callback境界そのものは省かない。

追加した検査は、辞書全体の差し替え、最初以外のbufferの`__class__`変更、
TensorMatch非対応時のidentity/dtype検査を対象にする。
既存のbuffer/登録/metadata変更、同値書き込み、Module walkerのcode/default変更、
custom property/callback、guard非対応fallback、参照寿命の検査も維持する。
旧版、State依存だけ削減、guardだけ削減、両方の4条件を同じ数値GPU binaryで比較し、
片方だけの効果と組合せの効果を分けて判断する。

#### 4条件の同時比較

比較元は`e4adbc70`。N128/B4/A819/M8/seed41、spacing1、base1/merge2、
direct levels2、32-site CTA、FP32 IEEE、native CSTAdamW/fused AdamW、
raw decay0/amp decay.01/check_finite=Falseは前節から固定した。
L4/Torch2.11.0+cu130/CUDA13.0/Python3.13.16/Triton3.6.0で、
各Planの旧版/Stateのみ/guardのみ/両方を2 fresh processesずつ、
往復する順序で計16回測定した。各回10 warmups/101 synchronized complete steps。
全回でbenchmarkの独立Y/dX/全atom勾配・quadrature・matched-state gateが通過した。

| Plan / 条件 | step中央値2回 ms | 中央値の平均 ms | 旧版からの短縮率 |
| --- | --- | ---: | ---: |
| parallel / 旧版 | 1.788814 / 1.786755 | 1.787785 | — |
| parallel / State | 1.803208 / 1.807631 | 1.805419 | -0.99% |
| parallel / guard | 1.792807 / 1.761931 | 1.777369 | .58% |
| parallel / 両方 | 1.865664 / 1.745646 | 1.805655 | -1.00% |
| shared-profile / 旧版 | 1.848337 / 1.846360 | 1.847348 | — |
| shared-profile / State | 1.806592 / 1.835082 | 1.820837 | 1.44% |
| shared-profile / guard | 1.765540 / 1.821022 | 1.793281 | 2.93% |
| shared-profile / 両方 | 1.809416 / 1.774560 | 1.791988 | 3.00% |

Stateだけの変更では、別の単一code-object計測で`_signature`が17.130→6.773us、
`is_current`が22.204→10.146us、`tensor_signature`が5→2 callsになった。
それでも両Planの完全step改善を確認できなかったため、局所時間を理由に採用しない。
両方を入れた版も同じ理由で採らず、guardだけを独立再測定する。
これはState変更が一般に遅いという証明ではなく、この比較の採用基準を満たさないという判断。

このjobのGPU-host suiteは472 passed。3候補×2 Plansの5実更新後35 Tensorはすべて旧版と
bit一致した。全16回のallocated/reservedは297984/2097152Bで、
matched denseの17313280/23068672B以下。matched denseのstepは.700507–.745767ms。
数値GPU binaryは全条件同じ`37a00de5c036`で、runtime全376ファイルの差は
State条件で`radius_cell_mass/algorithm.py`だけ、guard条件で
`operators/_native_metadata_guard.py`だけ、両方の条件でその2ファイルだけだった。

別code-object計測は各関数を1つずつ有効化し、測定前にP/Adamの数値状態を同じsnapshotへ戻す。
各関数10 warm monitored steps/51 samplesで、function/code/dictionaryの差し替えはしない。
定常prepared/program identityとmiss数も不変だった。
guard版のshared-profileで残るinclusive CPU費用は以下のとおり。

| 関数 | stepあたりcalls | 合計usの中央値 |
| --- | ---: | ---: |
| `CSTLinear.forward` | 1 | 588.542 |
| `Dispatcher._prepare` | 1 | 135.672 |
| `operator_declaration` | 3 | 94.377 |
| `fixed_state_signature` | 3 | 78.156 |
| `AtomState.protect_tensor` | 3 | 126.487 |
| `AtomState.parameters_for_execution` | 1 | 69.541 |
| `Registry.validate_plan` | 2 | 10.244 |
| `RadiusCellMassAlgorithm.validate_recipe` | 4 | 11.546 |
| `radius_linear` | 1 | 177.015 |
| `_BlockLocalRadiusLinear.forward` | 1 | 94.761 |
| `_BlockLocalRadiusLinear.backward` | 1 | 91.399 |
| `CSTAdamW._adamw_proposal` | 1 | 95.124 |
| `_ranges_from_layout_keys` | 2 | 45.549 |
| `AxisPlan.native_config` | 8 | 10.423 |

親子と異なる対象の計測が含まれるため足し合わせない。autograd wrapperやCUDA投入も
host呼び出しの一部で、残りをすべて宣言検証の費用と扱わない。
次の大きめのCPU候補は`protect_tensor`を軸にした読取寿命管理とautograd境界である。
Parameter借用前の保護、dX-only、retained backward、saved-tensor offload、relayout禁止を
保つ必要があり、guardを単純に外す案にはしない。

4条件×2 Plans×4 trace scopesでGPU kernelはすべて20、内部sync/item/DtoH/
cudaMalloc/cudaFreeは0。kernel busyは通常parallel約1005us、shared-profile約786usで、
数学演算や階層・responseの再利用を変更した短縮ではない。
inclusive host時間とこのbusy時間を足したり、primary stepから差し引いたりしない。

job=`l4job-52c0de5e9e54426c9f60a31abaeeaa94`、snapshot SHA256=
`0039dd913125e2fc7c0c716690b7dad4d0aac9ab8bab62f4ff5e35b23a0d8c5e`、
result archive SHA256=`9e0cef066c55385d22246c9a5b3d9e557918ae14e0bd0cd6992648dc794dd19b`。
先にqueuedだった`l4job-bfd6d0de69974b018247a351510ee713`は実行前にcancelし、
4条件比較へ置き換えた。GPU時間・性能結果として数えていない。

#### guardだけの独立再測定と採用

State変更を戻した`eb3fbe4b`で、旧版/guardを両Plan各2 fresh processes、
計8回独立再測定した。10 warmups/101 samplesと全benchmark correctness gateは同じ。
以下の結果に加え、両Planの5実更新後35 Tensorも旧新bit一致した。

| Plan | 旧版中央値2回 ms | guard中央値2回 ms | 中央値の平均の短縮率 |
| --- | --- | --- | ---: |
| parallel | 1.754909 / 1.788632 | 1.754601 / 1.699856 | 2.51% |
| shared-profile | 1.768765 / 1.718681 | 1.746267 / 1.728400 | .37% |

4条件jobの旧版/guardと合わせた4 process中央値の平均は、
通常parallel **1.779778→1.752299ms（1.54%）**、
shared-profile **1.795536→1.765307ms（1.68%）**。
独立2 jobsの平均では両Planとも短縮したため、guard重複の削減だけを採用する。
個別runには悪化もあり、shared-profileの2件目は小さな差である。
一般的な大幅改善や各runの保証とはせず、各job/反復の値も残す。
Stateの依存削減と両方の組合せは最終runtimeへ入れない。

全24 primary processesでCST allocated/reservedは297984/2097152B
（.284180/2MiB）、denseは17313280/23068672B（16.511230/22MiB）で不変。
採用する旧/guardの比較はこのうち16 processesで、残り8は不採用候補の比較である。
matched denseのstepは全24回で.699634–.745767msで、denseとの差はなお大きい。
GraphやGPU process全体の使用量を測った値ではない。

独立再測定の単一code-object診断でも、shared-profileの
`fixed_state_signature`は77.024→71.043us、`operator_declaration`は92.776→86.428us、
`Dispatcher._prepare`は133.497→126.560usだった。
呼び出し回数は各3/3/1のままで、設定確認やcallback入口の回数を減らした効果ではない。
`protect_tensor`は117.492→117.762us（各3 calls）で今回の削減対象外だった。
この2件目も全16 trace scopesでGPU kernel20、内部sync/item/DtoH/cudaMalloc/cudaFreeは0。

最終runtimeの差は`operators/_native_metadata_guard.py`だけで、
2 jobsのguard版と現ソース全376ファイルはbyte一致する。
両jobsの同variant/factory hash、および全24回の入力・target・初期P hashも一致した。
4条件のGPU-host suiteは472 passedを再利用し、独立再測定は毎benchmarkの
Y/dX/全atom勾配・quadrature・matched-state gateを実行した。
最終guard-onlyのCPU全体は2718 passed/6163 skipped、既存warning19件。
Ruff/diffと最終wheel/sdist全376 runtimeファイルの照合も通過した。

job=`l4job-c9c6bf00b9354825a33d4eee0fac3ac4`、snapshot SHA256=
`7c3892635472bd508f1fac8c24696ffcc503977de3b6a5f5e5f41e197d0ce187`、
result archive SHA256=`470d51eeb152cfcda29780f67083708f6aae32f54bfccfe73a02620cbc9df0a2`。
native binaryは両jobsとも`torchcst_radius_cell_mass_37a00de5c036`、SHA256=
`7c81476250b4ea1651a0e36f4afb8edb9ab5ebb88d3f77d4ad29969855ac70eb`。

driver/worker、4条件/独立再測定/未実行cancelのsource/results/receipt、24 records、
state/trace、集計script・JSON、CPU/build/package記録をignored
`output/radius-state-dependencies-20261030/`へ保全した。
共有poolは結果取得後のstoppedを確認した。
1024/MNIST、他GPU世代/Torch版、Graph/process memory、cold開始の速度は今回の対象外。

### 2026-10-30: 完全stepの時間・メモリ目標と次の比較

新しいgoalは、上記と同じN128/B4/A819/M8/seed41・FP32 IEEE・native
CSTAdamWの**未計装・同期済み完全step**で、CST時間をmatched denseの2倍以内、
ピークallocated/reservedをそれぞれdense未満にすること。
forwardだけやhost関数の局所短縮では達成としない。設定変更時の検証、Polar更新則、
階層moments、完成済みresponseの再利用、独立Y/dX/全atom勾配、read lease、
offload/retained backwardの契約を維持する。
128で達成したら、同じ目標をMNISTの完全stepと1024以上へ広げる。
MNISTは実際のradius/階層分布と学習条件を記録し、B4の結果を一般化しない。

比較候補は、(a) Parameter/出力の読取寿命を保つidentity autograd nodeのC++化、
(b) 既存CUDA stageをC++ autograd graphから直接呼ぶ明示的な
`native_autograd=True` recipeである。後者もmoments/Hを1回だけ保存し、
backwardはdYのmoments/G→dX→atom勾配の順とする。各CUDA計算式は共通で、
coeffsは引き続きshared scratchに置く。既存Python autograd routeは比較元として残す。
CPU側のsaved-variable寿命・offload・再backwardを確認しただけでは性能採用しない。

(a) snapshotは`c534c260`、GPU比較jobは
`l4job-efa9002130ff4b4dbcebf56954459449`。
(b) のGPU比較jobは`l4job-1004f07e73d24214aa862eb71299bc8d`。
初回queued job `l4job-bd290528c167442db9df9ffb48538c02`は、recipeに追加した
fieldがcatalogに欠けていることをCPU suiteで検出したため、実行前にcancelした。
catalog修正後のfixture/registry対象テストは6 passed。
この追記時点では両候補ともGPU比較待ちであり、目標達成・最終採用は未確認。


#### native read leaseの初回比較

job `l4job-efa9002130ff4b4dbcebf56954459449`は298 GPU-host testsが通過した。
同じCUDA binary `37a00de5c036`で、baseline/候補を各Plan 2 fresh processes、
各10 warmups/101 synchronized complete stepsで比較した。
通常parallelは平均中央値1.787915→1.753215ms（1.94%）、
shared-profileは1.784483→1.771731ms（.71%）。
全8回のallocated/reservedは297984/2097152B、denseは17313280/23068672B。
両Planの5更新後35 Tensor（Y/dX/dP/P/Adam状態）はbit一致した。
この差だけでは時間2倍以内の目標に届かない。小さな差なので、次の候補との
最終組合せも独立に再測定して採否を決める。

snapshot SHA256=`25423ead014a71a2cd616fcf720928fcb256473e3eb5dbfe24918dd6e0d0f353`、
result archive SHA256=`455311010d82267d667dd01764de6b319f9300b279fc6d4efa312fd5edf91046`。
source/results/receipt/8 primary JSON/単一code-object診断/trace/stateをignored
`output/radius-read-lease-20261030/`へ保全した。

#### C++ host autogradの初回比較

catalog修正後のjob `l4job-1004f07e73d24214aa862eb71299bc8d`は147 GPU-host tests、
各Planの5更新後35 Tensorのbit一致、8 primary processesのbenchmark gatesが通過した。
両条件が同じCUDA stage binaryを使い、recipeの`native_autograd`だけを切り替えた。
通常parallelは平均中央値1.786286→1.688557ms（5.47%）、shared-profileは
1.771156→1.688535ms（4.66%）。候補のdenseは.726257–.741576msで、時間条件は未達。
候補のallocated/reservedは287232/2097152B、denseは17313280/23068672B。
baseline allocatedは297984B。C++ backwardではdXのrouting rows/offsetsを
atom勾配の前に解放するため、同じ保存cacheのままピークの重なりも減る。

CPU全体はcatalog修正後2750 passed/6194 skipped、既存warning19件。
新native host nodeのoffload、retained/通常backward、dX-only/atom-only、
saved-version検査、二階微分の拒否はGPUテストでも対象にした。
snapshot SHA256=`771bd648f41399e7ee8ce0cb3e75b859794e9b4f49a09b19ef07e72dc8446220`、
result archive SHA256=`dacff5f0ee9fd9218e4d2161f930b504e3402d12a18d44b6ed199ad50d3ba0a3`。
source/results/receipt/8 primary JSON/stateと再利用可能なnative binaryをignored
`output/radius-host-autograd-20261030/`へ保全した。

次のGPU比較は、必要bankのwarp内集約・bank範囲だけの走査
（`l4job-9173c82f2f264e5fac51399ee8be55d6`）と、依存のないdirect momentsを
grid全体へ配る方式（`l4job-9972b42d198549098b7c0547d676cf47`）。
後者の追加global scratchはbank maskの4 bytes（allocator単位では別）だけで、
coeffsのglobal表やatomの複製を作らない。比較完了までは性能採用未確認とする。

#### bank集約だけの結果とlaunch候補のscreen

job `l4job-9173c82f2f264e5fac51399ee8be55d6`は147 tests、5更新後35 Tensorの
bit一致、全10 primary/screen benchmark gatesが通過した。
shared-profile/C++ hostの比較元2回は1.695940/1.680413ms、bank変更後は
1.697767/1.677755ms。平均1.688176→1.687761msで、完全stepの改善は確認できない。
別traceのmoments kernelは125.59→85.41usだが、局所GPU短縮を採用根拠にしない。
メモリは全primaryでallocated287232/reserved2097152Bのまま。

同じ初期値・profileのままexecution recipeだけを変えた各1 processのscreen：

| direct cutoff / output sites / moments | 完全step ms | matched dense ms | CST allocated B |
| --- | ---: | ---: | ---: |
| 1 / 32 / merge | 1.681964 | .731855 | 426496 |
| 2 / 8 / merge | 1.768103 | .725938 | 287232 |
| 2 / 16 / merge | 1.664208 | .715546 | 287232 |
| 2 / 64 / merge | 1.686941 | .739207 | 287232 |
| 3 / 32 / merge | 4.308174 | .734443 | 217600 |
| 2 / 32 / direct（batch毎CTA） | 1.832280 | .726255 | 287232 |

どれも時間2倍以内は未達であり、最小メモリの条件が最速ではない。
これはscreenで、最良の1件だけを選んで達成としない。
snapshot SHA256=`4b9ee0dc3a05f211298ff4e1c45363e8cf7e09d134af5b2d8d15dd33ff8d482c`、
result archive SHA256=`130329003b427b21ce032bff631aa5e0e506ef4ba5b39682d9abdc2c9100f011`。
原本・両compiled binaries・profile/state・10 recordsはignored
`output/radius-projection-banks-20261030/`へ保全した。

queuedの`l4job-9972b42d198549098b7c0547d676cf47`と
`l4job-d91a9b23b2434f6da05b0c91cce1dcde`は実行前にcancelした。
hash確認済みの既存native binaryを比較元へ再利用し、重複compileを減らした
`l4job-c2b30bbbdd534ad5b30961453318f1d3`へ置き換えた。
次はgrid direct moments・value-only M8係数・同じradius/periodの軸間再利用を
**組合せとして**比較する。Polarではtask radius微分を元から使わないが、
物理radiusのparameter化では既存のradius微分を維持する。

#### 組合せの結果とGPU変更の不採用

job `l4job-c2b30bbbdd534ad5b30961453318f1d3`は157 GPU-host tests、
5更新後35 Tensorのbit一致、8 primary/screen recordsのgatesが通過した。
同じdirect-moment recipeで、比較元1.672780/1.667976ms、組合せ版
1.695344/1.706612ms。平均1.670378→1.700978ms（1.83%悪化）だった。
全primaryのallocated/reservedは287232/2097152B、denseは17313280/23068672B。
2回のmatched denseは比較元.718268/.719992ms、候補.739092/.734991ms。
別traceではcached expansion130.31→108.77us、parameter kernel128.88→112.94us。
direct momentsは従来90.93usに対しmask63.54us＋grid6.53usになったが、
追加launchもあり、完全stepでは改善しなかった。

candidate内のsingle-process screenも、direct/tile16=1.702750ms、
direct/tile64=1.776467ms、merge/tile32=1.702985ms、merge/tile16=1.690705ms。
時間2倍以内は未達。GPUのbank集約・grid direct moments・value-only係数・
軸間profile共有は`d33013cf`でruntimeから戻し、
採用済みC++ hostの元のnative source digest `55a6bafe9bf2`へ復帰した。
候補は`97419371`/`7438135a`/`0dc46a77`とignored evidenceから復元できる。

snapshot SHA256=`41e9761fc7eef4b6633ffce23acea7c045618f649a5598886fafac3528989169`、
result archive SHA256=`cf253dc1e6d419a4338c7b0f19f9aea7743f032e14cb7993cd67d2819f8f9771`。
source/binaries/results/receipt/profile/state/screenはignored
`output/radius-combined-projection-20261030/`へ保全した。

#### 読取リスト整理と出力leaseの比較

CPU weakrefの整理を32件ごとにし、配置変更境界では従来どおり全件確認する候補を比較した。
死んだweakrefはTensor/storageを保持せず、live/retained graphの保護は変えない。
最初のjob `l4job-0b8c1774fdbb4086a87611b16fff3166`はresult-readyの後に
archiveのdownloadがfile-not-foundで失敗した。回収できていないため性能値は採用しない。
pool所有VMのstopped確認とrecoverを経て、明示的に新jobを投入した。

再測定job `l4job-980923884b194bc384ca0ee64807f308`は80 testsと35 Tensorのbit一致が通過。
同一GPU binary・shared-profile/C++ hostの2 fresh processesずつで、
比較元1.641184/1.584592ms、整理候補1.599014/1.598209ms。
平均1.612888→1.598611ms（.89%）で、各287232/2097152Bのまま。
matched denseは.686581–.704079ms。時間2倍以内は未達で、次の組合せでも再確認する。
snapshot SHA256=`2eb2d648c14ed45cbd2e6aa030e747ec0d1b282e26470905f727a32163793722`、
result archive SHA256=`9d20deeaee03d6eeeb763c9c8c29ebecdee265e7dd725b467884a3b09dd87314`。
失敗jobの運用記録と成功job原本をignored `output/radius-read-pruning-20261030/`へ保全する。

次の明示recipeは`native_autograd=True, native_read_lease=True`。
数学とnumeric SavedVariablesは標準CppNodeへ委譲し、実際の`release_variables`で
出力の配置read leaseも終了させる。backward hookで代用しないため、後続のuser hookが
例外を出した場合にsaved graphが残っているのにleaseだけを解除しない。
graph/ownerへの参照はweak、intrusive_ptr版ではstrong参照が0になった時にnumeric resourceを
解放する。Parameter借用前の既存保護とcustom callback fallbackは維持する。
CPUでoffload/retained/通常backward/hook例外/graph破棄を検証し、GPU job
`l4job-8037da8c052a47de9f86ccf8482f25d9`で比較中。未確認の性能採用はしない。

#### 出力leaseは不採用、native入口検証を継続

job `l4job-8037da8c052a47de9f86ccf8482f25d9`は205 testsと35 Tensorのbit一致が通過。
比較元1.700151/1.710475ms、出力lease統合版1.673881/1.718510msで、
平均1.705313→1.696196ms（.53%）。両者ともallocated287232/reserved2097152B。
`protect_tensor`は別診断で112.31→80.50usだが、`radius_linear`は160.55→185.26us。
統合するためのqualification/graph wrapperにも費用があり、完全stepの改善は小さい。
通常/retained/offload、hook例外、graph破棄を通したものの、この複雑さは採用しない。
prototypeは`ad4bc03d`とfrozen sourceから復元可能で、runtime/test/recipeの
`native_read_lease`を削除する。既存のParameter/出力identity保護は元どおり残す。
snapshot SHA256=`8d0258b3187560f438b09bccbf811ade381465b56cad88103ffdc5811800b06e`、
result archive SHA256=`d8a872c525999fa68d2ef55cadfaae1404a94526f23f643275995c31fe0d19c8`。

次の比較は`native_entry=True`。毎stepの検証を省かず、入力/atomのmatrix・dtype・device、
hierarchy設定、shared容量/index/grid上限、現在のautocast/決定論/capture状態を
C++入口で検査してからclassifyを開始する。公開Dispatcher/Algorithmの検証は維持し、
Tensor subclass/function transformは従来入口へ戻す。計算stage/保存cacheは共通。

初回job `l4job-67135bb6537a43ac9ad5000124819e5c`は222 tests、35 Tensorのbit一致が通過。
比較元1.712537/1.690881ms、native入口1.620235/1.640561msで、
平均1.701709→1.630398ms（4.19%）。CSTは287232/2097152B、
matched denseは17313280/23068672B、候補のdense時間は.708857/.737733ms。
時間2倍以内はまだ未達。独立したメタデータ拒否テストでは、誤shape/dtype/shared容量/
cutoff/tileをGPU計算の前に拒否し、現在のautocast/決定論設定も検査した。
snapshot SHA256=`eb0eede4b0cff29a1f8440e3a912a99a790bad4c208d0fb44a5ab36d4d88bdf6`、
result archive SHA256=`3bf29e5ac6b64d3996c1684d5246d99478f5180dcc853a9c74daa86203505171`。

出力lease prototypeを除いた候補のCPU全体は2753 passed/6219 skipped、既存warning19件。
最終候補の独立job `l4job-46d6577a27b344d58463baecd9abcc7a`も完了した。
206 testsと35 Tensorのbit一致が通り、同じ未計装・同期済み完全stepは次のとおり。

| 独立process | Python実行入口 ms | native実行入口 ms | nativeとmatched denseの時間比 |
|---|---:|---:|---:|
| 1 | 1.638342 | 1.583040 | 2.2376 |
| 2 | 1.644470 | 1.598065 | 2.1759 |

平均1.641406→1.590553ms（3.10%）。nativeのmatched denseは.707483/.734430ms。
全processでCSTのallocated/reservedは287232/2097152B、denseは17313280/23068672B。
メモリ条件は満たすが、**時間2倍以内はまだ未達**で、MNIST/1024への達成宣言はしない。

単一code-object診断ではnative候補のforward約479µs、autograd.backward約351µs、
optimizer本体約471µsだった。内訳のDispatcher._prepare約131µs、
fixed_state_signatureの3回合計約75µs、optimizer prepare/finish約172/149µsは
上位区間に含まれる。計装の影響もあるため、これらを合計して完全stepを説明しない。
optimizer programは`PreparedPolarOptimizerUpdate`で、generic経路に落ちていたわけではない。
source SHA256=`10fc209dbad0eba480c947f143be0778efbe877f1b24e9c6d6c9f57384cbbb21`、
result archive SHA256=`1f3b19474ab645fe784045ebb2ce73bef53c4d17f1778a10edb336fc8d8f7d09`。
結果・CPU/GPU trace・35 Tensor・同一native binaryは
`output/radius-native-entry-clean-20261030/l4job-46d6577a27b344d58463baecd9abcc7a/`に保存した。
ここまでのsource/binaries/results/receipt/state/profile/失敗を含むCPU診断はignored
`output/radius-output-lease-20261030/`と`output/radius-native-entry-20261030/`へ保全した。

#### 固定bufferバージョンの一括確認（不採用）

個々の`_version` getterを弱いTensorImpl/Python wrapper参照のC++ epilogueへ
まとめる候補を試した。Tensor値・GPU data・同期を追加せず、型/構造の検査は維持する。
初回job `l4job-b706ca5bb79145589f1bd00f1b330de9`はTorch 2.11のTensorMatchだけでは
同じbufferの`__class__`変更を検知できないケースでテストが失敗したため、計時に進まず停止。
weak wrapperからの明示型確認を追加して修正した。CPU全体は2763 passed/6219 skipped、
追加のdtype/shape/layout/getattribute変更テストも通過した。

修正版job `l4job-be14c6fded1b48acae9b209b928c31b3`は252 testsと35 Tensorのbit一致が通過。
同じnative入口を両側に使い、通常getter 1.608984/1.641977ms、
一括確認1.609876/1.596200ms（平均1.625481→1.603038ms、1.38%）。
候補のmatched denseは.697376/.733529msで、**時間条件は未達**。
allocated/reservedはCST 287232/2097152B、dense 17313280/23068672Bのまま。
局所guardの改善は約1µs、完全stepの片方は改善せず、別CPU拡張を追加する効果として小さい。
候補のsource/testsは研究checkpointとignored証拠に残し、実行経路からは外す。
source SHA256=`9a94092167e503bc0fb1799f31d7885496aa81433d22e0a2ae56a3c5414abb58`、
result archive SHA256=`2a1bc4105bd45af1543c341f62f282c6c2fb62649e4a41bfda5424c27b845d58`。
全結果は`output/radius-batched-versions-20261030/`に保全した。

#### 通常runnerからの再現

`radius-native-cached-parallel`を追加し、明示Planで`native_autograd=True, native_entry=True`を
選べるようにした。自動dispatcherへの採用は行わない。radius benchmarkのGPU optimizerは
公開`CSTAdamW`を使用し、CPU oracleは従来の`CSTOptimizer(AdamW)`を維持する。
以前の比較driverがGPU側へ注入していたものと同じconstructor/更新設定であり、今後は
benchmark sourceの書換えが不要になる。結果には実際のoptimizer class名も記録する。

```bash
PYTHONPATH=src python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-128-native-initial.json \
  --polar-update radius --output output/radius-native-complete-step.json
```

このCaseは128×128/B4/A819/M8/seed41、warmup10/rounds101で、前述の目標条件と同じ。
変更後のCPU全体は2754 passed/6219 skipped。通常runnerを無変更で実行する独立job
`l4job-b88134409a204d0f9895dd73a010b428`も212 tests、独立oracleと更新後の全勾配検証が通過。
独立processの完全stepはCST 1.567546/1.586360ms、matched dense .699269/.701042ms
（2.2417/2.2629倍）。メモリはCST 287232/2097152B、dense 17313280/23068672B。
native optimizer class名は`torchcst.optim.adamw.CSTAdamW`として記録された。
通常入口でもメモリ条件は満たすが、時間2倍以内はまだ未達。
source SHA256=`9ecf6e5cd6bf0ebe901d90d1480ef9f0f0f295ba5785766623f028f3b26fe073`、
result archive SHA256=`26fd7f41d9f96151e0c3a6e5e76e423d81c5525da04f2aa726440671c132e613`。
結果は`output/radius-public-runner-20261030/l4job-b88134409a204d0f9895dd73a010b428/`に保全した。
このsourceのPR commit `6cf0b5883b74b18ce1bf5631ab549c85d63a0568`のCPU validation CIも成功。

#### 固定Planの準備済み実行入口（検証中）

`native_entry=True`の標準CSTLinear/FixedSelectorに限り、一つのbinding-owned
execution programを試作する。現在の入力Tensor metadata、Parameter/optimizer fieldの
storage、登録、固定buffer version、Plan/recipe/state、標準callback定義、PyTorch global
stateをnative guardで確認する。callbackの確認を先に行い、差し替えられたaccessorを
guardの途中で実行しない。変更・custom callback・非対応API・Tensor modeは通常Dispatcherへ戻す。
初回入力Tensorは保持せず、新しい入力storageでも同じmetadataなら処理できる。
deepcopy/checkpointではprogramと試行stampを捨てる。

置き換えるのはDispatcher/Algorithmの純粋なmetadata prologueだけ。
`RadiusCellMassAlgorithm.execute`はそのまま呼び、Parameter借用、その後の宣言確認、
現在atomの分類、階層moments/responseの生成・共有、checked CUDA入口、出力read leaseは維持する。
数値計算のcacheをstep間で共有したり、検証なしで以前のroutingを再利用したりしない。
generic Dispatcherは演算を区別せず、Algorithmが任意に提供する準備済み入口だけを扱う。

CPU全体2771 testsが通過し、変更検知/弱い入力寿命の17 testsも通過。
CPU metadataだけの局所比較では通常prologue約31µs、guard約6.2µsだった。
これはCPU上の局所診断であり、GPU完全stepの改善や目標達成を示すものではない。
同一native入口・同一CUDA binaryで、通常prologueと準備済み入口の実GPU比較を次に行う。

その後の独立レビューで、準備前のinstance/class callback差替え、継承されたmemberの
shadow、登録atomへのdescriptor差替えをcold qualification前に拒否するよう修正した。
ライブラリ定義時のowner/member対応と関数本体を両方確認し、別のsealed関数を同じcallbackとは
みなさない。custom getterを呼んでから拒否する順番も修正した。変更を元へ戻した際には
準備の再試行を許し、別bindingのprogramを実行しない。これらのCPU回帰42件が通過し、
CPU全体は2796 passed/6230 skipped。単一勾配/offloadテストも再warm後のprepared hitを明示する。

job `l4job-7197d7ed0c114da09193460c1548aa2e`で281 tests、5step分35 Tensorのbit一致が通過。
通常入口1.567622/1.520113msに対し、準備済み入口1.530794/1.505848ms
（平均1.543868→1.518321ms、1.65%）。候補のmatched denseは.699678/.714281ms
（2.1879/2.1082倍）。CST allocated/reserved=287232/2097152B、dense=17313280/23068672B。
メモリ条件は満たすが**時間2倍以内は未達**。
計装ではDispatcher._prepare/state.is_currentがhot callから消え、fixed_state_signatureも
3回から1回へ減った一方、forward全体の短縮は小さい。次にguard自体の内訳を測る。
source SHA256=`76a4accbeaa611bba07531d8a3f7f388df0298bf175ef67dbaba0f2f5f562084`、
result archive SHA256=`8f267c09bfb21e6bd927f3109aa71fc7cbc0481fc82614fe5bc2ea48f9ef5521`。
証拠は`output/radius-prepared-entry-20261030/l4job-7197d7ed0c114da09193460c1548aa2e/`。

先行job `l4job-b4d1ae71570748e583cf41fa3145b68c`はdriverのcatalog書換えによる重複Plan、
`l4job-b9411d216506406e868920d5073a5f28`は古いoptimizer注入処理のlocal importによる
UnboundLocalErrorで停止した。後者の257 testsは通過したが、両jobとも性能結果から除外する。
通常catalog/optimizerを無変更で使うdriverへ修正し、variantごとにpytest設定も分離した。

class/moduleの辞書がCPythonで差替え不能であることと完全なMRO pointerのnative確認を利用し、
callback guard内の重複したMRO走査を定義時に収集した辞書version確認へまとめた。
job `l4job-c95de37852034bbcbb149532fcb4c4c1`で281 testsと5step35 Tensorのbit一致が通過。
通常入口1.542809/1.535098msに対し候補1.506404/1.482536ms
（平均1.538954→1.494470ms、2.89%）。matched dense .700190/.706308msで
2.15142/2.09899倍、時間の目標は未達。allocated/reservedは前記と同じでdense未満。
実L4での未計装metadata局所測定は通常prologue約94.26µsに対し、候補は
callback guard18.46µs、data guard15.78µs、mode確認1.91µsだった。
計装forwardは469.64→432.13µs、backward352.59→347.16µs、optimizer463.47→464.73µs。
局所測定/計装時間を足して完全stepとみなさず、独立processの未計装結果を採否に使う。
source SHA256=`cf5f2bab4c7e4079df654afa24d75fd0de361e09bfe617f6884d6f5ea650f2da`、
result archive SHA256=`0a6de881f43a22cc434f2b4c47af74b006cea381e9e533e4a54c2968a855d483`。
証拠は`output/radius-prepared-entry-20261030/l4job-c95de37852034bbcbb149532fcb4c4c1/`。

次の試作はCSTLinear.forward内部の2次元・連続した標準Tensorだけを同じprogramへ渡し、
reshape/contiguous、feature取得、Dispatcher構築の重複を省く。nn.Module.__call__は維持し、
pre/post/full-backward hooksとTensor勾配hooks、数値計算とread leaseは同じ経路を使う。
raw Tensor metadataと現在のModule.selectorを確認し、形状・stride・設定変更は通常入口へ戻す。
新しいLinearInputsは呼出しごとに生成して保持しない。constructorの__init__/__new__と
inputのreshape/contiguousの差替えを認めず、custom動作をshortcutで隠さない。
独立レビューで、準備前のLinearInputs.__new__差替えと、標準Tensor instanceのreshape/
contiguous差替えを新入口が隠せる問題を再現し、cold認証とraw input guardへ拒否条件を追加した。
__new__の意図的な変更はCPython slotを汚染するため、回帰検証は独立subprocessへ隔離する。
CPU全体2804 passed/6231 skipped（51.55s）。GPU比較と完全stepの採否は以下で別に記録する。

Module入口の先行job `l4job-cc6ab8bc120d42c2aa8c46dba223bf66`では285 testsと
5step35 Tensorのbit一致が通過。通常入口1.555658/1.569555msに対し
候補1.466828/1.429843ms、matched dense .692519/.671758ms（2.11811/2.12851倍）。
allocated/reserved=287232/2097152Bは維持。時間の目標は未達。
計装forward472.839→368.933µs、backward352.464→324.679µs、optimizer461.501→460.008µs。
source SHA256=`b0e8bfef2f9b30ae9464152ebb7b456eb36689afa3b9eef3265bbb31c36ea649`、
result archive SHA256=`e837557aafddc122a5e9d1a33f994100a6a2ca4fd42d8a577d4a8d47857fb927`。
証拠は`output/radius-prepared-entry-20261030/l4job-cc6ab8bc120d42c2aa8c46dba223bf66/`。
このsnapshotは上記allocator/input callbackの認証修正前であり、そのまま最終採用とはしない。
修正版のみの待機job `l4job-3e9c50b3053d4937b6f53c468113854c`は実行前に取消し、
同じ修正とoptimizer guardの辞書走査削減をまとめた比較へ置換した。取消jobに性能結果はない。
optimizerの更新則やpre/post-Adamの検査境界は変更せず、CPython checkerが完全なMROを
確認する場合だけraw dictionary version確認へまとめる。checkerなしのfallbackは従来通り。
独立レビューで必要な110 namespace辞書がすべて残ることを確認した。

統合版だけの待機job `l4job-b0b2ecf5a6884dd2a2c461e654d52a3a`も実行前に取消し、
同じsourceの出力タイル幅32/16/8/64をまとめたjob
`l4job-9c85563a9c24426bba73135e0d3ce1d2`へ置換した。旧jobの性能結果は存在しない。
新jobはまず通常入口/統合版をABBAで比較し、その後16/8/64を逆順も含めて各2回の
独立processで測る。各processで既存の初期/更新後Y・dX・全atom勾配gateも実行する。
radius階層のTと出力CTAの担当site幅は別の設定であり、階層やPの式は変更しない。
正しさと未計装完全stepが通るまで、タイル幅変更の採用や目標達成を宣言しない。
統合版のCPU全体2804 passed/6231 skipped（53.10s）、commit
`41fc25154c8cb3f78bfcccb9776b050ecdbf3a47`のCPU validation CIも成功。

大型化の事前見積り：A=floor(.05*N²)、FP64の完成済みH/Gを両方同時に保持すると
保存量は`16*B*A` bytes。N1024/A52428ではB32で25.60MiB、B64で51.20MiB。
これは実測peakではなく配列量の下限で、FP32 denseのweight/grad/Adam momentsの
`16*N²=16MiB`より大きい。MNISTの共通層/activation/workspaceも含む実測が別途必要。
shared moments/Cの分割だけではH/Gの全量同時保存は減らない。必要なら有効batchを保つ
勾配蓄積や明示的な再計算/offload方針を比較する。forwardを分割しても全chunkのHを
保存し続ければ合計保存量は減らない。128の結果をそのまま大型化の達成とみなさない。

#### 修正後の統合入口と出力幅の比較（2026-10-30）

job `l4job-9c85563a9c24426bba73135e0d3ce1d2`は正常終了・hash確認済みで回収した。
最終callback認証修正とoptimizer辞書走査削減を含むsourceで461 testsが通過し、
5step35 TensorのParameter/Y/dX/全atom勾配/Adam stateが通常入口とbit一致した。
同一N128/B4/A819/M8/seed41、FP32 IEEE、native CSTAdamW、10 warmup/101 samples。
通常入口は1.576297/1.593188ms、統合入口（幅32）は1.438426/1.436839ms。
matched dense .702707/.697241msで、2.04698/2.06075倍。時間目標は未達。

| 出力CTA幅 | CST完全step ms（独立process 1/2） | matched dense ms | CST/dense |
| --- | --- | --- | --- |
| 16 | 1.443524 / 1.428140 | .706232 / .710989 | 2.04398 / 2.00867 |
| 8 | 1.444024 / 1.406958 | .718434 / .712295 | 2.00996 / 1.97525 |
| 64 | 1.447762 / 1.427002 | .710081 / .722951 | 2.03887 / 1.97386 |

各processで初期/更新後の独立Y・dX・全atom勾配とmatched-state optimizer検証が通過。
CST allocated/reserved=287232/2097152B、dense=17313280/23068672Bは全幅で同じ。
幅8/64の片方だけ2倍を下回る結果や、process平均だけで達成とはしない。
出力幅だけでは安定した余裕は得られておらず、階層momentsの構築方式も次に比較する。
`moment_method=direct`は階層ごとのmomentsを入力から作る方式で、低階層atomのdirect経路や
階層分類・周期式・完成responseの共有とは別の設定。これらを維持して比較する。
そのjobは`l4job-acc85bd3dacc4406806c4f8f3fbd4737`（未採用）。
source SHA256=`dd6c088e977bd3de26a4d551ae22062cfa3b8b4d626df9fbfb6a339fa6ca681c`、
result archive SHA256=`46cbdbbead5035a320b74d682ef386cdfbdbb5fd71258873afeb36b1af6d65cb`。
証拠は`output/radius-prepared-entry-20261030/l4job-9c85563a9c24426bba73135e0d3ce1d2/`。

#### 未使用階層の走査削減と共通guardの配置（2026-10-31 JST、検証中）

`shared_projection_kernel`はrequired-bank maskに親が使う子bankを加えた後も、未使用levelを
全threadで走査・同期していた。commit `5c66dd65`ではmerge時のみ、そのlevelの2bankが
両方不要ならblock全体でloop本体と同期を飛ばす。maskは直前の同期後に不変で、全threadが
同じ分岐を通る。必要な子levelの同期、zero初期化、direct方式、保存配列は変えない。
独立レビューでも同期の妥当性を確認したが、GPUのbit一致/速度は別途検証する。
job `l4job-ad78884e92e64a93aa335734fe0db4c0`は、同じ準備済みPython入口で
元のCUDA binary `torchcst_radius_cell_mass_9dbaec7aa25f`と新sourceを比較する。
新binaryは実GPUで実際にcompileし、source digest名は`torchcst_radius_cell_mass_49af2a2b1ff9`。
source SHA256=`a291808afe99701d5c328f36dc28348c9cf2fc58e8a40215690fd5fdbfe7c666`。
実行中の比較に共通guard移動後のsourceを混ぜない。

別のmoments方式job `l4job-acc85bd3dacc4406806c4f8f3fbd4737`は旧CUDA source固定で、
source SHA256=`b56c18efde90fd78fdd68040924806282d8c3306ce2afc1b7927fd7225a7e4f8`。
DIRECT/MERGEはatomが消費するbankで数学的に同じmomentsを作るが、merge専用の子bankは
directではzeroのままで、保存buffer全体や異なる加算順序のbit一致を主張しない。
両方式の比較でも階層分類、period式、完成H/Gの共有を保ち、各processの独立oracleと
matched-state optimizer検証を通す。まだ採用/目標達成とは扱わない。

LinearとPolarが実際に共有するCPU helperは、配置規約に合わせ
`_backends/cuda/_host_function_metadata.py/.cpp`へ移動した（commit `034e7970`）。
旧private pathのshimは追加しない。C++は移動前とbyte一致
（SHA256=`25808b593fca09e5c25621a80c2457f2ee7303d934bf4dd3eb0a3fb79c66ced2`）、
Pythonもdocstring以外は同じ。関連CPU111件、CPU全体2804 passed/6231 skipped
（51.60s）、Ruff/diff検査とwheel/sdist buildが通過。配布物内の共通helperとCUDA headerの
byte一致も確認した。GPU比較sourceと現在sourceの配置差は最終再測定時に揃える。

共有poolは既存実験の正常終了とVM停止を確認してから`--workers 2`で再起動し、上記2jobを
別々のL4へ配置した。各job自身のmatched denseを使い、異なるVMの絶対時間だけを
差し引いて効果とみなさない。結果回収と停止確認までpoolの所有手順に従う。

#### 構築方式と未使用level削減の結果（2026-10-31 JST）

moments方式job `l4job-acc85bd3dacc4406806c4f8f3fbd4737`は正常終了・回収済み。
全8processの独立Y/dX/全atom勾配、small-atom quadrature、3step matched-state optimizer、
初期/更新後の全atom gateを確認した。設定変更で階層分類・profile・response再利用は変えない。

| moments / 出力幅 | CST ms（独立1/2） | dense ms | CST/dense |
| --- | --- | --- | --- |
| direct / 8 | 1.464112 / 1.445583 | .707472 / .698413 | 2.06950 / 2.06981 |
| direct / 16 | 1.452209 / 1.425558 | .690874 / .687263 | 2.10199 / 2.07425 |
| direct / 32 | 1.401239 / 1.418352 | .699111 / .685457 | 2.00432 / 2.06921 |
| merge / 4 | 1.427850 / 1.407744 | .721247 / .700088 | 1.97970 / 2.01081 |

全条件でCST allocated/reserved=287232/2097152B、dense=17313280/23068672B。
両独立processで2倍以内となる設定はなく、この比較だけでは変更を採用しない。
result archive SHA256=`c7c19756a43bade851a9179d29c6e4acc32107455cc2f0cf5a5e77a340d0b256`。
証拠は`output/radius-prepared-entry-20261030/l4job-acc85bd3dacc4406806c4f8f3fbd4737/`。

未使用level削減job `l4job-ad78884e92e64a93aa335734fe0db4c0`も正常終了・回収済み。
新CUDA sourceを実際にcompileし、461 tests、5step35 Tensorのbit一致が通過。
元kernel1.504444/1.498665msに対し新kernel1.502817/1.450439ms。
新kernel自身のmatched dense .709443/.722097ms、2.11831/2.00865倍で時間目標は未達。
新kernel・幅8も1.496848/1.529330ms（2.11768/2.08032倍）で安定した改善にならない。
メモリは前記と同じ。計装projection kernelの平均は125.54→91.33µsだったが、
完全stepの余裕には結びついていない。GPU単体の短縮だけで最終採用を決めない。
新binary SHA256=`74aed5fa9401bfb7dbe8998fae5c550337218dd8f11c0ccfbc5d9fa26568d598`、
result archive SHA256=`43d9042ded6d0b09aea38dcce72e29f111001f95ab66b0c365b62f91c8bb490f`。
証拠は`output/radius-prepared-entry-20261030/l4job-ad78884e92e64a93aa335734fe0db4c0/`。

CPU側の次の試作はgradient preflightの既存TensorMatch再利用（commit `04a28ff4`）。
Python fallback条件は元のpredicateとAST一致のまま、nativeで適合しない勾配は通常経路へ戻す。
custom detachのcold認証も加え、勾配/例Tensorを保持しない。関連138 tests、CPU全体
2831 passed/6231 skipped（72.71s）。localのfull preflightは8.295→8.059µsと小さい。
job `l4job-3e766d1bdc3f486b9f3586f08883be78`で同じCUDA49af・共通helper配置のまま
旧preflightと比較し、別passでCPU call profileも採る。現時点でL4の改善は未確認。

CSTAdamWのgroup検証cacheもlocalで調べたが、単一groupの削減は約.224µs、2groupでも
.395〜.427µsで、100行以上のcold認証/mutation/fallbackコードに見合わず本体変更を見送った。
既存optimizer CPU31件は通過。再現script/独立2processのraw logと安全性の前提は
`output/adamw-groups-guard/assessment.txt`ほかに保全した。L4の速度効果とはみなさない。

#### optimizer host guardの測定と辞書変更検知の試作（2026-10-31 JST）

gradient preflight再利用job `l4job-3e766d1bdc3f486b9f3586f08883be78`は488 testsと
5step35 Tensorのbit一致を通過した。候補の完全stepは1.483964/1.441370ms、
自身のmatched dense .729854/.695375ms、比率2.03323/2.07280で、時間目標は未達。
source SHA256=`4d8286a8829e18aed5f4e936eced0e5b8104018071b83712303b4c6eda1d3158`、
result archive SHA256=`4d15ddaa2c246d81ce09b5fa46a558a51ef070d40846871a0da97891f84776f7`。

Polarの固定importをmodule aliasへ移したjob `l4job-bddd099c60764ba58900760b3dacef68`も
492 tests、35 Tensor bit一致を通過。属性はその都度moduleから取得するので、callbackや
parameterizationの変更検証は維持する。元版1.429023/1.400714msに対し候補は
1.395532/1.432693ms、dense .692218/.705734ms、2.01603/2.03008倍。
メモリは両版ともCST allocated/reserved=287232/2097152B、dense=17313280/23068672B。
両独立processの2倍以内は未達で、import削減だけの安定した完全step改善とも断定しない。
source SHA256=`3fb3b62cec618a6c32f7dd99674e9b7354e0cfa433f17d6b33ae6b08774092dc`、
result archive SHA256=`bd2ddb11f2bf0006ed2b0a24221fcce4d7ceee4b7da78a066862786b74a853a2`。

同jobの未計装micro測定（外側でno_grad/finite無効、各5×1000回）では、候補の実際の
optimizer guardが28.334µs、validate_inputs 2.627µs、preflight 35.116µs。
重複importを除いてもguard走査が残る。cProfileの時間は計装の影響があるので、
それを未計装の関数時間や完全stepの内訳として加算しない。

snapshot方式job `l4job-e4d49eec73654198b34addef3171532c`では、A819×4 FP32の
detach().clone()がidle時13.47–19.62µs、clone()が12.05–17.08µs、pending CUDA時は
それぞれ約14.9/13.1µsだった。GPU copy自体は約1.1–1.34µs。専用copy kernelによる
大幅な短縮は支持されず、本体のsnapshot方式は変更しない。これはoptimizer全step測定ではない。
source SHA256=`12d0f104fed3766aa5aef4e02c1906e5f2ae5a3d18df8199282e3a3b32b5c0af`、
result archive SHA256=`d7f64d032c63998a3739aeb4cafb9871279f02a1bb08caeee9adec2a24ef9a40`。

次の試作はCPython 3.12+のdictionary watcherを使い、同じ辞書群の変更を1個のstickyな
native epochへまとめる（commit `3bc361f7`）。function slot・辞書pointer・MRO・Tensor
storageの検証は別に残す。対応しないPython、watcher枠不足などでは従来の辞書version
guardへ戻す。epochは辞書やTensorを強参照しない。辞書変更後に値を元に戻しても古い
epochを有効にせず、通常経路で改めて認証する。CUDA演算・階層・responseの寿命は変えない。

helper単体の実機job `l4job-46a170d616a9441faaf7d197eb91c2b8`はPython 3.13/L4で
57 passed/1 skipped（skipは旧Python専用fallback検査）。PYTHONMALLOC=debug下で
変更・clone・GC・アドレス再利用・循環Tensor・watcher枯渇を検査した。
110辞書のRootGuard microは10.1803→.8162µs。ただしこれはhelperの結果であり、
Linear/Polarへ接続した完全step改善や目標達成はまだ意味しない。
source SHA256=`604a4defc47ad6b537e898541d07d619ca1f10dc635c14b59e3a20e7a0210679`、
result archive SHA256=`1fa7125c7ae3ae708c97779028bbe68cfac7e119c74e6f2ebbda35692b7c145b`。
各jobのspec/receipt/resultsは既存の`output/radius-prepared-entry-20261030/<job-id>/`へ保全。
次は両consumerでepochが実際に有効なこと、独立oracle/全勾配/状態bit一致、未計装の完全stepと
denseに対する両ピークを同じ条件で確認する。

Linear/Polarのconsumerは同じepoch checkerをnative RootGuardへ直接組み込み、余分な
Python wrapperを追加しない。tuple identity・function slot・MRO・storage/data guard、
非対応時の従来loopを維持する。CPU全体2871 passed/6254 skipped（53.85s）、
Ruff/diff検査が通過。Python 3.11ではwatcher本体を使えないため、GPU側のPython 3.13で
実際の両consumerの有効化と完全stepを別途検証する。

併行して調べたcoordinate-stepのimport移動は、全体テストで宣言/実行のimport境界を
破ったため本体変更を取り消した。約1µsのlocal診断短縮を理由に境界を変更しない。
fresh import順とlive callbackの回帰検査だけを残した。失敗logも同じoutputに保全している。

#### N128完全step・メモリ目標の達成（2026-10-31 JST）

commit `b6a285d0a1d43de8b41d4e60c388c3e16bc12299`の両consumerへのepoch接続を
実GPUで検証した。N128/B4/A819/M8/seed41、FP32 IEEE、公開CSTAdamW/fused AdamW、
raw decay=0・amplitude decay=.01・check_finite=False、10 warmup/101 samplesを維持。
下表は未計装・同期済みforward/loss/backward/optimizer完全stepのmedianで、microやtraceから
推定したものではない。CUDA49afの階層・完成response共有・更新則は比較両版で同じ。

| 独立process | 旧consumer ms | epoch接続 ms | 接続版自身のdense ms | CST/dense |
| --- | --- | --- | --- | --- |
| ABBA 1 | 1.509955 | 1.375293 | .707463 | 1.943979 |
| ABBA 2 | 1.407406 | 1.317650 | .682119 | 1.931701 |
| 公開runner 1 | — | 1.366333 | .722017 | 1.892383 |
| 公開runner 2 | — | 1.312774 | .706161 | 1.859029 |

**全4独立processで時間はdenseの2倍以内。** 公開runnerの2回は別のL4で、既存の
`radius-cell-mass-128-native-initial.json`、Plan catalog、runtime sourceを変更せず実行した。
各process自身のmatched denseを使い、異なるVMの絶対時間を直接差し引かない。
全回のCST allocated/reserved=**287232/2097152B**、dense=**17313280/23068672B**。
両ピークとも厳密にdense未満。総GPU processメモリやGraph captureの達成は主張しない。

GPU **550 passed/1 skipped**（skipは旧Python専用fallback検査）、初期/更新後の独立
Y・dX・全atom勾配、small-atom quadrature、matched-state optimizer gateが通過した。
旧consumerと接続版の5stepのY/dX/dP/Parameter/Adam state、35 Tensorがbit一致。
実際のprogramを変更せずguard構造とnative統計を確認し、Linear・Polarの両consumerが
epochを使用していることを検証した。従来のtuple TYPE/IDチェックは残り、旧版の辞書子node
133+58/134個が接続版では0個、watch登録は276辞書/325登録。slot/MRO/storage検証も残る。

比較job=`l4job-a847e658c7e24b79969b74e89bb76895`、
source SHA256=`9e2d2e8115ed8cbce1e265cce741ebd5d533e466bd98a379e9dce286a62f287c`、
result archive SHA256=`44597e1de5117f1e65887ee76952e9eac18357c15f7ee382f5f20cfb793099c3`。
公開runner確認job=`l4job-e5797d337ef34d309d928a6b56a82f0a`、
source SHA256=`b6b70a94706924776d7f3f2da7b47784d17b8980c154e415ef5efb4d87e22606`、
result archive SHA256=`cd41d0b4b4b7a194b1daeab0aa8e1b0f106aad26b1eef5a80dee7099c3b26b01`。
両jobのspec/receipt/resultsを既存の`output/radius-prepared-entry-20261030/<job-id>/`へ保全。
同commitのCPU validation CI（run 631）も成功。wheel/sdist内の変更5fileのbyte一致を確認済み。

この結果に基づき、限定された明示Planの研究経路としてepoch接続を採用する。非対応Pythonの
fallback、設定変更時の再検証、通常PyTorch hooks、read-lease/offload/retained-backward契約は
維持する。自動dispatchへの採用やmainへのmerge、MNIST/B32/N1024以降の達成とは別である。
次は既存MNIST harnessの中間CST層をradius/native版へ接続し、真のbatch 4から完全model
step・学習・分布・ピークを比較する。batch 32のglobal moments/response保存量と、
N1024/A52428のshared/layout上限は別途検証・設計する。

#### 次段階：既存MNIST harnessへのradius接続（2026-10-31 JST、実測中）

`benchmarks/training/cell_mass_mnist.py`へ明示cohort
`polar_radius_native_128_v1`を追加した（commit `91567f6e`）。旧sigma fixtureや既存cohortの
意味は変えない。構成は既存の784→128→128→10/ReLUで、中間128→128だけをCST/denseで
比較する。公開radius初期化の実際のW、外側層state、seed/data順、FP32 scalar buffer、
Planとoptimizer設定をartifactに固定する。振幅の再調整やsigmaからの変換は行わない。

N128/A819/M8/spacing=1、radius bounds=(minimum .5,birth 4,maximum 50,upper_floor 1)、
amplitude_max=1/w_c=.3。最小support直径`2R_min=spacing`で、R_max/R_min=100。
seed41の初期radiusは[3.985522,4.000000]に集中し、input/outputとも全819 atomが
level2/T=4/k=3に所属する。周期は`P=kT=12`。この学習実験は均等な階層分布ではなく、
学習前後の実radius分位点・bank人数・owner偏りを記録する。

真のbatch4から開始し、B32は明示opt-inの別条件とする。CSTは公開CSTAdamWとnative Polar
Planを使い、outer raw decay=.01、中間atom raw decay=0/logical amplitude decay=.01。
denseも同じlr=1e-4とordinary/middle decay=.01。両者の学習軌跡の一致は前提にしない。
先に実MNISTの最初のbatchでlogits/dX、CST中間層の独立Y/dX/全atom勾配とquadrature、
1stepのCPU参照optimizerを確認し、それからfresh processで各1epochを測る。
初回epochはoptimizer/JIT/cold準備を含むと明示し、隠れたwarmup更新は行わない。
数epochの比較では後続epochも別に示す。memoryはresidentデータ/modelを含め、
初期evaluation/backward warmupとtrainingのピーク上限、training単独のピークをともに残す。

CPU全体2890 passed/6254 skipped（55.36s）、cohort固有の初期化・旧経路互換・CLI・hash/
optimizer設定テスト、Ruff/diff検査が通過。GPU pilot jobは
`l4job-a7ff212818bc4c519b1cf2d4d49dd6d5`。MNISTの速度・メモリ・精度の達成はまだ主張しない。

同pilotは学習開始前、評価用`torch.no_grad()`のnative forwardで停止した。
C++ Function::applyは勾配を記録しない場合next_edgesを持たず、forward内の
`ctx->needs_input_grad(0)`が`Index out of range`になっていた。これまでの勾配を有効にした
完全stepの測定結果とは別の、評価/全凍結経路の不具合である。forwardでrouting keyを
保存する条件を`x.requires_grad()`へ変更する。実行可能なautograd nodeでは従来と同じ条件、
記録しないcontextではSavedVariables自体がcommitされず破棄される。backwardは変更しない。
checked/unchecked、no_grad/inference、全凍結、FP32/FP64と通常のVJPを含む50 GPU回帰条件を
追加し、C++ syntax検査を通した。新binaryの実compile/GPU検査と完全step再確認が必要。
failed pilotのsource SHA256=`ada602c6c8e34f0b36fd0d52730992ac25cf26dd49d18157653e7242ce0db643`、
result archive SHA256=`99fc642a0b1c8d9125f776d305fa3ce595ff8888c12a8f54bff2fcf6c7b730ee`。
spec/receipt/resultsは`output/radius-mnist-native-20261031/<job-id>/`へ保全し、学習速度には使わない。

#### no-grad修正とMNIST学習の実測結果（2026-10-31 JST）

commit `94a8338e`をjob `l4job-f9b54db727a74f82aeb857daaa381987`で実際にcompileし、
新module=`torchcst_radius_cell_mass_8d716ff5fa0d`、binary SHA256=
`f45f27cc8c443212acd7c81146bf22648ac86bb7060013bd66fcc952b2ae4b30`を回収した。
GPU **619 passed/1 skipped**、追加50条件、実MNIST最初のbatchのlogits/dX・独立中間層
Y/dX/全atom勾配・quadrature・CPU参照との1回の完全model更新が通過。
CPU全体2890 passed/6304 skipped（57.47s）、wheel/sdistのhost source byte一致も確認した。
既存N128公開runnerを変更せず再実行し、CST1.331525/1.381614ms、dense.741259/.702659ms、
1.796302/1.966265倍。CST287232/2097152B、dense17313280/23068672Bで、
no-grad修正後もN128/B4の時間・両ピーク目標を満たした。
同jobのsource SHA256=`a4a5ece9c39b09c01a8eabf0a9cc2f3c21ca54d6245ddc3c630e7332b44f0ce8`、
result archive SHA256=`f8a50ffdab61d7c7110dadc5a52151ee7975e89fb11cf44f522de3bfbc83a3e8`。

MNIST pilotの真のB4・1epochはCST平均2.168916ms/step、dense1.265943ms、1.71328倍。
test accuracy91.26%/93.35%、allocated240972288/241155584B、reservedは両方381681664B。
速度は2倍以内、allocatedは少ないがreservedは同じなので、MNISTの両ピーク目標は未達。

続く独立fresh-processのB4・3epochも全60000/10000件、同じ初期W/outer state/data順で比較した。
初期test accuracyは両者とも8.92%。epoch時間はinput gather/metricsを含む実際の完全学習loop、
evaluation/diagnostics/checkpointは別。epoch1はoptimizer/JIT/cold準備を含む。

| epoch（真のB4） | CST平均step ms | dense ms | CST/dense | CST test accuracy | dense test accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.261365 | 1.285185 | 1.759564 | 91.26% | 93.35% |
| 2 | 2.160005 | 1.274387 | 1.694937 | 92.79% | 95.67% |
| 3 | 2.146149 | 1.339832 | 1.601805 | 93.92% | 96.68% |

epoch2/3のallocated CST/dense=241439232/241623040B、reservedは両方383778816B。
CSTのallocatedは183808B少ないが、reservedは同値。3epochの精度差は2.76 percentage pointsで、
dense同等精度や他seedへの一般化は主張しない。radius中央値は3.999312→3.515043、
inputのlevel1/direct atomは0→8→18→34、level2は819→811→801→785へ変化した。
完成response・階層分類は学習に追従しており、均等な階層分布を仮定していない。

別の真のB32・1epochはCST2.774184ms、dense1.331469ms、**2.083551倍**。
allocated242218496/241331712BでCSTが886784B多く、reservedは両方383778816B。
test accuracy87.97%/91.22%。同じlrでもB4とB32では1epochのoptimizer更新回数が異なるため、
この精度差をbatch変更だけの優劣や収束比較とはみなさない。B32の時間・メモリ目標は未達。
次は別のmemory診断passでresident data、optimizer、Z/H/Gの保存とallocatorのinactive領域を
切り分ける。計装したpassの時間を未計装の学習速度として使わない。

B4・3epoch job=`l4job-17f16b67d5e443a0bca21786bfa2ea70`、
B32・1epoch job=`l4job-91dd916704c849f08a38071b46dc1b41`。
B4 source/archive SHA256=
`e9c9415bb54797d35955d30dabe812e9f669e37fa6240f94c7022fa35e3cb3dd` /
`85f1e8bc3e1d09e1721a64ab662b18c2847c627d64fa72ba9b9469b9c5627342`。
B32 source/archive SHA256=
`b6f942179b960722fcc611421f9d0eef4bad657f5c0d3e1d1084eb2b1512121a` /
`600d43f220c31f413ba1dca9f7ce82c144895d98b639d45ecac307b53186ba40`。
初期化artifact SHA256は全比較で
`ae13d18df2b8df94ff032bdd2e9de2679c76a323bcca4f9e53df2e8ec4bfa9de`。
spec/receipt/results/checkpointsは既存`output/radius-mnist-native-20261031/<job-id>/`に保全。

#### MNISTのallocator原因と同値なデータ準備（2026-10-31 JST）

別の計装job `l4job-64d702e96e7047f7bc56061daeb9bb2c`で、CST/dense×B4/B32×
original/inplaceの8条件についてallocation/free履歴、native stack、段階ごとのstorage inventory、
segment snapshotを回収した。これは20 warmup後の1step診断で、学習時間には使わない。
元の正規化は、31360000Bのtest画像を188743680Bのsegmentへ残し、157383680Bのinactive
領域を同じsegmentに保持していた。resident画像があるためempty_cacheだけではそのsegmentを
返却できない。計算中のCST bufferだけがreserved増加の原因ではなかった。

CST固有の保存もnative stackで別に確認した。F=2M+1=17、retained nodes J=126なので、
各軸momentsは`8*B*J*F`、各responseは`8*B*A` bytes。B32では548352Bと209664Bの
allocをforward/backwardでそれぞれ1回、計2回ずつ確認した。forward保存Zx/Hが758016B、
reverseでZy/Gも加わると1516032B。coeffsはsharedで、global C表のallocではない。
この保存量とbatch拡大を、データのinactive segmentとは分けて扱う。

commit `4a03f85c`は既存harnessに`--data-normalization inplace`を追加する。
float32への独立転送`copy=True`の後にdiv/sub/divをin-placeで行い、元画像は変更しない。
default `original`と旧fixture/protocolは保持する。両モデルに同じ選択を適用する。
診断でGPU画像hashがoriginal/inplace間で一致し、さらにfresh-processの実1epochを回した。

| データ準備inplace | CST平均step ms | dense ms | CST/dense | CST/dense allocated B | 両者reserved B |
| --- | --- | --- | --- | --- | --- |
| 真のB4 | 2.105118 | 1.265322 | 1.663702 | 241069568 / 241252864 | 245366784 |
| 真のB32 | 2.764622 | 1.302600 | 2.122388 | 242315776 / 241428992 | 247463936 |

reservedは両モデルともoriginalより136314880B（130MiB）少ない。
allocatedの一律97280B増加は、test画像のrequested storageは同じ31360000Bのまま、
inplace側では31457280B block全体がactiveとして数えられることに対応する。
CST/denseの差やB32の中間保存問題を、この共通の削減で解決したとは扱わない。
実学習checkpointのTensorをraw bytesで比較し、各batchでCSTの48 Tensor、denseの20 Tensorが
originalとbit一致した（model/optimizerの数値state）。精度・lossも同じ。データ準備の選択肢として
採用するが、別VMの小さな時間差をkernel速度改善とは呼ばない。CPU全体2893 passed/
6304 skipped（58.41s）、Ruff/diff検査が通過。

実学習jobはB4=`l4job-02c26f8ee72b40589c0edbd024f06698`、
B32=`l4job-b179d388d4474c50a9cdf8413775f153`。
raw trace、8条件のinventory/snapshot、比較checkpointとbyte検証JSONは既存outputの各jobに保全。

#### N1024汎用経路の初回実機検証（2026-10-31 JST、速度目標未達）

別worktree/branch `kernel/radius-large-shapes`の実験commit `3ab77f6a`/`afb7301e`は、
N1024/A52428へ既存sequential/global compact moments経路を拡張する。N128 shared経路の
上限は外さず、大きいowner keyは既存stable ATen sortへ戻す。共有H/G、階層、周期式、
profile、更新則を維持し、packed atomコピーは作らない。primary実装への採用はまだ行わない。

初回job `l4job-d966d8e48da84bd1b5b670dfe57fd232`は254 GPU tests後、CPU参照との
更新座標差で停止した。711.1985473632812対711.198486328125の差6.103515625e-5は
FP32の1隣接値で、その座標の更新前勾配はbit一致していた。旧固定2e-5は128座標では
数ULPだが1024座標では1ULP未満になる。kernelを変えず、large中心だけ要素ごとに
`max(2e-5, expectedからactual方向のFP32隣接間隔)`で確認する。隣接間隔が基準となる大きい
座標では2ULPを拒否し、非有限値も拒否する。
振幅s/t・N128・勾配・Adam stateの基準は維持する。差を生んだprimitiveは未特定。
別に同一GPU/P/grad/Adam stateからCSTAdamWと互換CSTOptimizer(同じfused AdamW/native Polar)
を動かすraw-byte gateを追加し、CPU丸め差とwrapper一致を分離した。

job `l4job-5b0bc245c12349c08ccefe95a3f6c1fd`では全52428 atomの独立Y/dX/dP、
stratified quadrature、3回のCPU参照更新と同GPU wrapper/state byte gateが通過した。
Y/dX/dP max absは約1.14e-10/1.16e-10/1.19e-7。参照は256 atomずつのCPU public実装を使い、
全atomを評価しながらN×Aの巨大なfactor/graphを保持しない。実compileしたmoduleは
`torchcst_radius_cell_mass_6b1501d1da24`、binary SHA256=
`f64b649f2ec90ed298749566e6aed03582536afc1270962c35dcc0830a29e97d`。

完全stepは2独立processで**50.864570/50.864004ms**、dense .679625/.719782ms、
74.8421/70.6658倍。CST allocated/reserved=9211904/31457280B、dense33900032/44040192B。
両ピークは少ないが、速度目標は大幅に未達。N128 regressionは1.84111倍で通過した。
N1024の高速化・MNIST1024達成とは主張せず、次は別passの実kernel profileで原因を特定する。

上記追加jobのsource / verified result archive SHA256：

- memory診断64d702：`417b327fe73bb1f1d1e2b81956854f9bed41ade4a2c14e18de10f0b828505d75` /
  `d42f1f55eacb4f3080c1b3da6e5adbeba46de0b5ada14c3920d2a6a4ab8b5ab1`。
- B4 inplace 02c26f：`f7b6d65752e604be8272e6ae10bba17a832ddbcd94be70e39e4e8ae161824acb` /
  `d21e5fcdd312749d311128042b4ac733fbf87163d1f7c599633ce014ee4f555c`。
- B32 inplace b179d3：`15cd1d30570dfd1831860ff172dc09439fe04f29d69268e65de3b0d095a55439` /
  `68f4f98aa47502e3a1f9711dafcc62520f0ede2865aa305dc9b4e0cb495ccde6`。
- N1024 qualified 5b0bc2：`97b4825b341d872ae073d3c7226ffd15c4c86c21955ff8ff0a5ed3d8d89a347c` /
  `694dce89b66ba45aef8059909756703cd319f20ad62d96edab4e710562ec72be`。

いずれもspec/receipt/resultsを既存`output/radius-mnist-native-20261031/<job-id>/`へ保全。
byte検証JSONは回収済みcheckpointからlocalで生成した追加検証で、pool原本archiveの一部とは
扱わない。large失敗d966の原本も保持し、速度集計には含めない。

#### N1024の実kernel profileと次の候補（2026-10-31 JST）

job `l4job-2545d6d169794f04bc89315cfabc6590`は、同じN1024/5%/B4の20 warmup後に
未計装の同期済み5step、別passのCUDA event、さらに3stepのCPU/CUDA profilerを採取した。
未計装medianは51.062833ms。別event区間はforward/loss22.882303ms、backward27.976704ms、
optimizer.038912ms。eventにはlaunch gapも入り、別passの値を加算/差し引きして完全stepを
再構成しない。以下はChrome traceの`cat=kernel`実eventだけを3stepで割った値であり、
record_functionのinclusive device時間を二重に加えない。

| CUDA kernel family | ms/step（計装pass） | 3step中のcall数 |
| --- | --- | --- |
| compact_coefficients_kernel | 43.160672 | 6 |
| hybrid_response_kernel（moment側） | 2.189932 | 6 |
| hybrid_response_kernel（direct側） | .576019 | 6 |
| hybrid_parameter_kernel | 1.913549 | 3 |
| compact_moments_kernel | 1.413029 | 54 |
| hybrid_expand_kernel | .407796 | 6 |

主因はC生成で、対応コードは`hybrid.cuh::compact_coefficient_tile`。
batch/出力node担当threadがatom列を直列に走査し、同じatomのprofile係数を複数node/rowで
繰り返している。profileだけで理論的な上限達成を主張せず、実際の共有化候補を比較する。

別branch `kernel/radius-batched-tile-coeffs`の実験commit `8329702c`は、4batch行/CTAで
profile係数を共有し、1bankの担当出力node分だけCをsharedに置く。Yはregisterに足し込む。
FP64中間計算、完成H/Gの再利用、従来path/defaultは維持し、global C/packed atoms/出力atomic
は追加しない。N128/M8/start2/tile32のshared budgetは37152B。新経路は明示Planでのみ選ぶ。
研究説明は新しい追跡対象documentを増やさず、この正本とignored outputに保全する。

初回job `l4job-ad592e9cc600477e8d68339aa27f8ca5`は新sourceを実compileしたが、
VJP test fixtureのphysical cap=23.68をFP32で上側23.680000305へ丸め、device assertで停止。
その後の同一processの多数の失敗は汚染されたCUDA contextによるもので、別々のkernel故障とは
数えない。runtimeのradius検査は緩めず、新test入力だけをcap以下の最大表現値に直す
（`5d601bec`）。FP64ではexact capを維持し、CPUで妥当性/最大性を検査する。
新binary=`torchcst_radius_cell_mass_3a377ca61557`、SHA256=
`13e060df64ce08cac976e290aeff816722326b614f386bcc4d3012f74cb30b3f`。
この失敗jobの速度値は採用しない。修正fixtureのjob
`l4job-13e083242d594ff3b1903fa7f3a03e83`で独立GPU検査とB32完全stepを再確認する。

N1024への最初の移植は別branch `kernel/radius-large-batched`で、global projection/parameter
gradientをそのままに、H/GからC生成とsite展開の対だけをこの独立consumerへ置き換える。
以下の実測後も既定pathは変更せず、明示Planの候補として扱う。既存N128の達成と混同しない。

### batched係数生成の実測とCPU参照の境界修正（2026-10-31）

CPU参照`axis_masses`はFP32 radiusをPython float境界と直接比較すると、その境界自体を
FP32へ丸めてしまっていた。`routing_radius=radius.detach().double()`で分類とdomain検査だけを
物理境界の精度へ揃える。積分と微分には元のradiusを使う。CUDA式・検査は変更しない。
spacing=0.37のnextafter境界で独立積分との差0.03467524が発生した参照側の不具合を修正し、
20件のCPU境界・Y/dX/全physical勾配回帰を追加した。primary commit `dad138ce`。

job `l4job-c35ea2bdf5e0425baee2338ed0f41bb2`は、この参照修正とcap以下fixtureでGPU299件を通過。
N128/B32/A819/M8の未計装完全stepは、2つのfresh processで旧2.268914/2.261464ms、
新1.820384/1.808972ms、dense0.700490/0.719276ms。同じglobal中間配列を保ち、
allocated/reservedは両CSTとも1684992/2097152B（dense17384960/23068672B）。
約20%短縮だがdense比2.60/2.51倍なので、このB32単体の速度目標は未達。
B4既存pathの回帰は1.353711ms / dense0.686574ms=1.97169倍。
GPU検証済み候補をprimaryへ`fb99c7b7`と`dcb6f3e1`で取り込み、
`radius-native-batched-cached-parallel`を明示した時だけ使用する。自動dispatchは変えない。
source SHA256=`3d84704212acd2552cf39c8c8ddc239c8a54d82b8b6b1fd381acb33b35ec504c`、
結果archive SHA256=`05a42282cb5ab3e7ec263f4bd050234d0c5a8370bf0974b62e5ebe4d9d121087`。

別branch `kernel/radius-large-batched` / `d3f6dac0`のN1024/B4/A52428/M8は
job `l4job-62d2ca70e9f244aab496c00c7816eb83`で実compile、GPU351件、
全atomの独立Y/dX/dPとmatched-state更新を通過した。global moments/responseとparameter gradは
旧版のまま、完成H/Gのconsumerだけtile-local batched Cへ交換した比較である。
旧50.817124/50.817638ms→新9.000168/9.018308ms、dense0.706106/0.712809ms。
約5.64倍速いがdense比12.75/12.65倍で、1024の速度目標は未達。
両CSTピーク9211904/31457280B、dense33900032/44040192Bなのでメモリは両指標とも下回る。
この結果はLinear単体の完全stepであり、1024のMNIST学習結果ではない。
新binary=`torchcst_radius_cell_mass_2ee89e4aa200.so`、
SHA256=`7de2ba2302f61fae417f92b8b83bcfd34294ae05ee537950f7a95ca30fa0f096`。
source SHA256=`41f3b8a2a63c436410d06827e1c6b2bcb5565d87b7960e13ce9caaa1edcb2ec2`、
結果archive SHA256=`c609fe2d447222c7a82878856c972057c6635454329b93765f2a8dc2b60f51b2`。
rawは既存`output/radius-mnist-native-20261031/`と共有pool jobへ保全する。

### MNISTへのbatched Plan接続の完全step再測定（2026-10-31）

既存harnessへ`--radius-execution cached|batched`を追加した（既定cached）。初期化のPlanとhashは
変更せず、実際の実行Planを`execution_linear_plan`へ別記する。warmup/evaluationを含むメモリ
定義、データ順、Polar更新則も維持。両jobの初期化SHAは従来ae13d18d…のままである。

- 真のB4、inplaceデータ、1epoch：CST 2.105509 / dense 1.239231ms = 1.69905倍。
  allocated/reserved：CST [241069568, 245366784]、dense [241252864, 245366784]B。
  test accuracy 91.26% / 93.35%。job `l4job-ef1feee3b02e4798bb90701327e1dd5b`。
  source_sha256=`7c1e4aaae7dc2862eda3f1d4f617fb4b24664140a2659826526c8c92ea71eb4a`。
  archive_sha256=`457e7c9ea1e4b9d269056757079e34670f25ead7f39b21c7f9ee0dbb968fe064`。
- 真のB32、inplaceデータ、1epoch：CST 2.765070 / dense 1.365386ms = 2.02512倍。
  allocated/reserved：CST [242315776, 247463936]、dense [241428992, 247463936]B。
  test accuracy 87.97% / 91.22%。job `l4job-7c11bb77a5c4452f8c2872c2b8666b41`。
  source_sha256=`8efd6378c5bd54e511b35221dfa983553a80eceed2c51671b8ec9f20748bd428`。
  archive_sha256=`825e7cebc8c6f028008d51a3637ece36c13e56c92402268cffdf48c0c4973b09`。

両jobで最初の実MNIST batchのY/dX/全atom勾配・独立積分・full-model更新検査を通過。
単体B32の約20%短縮をfull-modelの短縮と読み替えない。MNIST完全stepは旧版とほぼ同じで、
B32の速度2倍以内およびメモリdense未満は未達。B4は速度を満たしallocatedは下回るがreservedは同値。
次は既存direct-level設定の損益分岐とglobal中間配列の寿命を別々に調べる。

### direct境界の損益とMNIST host/GPU分離診断（2026-10-31）

`--radius-direct-levels`は実行Planだけを変える既存recipeの調整入口（既定2、0..8）。
初期atom・profile・M・frozen Planは変えない。level5ならT32以上で階層を使い、
小さいradiusを直接計算する。Zpairは予算上1096704→121856Bに減る。

B32/inplace/1epochの実測はCST 10.127948 / dense 1.308486ms = 7.74020倍。
allocated/reservedはCST [241340928, 245366784]、dense [241428992, 247463936]B。
両メモリ指標を下回ったが速度が悪化したので未採用。初期実batchの独立勾配・更新検査を通過し、
accuracyは既存B32と同じ87.97/91.22%。job `l4job-1158bee098a74fbdbef9a0b9c651fd29`。
source_sha256=`eea13217837bed3c36292fec398c070c20d97508c223a9a88aefba79c8cdaedd`。
archive_sha256=`5f56ee67942a20cf2063089629b975bd7451776942adea6407c94c977fdbd6f9`。

混在radiusでdirectとretained T32/T64の両方を通るGPU VJP検査も追加した。
この追加ケースのGPU実行は次の候補比較で確認する（追加時点は未実行）。

別診断job `l4job-3621f468ad3e471a9895f360fe4c1a90`ではB32実MNIST先頭batchを継続し、30warmup後の
未計装15step、別の3step Torch trace、別の30step cProfileを測る。epoch速度証拠とは区別する。
未計装median CST2.298030/dense1.473265ms。traceのcat=kernelだけの実GPU集計では
hybrid_parameter_kernel0.770267ms/step、batched C/site0.288894、sharedmoments0.183231、
parallelresponse0.101738。parameter_gradのbatch loopが固定atomのphaseを繰り返す。
CPU cProfileでは30stepでvalidate_inputs60calls/12ms inclusive、prepared._execute30calls/14ms、
run_backward30calls/19ms。inclusive費用を足し合わせたりepoch速度から差し引かない。
同期eventには測定終端のcudaDeviceSynchronizeが含まれるため、これをruntime内部同期と断定しない。
source_sha256=`11f5d9eaf9941d9e106258f52064798cf07533c6e4679525a761db95679635c4`。
archive_sha256=`589d090097d2463691e789da49f8bde3948e711364a29ba291dd17833d3f164f`。

### M8多項式のHorner候補：正しさ通過、速度効果なし（2026-10-31）

`kernel/radius-m8-horner` / `725cac58`は、既存の正係数・斉次多項式を
割り算なしHornerへ変形する差分候補。独立Chebyshev Laurent多項式のFraction展開で
全係数と微分を厳密照合し、元のprofile・正規化・Mは変更しない。
job `l4job-00a24777337643aea8b0c1a2d70f4352`で実compile/GPU差分10件を通過。

execution子branch `kernel/radius-m8-horner-execution` / `5917ce5b`だけでM8の評価器を交換した。
job `l4job-5acca242c13142d5a709a9178db38347`で実compile、GPU309件通過。
B32単体のcached/batched完全stepはrun1 2.252948/1.854375ms、run2 2.244845/1.829956ms、
dense0.728902/0.728516ms。旧batched1.820384/1.808972msより改善せず、
MNIST B32/inplace/1epochも2.761022ms / dense1.332986ms=2.07131倍、
ピーク242315776/247463936B / dense241428992/247463936Bで変わらない。
未採用として元評価器をprimaryに保持する。演算数の理論減少だけで性能向上としない。

`l4job-00a24777337643aea8b0c1a2d70f4352`：
source_sha256=`f43e59e0a163c63d3c6d49192dbf3fe829fc8578b4a3eff48249c96c96a1c512`。
archive_sha256=`a63214b38aca4374dc8614959cf6f9fffbcbb38ffc3c8939e37f0f4e8ad932ba`。
`l4job-5acca242c13142d5a709a9178db38347`：
source_sha256=`96456d49d4243e6646d4d2fa47064b05f74ff738cb148d45f31473ef7ccbbc84`。
archive_sha256=`fe77a4a01514941a1138ff4e3d8cc6a996901205cfe083ed46b4e08c6da8c5ca`。

4行response候補job `l4job-46fe8eac364a42d8b483757cceb67fd3`は独立微分/更新を含む392件を
通過したが、新設したH/Gのbitwise比較15件で約1e-16の差が出て停止し、速度は未測定。
実binary PTXでは、同じsine再帰のFMAに取り込まれる積の側がscalarと4行版で異なることを
確認した。独立oracle/optimizerの既存許容値は変えず、新stage比較だけをFP64丸め誤差上界と
finite検査へ修正して別jobで再確認する。失敗結果も同じoutputに保全する。

### N1024 response4共有の再確認（2026-10-31）

test-only修正`edd56271`後、job `l4job-e31b11ba10894bd7ac9802b944c21584`で
前回実compile済み`torchcst_radius_cell_mass_2ac0d353250a`をSHA検証して再使用した。
このjobでの再compileではない。GPU152件と全atom独立Fourier/小部分quadrature、
matched-state更新を通過。実binary SHA256は
`de50e3da970074b42c864809761f1d88d5a33085ed8db045e35bfe0f11add264`。

N1024/B4/A52428/M8の2fresh processは、旧global-batched9.039005/9.019444ms、
response4版7.006179/7.005878ms、dense0.716053/0.730737ms。
allocated/reservedは両CST9211904/31457280B、dense33900032/44040192Bで変わらない。
約22%短縮だがdense比9.78/9.59倍なので速度目標は未達。mainへの自動採用はしない。

N128 native/batchedとlarge generalの統合候補は別branch
`kernel/radius-large-integration` / `f31084e1`へ保存し、CPU3062件と配布物確認を通過した。
fresh union CUDA binaryの128/1024両方のGPU再確認は次jobで行う。
この節の測定だけを統合版の実測としない。

source_sha256=`fea7f07892d4d7d78f32e183a4a15e9924b2a90f29b0f496d425edf6793ef22e`.

archive_sha256=`6a9d26043ea755957106d2f39d4f89f07e42197847be146ffe14bac059dc5035`.

### 統合版の実GPU確認と採用（2026-10-31）

job `l4job-9bf65b69e5764a4f8f9ebee14a9a65b1`で統合版を実compileし、GPU661件を通過。
module=`torchcst_radius_cell_mass_bcd98ac2a3cb`、binary SHA256=
`91d49385563731ab8cb6df92a2ee4c12bb9387069bb4d7feba7da9fc4c1887dd`。
N128/B4は1.378396/1.388928ms、dense0.709148/0.715507msで両方約1.94倍。
ピークは従来のCST287232/2097152B、dense17313280/23068672Bを維持する。
N1024/B4のresponse4版は6.993471/6.988053ms、dense0.711520/0.709722ms、
ピークCST9211904/31457280B、dense33900032/44040192B。large速度目標は未達。

primaryへ`295b8d79`で採用し、測定snapshotのruntime383ファイルがすべてbyte一致することを確認。
general経路はN1024/A65536までの明示Plan、native/block-local経路はN128/A4096/48KiBまで。
既定dispatchを変えず、N1024のMNIST学習は別に検証する。

### 4行分のparameter grad共有（2026-10-31）

別branch `kernel/radius-batch4-grad` / `581b914f`とtest-only修正`20a19e29`は、
1atom/threadを保ち、4batch行で位相を共有する。完成H/GとZ、batch加算順、FP64中間計算を維持し、
追加global配列や出力atomicを使わない。物理radius勾配も省略しない。
新設差分検査はFP32でbyte一致、FP64ではfiniteと明記した診断用丸めbudgetを検査し、
既存の独立oracle許容値は変えない。FP64の具体的なFMA差の原因は未証明とする。

job `l4job-356d240af2ad45338594563f099c1598`は、先行compile job
`l4job-11077ac601214b5f8637a23af9543da3`の同じbinaryをSHA検証して再使用し、GPU413件を通過。
module=`torchcst_radius_cell_mass_05524d1f41b5`、binary SHA256=
`13803067676af9fc49f64610943302f72edbc1b76d7bc9ff6fe561e54013ce6a`。
先行jobは新設FP64 bitwise差分20件で停止したが、独立検査とFP32 byte比較を含む372件は通過した。

B32単体の新batched完全stepは1.427050/1.427654ms、dense0.727122/0.717842msで
両方2倍以内。ピークは従来の1684992/2097152Bでdense17384960/23068672Bを下回る。
一方MNIST B32/inplace/1epochは2.783016ms / dense1.321053ms=2.10666倍で速度改善なし。
direct_levels5では5.605553ms（旧10.127948msより短縮）、dense比4.24324倍。
そのメモリ241340928/245366784Bはdense241428992/247463936Bを下回るが、速度との同時達成は未達。
この候補は統合large版への移植を別途GPU検証してから取り込む。

`l4job-9bf65b69e5764a4f8f9ebee14a9a65b1`:
source_sha256=`f5e07b43df39a255de19f7dcdc8dbb259c379226bbf1bf482fc85c94f0e0662a`.
archive_sha256=`2304f2837399805d472f7f1fa79629bb48f33cbb6c594bd95b59b9935bcbfde2`.

`l4job-356d240af2ad45338594563f099c1598`:
source_sha256=`41c9852a0579a04365cd59e163df3016bedc367c0613cbb12d163fac507db68f`.
archive_sha256=`c303e8cb1de571ceee7661ee8bd6e6169af958afe4250903fffe91f0dbf31a9a`.

### N1024 MNISTの3epoch実学習（2026-10-31）

既存harnessの別cohort `polar_radius_global_response4_1024_v1`を`9197e8e1`で採用した。
N128初期化と旧sigma fixtureは変更せず、N1024はA52428、spacing1、M8、物理radius bounds
0.5/4/50/1、同じPolar更新則、真のB4を明示opt-inして使う。
`global-response4`はnative host flags=falseのglobal経路で、N128 native版と混同しない。
初期Wは256atomずつのpublic CPU参照から一度だけ作り、両モデルに同じWと外側層を渡す。
CPU全体参照も完成したmiddle Yから一つの大域dYを作ってからchunk VJPし、optimizerは最後に一度更新する。

job `l4job-0acd946cd9904de9af6ae11523bcff06`は、実MNIST先頭batchの全model Y/dX/全parameter勾配・
全atom Fourier参照と層化8atomの独立積分・full-model CPU-reference更新を通過し、全splitを3epoch実行。
seed41、784→1024→1024→10の中間層だけCST、同じ初期W・外側層・data order・optimizer条件で比較した。
初期化artifact SHA256=`5ef7a7890367641835d1d5216ff2a9d21a2c4638f6d73fa7f2a3c2582319a539`。
以前に検証した実compile済みbcd98ac2a3cb binaryをSHA検証して再使用した。

| epoch | CST step ms | dense step ms | 比率 | CST test | dense test |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.688933 | 1.304621 | 5.12711 | 94.87% | 96.41% |
| 2 | 6.623426 | 1.243213 | 5.32767 | 96.47% | 96.94% |
| 3 | 6.756943 | 1.255705 | 5.38100 | 97.37% | 97.87% |

各epochのCSTピークallocated/reservedは258337792/274726912B、denseはepoch1で
270021120/287309824B、epoch2/3で270484480/287309824B。両指標ともCSTが下回る。
stepはgather/loss/backward/optimizer/metricsを含むepoch wall、最初のepochはoptimizer cold start込み。
1seed/3epochの結果で、精度同等・一般化や速度目標達成は主張しない。
radius中央値はepoch1/2/3で3.96607/3.95073/3.93860。3epoch後はlevel2/T4/K3が52423atom、
level1/T2/K3が5atomで、均等な階層分布ではない。input/outputの物理radiusは同じだがowner分布は別々に記録する。

source SHA256=`812249c459a0ffdde9542355cfb55faa66ecbe5cd638c0e887a9d1c2c44a119f`、
archive SHA256=`227c3c320e4c4f1d8dc9d89a7e172abcd920a156b6a60937c6920bee7197a214`。

### large grad4とactive momentsの段階別検証（2026-10-31）

job `l4job-58828a5eccac4e53805d80b2f97b0864`でlarge grad4版を実compileし、GPU795件を通過。
N1024完全stepは旧response4 6.978640/6.985781ms→grad4 6.255403/6.253121ms、
dense0.696032/0.724249ms。ピークは9211904/31457280Bで変わらず、large速度目標は未達。
small B32の2fresh processでも1.431758/1.413957ms（dense0.739813/0.755772ms）で2倍以内。
同候補をprimaryへ`1e20cfdf`で採用した。旧global aliasのgradient選択は変えず、
新`radius-global-response4-grad4`を明示した時だけlarge grad4を使う。

別branch `kernel/radius-active-moments` / `5dc47ff2`は4BのGPU bank maskで未使用moment計算を省く。
mergeでは選択bankの同じKの子をstart階層まで含め、完全なZ/H/Gだけをconsumerへ渡す。
Zのshapeとhostのlevel launch数は変えず、メモリ削減とは呼ばない。
job `l4job-435995bebefa47709ac781551c23b3af`で実compile、GPU718件を通過。
N1024旧6.987939/6.992331ms→active5.810253/5.808056ms（dense0.698520/0.717056ms）。
ピークは9211904/31457280Bのまま。これはgrad4とは別の比較で、改善率を単純に足し合わせない。
両者の統合は別候補で確認する。

### prepared declarationの単純省略は未採用（2026-10-31）

診断3621でaxis configとrecipe検査は合計約13us/step、execution_declarationのinclusive範囲は約0.137ms/step。
後者を既存prepared guardだけで省く案はCPU反例で中止した。同じnative RootGuardManagerに
`add_lambda_guard`で処理を追加してもprepared guardは通るが、通常のdeclaration呼出しは追加処理を実行する。
省略すると挙動が変わるため、この0.137msを安全に除ける時間とは扱わない。
proofはignored `output/radius-prepared-live-entry-audit-20261031/`に保全し、runtimeは変更しない。

`l4job-58828a5eccac4e53805d80b2f97b0864`:
source_sha256=`395bba85296134108a30f82926cc9a912ea8210794fc2c01b723a7175f545475`.
archive_sha256=`4d0142a796a5b6f775d32cd7307994f97e0d155bfe841deeea0bc1ac0d45bfe9`.

`l4job-435995bebefa47709ac781551c23b3af`:
source_sha256=`ae2e29a69e12195d2b807f3351201459864cfec6a65bd0b21aae3ab2d81a8ffb`.
archive_sha256=`70f01c6c96c1390a88e9a654ec43af938bf0018b843aba9e342e24f7b203499d`.

### active moments＋grad4の統合とMNIST再測定（2026-10-31）

`18941c11`で明示`radius-global-active-grad4`を採用した。旧global Planの選択は変更しない。
実compileしたjob `l4job-6827da61ce564c0cafb06a7572ddec02`でGPU884件を通過し、
N1024/B4単体は非active grad4の6.463579/6.454866msから5.181212/5.219086msへ短縮した。
denseは0.707896/0.712054ms、allocated/reservedは9211904/31457280Bのまま。
active maskは4Bで、選択bankとmergeに必要な子bankの計算だけ残す。Z配列を縮小した結果ではない。

MNISTは既存harnessに`--radius-plan`で登録済みPlanのJSONを渡す。設定は実行前に検証し、
raw/canonical Plan hashを結果に残す。データ処理・初期化・optimizer条件は変更しない。
job `l4job-812252a12de64eabae44a6f0dfe993a9`は、前回0acdの初期化artifactを同じbyte列で再利用し、
真のB4、784→1024→1024→10、seed41、inplace normalizationで3epochを実行した。
実compile済みnative `e73fa314ccd1`をbinary SHA256
`d15c4b07689079c2ec85833d8052ab45a329087c8f02e8945cb5a3e7221815a9`で照合して再使用した。
実先頭batchの全model Y/dX・全Parameter勾配・public CPU-reference更新、全atom Fourierと
層化8atomの独立積分を別processで通過してから学習した。

| epoch | CST step ms | dense step ms | 比率 | CST test | dense test |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.774085 | 1.268861 | 3.76250 | 94.87% | 96.41% |
| 2 | 4.706828 | 1.318559 | 3.56968 | 96.47% | 96.94% |
| 3 | 4.745863 | 1.317406 | 3.60243 | 97.37% | 97.87% |

CSTピークは全epochで258337792/274726912B。denseはepoch1で270021120/287309824B、
epoch2/3で270484480/287309824B。両指標とも下回るが、速度2倍以内は未達。
stepはgather/loss/backward/optimizer/metricsを含むepoch wall、epoch1はoptimizer cold start込み。
同じ学習の別実行なので、baselineとの時間差と各job内のdense比を分けて読む。
1seed/3epochの結果であり、精度同等・一般化を主張しない。

`l4job-6827da61ce564c0cafb06a7572ddec02`:
source_sha256=`5f3439ec816076684a2b6f89a561ae21c44319baf679dec1f6a007a2a910b4d2`.
archive_sha256=`54409efb7f56e0d01ce3725f01d15b530dd168568bb88c803da5cbf1bc5d85f7`.

`l4job-812252a12de64eabae44a6f0dfe993a9`:
source_sha256=`632f30269cb49fb86f0f4df0b81fed6f3db1a3c060210b1fc693e125b395a6b4`.
archive_sha256=`7373e67ae9941dbf8e3bc93ec4108cfe630aa2cdcdfb21d132d701272b0de9ac`.

再測定後のcheckpoint照合では、CSTの保存済み最終model33／optimizer15 tensor、
denseのmodel5／optimizer15 tensorが前回0acdとbyte一致した。全epochの学習・評価metric、
radius診断とdata-order hashも一致した。epoch1/2のtensor checkpointは保存していないため、
途中の全tensor一致とは主張しない。照合記録はignored
`output/radius-mnist-native-20261031/checkpoint-active-grad4-audit.json`、SHA256
`4268b1e9e110b3887c3fe0452a628d3ade5f9a958631b8f8b1b6218953851cb9`。

`18941c11`統合状態でCPU全体は3163 passed／6592 skipped／既存警告19件。
CUDA skipは追加のGPU検証件数に数えない。再現はcommand-localの
`CXX=/usr/bin/clang++ TORCH_NO_COMPILER_WRAPPER=1 MAX_JOBS=1 CST_TEST_NATIVE_READ_LEASE=1 CST_TEST_NATIVE_FUNCTION_GUARD=1 PYTHONPATH=src:tests python -m tools.kernel_dev test --suite cpu`。

### value-only統合の段階別検証と、実MNIST profile（2026-10-31）

研究branch `kernel/radius-active-value-profiles` / `fd0854a0`は、H/Yなど値だけが必要な箇所で
radius微分の係数生成を省き、Polar taskが使わないradius勾配も作らない。物理radius経路には微分を残す。
階層moments・完成H/G・既存の中心／振幅勾配・Polar更新則は維持する。
job `l4job-de59e907480b496dbd41daf40debd896`でnative `8b4a8fd887b6`を実compileし、
GPU960 passed／1 old-Python-only skipped、公開runnerのN128/B4、B32、N1024/B4を通過した。
その後のMNIST用driverで`zlib`のimport不足により停止したため、**job全体はfailed**。
保存済みの上記stage成果は個別に確認し、MNIST実学習の証拠としては扱わない。

N1024単体のactive+grad4は4.300627/4.302096ms、同jobの非active grad4は
5.483575/5.484421ms、denseは0.721757/0.719460ms。allocated/reservedは従来の
9211904/31457280Bで変わらない。N128/B4は1.378838/1.351073ms対dense0.713845/0.724613ms、
B32 batchedは1.348727/1.347308ms対dense0.740251/0.726456msで、いずれも単体目標を維持した。
これは別binaryとの同時AB比較ではなく、このsnapshotのPlan比較とfresh controlsである。

実compile済みbinary SHA256=`9e464d0918ba80833b3cc324cd48f6a4e26e7425eb5c6d94d11e11de4aa9c5a5`。
修正後の学習はこのbinaryと全native source hashを照合して別jobで行う。

独立診断job `l4job-8001b86aab544ac89a31e79d525047f4`は同じ8b binaryと初期化5ef7を照合し、
実MNIST N1024/B4の同じ30更新prefixを復元したfresh processで、未計装control、Torch profiler、
cProfile、memory履歴を別々に取得した。未計装300更新prefixはCST平均3.954423ms対dense1.303753ms、
ピーク258337792/274726912B対270021120/287309824B。これは3epoch平均の代用ではない。

Chrome traceの`cat=kernel`だけを5step分集計した時間は以下。GPU user annotationの包含時間は除いた。

| CUDA kernelの処理 | 対応する実装 | ms/step（診断） |
| --- | --- | ---: |
| coeffsとY/dX展開の2回分 | `batched_cached_coefficients.cuh` | 1.54653 |
| Polar parameter勾配 | `batched_parameter_grad.cuh` | 0.83805 |
| 完成H/Gの2回分 | `parallel_projection.cuh` | 0.62048 |
| 入出力の再分類・2回分 | `pipeline.cu`のclassify | 0.34366 |
| active bank mask・2回分 | `active_moments.cuh` | 0.16397 |
| 18回のmoment level launch合計 | compact moments | 0.07056 |

CPUの30step cProfileではlayout_ranges60呼出しが約9ms inclusive、forward/backward wrapperが
それぞれ約10/11ms、optimizer validation60呼出しが約15ms inclusiveだった。
これらは親子区間と非同期投入を含み、GPU時間との足し算・未計装stepからの差し引きはできない。
最大のGPU処理はmoments構築ではなくcoeffs/展開である。CPU同期が新たに見つかった、とは結論しない。

Zx/Zyはそれぞれ[4,1022,17] FP64=555968B、H/Gはそれぞれ[4,52428] FP64=1677696B。
4配列の同時payloadは4467328Bで、sort・元Parameter・optimizer・resident dataは別に存在する。
Zx/HはforwardからdPまで、Zy/Gはbackward生成からdX/dPまで保持する。global coeffs配列はない。
保存hookでZx/H、allocation履歴でbackward peakを確認し、optimizerはそのpeakを上回らなかった。
cuobjdumpはFP32 expansion REG128/STACK704、Polar grad4 REG128/STACK704、response REG80/STACK568を報告した。
LOCAL0はstack traffic/spillがない証拠ではなく、動的sharedはこの静的dumpのshared0にも含まれない。

現統合`ac55f9e6`のwheel/sdistはruntime386ファイルをsourceへbyte照合済み。
同headのGitHub CPU validation run644もsuccess。新候補のGPU資格をこのCPU CIで代用しない。

### value-only統合後のMNIST完走と採用（2026-10-31）

import不足を直した学習専用job `l4job-3bee545a736a459ab7881270e3973f9a`は、
前節と同じ実compile済み8b binaryと初期化artifact5ef7のbyte／native source／ABIを検証し、
独立先頭batch gateを通過後、既存harness・同じPlan／データ順で真のB4を3epoch実行した。

| epoch | CST step ms | dense step ms | 比率 | CST test | dense test |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.990713 | 1.271784 | 3.13789 | 94.87% | 96.41% |
| 2 | 3.917658 | 1.351526 | 2.89869 | 96.47% | 96.94% |
| 3 | 3.963593 | 1.263152 | 3.13786 | 97.37% | 97.87% |

CSTのallocated/reservedは258337792/274726912Bで、denseのepoch1
270021120/287309824B、epoch2/3 270484480/287309824Bを下回る。
速度2倍以内は依然未達。epoch1はoptimizer cold startを含み、評価と診断はstep時間から除く。
前回0acdと最終CST model33／optimizer15、dense model5／optimizer15 tensorが全てbyte一致した。
途中epochのtensor一致は未検証。独立gate・学習metricを変えずに約6.7ms→約4.0msへ短縮できた範囲を採用し、
primaryへ`abdb2c88`で取り込んだ。runtime/tests/benchmark全byteはGPU検証snapshot fd0854a0と同一である。
照合記録`output/radius-mnist-native-20261031/checkpoint-active-value-audit.json`のSHA256は
`7effc82817af8a40132d74c73f6b3224c212eb6480843e03165d761bfb64e2fc`。

`l4job-de59e907480b496dbd41daf40debd896`（部分成果のみ。job全体はimport不足でfailed）:
source_sha256=`3a99781dada8f9f6517b080445896b7cfa455a714fbadba9378ce4d25f1936bf`.
archive_sha256=`a80fbccfde8180333c30fc8abc2fa514fc09fced400e833dbb036b9c5b79ba57`.

`l4job-3bee545a736a459ab7881270e3973f9a`:
source_sha256=`684b61a614fa021ab3b6a417d0f211ccaf55e44b8ef38514ea128a7a2621a74f`.
archive_sha256=`cfb57967d9d9f899cdafd94ad53994531d98aa2367f783488a5dc533207253f2`.

`l4job-8001b86aab544ac89a31e79d525047f4`（別processの診断と未計装prefix control）:
source_sha256=`57c7d8125d6e594d9cb8203b951f75bb4a57456f5d882e9eb8315d5b3ac5e49a`.
archive_sha256=`5a6ce5143f37d2e24667d22748f71f9b2e397434a30d85a03062690c49324349`.

`abdb2c88`採用状態のCPU全体は3168 passed／6664 skipped／既存警告19件。
wheel/sdistはruntime388ファイルをsourceへbyte照合した。検証記録はignored
`output/radius-mnist-native-20261031/{primary-abdb2c88-cpu.log,package-abdb2c88-audit.json}`。
GPU native source・Python runtime・tests・benchmarkはfd0854a0との全byte一致を確認済み。

GitHub CPU run645は新しいCPU用C++ smokeの`sincos`宣言がglibcの例外指定と衝突し、
1 failed／3035 passed／6796 skippedとなった。CUDA本体の失敗ではない。
CPU shimを`radius_test_sincos`へ分離し、標準cmathを先に読み込んでからCUDA呼出しだけmacroで向ける。
macOS clangの該当実算術テストは1 passed。Linuxは次のCIで再検証する。
この変更はtestsのみで、qualified8b native sourceとruntimeのbyteは変更しない。
元ログはignored `output/radius-mnist-native-20261031/ci645-github-job.log`へ保全した。

### global native hostの採用と、独立の候補比較（2026-10-31）

`5143b4ee`で新しい明示Plan `radius-global-native-active-grad4`を採用した。
CUDAの算術kernelは8b版と同じで、C++ autogradがactive Z→完成H/G→shared C/展開を呼び、
backwardはdXの後で保存Zx/H・Zy/Gを使うgrad4を呼ぶ。Pythonのstage間呼出しとbackwardの再分類を減らす。
小型nativeのshape/shared制約は変えず、global専用mode5だけN<=1024/A<=65536へ対応する。
各入口でmetadataを検査し、SavedVariables、保存hook/read lease、version検査、once_differentiable、
no_grad/inference、atom-only/dX-only、custom Tensor/モードのfallbackを保つ。mode4はこの版では未対応のまま。
旧Plan・自動dispatch・MNIST初期化の既定選択は変えない。

job `l4job-86ba6e64c49442a4b53ba0c3bd452177`でnative3b5305691809を実compileし、
GPU991 passed／1 skipped、実MNIST first-batchの全model／全atom／積分／CPU-reference更新gateを通過した。
公開N1024/B4のpaired比較は旧Python active 4.362189/4.347841ms→native4.016808/4.008405ms、
dense0.721831/0.721784ms。nativeピーク9179136/29360128Bは旧9211904/31457280Bより低かった。
保存key自体は8A bytesだが、実peakは他の一時Tensorの寿命も含む。keyの容量だけからpeak増加とは判断しない。

同じ初期化5ef7・真のB4・データ順・optimizerでMNISTを3epoch比較した。

| epoch | CST step ms | dense step ms | 比率 | CST allocated/reserved B |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.824976 | 1.302248 | 2.93721 | 258305024 / 272629760 |
| 2 | 3.752841 | 1.282409 | 2.92640 | 258305024 / 274726912 |
| 3 | 3.800630 | 1.304780 | 2.91285 | 258305024 / 272629760 |

denseのpeakはepoch1で270021120/287309824B、epoch2/3で270484480/287309824B。
両メモリ指標は下回るが、速度2倍以内は未達。最終精度97.37%対97.87%は変わらず、
最終CST model33/optimizer15・dense model5/optimizer15 tensorは初版0acdと全byte一致した。
途中epochのtensorは未保存。照合記録`checkpoint-global-native-audit.json`のSHA256は
`6d0d7cbbde4a88ed885097e30d867048ba76a2e06422277dc5015ba9777483e5`。
binary SHA256=`aee6e775d967a8774119033983d030b76a2dee674309fe85f81212ad3eb9ff94`。
source_sha256=`d56ccb125086fc7377e42ad24050745a9a6aa3d200179c36fd2607f82a8aaef1`.
archive_sha256=`904041942d11c24c3740240242e43484922cbae670848a7d47c086c9fb54683c`.

**same-profile候補（未採用）：** 別branch6e85c5a0はPolarの入出力radius則・bank geometryが完全一致するときだけ
parameter勾配のprofile生成を1回にする。異なるgeometryや物理radiusは既存経路のまま。
job `l4job-a33351a4c0a1440e9a9a3ef54efc6592`の実compileでGPU931 passed／1 skippedと
generic/optimized FP32勾配のbyte一致24ケースを通過した。N1024 activeは4.151259/4.146184ms、
以前の8b 4.300627/4.302096msより短い一方、小型B32は1.429740/1.343383msとばらついた。
MNIST・新nativeとの統合は未測定で、改善率を加算しない。
source_sha256=`61c9ed3cb5e69d60bd984e5ce6cc07d5f8f993c989a3e5bd1d0467de7fbd4dde`.
archive_sha256=`e793af5c68fafcf82bd97ac3d7598f2bd9cde50e738a9651fae3656a49c3d064`.

**local tile moments候補（速度未達で未採用）：** 別branch abfbecc5/native6fafは、低い階層のmomentsをowner CTA内の
sharedで作り、高い階層だけglobalへ保存する。完成FP64 H/Gとforward Hのbackward再利用は保つ。
初回30b25は新設テストの空batch保存hook・整数0atom fixture・エラー文regexで8件失敗した。
数値基準を変えずテスト側の前提だけ直し、検証済みbinaryを照合してretryした。
job `l4job-e86f7f30ff954c3aa9664fb71e920d7c`でGPU575 passed／2 skippedと独立MNIST gateを通過。
B32単体は旧1.422358/1.435008ms→新1.609435/1.608798msと遅くなったが、allocatedは
1684992→731648B。reservedは2097152Bで同じ。B4は約1.369msでallocated186880B。

MNIST N128/B32/inplace・同じae13初期化の1epochは2.759727ms対dense1.321156ms=2.08887倍。
allocated241362432Bはdense241428992Bを下回るが、reserved247463936Bは同値。
test87.97%対91.22%で、速度・両メモリ同時目標は未達。メモリ削減を速度向上とは扱わない。
source_sha256=`027c9f24285703234cface8514f1ae1234aee1510f2b60cf597762bf951675f9`.
archive_sha256=`2e3c4fbfcea38e84bf4ee4176bd8aafbaaedcd51740cf055b82e610965ddfe47`.

**output tile幅の比較（既定32を維持）：** 検証済み8bを使うjob
`l4job-dbe5065b4fd146c385ed62dff524c910`は、同じN1024/B4/A52428で既存Recipeの担当幅だけ変え、
全候補の独立Y/dX/全atom勾配・optimizer gateの後、2fresh public runnerで測定した。

| output_tile_sites | 完全step ms（2process） |
| ---: | ---: |
| 4 | 4.632311 / 4.630001 |
| 8 | 4.566202 / 4.561172 |
| 16 | 4.286583 / 4.261803 |
| 32 | 4.299146 / 4.292907 |
| 64 | 5.667212 / 5.660681 |

全候補のpeakは9211904/31457280B。16の改善は1%未満で、別のMNIST学習を増やす根拠にはしない。
32 blockしかないことから大幅改善を予測した仮説は支持されなかった。担当幅を減らすと読み取り／profileの重複も増える。
source_sha256=`76d2b797db3a014bfea57b90a94381365278d8217555b5c6b10a3d1eb18f6856`.
archive_sha256=`838b0ed7644545b872e24f15f937ea64b5f3cd456beb2c581c01cecccb35dcb6`.

CPU sincos shim修正後のGitHub run646はsuccess。`5143b4ee`統合状態は、両native guardを有効にした
CPU全体3172 passed／6711 skipped／既存警告19件。runtimeとbenchmarkはGPU検証23125266とbyte同一、
testsは前述のportable shim修正のみ追加されている。wheel/sdistのruntime388ファイルもsourceへbyte照合した。

### 位相seed再利用と小型診断の限定結果（2026-10-31）

`kernel/radius-fixed-k-phase` / b93ea5b7はP=K*T*spacingを使い、response/compact parameter微分の
part0でsin/cosを求め、K2は符号反転、K3は固定回転で他partのfundamentalを作る。
C cacheは変えず、part→harmonic→batchの加算順序を保つ。数学的には同じだがFP64の丸めは変わる。
job `l4job-1831538cb2754820a48f0b9182c7d047`で817f645dcea3を実compileし、
GPU955 passed／2 skippedと既存の厳密byte/state gateを通過した。
N1024 activeは4.165755/4.160563msで、base same-profileの4.151259/4.146184msより速くならなかった。
ピーク9211904/31457280Bは同じ。演算回数の削減を速度改善と扱わず、この候補は未採用とする。
source_sha256=`32ff704e38d57729af0798b5e1eaadab4bd18b0bc4f246fe108b1ba35e34aeba`.
archive_sha256=`7b93929297c7081683af3cc14a00d28c1dba30ba4bd9083b5db0fc153d28ab75`.

N128 local-tileの診断job `l4job-8ca4f82ef95f4b548aaf1aa7768ae838`は、実compile済み6fafとae13初期値を照合し、
30更新後の300-update prefixで2.513884ms対dense1.297071msとなった。
ただしpeak allocatedはCST241362432B対dense241345536BでCSTが16896B高く、reservedは247463936Bで同じ。
この短いprefixの時間を前節の1875更新・全epoch2.759727msと置き換えない。
補助hostコードのコンパイル環境・prepared経路の可用性も全epoch driverと照合中で、
そのCPU profileを全学習の内訳として断定しない。
source_sha256=`3a304f7f5576d8b012ae20a6ad2889e6f5134668b7e7c7194d0f942dfb53b49f`.
archive_sha256=`ca4e1a49ae146a4f495d234b3b8427fefccb94a9cdf269cb4c6a9ec2d7bc9723`.

現在の採用状態3113d6d5のGitHub CPU validation run647はsuccess。
owner CTA内で完成responseとC生成をまとめる次のmode6は別branchの試作中で、GPU性能・peakは未検証。

### shared C seed候補の厳密置換gate失敗と、診断環境の修正（2026-10-31）

別branch653198ed/native566167b2cde8は、C cacheへ2つのFP64 fundamental seedを加え、
K2/K3のpart位相を回転で再利用した。容量不足時は元のunseeded経路へ戻し、旧chunk容量・順序を保った。
job `l4job-b65c07f2248a4689971e6939c68f78d3`は実compile後、GPU2 failed／1011 passed／2 skipped。
B4/B32の3更新における「同じPlanの出力bytesを全て保つ」gateでdXが不一致となり、速度測定は行っていない。

全42記録のraw snapshot再計数では、5つのdX記録に計50/13824要素・51bytesの差があり、
最大絶対差8.470329472543003e-22、最大29 ULP、全値finiteで符号変化なしだった。
Y/dP/P/Adam moments/stepは保存したA41・B4/B32の各3更新でbyte一致した。
これは外側層を含む学習や一般のcheckpoint一致の証拠ではない。
同じPlanの互換置換としてはfailedのまま保持し、gateを緩めたり測定済み速度を主張したりしない。
元の候補と証拠は凍結し、旧Planをliteralにunseededへ保つ別の明示Planを作る場合は、
既存の独立Y/dX/全atom勾配・CPU-reference更新基準で新しい数値経路として検証する。
新経路の数値同値と、既存経路のbyte互換を混同しない。
再計数はignored `output/cached-phase-seed-negative-audit-20261031/`へ保存した。
source_sha256=`712527f063d39820226160eb74684919e878a456eeb0eb57cb2453633d2dae76`.
archive_sha256=`97b4d36910a5535f824fd8b3f6bc6e3a26af8677b5af0fc63cdd83ed6ae2f616`.

小型診断8caは、全epoch driver e86にあったjob-local Ninjaのinstall/PATH設定を省いていた。
実cProfileではPython ReadParameter forward/backwardが60回、prepared Linear本体は0回だった。
8caはhelperの失敗例外・PATH・loaded状態を保存していないため、Ninja不足を歴史的な確定原因とはしない。
補助factory/leaseのoptional loadがNoneをcacheする仕組みとNinja依存は別CPU実験で確認した。
修正した診断driverはe86と同じNinja環境を用意し、自然warmup後のhelper状態を非破壊で記録する。

また、prefix driverは30回の実更新後にsteps31–330を測り、各30stepで同期する。
全epoch harnessは初期評価とbackward-only warmupの後、cold optimizerを含む1875stepを一括で測る。
後者ではloopのinputs/logits/loss参照が次の代入まで生存するが、prefixのstep関数はreturn時に解放する。
データgather/zero_grad/CE/backward/optimizer/metricsの演算は同じでも、Tensor寿命と測定境界は同じではない。
最終radius bankは全819atomがL2/K3/T4のままで、endpointのtier移動は観測されていない。
短いprefixから全学習の時間差やメモリ差の原因を断定しない。

### N128/B32の3epochと初回optimizerの実測（2026-10-31）

修正環境の診断dd5は、同じNinja準備でnative lease/factory/storage/dictionary epoch helperのloaded状態と
実際のprepared Linear呼出しを記録した。300-update prefixは2.111300ms対dense1.318120ms、
全1epochは2.857667ms対1.323668msだった。prefixを全epochへ置き換えず、3epochで切り分けた。

job `l4job-d19d7541c69b49049195460d188e2a22`はfrozen abf/6faf、ae13、同じdata order、
trueB32、inplace、独立先頭batch gate、通常harnessを保ってCST/denseを3epoch実行した。

| epoch | CST step ms | dense step ms | 比率 | CST/dense test |
| --- | ---: | ---: | ---: | --- |
| 1（cold込み） | 2.896591 | 1.352565 | 2.14155 | 87.97% / 91.22% |
| 2 | 2.161973 | 1.336248 | 1.61794 | 90.51% / 92.67% |
| 3 | 2.174449 | 1.348096 | 1.61298 | 91.34% / 93.92% |

5625更新の合計時間/更新数はCST2.411004ms対dense1.345637ms=1.79172倍。
初回を除外していない。3epochのmax allocatedは241625088B対241808896BでCSTが183808B低い。
reservedは247463936Bで同値。時間とallocatedの条件は満たすが、両ピーク厳密未満の条件は未達。
小型単体ではlocaltileが遅くなることも前節のままで、万能な既定選択にはしない。

別のfresh診断processで実optimizer.stepを同期して測ると、CSTは1回目1282.244302ms、2回目1.081057ms。
denseは9.266584ms、0.407486ms。別cProfileの初回CSTはTritonのtriton_keyに約1.20s、
うちhashlib.updateに約1.124sを費やした。初回の大部分は更新の算術ではなくcache key準備である。
filesystem/JIT cacheの状態は診断に記録し、cache削除や学習時間からの差し引きは行わない。
1回目と継続時を混同せず、後者もfull forward/loss/backward/optimizerを含める。

`l4job-dd5cf63952ff4690bcf8af390607101a`:
source_sha256=`54602d5b923ede846221304553c5413bd0755975907ece73db6267cdd9ea006b`.
archive_sha256=`ee5741b80452e416268f6238da6d878704efd8a313d6bab660fbc922a1062ed2`.

`l4job-d19d7541c69b49049195460d188e2a22`:
source_sha256=`e1dc1e8413da8af77ac1654e340ab2f2399f10c5652bdb1455f1a6fe9b8dbacc`.
archive_sha256=`90f0472f4a86ba991bc16d13640e0522d054f83ba234bceed6559a7b1949bb95`.

### 小型denseの大きなメモリ差は主にcuBLAS workspace（2026-10-31）

N128単体denseの総peak17313280Bには、Parameter/grad/Adam状態だけでは説明できない容量があった。
8ca/8001のCST/dense4つの実snapshotには、それぞれ8519680Bのlive blockが2個あり、
同じTorch v2.11.0の[commit固定のCublasHandlePool.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cuda/CublasHandlePool.cpp#L147)の既定容量と一致した。
frameがないblockなので、サイズ一致だけで所有者を確定せず、Torchのみの隔離processで因果確認した。

job `l4job-a1b1fef382a244eab1634d326b6c2a2a`はN128/N1024の各fresh processでX/Wを保持し、
そのpointerと値hashを維持したままprivateなclearCublasWorkspaces hookだけを呼んだ。
forward後は8519680B、forward/backward後は17039360Bのallocatedが減り、両サイズで同じだった。
行列演算を一度も使わないfresh fusedAdamW processでは、このhookによる減少は0Bだった。
GPU Torch git hashも上記commitと一致した。reservedの減少は0Bで、allocator cacheに残ることも区別する。

したがって単体dense対CSTで見えた約17MBの差の大部分は、このライブラリworkspaceの有無である。
MNISTのCSTも外側のnn.Linearで同じworkspaceを使うため、その差は共通化する。
activation/H/G/Zのbatch比例分だけが、単体と全modelの差の原因だったわけではない。
既存の測定済み総peakはそのまま保持するが、それを固有の中間配列payloadや総GPU process使用量とは呼ばない。
この診断のworkspace解放を実装・benchmark最適化として採用せず、clear後の値を元のpeakから引いて達成判定もしない。

公式sourceはignored `output/cublas-workspace-attribution-20261031/official-pytorch-v2.11.0/`へhash付きで保全し、
snapshot照合・raw causal記録は同じ研究outputとpool結果に保存した。
source_sha256=`5f4803005a7102ac493a84448e4b11f6e82f215eb5c036d0f6ce5564729a9042`.
archive_sha256=`fe8760ab6929d6e551ff5c7e524f000a9b9de0c0fd3f06562e324bd90cea2ac0`.


### 非ゼロoriginのtorus座標とCPU参照の修正（2026-10-31）

owner融合候補6660の初回job `l4job-016bd35815334cbb9fa0a328065a3789`は
1049 passed/7 failed/2 skippedで停止し、性能測定へ進めなかった。
うち6件はFP64の非ゼロorigin/skew fixture、1件は新規empty fixtureの誤指定だった。
失敗した原本は保持し、独立診断job `l4job-5e77db8f6de247cca249f3e0dd7213d4`で切り分けた。

旧native mode5と候補mode6の両方に同じCPU参照との差があり、両native同士は約1e-16で一致した。
encoded FP64のcenter=`79*13.7`、origin=`-1.2`、chart period=`1024`では、
先にcenter-originを計算すると59.5へ丸まり、T4/K3のfirst tileは14になる。
各座標を先にfmodで縮約する既存native式では59.49999999999996でfirst tileは13となる。
入力floatそのものをFractionへ変換した独立計算も後者に一致した。epsilonや境界のsnappingは追加しない。
またcenter=1e300、origin=0.5、period=1024では正しい相対座標1023.5が、先行減算だと0になる。

CPU referenceのaxis_massesだけを既存nativeと同じmodular-relative座標へ合わせる。
同じ座標でwindow選択とcell phaseを計算し、中心の単位VJPを保つ。
CPU fmodの極端な有限値でのNaNはscalar libmで修復し、DBL_MAX近傍のperiodも別途検証する。
このreferenceは既にCPU専用であり、追加したCPU分岐がnative高速経路へ入ることはない。
CUDAの式、radius、profile、正規化、既存oracleの許容誤差、legacy Planは変更しない。

保存済みの実GPU tensorを修正CPU参照で再評価するとH/Y/dX/全parameter勾配は最大1.11e-15で一致した。
これは保存tensorの再評価であり、新しい実GPU qualificationの通過とは別に記録する。
新しいテストはexact/nextafter/巨大center/逆符号/overflow/符号付きzero/中心VJPと、
既存global-native Planの元の境界fixtureを対象にする。候補Planへの依存は持ち込まない。
focused CPU検査は73 passed/48 skipped、全CPU検査は3191 passed/6712 skipped/19 warnings。
全CPU raw logは`output/radius-mnist-native-20261031/primary-canonical-reference-cpu.log`。GPU再qualificationは別途実行中。
be52decfのGitHub CPU validation run41146807786は成功した。
be52decfのwheel/sdistはruntime388ファイルがsourceとbyte一致し、
`output/radius-mnist-native-20261031/package-be52decf-audit.json`にhashを保持する。
証拠はignored `output/radius-mnist-native-20261031/owner-canonicalization-audit/`と上記pool結果に保持する。
実GPU binary SHA256=`48557033186f1ee1fd7dfcd9081a0679cafa2b9efd6919eaf76ac9407fb56b02`。
Fraction proofと再評価JSON SHA256=`b1b79e66263ef4ec075213747ee3b4c166da08b12eca0b06f92b1b3cc163d482`。


### Cのshared paddingと別Planのseeded Cは速度採用しない（2026-10-31）

両候補とも独立oracle/legacy byte gateを通過したが、完全stepの有効な改善は得られなかった。
rawはprimary ignored `output/radius-mnist-native-20261031/<job>/results/`へ保全し、
source/result全manifestと実binaryの照合は`output/cache-layout-terminal-audit-20261031/audit.json`に残す。

| 候補 / job | suite | paired baseline → candidate ms | allocated / reserved B |
| --- | --- | --- | --- |
| shared cache stride14→15 / `l4job-d8f5b296ee2043c38732542b11b5655e` | 1079 passed/2 skipped | 3.960352→3.958454、3.954693→3.966433 | 両者9179136 / 29360128 |
| 明示seeded C別Plan / `l4job-bfcbb80b7d5f4c70a6e2dcdae30da899` | 1041 passed/3 skipped | 4.294860→4.270823、4.301858→4.293986 | 両者9211904 / 31457280 |

各workerは101標本、fresh process2組、L4/Torch2.11/CUDA13、N1024/B4/A52428。
denseのallocated/reservedは両jobで33900032/44040192B。candidate/dense時間は約5.6〜6.0倍。
paddingはnative5、seededは旧Python hostを基準とするため、候補同士の絶対値差を効果としない。
paddingは一方が0.05%改善、他方が0.30%悪化。seededは0.56%/0.18%改善に留まり、両方メモリ改善なし。
主goalの大幅な短縮に寄与せず、既定Planやprimary runtimeへ取り込まない。MNIST測定は未実施。

元binaryとの旧Plan比較は両jobとも63/63 recordでY/dX/dP/P/Adam/clockがbyte一致した。
別Plan seededの観察用cross-route比較はY/dXの11recordが異なり、A41/3stepでdP/P/Adam/clockは一致した。
これは全学習の同一性を主張する証拠ではない。旧653/b65の同一Plan置換失敗とed/bf90のfallback失敗は
負の原本として保持し、新Planの通過で書き換えない。fallbackはliteral旧kernelへdispatchして修復した。

padding source/archive SHA256=`8616ee105fa0a7d8930e75e175244e60f23f2a04f0998c9663684f3139a5e229` /
`78789a9a22b240d9a2595f1ba7740796e4aead160a4b1706ce9e4f7f4a0f9c08`。
seeded source/archive SHA256=`c75ee8c5f82ab924dd3da06d360ad551feadbd030f971216f422050b4a12000e` /
`56dbc8457d56d9050bf89b7af70ad046b22372be9bc427c6d8bf576f9109defb`。


### ownerごとの完成response/C融合で1024を短縮（2026-10-31）

明示Plan `radius-global-owner-response-coeffs` / schedule `global_owner_response_coefficients`
をprimaryへ追加する。N1024/B<=4/M8/base1/merge2などの宣言条件を満たす場合だけ選べる。
自動dispatchと従来Planは変更しない。式P、normalization、全atom、Polar更新則も同じである。

`owner_response_coefficients.cuh::owner_response_coefficients_kernel`は、destination ownerごとに
128thread CTAを使い、各threadがatomを担当して完成H（逆方向は完成G）を一度だけ計算する。
profile/radius/phaseをblock内で保持し、同じ完成responseをshared Cの集計へ使う。
低階層は積分済みsite contributionを直接保持し、高階層はFourier係数を使う。
`owner_expand_kernel`が各siteを一意に担当してowner partialsを集め、Y/dXへ展開する。
ブロック間のspin待ちやatomicsは使わない。dX完了後のatom勾配は従来grad4を使う。
完成していないHからCを導かず、forward Hとmomentsはbackwardまで保存して再利用する。

CTA sharedは34432B。ここで完成したowner partialsは一時global配列[B,node,3,F]へ書く。
N1024/B4では6681408Bで、永続的なatom/profileコピーではない。
裸の完全step allocatedは9179136→15474176Bへ増えるため、速度だけの無償な改善とは説明しない。
MNISTでも一時配列を含む全peakを測り、dense未満を確認した。

job `l4job-2e2ddf55f2b94425b626d72650621398`はsource5aから実module6660を新規compileし、
独立Y/dX/全atom勾配・metadata/hooks/lease・retained/offload・optimizer検査を含む
1076 passed/3 skippedを通過。旧016の失敗を上書きせず、CPU参照修正後の別qualificationとする。
2fresh bare比較は旧native5の3.982109/3.979447msから3.132372/3.134375msへ約21%短縮した。
対応denseは0.732583/0.717382ms。CST allocated/reserved=15474176/29360128B、
dense=33900032/44040192B。裸1024の速度2倍目標は未達である。

同じ5ef7初期bytes・外側層・data order・optimizer条件で、trueB4の全MNISTを3epoch回した。

| epoch | owner融合 CST ms | dense ms | 比率 | CST allocated/reserved B | dense allocated/reserved B |
| --- | ---: | ---: | ---: | --- | --- |
| 1（cold込み） | 2.971069 | 1.299282 | 2.28670 | 264600064 / 272629760 | 270021120 / 287309824 |
| 2 | 2.887671 | 1.315609 | 2.19493 | 264600064 / 274726912 | 270484480 / 287309824 |
| 3 | 2.910388 | 1.311122 | 2.21977 | 264600064 / 272629760 | 270484480 / 287309824 |

最終accuracyはCST97.37%対dense97.87%。1seedでの精度同等・一般化は主張しない。
旧native5の実学習3.75〜3.82msより短縮したが、時間2倍以内は未達のまま。
独立にfinal.ptの全CST model33/optimizer15、dense model5/optimizer15 tensor leafを比較し、
旧86結果とbyte一致した。A41/3stepのcross-route診断も全byte一致だったが、
新Planを全入力でbit-compatibleとする保証にはしない。独立数値gateを採用条件とする。

統合primaryの全CPU検査は3198 passed/6771 skipped/19 warnings。
runtime/benchmarkの全hashは実5a snapshotに一致し、追加のCPU回帰fixtureだけowner Plan依存を避けた形にしている。
rawはignored `output/radius-mnist-native-20261031/l4job-2e2ddf55f2b94425b626d72650621398/results/`。
source SHA256=`d89e31876ec705e599f6557974073cfc1d9cdf8d924518cb17706ce2da45e501`。
archive SHA256=`4ccceb01ce7caeee02a9de63f90c891b9bbc536ca0bf1223cf605dbaea38c8ae`。
実binary SHA256=`6062eb966c340b8f15d87f5b5e7677a08183416749ab91526c46f2fec786655e`。

### local tile/global native併存sourceの再qualification（2026-10-31）

別branch799f7b2のunion module0724を新規compileした
job `l4job-49e992a4765746c39cf98fada2520887`は1602 passed/3 skippedを通過した。
mode4のlocal tileとmode5のglobal nativeは明示的に併存し、既存modeの挙動を変えない。
このunion sourceはprimaryへ未統合で、N128/B32のメモリ削減候補として保持する。

trueB32/ae13/同じdata orderの全MNIST3epochはCST2.791092/2.084941/2.084597ms対
1.331040/1.325906/1.335662ms。cold込み5625step平均は2.320210対1.330869ms=1.74338倍。
max allocated=241625088対241808896Bで下回るが、reservedは247463936Bで同じ。
速度とallocatedの条件は満たし、両ピーク厳密未満は未達。最終accuracyは91.34%対93.92%。
final.ptの全model/optimizer tensor leafは旧d19 localtile結果とbyte一致した。
既知の裸B32の速度低下を隠さず、万能な既定にはしない。
rawは同じprimary ignored output配下のjob別resultsへ保持する。
source SHA256=`233c527f4cd53e7bed4dc6137fa502b6c632a80a93f1a272fbaf1eb35d54054a`。
archive SHA256=`7c705b48ea702ed8a120a9e8989e322f74286206bbd4dae4fddebf31867c5f22`。
実binary SHA256=`ba0fc422bd4354d9a9c90522fd720b265fd364d6b59eef7e14002b8de787fe0a`。
