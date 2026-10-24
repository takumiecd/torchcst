# block128：CUDA C++での最小forward / backward実装

2026-10-24。[init128の具体案](hierarchical-fourier-init-128-20261024.ja.md)を実装した研究用候補。
source checkpoint：`dcb12057`。公開CSTLinearへの採用や速度改善の結論ではない。
以下の検証結果は、動的job版・複数集団版・split_groups版へ一般化しない。

## 実装した範囲

N128 / M8 / A128、FP32のX・P、batch1..64、連続な二次元Xだけを受け付ける。
一つのblockは128 threads、一集団。小バッチ4件とYまたはdXの32 sitesを所有する。
各threadが別atomの全支持を縮約し、backwardのtile 0だけがH/GとdP部分和を計算する。
他のblockはGのvalueからdXだけを作る。

処理はZ準備 → atom縮約 → C集約 → 最終出力のphase型。
全atomのZ利用終了をblock同期で確認してから、同じshared領域をCへ再利用する。
H/Gの微分項目はatom担当threadのローカル値で、valueだけをshared経由で集約へ渡す。
一つのatomのH/Gを複数threadの部分和へ分解しない。

H/Gは未正規化・振幅適用前。正規化は混合後の生profileを全128 sitesで評価した離散L2とfloor。
中心・live幅・正規化分母の微分を保ち、振幅0でもdAmpを計算する。
元Tensorのversion検査とretained backwardを保つため、XとPの元Tensorをautogradに保存する。
高階微分は対象外。

H保持と割り当ての比較案は設計文書に残すが、最初のrecipeは
`same_thread / recompute / phased`だけを許す。未実装の選択肢を受け付けて黙って別方式で動かさない。

## 関数とコードの対応

| 設計上の処理 | 実装 |
| --- | --- |
| init・block/threadの担当番号 | `block128<BACKWARD>()`の先頭、source読込と最初のblock同期 |
| 階層Z生成 | `build_z_tile()`。互換な子bankの偶数modeを継承し、残りをsourceから計算 |
| atomごとのH/G | `contract_component()`。担当threadが直接残差とZ項を最後まで縮約 |
| atomごとのパラメータ勾配 | `write_parameter_partial()`。全batch方向の和とnorm微分 |
| 入力側／出力側C | `build_c()`。一つのthreadが一つのC要素の全atom寄与を所有 |
| 直接部分とCの展開・出力保存 | `store_output()`。各threadが一つの出力要素を保存 |
| batch間のdP合算 | `reduce_parameters()`。最後にFP32へ変換 |
| 係数・支持・基底の数式 | `block128_math.h`。同じheaderをhost C++でも検査 |
| PyTorchと接続・保存状態 | `block128_executor.py`と`block128_bindings.cpp` |

[CUDA本体](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_kernels.cu)、
[スカラー数式](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_math.h)、
[autograd入口](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_executor.py)。

CUDA C++をPyTorchの[JIT拡張load](https://docs.pytorch.org/docs/stable/cpp_extension.html)で遅延コンパイルする。
初回実行にはnvcc・host C++ compiler・ninjaが必要。Planのimport・宣言検査ではコンパイルしない。
wheel/sdistの両方へ`.cpp`・`.cu`・`.h`が含まれることを確認した。

## 保持する配列

| 配列 | 場所・量・寿命 |
| --- | --- |
| X・P | 呼出側のglobal Tensor。autogradは元Tensorを保持 |
| dY | backwardの入力global Tensor。各blockの小バッチ分をsharedへ読む |
| fit | global FP64 `[32,9]`、2,304 bytes。shapeだけに依存する係数表 |
| norm | global FP64 `[128,2,3]`、6,144 bytes。forwardごとに再計算し、backwardへ保存 |
| H_value / G_value | shared FP64 `[128,4]`。各blockで生成して同じblockの出力集約へ使用 |
| ZとCのarena等 | forward 21,376 bytes、backward起動は各blockに38,656 bytesを指定 |
| dP_partial | global FP64 `[ceil(B/4),128,4]`、batch4でも4,096 bytesを確保 |
| Y・dX・dP | 最終global Tensor。X/Pと同じFP32 |

新しい候補では全atomのNative・Co/Ci・Betaや全域Phiをglobal配列として作らない。
fitから必要な係数を生成する計算費用は残り、出力集約時の再計算もある。
6,144 bytesのnormを作る独立したprepare kernelは必要。
batch4でもdP_partial→最終dPの別kernelを使う点は、init文書の「直接保存も可能」という案より単純な実装。
registerからの退避、CUDAのcontext/module、Graph pool等を含む総メモリは別に測定する。

一つのforward呼出しはnorm準備と本体の2 kernel、backwardは本体とdP合算の2 kernel。
依存は呼出側の現在streamで保証し、GPU全体の同期やglobal Hへの受け渡しを追加しない。

## CPUと宣言・packageの検査

- host C++で同じ数式headerをコンパイルし、支持境界・階層／周期接続前後のprofileと全微分項目を
  Torchの独立した解析係数参照と照合した。
- global座標のbeta・中心／幅微分が、中心をずらしたseriesと一致することを確認した。
- 宣言・既存回帰を合わせた対象テストは44 passed / 11 skipped。skipはCUDAが必要な項目。
- 専用Case/Planの宣言検査はPASS。wheel/sdist buildとnative sourceの同梱検査もPASS。
- CPU全体検査は、後述のnative source provenance追加後の最終状態で
  2,266 passed / 5,751 skipped。Torch JIT廃止予定の既存警告18件。
  provenance・DB adapterを含む対象検査も69 passed / 5 skipped。

既存runner用の
[Plan](../../../benchmarks/cuda/linear/plans-hierarchical-fourier-block128.json)と
[Case](../../../benchmarks/cuda/linear/cases/block128-fourier-m8-mixed-128-a128.json)を追加した。
baselineは既存direct、dense対照も同じCaseから実行する。

```bash
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-block128.json \
  --case benchmarks/cuda/linear/cases/block128-fourier-m8-mixed-128-a128.json
python -m pytest -q tests/test_hierarchical_fourier_block128.py
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-block128.json \
  --case benchmarks/cuda/linear/cases/block128-fourier-m8-mixed-128-a128.json \
  --source-commit "$(git rev-parse HEAD)" \
  --output output/block128-comparison.json
```

## GPU検証の記録

最初のjob `l4job-f2f2493c72cb463cb2fcd871d4499ceb` は、依存準備のvenv/ensurepipで失敗。
CUDAコードへ入っておらず、正しさ・性能の結果に含めない。
system packagesを変更せずjob専用targetディレクトリへ依存を入れるdriverに変更し、
`l4job-48a1059ed2f441489ba462f7efbe1de8`へ再提出した。

再提出jobはreturncode0、timed_out=falseで回収できた。
NVIDIA L4、GPU UUID `GPU-277fcf85-536e-1d2c-c28a-d9d39ec1554c`、driver580.82.07、
Python3.13.16、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。
新しい実装自体はCUDA C++で、対照のdirectに既存Tritonを使用した。

### 正しさ

専用pytestは9 passed。batch1/4/5のY・dX・全atom勾配、retained backward、
X/Pのin-place変更のversion検査、host数式と宣言を含む。
既存runnerのcorrectnessは独立FP64 quadrature/KKT参照と、3つのlive状態・retained backward・
幅と中心変更後のCUDA Graph replayをatol=rtol=4e-4で照合してPASS。
新実装の最大絶対誤差はY 2.94e-8未満、dX 2.96e-8未満、全dP 1.39e-6未満だった。
これはN128/B4/A128・mixed初期化と上記状態の検証であり、全分布・全GPUの保証ではない。

### 完全stepとメモリ

同じ初期P・X・targetのhashが一致する独立workerで実行した。
準備・norm・forward・backward・AdamWを含む未計装Graph完全step、7 replay samplesの中央値。
1 jobでの比較であり、7独立runではない。JITコンパイルの初回費用はstep時間に含まれない。

| 実装 | Graph完全step ms | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| direct-samples-h | 0.572416 | 1,061,888 | 6,291,456 |
| cuda-block128 | 44.285889 | 33,792 | 6,291,456 |
| Dense | 0.031744 | 34,351,616 | 48,234,496 |

初期CUDA C++版はdirectより約77.4倍遅い。**正しさを読むための比較基準として残し、速度候補として採用しない。**
allocatedは小さいがreservedはdirectと同じ。これだけでGPU全体の使用メモリや転送量が減ったと結論しない。
計測にはGraph capture/replayを含み、oracle Tensorは除外した。

### コンパイル属性

`cudaFuncGetAttributes`で次を取得した。static sharedは全て0で、別途動的sharedを起動時に指定する。

| kernel | registers / thread | localSizeBytes / thread |
| --- | ---: | ---: |
| prepare_norms | 255 | 816 |
| forward128 | 82 | 112 |
| backward128 | 162 | 184 |

local memoryの存在を確認した。これはregister退避だけを分離した測定や、実際のアクセスbyte数ではない。
local memoryやmodule等はPyTorch allocated値だけではGPU全体のピークとして数えられない。
shared内の値が正しく使われても、全処理がregister/sharedだけで終わると主張しない。

### 保全とsource provenance

- source archive SHA256：`9f898393d06322d449a0060ad65fbe1670df4c4256fd741813fba15ed14fcb4c`
- result archive SHA256：`cbf42ccbcbc2d9a0bdf25c9c9398ecfdc1dbea21ef99481e25a1e7aa44c63a62`
- comparison.json SHA256：`cc607e9d5fb439dcf4a423c3aa27037c2a9a53c34a77395685d08a748c091a9a`
- `.cu` SHA256：`736ddfccf4c90143dd914d9eb4575c329949b7e2c2c1b1ca065ee742b5a1dc38`

receiptとsource/result archive、17 artifact、全workerのsource hash計2,040件、
別途記録したnative source等6件を照合した。runner自身のsource_commit欄は`unrecorded`であり、
Git SHAを記録した結果へ書き換えていない。poolの凍結sourceと上記checkpoint・hashで対応を残す。

この実行時点のrunnerは`.py`だけをhash対象にしていたため、native sourceはdriverの
`resources.json`とsource archiveで別途確認した。以後の実行では`.cpp/.cu/.h/.cuh`も
runnerのsource_hashesへ含めるよう修正し、中央検証側の対象とCPUテストも合わせた。
このmetadata修正後にCUDA本体の再測定はしておらず、nativeの実装hashは変わっていない。

raw driver・CPU logs・package buildと、検証済みのsource/results/receiptのコピーは
ignored `output/block128-cuda-20261024/`に保持する。
実験終了後にpool全slotがstoppedであることを確認した。追加のGPU探索・最適化は開始していない。

## 次に比較する項目

現在のコードは出力のC／直接部分を作るときにも係数を繰り返し生成し、全28 nodeを準備する。
速度差の寄与を分離したtrace/counterはまだ採っていないので、これだけを遅さの原因と断定しない。
次は係数の再計算と小さな保存領域の交換、必要nodeだけの準備、norm準備の局所配列と担当数を
一つずつ比較する。G/Hのどちらが軽いか、同時実行が有効かはこの結果だけで決めない。
