# Cell-mass CUDA: 1024サイト・約5% atom

128から1024サイトへ拡張し、同じ面積正規化・whole-tile窓で検証する。
公開dispatcherへの登録とは分離した数値研究入口。

## 固定する数学と変える実行設定

- 入出力は各1024サイト、atom数はfloor(1024²×0.05)=52,428。
- batch4、FP32、ChebyshevSquare M8 / tail_bound1e-3、seed120。
- 物理的なchart周期Q=128を維持し、spacing=128/1024=0.125。
  入力tile128サイト（物理幅16）、出力tile256サイト（物理幅32）、K=3。
  したがってprofile周期48/96とsigma初期値0.2..12は128のcaseと同じ。
  site数の増加と物理的な領域拡大を混同しない。全sigma分布の代表とは扱わない。
- Y/dXのblock幅は8..256の2のべき乗。最大4つのprofile tileを担当する。
  広いタイルのsite展開は各threadが複数の(batch,site)を担当する。
- atom splitは1..64の2のべき乗。32atom単位で分担し、
  globalの部分出力を一意writerの次kernelで合計する。
- H/G/Parameter勾配は各threadが別atomを担当し、block幅16/32/64/128は別設定。

モデルのatomをglobalに並べ替えコピーせず、元のParameter配列とsorted_rowsを使う。
全site×atomのprofile表、全Fourier係数表、全C表をruntimeで作らない。
CPU oracleのprofile表は検証専用で、256atomずつに制限する。

## 中間値の丸めと更新量の検証

最初の拡張`1c84dc67`はCUDA/math317件が通ったが、全52,428atomの
CPU float64 oracleから同一GPU fused AdamWへ渡す更新量検査で停止した。
入力中心151.425のatomでは、4バッチの寄与が相殺され、勾配が約7.21e-10となる。
FP32 Parameter更新に1 ULP（1.526e-5）の差が生じ、固定したatol2e-6を超えた。
全atom勾配が通常の絶対誤差検査に通っても、更新まで一致するとは限らない。
このjobには採用できる速度値がない。許容差は緩めない。

再計算経路では、すでに再投影で得ているdouble値をParameter勾配へ使い、
保存済みFP32 H/Gを再読込する丸めを避ける。
さらに`double_intermediates=True`を選べるようにした。

| 配列 | 既定False / FP32入力 | True / FP32入力 |
|---|---|---|
| 元X・dY・atoms、Y・dX・atom勾配 | FP32 | FP32 |
| Zx・Zy | FP32 | FP64 |
| 微分保存なしのH/G | FP32 | FP32（Parameter勾配は再投影のdouble値を直接使用） |
| 微分保存ありのH/G | FP32 | FP64 |
| 保存する幅・中心の投影微分4本 | FP64 | FP64 |
| Cのregister/shared、内部profile演算 | FP64 | FP64 |
| 部分Y/dX scratch | FP32 | FP32 |

Trueはglobalの保存精度を上げる設定であり、係数やJacobianの全表は追加しない。
追加容量も完全stepで測り、dense以下という条件を満たす設定だけを採用する。
元のX/atomsのautograd version検査とretained backwardの契約を維持する。

## 検証と測定の境界

CPU oracleは256atomずつfloat64のセル確率U/Vを作り、全atomを合計する。
Y、dX、振幅・幅・出力中心・入力中心の全4列の勾配を照合する。
初期dense Wも同じoracleから構成し、X@W.TとdY@Wとの一致を確認する。
fixture構築は時間測定とGPUピーク測定の外。測定に使うCPU tensorを保存して子プロセスで再利用する。

AdamW検査はCPU oracleの勾配を同一GPU fused optimizerへ渡して比較する。
CPU optimizerとGPU融合optimizerの丸めは混ぜない。
固定のrtol5e-5/atol2e-6で、Y/dX/全atom勾配・Parameter更新delta・moments・stepを検査する。

時間はJIT・oracle・初期W構築を除外し、各CST call内の値検査・stable sortを含む。
固定X/dYのforward / backward / fused AdamW、TF32無効、CUDA Graphなし。
学習するパラメータ空間はdenseのWとCSTのatomsで異なり、更新後の軌跡一致は主張しない。
時間は通常dense、メモリはworkspaceなしdenseを保守的な基準として別々に示す。
これらは一つのdense設定の時間とメモリではない。

## 失敗・未回収の実行も残す

- `l4job-cf964eb343d342b89677588cab288b85` / source `1c84dc67`:
  CUDA/math317 passed、全52,428atomのCPU oracleを28.94秒で構築。
  最初のCST更新delta検査で上記の1 ULP差により停止。性能測定なし。
  source SHA256 `1ba3f968f090376047bd8fa2cdca1a188902a492b246e7eb1970141b95c183f1`、
  result SHA256 `8e1a22d88d34800ee9322e22b932c5102b6fbfa4c70ef10b694ef9094a5a01fb`。
  manifest17ファイル、321runtime source、binaryを照合した。
- `l4job-af2b449c016849379b26c9cb5b907bf9` / source `b98f7e80`:
  remote execの完了応答が1,140秒でtimeout。結果archive未回収のため、
  正しさにも速度にも数えない。source SHA256は
  `2d8b678b6de8a42d5526a847ff39ffb2e48f988ee89bd9cc76179cb9dec270ef`。
  source・要求・transport logを保存し、pool所有VMの停止とserver上の割当消失を確認後recoverした。
- 長い後続2jobはqueuedのままcancelし、fixtureを埋め込まない短い検証へ分割した。
  `l4job-14adf2dd54534c97836a9565e852ace3`、
  `l4job-746c251e267246c6b7cb86f5e93f9607`には測定結果がない。

raw evidenceはignored `output/cell-mass-1024-20261027/`とhost共通poolに保持する。

## 拡張した実装の回帰検証

source `8c7378dc`。
CPU全体は2006 passed / 3075 skipped / 既存JIT警告18件（42.72秒）。
Ruff・clang-format・差分検査・wheel/sdist buildが通り、
321 runtime source filesが両packageとbyte一致した。

L4 / Torch2.11.0+cu130 / CUDA13.0でCUDA・数学327 passed、failed/skippedは0。
旧128の回帰に加え、1024サイト・257atom・batch5、幅128/256、split16/64、
FP32/64・微分保存有無・double中間値有無、極小中心勾配のAdamW更新を含む。
この327件と、全52,428atomの性能用fixtureの検証は別の範囲。

短い検証job `l4job-f85d7e1167734662ab3928b0cce5260d`:

- source SHA256: `f82dbf2b5851cd3b63ed921ffdae71fb80bac958fac03035e5560b58abb31884`
- result SHA256: `0ed1c94e8c2f8e00be783c96fc988ff629ae0b8157e2903655279e86ce5bf408`
- driver SHA256: `0736f434283264c995319eaa1a2cd7c21bdd55291fef467ab136d9069600c671`
- binary SHA256: `8aa9cc20152661fb54e2104b023063b9c225d97f89ba1201d84ade23377caf68`

manifest11ファイルと321runtime sourceを照合済み。
raw `precision-gates/`に結果・source・binary・archive・verificationを保持する。
再検証はCUDA/nvcc/Ninja環境で
`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。

## 全52,428atomの更新検証と初回完全step

source `8c7378dc`、job `l4job-f2f184fa0de141a79dad88b177fa984e`。
全atomをCPU float64のU/V行列で256atomずつ評価し、初期W、Y/dX、全4列の勾配を確認した。
`double_intermediates=True`では、再計算・保存・幅128/256・split32/64の対象設定で
Parameter更新deltaとstepの差が0。CSTの最大絶対誤差はY1.49e-8、dX2.98e-8、
atom勾配2.91e-11、exp_avg3.64e-12、exp_avg_sq2.78e-17。
固定した許容差を変更していない。

比較用に実行したFalseの4設定は、全て同じatomの更新deltaで1 ULP差となった。
再計算案でも、入力中心勾配の期待値7.2077e-10に対して1.2465e-9となる。
通常の勾配atol内でも更新へ影響するため、これらは時間測定の採用候補から外した。
Trueにすると同じ検証が通り、中間保存の精度が重要だと確認できた。

各設定2fresh process、逆順での再測定、warmup10/同期wall30標本。
次の表はforward/backward/fused AdamWを含む。時間の基準は通常dense。

| N1024/B4 | step ms run0 / run1 | allocated MiB | reserved MiB | workspaceなしdense以下 |
|---|---:|---:|---:|---|
| dense通常 | 0.453 / 0.446 | 32.313 | 42 | 時間の基準 |
| dense workspaceなし | 0.644 / 0.620 | 16.063 | 22 | メモリの基準 |
| 再計算・幅128・split32・128thread | 9.639 / 9.651 | 5.530 | 8 | ○ |
| 微分保存・幅256・split32・128thread | 15.278 / 15.255 | 13.265 | 26 | reserved超過 |

再計算案の時間は通常denseの21.3/21.6倍で、目標の約2倍には未達。
メモリはallocated/reservedとも制約を満たす。
保存案は幅も異なるため、この時間差を「微分保存そのものが遅い」とは解釈しない。
ただしreservedが上限を超えるので、この設定は採用しない。
モデルのatom Parameter本体は約0.80MiBで、dense Wの4MiBの約20%。
atom密度5%は、4列のphysical Parameterを含めるとパラメータ数5%にはならない。

### 重い箇所

同じjobでkernelを別測定したevent中央値。全体wall時間の代わりにはしない。

| 処理 | ms |
|---|---:|
| 検査・key・sort | 0.190 |
| Zx / Zy | 0.066 / 0.066 |
| H / G（微分を含む投影） | 0.717 / 0.839 |
| Y・幅128 / 256 | 3.727 / 2.882 |
| dX・幅128 / 256 | 3.167 / 10.393 |
| Parameter勾配・再計算 / 保存 | 1.428 / 0.029 |

dXの幅256は入力profile tile128を2枚含み、位相評価をCの内側で繰り返すため重い。
Y/dXではatom係数の再生成と、splitごとのsite展開が主な次の削減対象。
[所属ownerのCとbegin/endによる再利用](cell-mass-owner-coefficients.ja.md)を候補として追加する。

### 128の再確認

同じruntimeのN128/A819/B4を独立した子プロセスで再確認した。
入力dtypeの中間保存、微分保存True、幅16/split8/16thread。

| 設定 | ms run0 / run1 | allocated KiB | reserved MiB |
|---|---:|---:|---:|
| dense通常 | 0.462 / 0.454 | 16904.5 | 22 |
| dense workspaceなし | 0.642 / 0.647 | 264.5 | 2 |
| CST微分保存 | 0.838 / 0.836 | 200 | 2 |

時間比は1.81/1.84、メモリもworkspaceなしdense以下。
この128/B4の達成を1024や他batchへ一般化しない。

### 証拠

- source SHA256: `52d7bf0c27cf6166c6d3d6bdaf81e762d1027189b4be6aab08db1e37dedb27bb`
- result SHA256: `646b3cbe3e3c2a25d2903e0dd0f4d8e0a762b5f1b537f8f35257bb80150c2ea5`
- driver SHA256: `5386568d7d27423ea6290ddf843641f23fed91feff7bf93c8214692a0d1acd0e`
- binary SHA256: `1329e7399dd08c3fb654296d4d5e66ab8ad28f4096efbfa6c9f0ddc6b3a3b7fb`

96 manifest files、321 runtime sources、driver、埋め込み128 helper、binaryを照合した。
rawは`output/cell-mass-1024-20261027/step/`。
再実行はそのsource archiveを展開し、srcとrootをPYTHONPATHに設定、
`CST_JOB_OUTPUT`を空directoryに指定して`__pool_driver__.py`を実行する。
所有pool slotは回収後に停止確認した。
