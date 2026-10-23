# N128: profile保存容量と再計算の比較

2026-10-23。`kernel/hierarchical-fourier-tiles` の研究候補。
直前の[packed実験](hierarchical-fourier-packed-128-20261023.ja.md)から、
cache容量だけを分離して比較する。公開dispatcherへは未採用。

## 配置と寿命

ここでいうNative/profile cacheは、準備kernelがglobal memoryへ明示的に保存する配列。
ハードウェアのL1/L2、block内shared memoryとは別物である。
この比較でL2 hit率・実DRAM転送量・SM同時実行は測らない。

| 値 | 保持場所・寿命 | 今回の扱い |
| --- | --- | --- |
| Meta/係数/Norm | global、forwardまたはbackward呼び出し内 | 全atomの正規化・支持を維持 |
| Native | global、準備から局所出力/縮約/残差出力の最終利用まで | 軸ごとの支持が容量以下のときだけ保存 |
| 局所H | local output kernelの作業値、出力へ直ちに消費 | globalへのstoreなし、既存kernelを維持 |
| 非局所H | global `[B,A,3]`、そのforward内で消費 | 容量は据え置き |
| Z/C | global、階層/複数blockで利用 | 必要nodeと既存の階層依存を維持 |
| saved X/P/routes/順列表 | backwardまで | 同じforward状態で再計算し、元atom IDへ勾配を戻す |

Nativeの形状は`[A,2,3,S]`、FP64。3 fieldは値・中心微分・幅微分である。
S=128/64/32で、A128なら768/384/192 KiBとなる。矩形容量を確保するため、
実際に保存しないatomのslotも残る。atomごとの可変長圧縮やsharedへの移動は今回含めない。

Normはすべての観測siteについて混合後profileの離散L2とその微分を計算する。
Nativeを保存できない軸はBoundsに`-1`を記録し、既存の完全な係数評価へ戻る。
共有bankの完全タイルはZから、direct成分と境界はprofileから計算する。
これはcacheの保存上限であり、支持を32/64 siteで打ち切る変更ではない。
保存できない残差cacheの生成も省く。幅・中心が更新されたら毎回判定し直す。

旧Packed/Local/Fourier Recipeのschemaを変えず、独立Recipe/Algorithmとして登録する。
同じP/routesの並べ替え、32 group worker、8 local worker、stream分岐位置を保つ。
Hのstride変更、forward微分cache省略、normの係数化、FP32 cache化は混ぜない。

## 検証

新たにA7/B3で空支持、ゼロ振幅、domain端、floor付近、cache容量を超える広い支持、
幅を交換して保存対象が入れ替わる場合のY/dX/全P勾配、retained backwardを検査する。
既存独立FP64 quadrature/KKT oracleと従来cache128経路の双方を対照とする。
既存runnerでN128/B32/A128/M8 mixed seed41、direct/cache128/cache64/cache32/Denseを
同じ完全step条件で比較し、Graph replayによる動的な支持変更も確認する。

CPU全suiteは2261 passed / 5746 skipped / 18 warnings (52.37秒)。宣言検査、Ruff、
diff検査もPASS。GPU追加testsは3 passed (249.30秒)、既存runnerの
4 correctness / 4 measurement / 1 Dense workerは全PASS。
Y/dX/全P勾配、retained backward、変更後Graph replayを従来のatol=rtol=4e-4で検証した。
新候補の最大絶対誤差はY <= 6.03e-7、dX <= 4.34e-7、dP <= 4.94e-6。
別診断でMeta/Norm/atom順序がcache128とbitwise一致し、保存対象が容量内の軸と
正確に一致することを初期状態と幅・中心変更後の双方で確認した。

## 実測結果と判定

N128/B32/A128/M8 mixed seed41、FP32、TF32無効、AdamW fused/capturable、
warmup2・7 replayの中央値。完全stepは準備/forward/backward/optimizer込み。
ピークはGraph capture/replay込みで、oracle/同期診断は別process・別測定。
独立jobは1回。小さい改善の採用判断をする比較ではなく、保存容量を減らした負の結果。

| Plan | Native KiB | Graph median ms | Eager median ms | Peak allocated B | Peak reserved B |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct-samples-h | 768 | 0.744448 | 1.726464 | 1205248 | 6291456 |
| packed-overlap32 (cache128) | 768 | 1.037312 | 3.621888 | 1529344 | 6291456 |
| cache64-overlap32 | 384 | 1.501184 | 3.599360 | 1136128 | 6291456 |
| cache32-overlap32 | 192 | 1.605632 | 3.719168 | 939520 | 6291456 |
| dense | — | 0.043008 | 0.746496 | 34408960 | 48234496 |

cache32はcache128よりNativeを75%、完全step allocatedを約38.6%減らしたが、
Graph時間は約54.8%悪化。cache64もGraph時間が約44.7%悪化した。
reservedは全CST候補で同じ。Eagerのcache64の微小差を速度改善とは採用しない。
このshapeの高速化候補として容量上限だけを32/64へ下げる案は**不採用**。
省メモリ対照として実装・測定を研究branchに保持する。

初期状態の保存対象は以下。数はatom×片側軸、出力/入力を別に数える。

| 容量 | direct cache 出力/入力 | 境界cache 出力/入力 | 保存なし 出力/入力 | 局所atom |
| --- | --- | --- | --- | ---: |
| 128 | 105 / 105 | 23 / 23 | 0 / 0 | 31 |
| 64 | 82 / 81 | 0 / 0 | 46 / 47 | 31 |
| 32 | 60 / 61 | 0 / 0 | 68 / 67 | 31 |

幅logit+0.31、中心+0.071の診断ではgroup数39→41、局所atom31→24へ変化。
cache32の保存なしは出力77/入力75、cache64では両側55となった。
局所の所属は3容量で一致する。正規化の計算対象や分類の固定化による差ではない。

現在の上限は**残差の非ゼロsite数ではなくprofile全支持のspan**で判定するため、
初期23 atomの境界cacheが32/64版で全て失われた。
その成分はZの完全タイルを使い続けるが、残差・direct profileはconsumerで再評価する。
従来もNativeの未使用末尾を毎回読み書きしていたわけではなく、確保容量と実転送量は
同じではない。今回の実測だけでDRAM/L2/occupancyのどれが律速かは確定しない。

Nativeだけでもcache32は192 KiB、dense W128² FP32の64 KiBを超える。
準備payloadは1028200→438376 Bへ減ったが、構造的な密行列以下の目標は未達。
Dense全stepのライブラリworkspaceを根拠にこの目標を達成したとは扱わない。

## 次の設計に残すこと

支持幅に加え、再利用回数、値/微分ごとのconsumer、残差を実際に保存する長さ、
その場で計算して使い切れる範囲を選別に使う。
Normで既に計算されるprofileを保存するかの目安は、共通のNorm費用を除くと、

\[
\Delta T_a\simeq r_a(C_{\mathrm{eval},a}-C_{\mathrm{read},a})
 -C_{\mathrm{write},a}-C_{\mathrm{extra\ prep},a}.
\]

これは設計上の近似であり、各費用・rを今回個別測定した式ではない。
起動、集約、並列性、register/shared容量も含めた完全stepで最終判断する。
次は再利用の多い広いprofileを一律に捨てる前に、未使用slot・残差の穴・不要な
保持fieldを詰め、実際に使う値を保持したまま確保容量を減らす候補を検討する。
局所Hのglobal保存は引き続き不要で、非局所Hの容量は今回据え置き。

## 証拠と再現

- runtime checkpoint: `656c1c7f`、job `l4job-1ae2bfbaec6a48e3bf6e38bffc9648f7`。
- source archive SHA256: `55ae2145111916ed0f4e4b6056a5660bb0cbd089764e6865e98706f8254ea5be`。
- result archive SHA256: `fc9629db32594b7b0d82ccd373ac4754cb8a43271c92cd6ec4fdffb6811a4e22`。
- NVIDIA L4 / 58 SM / UUID `GPU-5ba1880a-5114-45ab-fe98-fbe0910cb714` / driver580.82.07。
- Python3.13.15、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。
- archiveと全artifact、9 workerの3663 source file照合を通過。
- raw: worktree内 `output/hierarchical-fourier-cache-20261023/` と
  `~/.local/state/colab-l4-pool/jobs/<job>/results/artifacts/`。
- pool全slot stopped、serve終了、Colab server active session無しを確認。

```bash
python3 output/hierarchical-fourier-cache-20261023/collect.py l4job-1ae2bfbaec6a48e3bf6e38bffc9648f7
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-cache-128.json \
  --case benchmarks/cuda/linear/cases/cache-fourier-m8-mixed-128-a128.json \
  --source-commit 656c1c7f --output output/cache-128-repeat.json
```

GPU実行は共有poolを使う。driver/collectorもignored rawに保存済み。
