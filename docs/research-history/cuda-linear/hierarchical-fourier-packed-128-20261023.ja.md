# N128: live atom packing とグループ一括縮約

2026-10-23。`kernel/hierarchical-fourier-tiles` の研究候補。
公開dispatcherへの採用ではない。

## 実装と比較範囲

支持の階層集合・使用bank集合・local/direct/shared経路で分類し、入力支持の
開始位置、出力支持の開始位置、元atom IDの順に並べる。同分類の最大4 atomを
1グループとする。N128/M8/A<=128専用の1 CTAで分類、sort、P/routesの物理配置、
group開始表を作る。係数・正規化・cacheは並べたPから一度だけ生成する。
group数はdeviceに残し、固定worker数の1 launchで全groupのHを縮約する。
出力側のlocal tile参照表は既存経路を使う。Z生成は既存の階層依存順を保つ。

Parameterとoptimizerの並びは変えない。packed row→元atom IDを保存し、
backwardで全4 fieldの勾配を元の順序へ戻す。元X/Pのversion検査も維持する。
学習による支持・階層変更は毎回GPUで再分類し、Graph replayでも更新する。

小さいlocal atomのHは既存kernel内で消費する。一方、今回は比較を切り分けるため
非局所Hの割当 `[B,A,3]` とNative cache `[A,2,3,128]` を据え置く。
N128/A128では後者だけで768 KiB、dense W FP32の64 KiBを超える。
したがって今回の候補だけでは構造的なメモリ目標を満たさない。
追加のpacked P/routes、逆順表、group表も完全stepピークの比較に含める。
微小なH scratchを強制して多段のwaveに分けることはせず、必要量を測って決める。

## 検証結果

- CPU全suite: 2260 passed / 5744 skipped。CUDAのskipはGPU検証に数えない。
- live分類、A7/B5端数、retained backward、元入力のversion検査を追加。
- 既存local経路の回帰と新候補のGPU tests: **13 passed** (300.92秒)。
- 既存Linear runnerのN128/B32/A128/M8 mixed seed41で、direct、従来local、
  packingのみ、grouped serial/overlapを同じ独立oracle・完全step条件で比較した。
- 6 correctness / 6 measurement / 1 Dense workerが全PASS。
  Y/dX/全atom勾配、retained backward、動的支持のGraph replayを通過。
  新候補3種の最大絶対誤差はY <= 4.24e-7、dX <= 4.72e-7、dP <= 4.94e-6。
  閾値は既存のatol=rtol=4e-4を維持。

初期配置はGPUでも39 group。packingで追加した5 tensorのpayloadは合計5640 B。
Graph内でgroup数をCPUへ読み出さず、幅・中心の変化後も正しく再配置した。
この実装のG番号は階層bitmaskの数値順なので、先行の説明図のG番号と同一視しない。

| Plan | 完全step Graph median ms | Eager median ms | Peak allocated B | Peak reserved B |
| --- | ---: | ---: | ---: | ---: |
| direct-samples-h | 0.742368 | 1.776640 | 1205248 | 6291456 |
| local-serial8 | 1.165312 | 3.777440 | 1522688 | 6291456 |
| local-overlap8 | 1.126400 | 3.472384 | 1522688 | 6291456 |
| pack-only | 1.062912 | 3.556352 | 1529344 | 6291456 |
| packed-serial32 | 1.064960 | 3.535872 | 1529344 | 6291456 |
| packed-overlap32 | 1.036288 | 3.568640 | 1529344 | 6291456 |
| dense | 0.043008 | 0.859136 | 34408960 | 48234496 |

N128/B32/A128/M8 mixed seed41、同一X/target/P、FP32、TF32無効、
AdamW fused/capturable、warmup2・7 replayの中央値。完全stepは準備、
forward/backward、optimizerを含み、ピークはGraph capture/replay込み。
独立runは1回であり、他shape・GPUへの一般化やdispatcher採用の証拠ではない。

従来local-overlap8に対してGraphでは8.0%短縮したが、Eagerでは約2.8%悪化。
packingのみでもserial経路比約8.8%短縮し、group境界を守るpersistent contractionの
serial結果はpackingのみとほぼ同じだった。**今回の測定から、group一括縮約そのものの
追加効果や、SMでの同時実行を実証したとは言わない。** profilingは追加していない。
direct/denseには未達。allocated増分は従来local比6656 Bで、Native容量は据え置き。
Denseの完全stepピークに含まれるライブラリworkspaceを根拠に、構造的なメモリ目標を
達成したとは扱わない。局所計算開始前の共通準備も残る。

profile cacheの保存対象・容量とHのvalue/gradient保持、準備の分岐位置は次の別比較。
この候補は研究branchに保持する。

## 再現と証拠

- runtime source checkpoint: `3327f8b3`。
- job: `l4job-4ea3735ace71444f9e227e04372946a3`。
- source archive SHA256: `bb5dc6e1563520492a31eb48bb7240ecdc51d9a83ed91973cfef7d2e7e2da578`。
- result archive SHA256: `ba76d4d8678b4c4e5e1bb89940b89a25fe4e4d464927740d7aca1cb654d9ef33`。
- NVIDIA L4 / 58 SM / UUID `GPU-7e00540b-c067-6950-b251-9962efc24871`、driver580.82.07。
- Python3.13.15、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。
- raw: worktree内 `output/hierarchical-fourier-packed-20261023/` と
  `~/.local/state/colab-l4-pool/jobs/<job>/results/artifacts/`。
  archive、全driver artifact、全13 workerの5291 source file照合を通過。
- pool slotは全stopped、serve終了、Colab serverにもactive session無しを確認。
  待機job `l4job-003cd438754042dd86fac2691f552675` はcommit引数を直すため実行前に取消。

```bash
python3 output/hierarchical-fourier-packed-20261023/collect.py l4job-4ea3735ace71444f9e227e04372946a3
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-packed-128.json \
  --case benchmarks/cuda/linear/cases/packed-fourier-m8-mixed-128-a128.json \
  --source-commit 3327f8b3 --output output/packed-128-repeat.json
```

GPU実行は共有pool経由で行う。再現driverとcollectorもignored raw内に保存済み。

初回job `l4job-c36cc4846ca54127901d134680c18df1` (source `dee32f19`) は
コンパイル段階で失敗。既存JIT内の`first`との変数名衝突とassociative_scanの
引数指定を修正した。10 failed / 3 passedで、性能値は取得していない。
