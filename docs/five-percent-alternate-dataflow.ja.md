# atom 密度 5%：小分け重み生成と GEMM の分離

更新後の[学習ステップ基準値と重み再利用の測定](five-percent-training-and-reuse.ja.md)も参照。

2026-09-27。前回の [X 再利用と atom 群](five-percent-atom-group-and-x-reuse.ja.md) の後、5% を主対象に計算経路を変えた。列方向の距離項を共有して論理重みを **1024 出力行ずつ**作り、その都度 cuBLAS GEMM に渡す `triton_streamed` 試作 backend が有望だった。さらに固定siteの箱と陽性 witness を使い、不要な支持域検査を省いた。dispatch は変更していない。これらは forward 専用であり、学習全体の高速化を示す結果ではない。

## 条件と主結果

A100 80GB PCIe MIG 3g.40gb（42 SM）、FP32、TF32 無効、seed 21、入力 128 行、64×64 tile、Triweight、atom 数 `round(N² × 0.05)`。各形状で同じ atom・入力・dense oracle を用い、CUDA Graph 3 round の中央値を比較した。`full` は毎回 `prepare` を含む。出力全体を atol=rtol=3e-5 で照合し、全候補が合格した。生成済み dense W の速度は重み生成を含まないため、主比較には使わない。

| 形状 | 現行 `triton_fused` full | 小分け方式 full | 箱付き full | 箱＋陽性 witness full | 現行比短縮 | 一時重み |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128×4096×4096 | 14.148 ms | 12.460 ms | 12.159 ms | **11.736 ms** | **17.0%** | 16 MiB |
| 128×8192×8192 | 55.859 ms | 49.719 ms | 48.847 ms | **46.763 ms** | **16.3%** | 32 MiB |

CUDA Graph を使わない通常呼び出しの同期 wall time 中央値は、4096² で現行14.266 ms、小分け12.543 ms、箱＋陽性 witness 11.817 ms。8192² で56.109/49.980/46.950 ms。各経路 10 回連続を 3 round、順序を交互にした。箱のない小分け方式と現行方式については、単発 forward 時に増えた PyTorch allocated bytes のピークが4096²で双方58,721,280 byte、8192²で双方231,106,560 byteだった。箱＋陽性 witness 版のピークは別途測っていない。これらは測定runの既存tensorを差し引いた増分であり、学習全体のピークやallocator reserved bytesではない。

生成済みdense重みを掛けるだけの同一run参考値は0.659/3.026 msで、陽性witness付き方式も重み生成に大きな時間を使っている。dense参考値には重み生成とそのメモリ確保を含まない。

`triton_streamed` は `prototypes/block_strip_linear.py` から明示的に選択できる。既定の重み生成 tile は64×32、重み窓は1024行。4D の chart、64×64 tile、行数が64の倍数という条件に限定した forward-only 試作で、勾配が必要な呼び出しは拒否する。重み窓を生成して同じ buffer を使い回し、`torch.mm` で出力の該当列へ直接書く。atom 更新後の CUDA Graph 再実行を含む A100 テスト `tests/test_streamed_materialization.py` は 1 件合格した。これを既定 dispatch に組み込んでいない。

## 固定site箱による準備の改善

現在の支持域分類はatomごとにownerと隣接2 stationを調べる。固定したsiteの4D AABBと、現在のatom中心・現在のprecisionから距離下界を求め、支持半径を確実に超える隣接stationの64列走査を省く。重なる場合は従来どおり厳密な最近行×全列検査へ戻す。ownerは必ず厳密検査する。箱は固定幾何から一度作り、4096²で128 KiB、8192²で512 KiB。今回のfull測定には毎forwardのatom配置準備を含み、固定箱の一度限りの生成は含まない。

初期配置の全atom診断では、除外可能な隣接走査が4096²で1,677,722/1,677,722、8192²で6,710,886/6,710,886だった。箱だけの診断値を速度とは見なしていない。実測の配置準備は4096²で2.770→2.471 ms、8192²で11.348→10.478 ms。両形状とも準備済みのatom配列・offsetsが従来方式と完全一致した。境界共有、seam、idle、atom移動とCUDA Graph更新を含むA100テスト `tests/test_support_box_routing.py` も合格した。箱付き準備は `prototypes/support_box_routing.py` の `boxed_prepare` を明示的に呼び、`triton_streamed` に `prepared` として渡す試作である。

## owner の陽性 witness

balanced 初期化時の各atomの列番号を固定hintとして保存する。owner決定カーネルが既に求める行候補と組み合わせ、その1 siteへの距離が支持半径内で十分余裕を持つとき、ownerの64行最近点探索と64列支持検査をともに省く。陽性を証明できない場合は従来の厳密検査へ戻す。毎forward現在のatom中心とprecisionを読むため、atomが移動しても古いhintによる取りこぼしは起きない。初期位置を保証するためにhintを使うのではなく、単なる候補列として使う。固定したsiteの箱は幾何を変更した場合に再生成が必要である。

同一runで、箱付き準備2.473→陽性witness付き2.044 ms（4096²）、10.473→8.373 ms（8192²）。陽性witnessは箱付き方式からfullを3.5%/4.3%短縮した。準備済みatom配列とoffsetsは現行方式に完全一致、dense oracleとの全出力検査は両形状で違反0。境界共有・seam・idle・atom移動・CUDA Graph更新を含むA100テスト2件も通過した。固定hintの追加領域は4096²で約3.2 MiB、8192²で約12.8 MiB。箱は128/512 KiB。箱とhintの初回生成はtimed forwardに含めていない。明示的な試作呼び出しは `boxed_prepare(site, p, boxes=boxes, witness_cols=hints, fast_witness=True)` である。

## 経路の内訳とメモリの調整

5% で現在の fused kernel は 4096² の prepared forward のうち約11.4 ms を占め、`support_buckets_batched` は約2.0 ms。fused kernel は167 registers/thread、spill なし。論理重み生成を別 kernel に分けると、列座標だけで決まる距離の z/w 項を atom×列について一度だけ評価でき、X の再利用は cuBLAS に任せられる。

最初の全重み materialization（32×32タイル）は生成のみ 9.276/36.955 ms（4096²/8192²）、cuBLAS を含む prepared forward は 9.969/39.972 ms、full は 12.743/51.336 ms。全重みの一時領域は64/256 MiB。小分け方式なら 1024 行で16/32 MiB、512 行では8/16 MiBとなる。512 行の full は13.066/50.567 msで、メモリを半分にすると速度が少し落ちる。64行まで細かくすると launch/GEMM の繰り返しが増え、4096² の prepared で14.481 msとなった。

重み生成の tile を32×32から64×32へ変えると、全重み prepared の生成＋cuBLAS は 4096² で9.942→9.560 ms、8192² で39.883→38.463 ms。小分け full でも、4096² の1024行では32×32が12.850 ms、64×32が12.461 ms、64×64が12.790 ms。8192² では51.284/49.883/49.872 ms。64×64の単独生成は速いが register 使用が168/thread まで上がり、4096² の小分け full では64×32より遅いため64×32を試作の既定値にした。64×32は96 registers/thread、spillなし。

## 別案の測定

- 8 atom を近接順に並べた理想群で site の箱との距離下界を判定すると、4096² prepared kernel は現行32行11.447 ms、群内分岐11.195 ms、事前CSR候補列挙10.271 ms。全出力は合格。ただし群作成がCPUで10.77秒、CSR作成が0.156秒かかり、timed forward から除外した。CSRは候補群の77.45%を保持した。このままでは full forward に採れない。
- BF16 を high/low に分けた3項・4項の Tensor Core dot は正しさに合格したが、4096² full は14.971/14.983 msで現行14.166 msより遅い。実験 kernel は取り除いた。
- fused kernel の batch tile BM を128から64/32へ減らすと、4096² full は14.215→23.205/41.465 msに悪化した。8 warps も16.832 msで遅い。これは別資料の出力行 tile BN=32 の選択とは別の sweep。
- Torus 上の chord から support row を予測し近傍だけ検査する準備案は、出力と準備済み tensor が一致したが、4096² の準備時間が2.779→6.111 ms、full が14.220→17.554 msに悪化した。実験コードは履歴に残し、現行実装から除いた。
- 極座標chordで距離項を行と列へ分ける重み生成は、4096² prepared の生成＋cuBLAS が現行9.552→11.226 msに悪化し、出力の相対L2誤差が約1.97e-4へ増えた。円方向への射影式は14.124 msまで遅く、全出力488箇所が許容差を超えた。大きなTorus半径のFP32丸めが問題になるため、どちらも試作履歴に残して現行コードから除いた。
- 重み生成64×32 tileに固定site箱を付けてatom距離計算を省く案は、4096²で生成単体8.893→9.956 ms、準備済みforward9.679→10.714 msに悪化した。正しさは合格したが、この構成には採らず実験commit `0f11dca` をrevertした。

## 再現

試作 backend とテストは commit `b589248`、最終タイル比較は `b93bef4`。隔離した git archive の SHA256 はそれぞれ `591a63ad31810e570677a219b80258f0c617992cb489b88354b3506ca6adc149` と `6d13e18e018aa2a7967e071faaf89ed2801bf25bbfa03dd8bcf12199ca7df000`。いずれもA100側で一致を確認した。結果 JSON は gitignored の `output/triton-a100-20260927/` にある。

箱付き準備の境界・更新テストは commit `1a5a389`、最終同一run比較は `d30be48`。git archive SHA256 は `5fc30970e81efc0ebdcf9a5f0ed282692289be09a9cb117f35ccb64a070050a8` / `77b82b057f6b6730b4365fe1c3878533fbd4c64d127192552c4fc4fff6e5a913` で、A100側と一致した。CUDAテストは2件合格。

陽性witnessの現行最終比較は commit `bbc9ccb`、git archive SHA256 は `f3bf0ada136548d4b631b4cc249aa804c46d3bdcfb36b9466bab5735833db451` でA100側と一致した。現行経路を残したまま箱付き試作を明示的に選択した。A100で2件のCUDAテストが合格した。

| JSON | SHA256 |
| --- | --- |
| `profile-4096.json` | `e0153e4962e7cc1d89b9e58fecae044987b95a43c6ff83af5daf1c1bca599722` |
| `bf16-4096.json` | `db8b3afa2adb6a71f5280a8043340d9d81b1ee8ad510a297685ea0db499ef70d` |
| `warps-4096.json` | `caf56a29b737a2afa5998dfae9295ebe6a7eb36a4260ae09215e093ba4e00ea8` |
| `chord-4096.json` | `2e2a8ff59ac4746118c0f4eb44a30948c685690a98aa1e91dbc8d984f9de2291` |
| `grouped-csr-4096.json` | `3b8fbeb230c8297087fd3f7febed3b66ec1f4c2116e5ec8033a333fd1e112977` |
| `materialize-4096.json` / `materialize-8192.json` | `df0f26d141104343702887766b923973d4f9b8c9c38d6028b8a5b1a7588c22dc` / `c87d8d06bf4d2128caac09ae3fcd919c64ea14c001695b00c922b985ab230407` |
| `factored-sweep-4096.json` / `factored-sweep-8192.json` | `f7f8e967a10aa16ee7198fe3addd9bce1e7a8e2d98d06305d2f86cffb4efdb15` / `a3e99e09f689c919dc58ffbad1a00a9edd34649221e3417435176c1aa5390173` |
| `streamed-tile-4096.json` / `streamed-tile-8192.json` | `322671c5619be7f44bb66e52c8c353765d704e52e5fbe36264b88685f40c6f48` / `2b90da578279adea6fcc3d6e8d8e585bc47b99d7114a76b8b2e71405d5397c63` |
| `neighbor-box-4096.json` / `neighbor-box-8192.json` | `8bec6954248fe1d3e31e88c65947874f7d6699d315411ca375e3d2caf9d7e287` / `edf5e73f773a410d4fa56b4988ea5e000444f7a3085b183faad6c3dc5c551c6d` |
| `polar-4096.json` / `projected-4096.json` | `88f590c308e44bbcca8480057246db7fd431e0afb010f0450ae1407a83c2e1ce` / `cdf33248ce62510ee1972b8b3f680e5de0a03ac48a45593eb198b874631a2b37` |
| `box-final-4096.json` / `box-final-8192.json` | `6b9cfde45c6587fc8f8a45a9328e9de59bba0a0533cc54354b26d7a5aec5163d` / `a191723f3c3ca586aff75e6edc03ebbecb77d703276f5aad5233355379bf639d` |
| `weight-box-4096.json` | `3a752d87b9ee6d02d9ae459f28d81d8786fb4b4270a3775233bfe4337382e991` |
| `final-witness-4096.json` / `final-witness-8192.json` | `bed923751841a2a460d733685e2cdd14102a047fa762942ff23aedaa18ca8f2c` / `fab4a7b084fea6876a9461d24f9e2081e7ce30fa4f83378cc003200555f9f02f` |

隔離 checkout で次を実行する。最後の pytest は CUDA/A100 が必要。

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_streamed_materialize \
  --size 4096 --eager --source-commit b93bef4 --output streamed-tile-4096.json
PYTHONPATH=src:. python -m prototypes.benchmark_streamed_materialize \
  --size 8192 --eager --source-commit b93bef4 --output streamed-tile-8192.json
PYTHONPATH=src:.:tests python -m pytest -q tests/test_streamed_materialization.py
PYTHONPATH=src:. python -m prototypes.benchmark_box_support \
  --size 4096 --eager --source-commit d30be48 --output box-final-4096.json
PYTHONPATH=src:. python -m prototypes.benchmark_box_support \
  --size 8192 --eager --source-commit d30be48 --output box-final-8192.json
PYTHONPATH=src:.:tests python -m pytest -q tests/test_support_box_routing.py
PYTHONPATH=src:. python -m prototypes.benchmark_witness_support \
  --size 4096 --source-commit bbc9ccb --output final-witness-4096.json
PYTHONPATH=src:. python -m prototypes.benchmark_witness_support \
  --size 8192 --source-commit bbc9ccb --output final-witness-8192.json
```

次の技術課題は backward と学習時ピーク、学習後に移動した atom の分布、長い系列で重み窓を使い回せる条件の確認。CSR案を採るなら群と候補を更新するGPU処理の総時間を full forward に含める必要がある。
