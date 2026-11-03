# owner 窓共有・標本化 fold の実装検証（2026-11-03）

対象は `main` の `4a1b7a4a09e4908cb9273fbbf1c617a066f89bf5`。
参照提案は ChatGPT 会話「高速化設計の提案」
（`6ae8ac4a-79b0-83ec-9bf3-a32bfde47123`）。
研究 branch は `kernel/owner-window-fold`、worktree は
`/Users/ware10sai/.codex/worktrees/owner-window-fold/torchcst`。
検証済みの明示的 Plan を PR で統合する。公開 auto と既存 Plan の既定選択は維持する。

## 数学的契約

atom の学習 Parameter は従来の4列のまま。Polar の radius は task 勾配から detach する。
正規化、cell integration、中心の物理単位、支持の分類は従来と同じ。
M=8 を減らす近似ではなく、有限格子上で同じ基底になる周波数を合算する。

bank の窓長を L=KT サイト、セル間隔を Δ、物理周期を P=LΔ とする。
atom ごとに分類した **unwrapped** first tile からの中心変位を δ とすると、

$$
a_{k,c}=2c_k\Delta\operatorname{sinc}(\pi k/L)\cos(2\pi k\delta/P),\qquad
 a_{k,s}=2c_k\Delta\operatorname{sinc}(\pi k/L)\sin(2\pi k\delta/P).
$$

ここで sinc(x)=sin(x)/x。セル積分をこの係数に一度だけ含める。
窓の共有 moment は積分していない標本値の Fourier moment とする。

$$
Z_0=\sum_j x_j,\quad Z_{k,c}=\sum_j x_j\cos(2\pi kj/L),\quad
Z_{k,s}=\sum_j x_j\sin(2\pi kj/L).
$$

k mod L を使い、L/2 より大きい周波数では sine の符号を反転して fold する。
偶数 L の Nyquist sine は存在しない。DC と Nyquist cosine は一つずつ保持する。
実基底の数は r=min(L,2M+1)。L=12 では r=12 となる。
値と中心微分の係数は **元の k でそれぞれ生成してから** fold する。

$$
\partial_\mu a_{k,c}=-(2\pi k/P)a_{k,s},\qquad
\partial_\mu a_{k,s}=(2\pi k/P)a_{k,c}.
$$

H=Z_in・q_in、G=Z_dY・q_out と書けば、

$$
\partial_{\alpha_a}\mathcal L=\sum_b H_{ba}G_{ba},\quad
\partial_{\mu_a^{in}}\mathcal L=\alpha_a\sum_b H_{\mu,ba}G_{ba},\quad
\partial_{\mu_a^{out}}\mathcal L=\alpha_a\sum_b H_{ba}G_{\mu,ba}.
$$

その後、従来の `store_gradient` で Polar 振幅座標へ chain rule を適用する。
振幅0の atom でも dα を計算し、振幅で割り戻さない。

## 比較候補

| Plan の末尾 | 内容 |
| --- | --- |
| compact-keyed | 現行 baseline |
| compact-fused | 既存の complete G/DG を reverse owner CTA 内で直接 dP に使う。joint backward では G/DG を保存しない。atom-only は旧経路 |
| window17 | 窓全体17成分を共有。H/DH 保存、G/DG と dP は融合 |
| window-fold | 窓全体 r 成分を共有。H/DH 保存、G/DG と dP は融合 |
| window-recompute | folded 入力窓を保存し、reverse CTA の shared memory から H/DH を再計算。H/DH/G/DG の global 配列なし |

明示的 Plan のみ。適合条件は従来 owner と同じ N1024、同一 Polar M8/base1/merge2 軸、B<=4。
N128 や大バッチ用の公開経路へ一般化していない。
raw hierarchy と完成窓 moment を分け、親への merge は従来の raw hierarchy のみを使う。
長い窓では既存 tile moment を回転して合算し、短い窓では共有変換をセルから作る。
出力は窓ごとに係数を縮約してから一度展開する。chart 周期を超える窓の重複像も合算する。

## CPU の独立検証

`tests/test_radius_owner_window.py` は実際の CUDA header の係数 helper を C++ で取り出して実行する。
独立な全 M8 harmonic のセル直接評価と中心微分に照合。
N=8/32/128/1024、spacing=.125/.37/2、origin=0/-1.2、全 bank、first tile=-1/0/last、
異なる中心変位を確認した。
428,400 セルで値の最大絶対差 `4.651834473179406e-14`、
中心微分 `3.4927616354707425e-13`。
固定容量は全 bank で23,902 field、B4/FP64で764,864 byte。
これは係数領域の容量であり、GPU peak allocated ではない。

## 再現と証拠

既存 runner と case を使用する（別の測定 runner は追加しない）。

```bash
PYTHONPATH=src python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-initial.json
PYTHONPATH=src python -m pytest -q tests/test_radius_owner_fused.py tests/test_radius_owner_window.py
PYTHONPATH=src python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-radius-cell-mass.json \
  --case benchmarks/cuda/linear/cases/radius-cell-mass-1024-owner-window-initial.json \
  --polar-update radius --phase-diagnostics --output output/owner-window-fold/benchmark.json
```

GPU は default host-wide Colab L4 pool を使う。
ignored driver と補助ログの保全先は `output/owner-window-fold/`。
ジョブ原本は `~/.local/state/colab-l4-pool/jobs/<job-id>/` に残す。

## GPU 検証と初回の完全 step

L4 / Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0 / Python 3.13.16。
N1024、A52,428、B4、FP32 Parameter/入出力、FP64 内部、TF32なし。
各 Plan は fresh process、初期値・入力・target・AdamW 契約を一致させた。
10 warmup 後の51個の同期 eager 完全 step の中央値。
51 sample は51独立 run ではない。capture/replay は既存 radius と同じく非対応で、Graph は未測定。

| Plan | 完全 step ms | peak allocated MiB | peak reserved MiB |
| --- | ---: | ---: | ---: |
| compact-keyed | 2.572036 | 12.691406 | 28 |
| compact-fused | 2.553971 | 10.291504 | 28 |
| window17 | 2.454141 | 12.371094 | 28 |
| window-fold | 2.444905 | 9.582031 | 30 |
| window-recompute | 2.443295 | 7.111328 | 8 |
| dense | 0.702606 | 32.329590 | 42 |

`window-recompute` はこの run で時間5.005%減、allocated43.967%減。
単独層の完全学習 step であり、MNISTモデル全体や3 epoch の結果ではない。
dense に対する所要時間は約3.48倍であり、dense と同等の速度には達していない。
fold だけと recompute の時間差は0.1%未満で、この一回では順位を付けない。
G/DG の融合だけの0.7%差も初回値として扱う。

268 GPU/宣言テストが PASS、1 CPU helper test は Colab に clang++ がないため skip。
同 helper は Mac で428,400セルの実コード検証を通過済み。
FP32/FP64、B=0/1/3/4、joint/input-only/atom-only、境界/重複像、129 atom の owner 偏り、
retained backward、保存TensorのCPU offload、public hook/lifetime/version/二階微分禁止を確認。
融合のみの候補では3 AdamW step の Y/dX/dP/Parameter/moments/step が全48条件で byte 一致。
窓版の一般の契約は独立 FP64 oracle への数値照合であり、全条件の byte 一致は主張しない。

既存 runner では全52,428 atom の初期/更新後 Y/dX/全 dP、独立 polynomial quadrature の小集合、
3回の matched optimizer step、CSTAdamW と既存 wrapper の同GPU byte gate を全候補で通過。
初期/更新後 full-size oracle 誤差の最大値は Y `6.64e-9`、dX `1.83e-9`、全 dP `1.19e-7`。
更新後は51+warmupの短窓であり、自由走行の長期学習同等性を意味しない。
全候補の主な初期/短窓後 bank は T4/K3。広い bank と owner 偏りの正しさは小fixtureで確認したが、
それらの分布の性能は測っていない。

ローカル標準 CPU suite は `1704 passed, 2117 skipped`、wheel/sdist build と lint/declaration check は PASS。
GPU/DB の CPU skip を GPU/DB 検証済みとは扱わない。

### 工学的な失敗と修正

- `l4job-1c25eeb6069d47f9aed7e7ef11560549`: job-local ninja の bin が PATH にないためビルド未実施。探索パスを修正。
- `l4job-11f0f736899046bfb31797bf331d8c83` と `l4job-092b124f2a554a168d3bdcba54c4185c`: AT_DISPATCH 内のマクロと host/CUDA translation-unit 境界のコンパイル不備。generic lambda と CUDA 側検証関数へ修正。
- `l4job-ff73e25ff1d44fa19851f3f7a7935b06`: 独立VJP48条件は通過。空バッチの融合 dP を zeros_like で返したことで -0/+0 の byte 差を検出。空バッチは従来の chain rule の kernel を使用して修正。性能値はない。

これらを性能の負の結果として数えない。原本は pool job directory に保持する。

### 初回 job/hash

- Job: `l4job-f1054799f0134ca08a5595faa1563361`
- Source archive SHA256: `12c879029d248371851c9461c1b76c2b72227c8379579dd32848524289015638`
- Result archive SHA256: `ddb25f315eb5248f4d78e666859df69e452e1206bc1c221653cbf4298a757201`

## 独立再測定と判断

別の L4 VM の fresh run でも同じ runtime source、入力、最適化条件を使用した。
source archive は driver の違いにより別 hash。各 runtime ファイルの SHA256 は両 job と最終ソースで一致する。

| Plan | run 1 ms | run 2 ms | run 2 allocated MiB | run 2 reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| compact-keyed | 2.572036 | 2.565521 | 12.691406 | 28 |
| compact-fused | 2.553971 | 2.559356 | 10.291504 | 28 |
| window17 | 2.454141 | 2.488834 | 12.371094 | 28 |
| window-fold | 2.444905 | 2.481155 | 9.582031 | 30 |
| window-recompute | 2.443295 | 2.453196 | 7.111328 | 8 |

全候補の full-size Y/dX/dP、quadrature、matched optimizer と同GPU wrapper byte gate が再度 PASS。
全候補の allocated/reserved は初回と一致した。
`window-recompute` は完全stepで **4.38–5.01%短縮**（約1.046–1.053倍）、
peak allocated **43.97%減**、reserved 28→8MiB。
速度改善は窓共有の段階に大きく現れ、圧縮と再計算の効果はメモリ側が大きい。
G/DG融合だけの時間差は0.24–0.70%と小さく、単独の強い速度改善とは扱わない。

数学的な fold と現在の radius 実装への接続は成立し、この限定条件では時間とメモリの両方が改善した。
2026-11-03 の所有者の指示を受け、検証済みの明示的 Plan と比較経路を PR から main へ統合する。
公開 auto と既存 Plan の既定への適用は今回の統合範囲に含めない。
今回の問いを超える追加最適化や長期学習は実施していない。
B300 の一時W/GEMM、N128、小規模/広幅/集中分布の性能、長期MNIST学習、Graph は未検証。
B>4 と N!=1024 はこの新 Plan の適合条件外で、未対応を高速化済みとはしない。

### コンパイル資源

二回目の実際の共有ライブラリを cuobjdump `--dump-resource-usage` で検査した。
FP32/FP64 とも、既存 compact CTA / 融合 CTA の REG は136のまま、STACK は704→784 byte。
窓 CTA の REG は forward/saveH=128、forward/recompute=106、reverse/saveH=107、reverse/recompute=125。
STACK は1056–1088 byte、LOCAL フィールドは0。
動的 shared scratch は旧34,432 byte、窓22,048 byte。
STACK が残るため、LOCAL=0を根拠に局所メモリ交通やspillが皆無とは主張しない。
命令単位の spill/L2/DRAM profile は未取得。
`cuda-resources.txt` と実際の native binary SHA256 を raw evidence に保持する。

### 再測定 job/hash と保全

- Job: `l4job-73a33bf0090e4256ad64fe189f03d989`
- Source archive SHA256: `f5b29b6ed46e6f9864582057362b37f2323224f930f5a778e6495654ab0ef18a`
- Result archive SHA256: `790de9b79e505a7fedb419d48c238b5f26a212e20a6ade5892103bd93b5be46f`
- 2独立job。各51 timing sample、fresh correctness/measure workers。異なる seed の統計や長期学習ではない。
- 2jobの spec/source/result/receipt、全 worker JSON、全ログ、資源記録を
  `output/owner-window-fold/<job-id>/` に複製して保全。原本は pool job directory にも保持。
- `run1-summary.json` / `run2-summary.json` は raw JSON から抜き出した補助表。
- pool の source/result archive を SHA256 で再照合し、runtime source の一致も検査する。



所有 L4 runtime は pool の通常 cleanup/idle shutdown を完了し、最終 status で全 slot が stopped。
GPU process usage の計測値はなく、上記 allocated/reserved と区別する。
